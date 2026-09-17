#!/usr/bin/env python3
"""Controlled nonlinear-readout experiments for SmolLM2 and Qwen3.5.

Run with --help. Immutable revisions, document-disjoint splits, resumable adapter
training, validation-only selection, and full-model cached timing are mandatory.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import logging
import importlib.util
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch
import torch.nn.functional as F
from tinycenn_lm.nonlinear_readout import (
    ReadoutConfig, adapter_payload, build_student, eligible_layers, load_adapter,
    new_cache, restore_student, wrappers,
)

class KernelWarnings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if "falling back to its reference PyTorch implementation" in message and message not in self.messages:
            self.messages.append(message)


def baseline_backend(family, warnings):
    packages = {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}
    native = family != "qwen35" or (all(packages.values()) and not warnings.messages)
    return {"qwen_native_packages_available": packages, "reference_fallback_warnings": warnings.messages,
            "baseline_backend_eligible_for_target": native}


MODELS = {"smollm2": "HuggingFaceTB/SmolLM2-135M", "qwen35": "Qwen/Qwen3.5-0.8B"}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def save_torch(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def dtype_for(device):
    if torch.device(device).type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def amp(device):
    return torch.autocast("cuda", dtype=dtype_for(device)) if torch.device(device).type == "cuda" else contextlib.nullcontext()


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def collect_documents(rows, tokenizer, counts, length, excluded=()):
    """Fresh bucket ranges; one sufficiently long block per normalized document."""
    seen, excluded = set(), set(excluded)
    blocks = {key: [] for key in counts}
    hashes = {key: [] for key in counts}
    for i, row in enumerate(rows):
        if i >= 200000:
            break
        text = " ".join(str(row.get("text", "")).split())
        digest = hashlib.sha256(text.encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if not text or bucket < 60 or digest in seen or digest in excluded:
            continue
        split = "test" if bucket < 70 else "validation" if bucket < 80 else "train"
        if len(blocks[split]) >= counts[split]:
            continue
        seen.add(digest)
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=length)["input_ids"]
        if len(ids) < length:
            continue
        blocks[split].append(torch.tensor(ids, dtype=torch.long))
        hashes[split].append(digest)
        if all(len(blocks[key]) == counts[key] for key in counts):
            return blocks, hashes
    raise RuntimeError(f"Insufficient documents: { {k: len(v) for k,v in blocks.items()} }")


@torch.no_grad()
def nll(model, blocks, context, device):
    model.eval()
    scores = []
    for block in blocks:
        ids = block[:context][None].to(device)
        targets = block[1:context+1][None].to(device)
        with amp(device):
            logits = model(input_ids=ids, use_cache=False).logits
        total = 0.0
        for start in range(0, context, 32):
            end = min(context, start+32)
            total += float(F.cross_entropy(logits[:, start:end].float().reshape(-1, logits.shape[-1]),
                                          targets[:, start:end].reshape(-1), reduction="sum"))
        scores.append(total / context)
    if not all(math.isfinite(x) for x in scores):
        raise RuntimeError("Nonfinite NLL")
    return scores


def objective(prediction, target, labels, kl_weight, temperature=1.5):
    ce = prediction.new_zeros((), dtype=torch.float32)
    kl = ce.clone()
    count = labels.numel()
    for start in range(0, prediction.shape[1], 32):
        s = prediction[:, start:start+32].float()
        t = target[:, start:start+32].float()
        ce = ce + F.cross_entropy(s.reshape(-1, s.shape[-1]), labels[:, start:start+32].reshape(-1), reduction="sum") / count
        kl = kl + F.kl_div(F.log_softmax(s/temperature, -1), F.log_softmax(t/temperature, -1),
                          log_target=True, reduction="sum") * temperature**2 / count
    return ce + kl_weight * kl, ce.detach(), kl.detach()


@contextlib.contextmanager
def attention_outputs(model, layers):
    outputs, handles = {}, []
    def hook(index):
        def capture(module, inputs, result):
            outputs[index] = result[0] if isinstance(result, tuple) else result
        return capture
    try:
        for index in layers:
            handles.append(model.model.layers[index].self_attn.register_forward_hook(hook(index)))
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def validation_score(model, blocks, args):
    return statistics.fmean(statistics.fmean(nll(model, blocks, c, args.device)) for c in args.train_contexts)


def train_candidate(teacher, model, blocks, args, out, name):
    best_path = out / "checkpoints" / f"{name}.pt"
    last_path = out / "checkpoints" / f"{name}-resume.pt"
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    use_scaler = args.device == "cuda" and dtype_for(args.device) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    history, start, best = [], 0, float("inf")
    total_steps = args.warm_steps + args.joint_steps
    metadata = {"candidate": name, "base_model": args.base_model, "model_revision": args.model_sha,
                "layers": args.layers, "seed": args.seed}
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=args.device, weights_only=True)
        load_adapter(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, start, best = saved["history"], saved["step"], saved["best"]
    else:
        best = validation_score(model, blocks["validation"], args)
        save_torch(best_path, adapter_payload(model, {**metadata, "step": 0, "validation_nll": best}))
        history.append({"step": 0, "validation_nll": best, "phase": "initial"})
    order = list(range(len(blocks["train"])))
    random.Random(args.seed).shuffle(order)
    started = time.perf_counter()
    for step in range(start, total_steps):
        model.eval()  # Deterministic frozen base: dropout remains disabled during adaptation.
        context = args.train_contexts[step % len(args.train_contexts)]
        block = blocks["train"][order[step % len(order)]]
        ids = block[:context][None].to(args.device)
        labels = block[1:context+1][None].to(args.device)
        warm = step < args.warm_steps
        optimizer.zero_grad(set_to_none=True)
        with attention_outputs(teacher, args.layers) as targets, attention_outputs(model, args.layers) as predicted:
            with torch.no_grad(), amp(args.device):
                teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            with amp(args.device):
                student_logits = model(input_ids=ids, use_cache=False).logits
            fraction = max(0, step-args.warm_steps) / max(1, args.joint_steps-1)
            kl_weight = args.kl_weight * (1 - 0.8 * fraction)
            loss, ce, kl = objective(student_logits, teacher_logits, labels, kl_weight)
            transfer = sum((predicted[i].float()-targets[i].float()).square().mean() /
                           targets[i].float().square().mean().clamp_min(1e-6) for i in args.layers) / len(args.layers)
            loss = loss + (1.0 if warm else 0.05) * transfer
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: nonfinite loss at step {step+1}")
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        record = {"step": step+1, "phase": "warmup" if warm else "joint", "context": context,
                  "ce": float(ce), "kl": float(kl), "kl_weight": kl_weight,
                  "transfer": float(transfer.detach()), "grad_norm": float(grad)}
        del teacher_logits, student_logits, loss, transfer, targets, predicted
        if (step+1) % args.eval_every == 0 or step+1 == total_steps:
            score = validation_score(model, blocks["validation"], args)
            record["validation_nll"] = score
            if score < best:
                best = score
                save_torch(best_path, adapter_payload(model, {**metadata, "step": step+1, "validation_nll": best}))
            history.append(record)
            save_torch(last_path, {"adapter": adapter_payload(model, metadata), "optimizer": optimizer.state_dict(),
                                   "scaler": scaler.state_dict(), "step": step+1, "best": best, "history": history})
            write_csv(out / f"{name}_history.csv", history)
            write_json(out / "progress.json", {"status": "training", "candidate": name, **record})
        if (step+1) % args.log_every == 0 or step == start:
            print(json.dumps({"candidate": name, **record}), flush=True)
    saved = torch.load(best_path, map_location="cpu", weights_only=True)
    load_adapter(model, saved)
    return {"candidate": name, "checkpoint": str(best_path.relative_to(out)), "validation_nll": best,
            "best_step": saved["metadata"]["step"], "trainable_parameters": sum(p.numel() for p in params),
            "token_presentations": sum(args.train_contexts[s % len(args.train_contexts)] for s in range(total_steps)),
            "this_session_training_seconds": time.perf_counter()-started}


@torch.no_grad()
def cache_equivalence(model, block, args):
    length = min(len(block)-1, max(2*args.window+5, 41))
    ids = block[:length][None].to(args.device)
    with amp(args.device):
        full = model(input_ids=ids, use_cache=False).logits
        cache = new_cache(model)
        split = min(7, length-1)
        parts = [model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True).logits]
        for i in range(split, length):
            parts.append(model(input_ids=ids[:, i:i+1], past_key_values=cache, use_cache=True).logits)
    cached = torch.cat(parts, dim=1)
    error = float((cached.float()-full.float()).square().mean() / full.float().square().mean().clamp_min(1e-8))
    if not math.isfinite(error) or error > args.cache_nmse_limit:
        raise RuntimeError(f"Cache equivalence failed: NMSE={error}")
    if cache.get_seq_length() != length:
        raise RuntimeError("Cache position mismatch")
    return {"cached_logits_nmse": error, "cache_test_tokens": length}


@torch.no_grad()
def benchmark(model, blocks, context, args):
    timings, state_bytes, peaks = [], [], []
    for block in blocks[:args.timing_documents]:
        ids = block[:context][None].to(args.device)
        continuation = block[context:context+args.decode_tokens][None].to(args.device)
        for repeat in range(args.timing_repeats+1):
            cache = new_cache(model)
            sync(args.device)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with amp(args.device):
                start = time.perf_counter()
                model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                sync(args.device)
                prefill = (time.perf_counter()-start)*1000
                initial_state = cache.nbytes
                start = time.perf_counter()
                for i in range(args.decode_tokens):
                    model(input_ids=continuation[:, i:i+1], past_key_values=cache, use_cache=True, logits_to_keep=1)
                sync(args.device)
                decode = (time.perf_counter()-start)*1000 / args.decode_tokens
            if repeat:
                timings.append({"prefill_ms": prefill, "decode_ms": decode})
                state_bytes.append(initial_state)
                if args.device == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated())
            del cache
    return {"prefill_ms": statistics.median(t["prefill_ms"] for t in timings),
            "decode_ms": statistics.median(t["decode_ms"] for t in timings),
            "cache_bytes": statistics.median(state_bytes),
            "peak_allocated_bytes": max(peaks) if peaks else None,
            "timing_samples": timings}


def paired(values, reference, seed):
    import numpy as np
    delta = np.asarray(values)-np.asarray(reference)
    samples = np.random.default_rng(seed).choice(delta, size=(2000, len(delta))).mean(axis=1)
    return {"delta_nll": float(delta.mean()), "ci_low": float(np.quantile(samples, .025)),
            "ci_high": float(np.quantile(samples, .975)), "ppl_ratio": math.exp(float(delta.mean()))}


@torch.no_grad()
def generate(model, tokenizer, prompt, args):
    # Qwen instruction model receives its actual chat template, with thinking disabled.
    if args.family == "qwen35":
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
    cache, tokens = new_cache(model), []
    with amp(args.device):
        logits = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        for i in range(args.generation_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(int(token.item()))
            if tokens[-1] == tokenizer.eos_token_id or i+1 == args.generation_tokens:
                break
            logits = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    return tokenizer.decode(tokens, skip_special_tokens=True)


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=MODELS, required=True)
    p.add_argument("--base-model")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=None)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--refinement-steps", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    p.add_argument("--state-dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--train-contexts", type=lambda s: [int(x) for x in s.split(",")], default=[256, 512])
    p.add_argument("--test-contexts", type=lambda s: [int(x) for x in s.split(",")], default=[256, 512, 1024])
    for name, default in [("train-documents",96),("validation-documents",16),("test-documents",32),
                          ("warm-steps",20),("joint-steps",150),("eval-every",25),("log-every",10),
                          ("decode-tokens",32),("timing-documents",3),("timing-repeats",3),
                          ("generation-tokens",40),("seed",2031)]:
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--cache-nmse-limit", type=float, default=0.001)
    p.add_argument("--nll-margin", type=float, default=0.02)
    p.add_argument("--exclude-manifest", action="append", default=[])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Only for correctness smoke tests; cannot earn speed-win labels")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    kernel_warnings = KernelWarnings()
    logging.getLogger("transformers").addHandler(kernel_warnings)
    args.base_model = args.base_model or MODELS[args.family]
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("Select a GPU runtime; --allow-cpu is for offline smoke tests")
    if any(x < 1 for x in args.train_contexts+args.test_contexts) or not args.refinement_steps or any(x not in [0,1,2] for x in args.refinement_steps):
        raise ValueError("Invalid contexts or refinement steps")
    if len(set(args.refinement_steps)) != len(args.refinement_steps):
        raise ValueError("Duplicate refinement steps")
    for name in ["train_documents","validation_documents","test_documents","eval_every","log_every",
                 "decode_tokens","timing_documents","timing_repeats","generation_tokens"]:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warm_steps < 0 or args.joint_steps < 0 or args.lr <= 0:
        raise ValueError("Invalid training budget")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty: choose a new directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    from huggingface_hub import HfApi
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    api = HfApi()
    old = json.loads((out / "manifest.json").read_text()) if args.resume and (out / "manifest.json").exists() else None
    args.model_sha = old["model_revision"] if old else api.model_info(args.base_model, revision=args.model_revision).sha
    dataset_sha = old["dataset_revision"] if old else api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    signature = {k:v for k,v in vars(args).items() if k not in {"output_dir","resume","device","allow_cpu"}}
    source_paths = [Path(__file__), ROOT/"src/tinycenn_lm/nonlinear_readout.py",
                    ROOT/"src/tinycenn_lm/research_layers.py", ROOT/"src/tinycenn_lm/pdelta3_frontier.py",
                    ROOT/"src/tinycenn_lm/qwen35_integrated_memory.py", ROOT/"src/tinycenn_lm/integrated_memory.py"]
    source_digest = hashlib.sha256(b"".join(path.read_bytes() for path in source_paths)).hexdigest()
    if old and old.get("source_digest") != source_digest:
        raise ValueError("Resume source code differs; use the recorded source commit or start a new run")
    if old and old["configuration"] != signature:
        raise ValueError("Resume configuration differs from the saved manifest")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_sha)
    length = max(args.train_contexts+args.test_contexts)+args.decode_tokens+1
    if old:
        blocks = torch.load(out / "token_blocks.pt", weights_only=True)
        hashes = old["document_hashes"]
    else:
        exclusions = []
        paths = [ROOT / "configs/smollm2_v3_prior_exclusions.json"] + [Path(p) for p in args.exclude_manifest]
        for path in paths:
            data = json.loads(path.read_text())
            if not isinstance(data.get("document_hashes"), dict):
                raise ValueError(f"Invalid exclusion manifest {path}")
            exclusions.extend(h for group in data["document_hashes"].values() for h in group)
        rows = load_dataset(args.dataset, name=args.dataset_config, split="train", revision=dataset_sha,
                            streaming=True).shuffle(seed=args.seed, buffer_size=10000)
        counts = {"train":args.train_documents,"validation":args.validation_documents,"test":args.test_documents}
        blocks, hashes = collect_documents(rows, tokenizer, counts, length, exclusions)
        save_torch(out / "token_blocks.pt", blocks)
        try:
            commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
        write_json(out / "manifest.json", {"configuration": signature, "model_revision": args.model_sha,
            "dataset_revision": dataset_sha, "document_hashes": hashes, "source_commit": commit, "source_digest": source_digest,
            "torch":torch.__version__, "transformers":transformers.__version__, "python":platform.python_version(),
            "device":args.device, "gpu":torch.cuda.get_device_name() if args.device=="cuda" else None,
            "native_dtype":str(dtype_for(args.device)), "state_dtype":args.state_dtype,
            "warning":"Adaptation holdouts; absence from original model pretraining is not established."})
    loader = AutoModelForCausalLM
    if args.family == "qwen35":
        from transformers import Qwen3_5ForCausalLM
        loader = Qwen3_5ForCausalLM
    teacher = loader.from_pretrained(args.base_model, revision=args.model_sha,
        torch_dtype=dtype_for(args.device), attn_implementation="sdpa").to(args.device).eval().requires_grad_(False)
    args.layers = args.layers if args.layers is not None else eligible_layers(teacher)[:3]
    print(f"Model={args.base_model} layers={args.layers} GPU={args.device} FP32 reference recurrence", flush=True)
    native_cache = cache_equivalence(teacher, blocks["validation"][0], args)
    write_json(out / "native_cache_check.json", native_cache)
    configs = [("attention_control", ReadoutConfig(variant="attention_control"))] + [
        (f"recurrent_r{steps}", ReadoutConfig(feature_dim=args.features, window=args.window, rank=args.rank,
                                            steps=steps, state_dtype=args.state_dtype)) for steps in args.refinement_steps]
    candidates = []
    for name, config in configs:
        torch.manual_seed(args.seed)
        model = build_student(teacher, args.layers, config)
        record = train_candidate(teacher, model, blocks, args, out, name)
        record["config"] = asdict(config)
        record.update(cache_equivalence(model, blocks["validation"][0], args))
        candidates.append(record)
        write_json(out / "candidates.json", candidates)
        del model
        release()
    selected = min((r for r in candidates if r["candidate"] != "attention_control"), key=lambda r:r["validation_nll"])["candidate"]
    write_json(out / "selection.json", {"selected":selected, "test_not_used_for_selection":True,
                                       "criterion":"lowest validation NLL across training contexts"})
    results, documents, examples, reference, controls = [], [], [], {}, {}
    all_records = [{"candidate":"original"}] + candidates
    for record in all_records:
        name = record["candidate"]
        if name == "original":
            model = teacher
        else:
            teacher.cpu()
            release()
            model = restore_student(teacher, torch.load(out / record["checkpoint"], weights_only=True, map_location="cpu")).to(args.device)
        model.eval()
        for context in args.test_contexts:
            values = nll(model, blocks["test"], context, args.device)
            release()
            timing = benchmark(model, blocks["test"], context, args)
            backend = baseline_backend(args.family, kernel_warnings)
            row = {"candidate":name,"context":context,**backend,"test_documents":len(values),"test_nll":statistics.fmean(values),
                   "selected":name==selected, **timing}
            if name == "original":
                reference[context] = (values, timing)
            if name == "attention_control":
                controls[context] = values
            ref, speed = reference[context]
            row.update(paired(values, ref, args.seed))
            row.update({"prefill_speedup":speed["prefill_ms"]/timing["prefill_ms"],
                        "decode_speedup":speed["decode_ms"]/timing["decode_ms"],
                        "cache_ratio":timing["cache_bytes"]/max(1,speed["cache_bytes"]),
                        "peak_memory_ratio":timing["peak_allocated_bytes"]/speed["peak_allocated_bytes"] if speed["peak_allocated_bytes"] else None})
            if name not in ("original","attention_control"):
                row.update({"control_"+k:v for k,v in paired(values,controls[context],args.seed).items()})
                quality = len(values)>=8 and row["ci_high"]<0 and row["control_ci_high"]<0
                within = len(values)>=8 and row["ci_high"]<=args.nll_margin and row["control_ci_high"]<=args.nll_margin
                row["strict_quality_win"] = quality
                row["quality_within_margin"] = within
                row["meets_requested_target"] = bool(quality and args.device=="cuda" and backend["baseline_backend_eligible_for_target"] and row["decode_speedup"]>=2
                    and row["cache_ratio"]<1 and row["peak_memory_ratio"]<1)
            results.append(row)
            documents.extend({"candidate":name,"context":context,"document_hash":h,"nll":v} for h,v in zip(hashes["test"], values))
            print(json.dumps({k:v for k,v in row.items() if k!="timing_samples"}), flush=True)
            write_json(out / "results_partial.json", results)
        for prompt in ["Explain why the sky is blue.", "Write three tips for reliable software.", "Once upon a time, a small robot"]:
            examples.append({"candidate":name,"prompt":prompt,"text":generate(model,tokenizer,prompt,args)})
        if name != "original":
            del model
        teacher.cpu()
        release()
    write_csv(out / "summary.csv", [{k:v for k,v in r.items() if k!="timing_samples"} for r in results])
    write_csv(out / "test_document_nll.csv", documents)
    write_json(out / "generation_examples.json", examples)
    write_json(out / "report.json", {"status":"completed","selected":selected,"candidates":candidates,"rows":results,
        "claims":"Exploratory single-seed adaptation. Reference PyTorch kernels; no assumed speedup. Timing uses identical teacher-forced decode inputs, batch one.",
        "document_hashes":hashes})
    write_json(out / "progress.json", {"status":"completed","selected":selected})
    write_json(out / "backend.json", baseline_backend(args.family, kernel_warnings))
    logging.getLogger("transformers").removeHandler(kernel_warnings)
    print(f"Completed. Results: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
