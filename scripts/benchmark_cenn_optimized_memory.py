#!/usr/bin/env python3
"""V2: parallel normalized CeNN memory, calibrated controls, fresh holdouts."""
import argparse
import contextlib
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tinycenn_lm.optimized_memory import OptimizedMemory, VARIANTS, ridge_calibrate
from scripts.benchmark_cenn_research_layers import (
    write_json, write_csv, save_checkpoint, nmse, capture_samples, evaluate_nll,
    evaluate_transfer, fit_transfer, fit_language_loss, replace_attention,
    paired_interval, quality_label, gradient_fidelity, time_kernel,
)


def collect_documents(rows, tokenizer, counts, length, max_documents=100000, excluded=None):
    # V1 validation/test used hash buckets 0..19. Exclude those buckets entirely.
    # Optional prior manifests additionally exclude every previously used document.
    excluded = set(excluded or [])
    blocks, hashes = {k: [] for k in counts}, {k: [] for k in counts}
    seen = set()
    for scanned, row in enumerate(rows, 1):
        if scanned > max_documents:
            break
        text = " ".join(str(row.get("text", "")).split())
        digest = hashlib.sha256(text.encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if not text or bucket < 20 or digest in seen or digest in excluded:
            continue
        split = "test" if bucket < 30 else "validation" if bucket < 40 else "train"
        if len(blocks[split]) >= counts[split]:
            continue
        seen.add(digest)
        tokens = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=length + 1)["input_ids"]
        if len(tokens) < length + 1:
            continue
        blocks[split].append(torch.tensor(tokens, dtype=torch.long))
        hashes[split].append(digest)
        if all(len(blocks[k]) == count for k, count in counts.items()):
            return blocks, hashes
    raise RuntimeError(f"Insufficient unique documents: { {k: len(v) for k, v in blocks.items()} }")


def train_candidate(model, layer, core, blocks, train, validation_blocks, validation,
                    args, checkpoint, key):
    history, extras = [], {}
    if any(p.requires_grad for p in core.parameters()):
        before = evaluate_transfer(core, validation, args.device)["output_nmse"]
        original = core.readout.detach().clone()
        extras.update(ridge_calibrate(core, train, args.device, args.ridge))
        after = evaluate_transfer(core, validation, args.device)["output_nmse"]
        accepted = after <= before
        if not accepted:
            with torch.no_grad():
                core.readout.copy_(original)
        extras["ridge_accepted_on_validation"] = accepted
        history.extend(fit_transfer(core, train, validation, args, checkpoint, key))
        _, second = fit_language_loss(
            model, layer, core, blocks, train, validation_blocks, args, checkpoint, key)
        history.extend(second)
    else:
        extras["ridge_accepted_on_validation"] = False
    # Train in FP32. Validate the actual faster inference precision before selecting.
    drift = []
    with torch.no_grad():
        for q, k, v, _ in validation[:4]:
            q, k, v = (x.to(args.device) for x in (q, k, v))
            core.compute_dtype = "float32"
            reference = core(q, k, v)
            core.compute_dtype = args.compute_dtype
            prediction = core(q, k, v)
            drift.append(float(nmse(reference, prediction)))
    extras["inference_precision_nmse"] = statistics.fmean(drift)
    if not math.isfinite(extras["inference_precision_nmse"]) or max(drift) > 0.001:
        raise RuntimeError(f"{key}: inference precision drift is too large; rerun with --compute-dtype float32")
    with replace_attention(model, layer, core):
        score = statistics.fmean(evaluate_nll(model, validation_blocks, args.context, args.device))
    return score, history, extras


@torch.no_grad()
def benchmark_kernels(core, sample, device, native_dtype=torch.float32, compile_kernel=False):
    q, k, v, _ = (x.to(device) for x in sample)
    h = core.groups
    def exact(dtype):
        return lambda: F.scaled_dot_product_attention(
            q.to(dtype), k.to(dtype).repeat_interleave(h, 1),
            v.to(dtype).repeat_interleave(h, 1), is_causal=True)
    native_ms, native_peak = time_kernel(exact(native_dtype), device)
    fp32_ms, _ = time_kernel(exact(torch.float32), device)
    eager_ms, peak = time_kernel(lambda: core(q, k, v), device)
    _, final = core(q, k, v, return_state=True)
    _, prefix = core(q[:, :, :-1], k[:, :, :-1], v[:, :, :-1], return_state=True)
    candidate_step = lambda: core(q[:, :, -1:], k[:, :, -1:], v[:, :, -1:], state=prefix)
    native_step = lambda: F.scaled_dot_product_attention(
        q[:, :, -1:].to(native_dtype), k.to(native_dtype).repeat_interleave(h, 1),
        v.to(native_dtype).repeat_interleave(h, 1), is_causal=False)
    step_ms, _ = time_kernel(candidate_step, device)
    reference_step_ms, _ = time_kernel(native_step, device)
    kv_native = 2 * k.numel() * torch.empty((), dtype=native_dtype).element_size()
    result = {
        "prefill_ms": eager_ms, "transformer_native_prefill_ms": native_ms,
        "transformer_fp32_prefill_ms": fp32_ms, "prefill_speedup": native_ms / eager_ms,
        "decode_step_ms": step_ms, "transformer_native_decode_ms": reference_step_ms,
        "decode_speedup": reference_step_ms / step_ms,
        "state_bytes": final.nbytes, "transformer_kv_bytes": kv_native,
        "state_ratio": final.nbytes / kv_native, "peak_extra_bytes": peak,
        "transformer_peak_extra_bytes": native_peak,
        "native_reference_dtype": str(native_dtype), "candidate_compute_dtype": core.compute_dtype,
        "state_dtype": "float32", "compiled_status": "not_requested",
    }
    if compile_kernel:
        try:
            start = time.perf_counter()
            compiled = torch.compile(core, dynamic=False)
            actual = compiled(q, k, v)
            torch.testing.assert_close(actual, core(q, k, v), atol=0.005, rtol=0.005)
            result["compile_seconds"] = time.perf_counter() - start
            ms, _ = time_kernel(lambda: compiled(q, k, v), device)
            result.update(compiled_status="validated", compiled_prefill_ms=ms,
                          compiled_prefill_speedup=native_ms / ms)
        except Exception as error:
            result.update(compiled_status="failed_eager_retained", compile_error=str(error)[:400])
    return result


def gradient_diagnostic(core, sample, device):
    precision = core.compute_dtype
    try:
        core.compute_dtype = "float32"
        result = gradient_fidelity(core, sample, device)
        result["gradient_diagnostic_dtype"] = "float32"
        return result
    finally:
        core.compute_dtype = precision


def compare_adapted(values, reference, count, seed):
    delta, low, high = paired_interval(values, reference, seed)
    return {"adapted_delta_nll": delta, "adapted_delta_nll_ci_low": low,
            "adapted_delta_nll_ci_high": high, "ppl_ratio_vs_adapted": math.exp(delta),
            "beats_adapted_quality": count >= 8 and high < 0}


def load_core(record, outdir, device):
    payload = torch.load(outdir / record["checkpoint"], map_location="cpu", weights_only=True)
    core = OptimizedMemory(**payload["config"]).to(device).eval()
    core.load_state_dict(payload["state_dict"])
    return core


def evaluate_joint(model, candidates, winners, outdir, blocks, context, device, teacher, seed, margin=0.02):
    """Compose validation-selected replacements; no selection or adaptation on test data."""
    if len(winners) < 2:
        return [], []
    records = {r["candidate"]: r for r in candidates}
    groups = {
        "joint_adapted_transformer": [r for r in candidates if r["variant"] == "transformer_readout"],
        "joint_bounded_selected": [records[key] for key in winners.values()],
    }
    original_ms, _ = time_kernel(
        lambda: model(input_ids=blocks[0][:context][None].to(device), use_cache=False).logits,
        device, repeats=5)
    observations, rows = {}, []
    with torch.no_grad():
        for name, group in groups.items():
            with contextlib.ExitStack() as stack:
                for record in group:
                    stack.enter_context(replace_attention(
                        model, record["layer"], load_core(record, outdir, device)))
                values = evaluate_nll(model, blocks, context, device)
                ms, _ = time_kernel(
                    lambda: model(input_ids=blocks[0][:context][None].to(device), use_cache=False).logits,
                    device, repeats=5)
            observations[name] = values
            delta, low, high = paired_interval(values, teacher, seed)
            row = {"scope": "joint", "candidate": name, "variant": name, "context": context,
                   "layer": ",".join(str(r["layer"]) for r in group),
                   "test_nll": statistics.fmean(values),
                   "test_perplexity": math.exp(statistics.fmean(values)),
                   "ppl_ratio": math.exp(delta), "delta_nll": delta,
                   "delta_nll_ci_low": low, "delta_nll_ci_high": high,
                   "full_model_forward_ms": ms, "original_full_model_forward_ms": original_ms,
                   "full_model_forward_speedup": original_ms / ms,
                   "selected_on_validation": name == "joint_bounded_selected",
                   "quality": quality_label(delta, low, high, len(values), margin),
                   "composition": ",".join(r["candidate"] for r in group)}
            if name == "joint_bounded_selected":
                row.update(compare_adapted(values, observations["joint_adapted_transformer"],
                                           len(values), seed))
                row["beats_both_quality"] = len(values) >= 8 and high < 0 and row["beats_adapted_quality"]
            rows.append(row)
    docs = [{"candidate": name, "context": context, "document_index": i,
             "candidate_nll": value, "transformer_nll": teacher[i], "delta_nll": value - teacher[i]}
            for name, values in observations.items() for i, value in enumerate(values)]
    return rows, docs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--layers", default="0,18,29")
    p.add_argument("--feature-dims", default="64")
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--test-contexts", default="256,512,1024")
    p.add_argument("--train-documents", type=int, default=96)
    p.add_argument("--validation-documents", type=int, default=16)
    p.add_argument("--test-documents", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lm-steps", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--ridge", type=float, default=0.01)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--sink-tokens", type=int, default=4)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--dataset-seed", type=int, default=9208)
    p.add_argument("--nll-margin", type=float, default=0.02)
    p.add_argument("--compute-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--compile-kernels", action="store_true")
    p.add_argument("--exclude-manifest", action="append", default=[])
    p.add_argument("--output-dir", default="result/cenn-optimized-memory-v2")
    args = p.parse_args()
    for key in ("layers", "feature_dims", "test_contexts"):
        setattr(args, key, list(dict.fromkeys(int(x) for x in getattr(args, key).split(","))))
    args.variants = ["transformer_readout", "sink_window", "cenn_linear", "cenn_partition"]
    if min(args.context, *args.test_contexts) <= 2 * args.block_size + args.sink_tokens:
        p.error("Every context must exceed two blocks plus sinks; otherwise no compression is exercised")
    if (min(args.train_documents, args.validation_documents, args.test_documents) < 2
            or min(args.steps, args.lm_steps) < 0 or args.eval_every < 1
            or min(args.lr, args.ridge, args.nll_margin) <= 0 or min(args.layers) < 0
            or min(args.feature_dims) < 2 or any(f % 2 for f in args.feature_dims)
            or args.block_size < 1 or args.sink_tokens < 0):
        p.error("Invalid dimensions, data counts or optimization settings")
    return args


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import datasets
    import transformers

    args = parse_args()
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Use a new output directory; existing experiments are never overwritten")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.compute_dtype == "auto":
        args.compute_dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16"
                              ) if args.device.type == "cuda" else "float32"
    if args.device.type == "cpu":
        args.compute_dtype = "float32"
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
             ) if args.device.type == "cuda" else torch.float32
    api = HfApi()
    model_sha = api.model_info(args.base_model, revision=args.model_revision).sha
    data_sha = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=model_sha)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=model_sha, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(args.device).eval()
    model.requires_grad_(False)
    if model.config.model_type != "llama":
        raise ValueError("This wrapper supports Llama-family models (including SmolLM2) only")
    if max(args.layers) >= len(model.model.layers):
        raise ValueError("A requested layer is outside the model")
    counts = {"train": args.train_documents, "validation": args.validation_documents,
              "test": args.test_documents}
    max_context = max(args.context, *args.test_contexts)
    raw = load_dataset(args.dataset, args.dataset_config, revision=data_sha,
                       split="train", streaming=True).shuffle(
                           seed=args.dataset_seed, buffer_size=2048)
    excluded = set()
    for path in args.exclude_manifest:
        prior = json.loads(Path(path).read_text())
        for group in prior.get("document_hashes", {}).values():
            excluded.update(group)
    blocks, hashes = collect_documents(raw, tokenizer, counts, max_context, excluded=excluded)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "status": "running", "experiment": "cenn-optimized-memory-v2",
        "args": {key: str(value) if key == "device" else value for key, value in vars(args).items()},
        "model_revision": model_sha, "dataset_revision": data_sha,
        "source_commit": git, "document_hashes": hashes,
        "unique_train_tokens": args.train_documents * args.context,
        "transfer_token_presentations_per_candidate": args.steps * args.context,
        "lm_token_presentations_per_candidate": args.lm_steps * args.context,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "datasets": datasets.__version__,
        "device": str(args.device), "teacher_dtype": str(dtype), "training_core_dtype": "float32", "inference_compute_dtype": args.compute_dtype,
        "gpu": torch.cuda.get_device_name(0) if args.device.type == "cuda" else None,
        "scope": "single-layer screening plus joint composition of validation-selected layers",
        "split_protocol": "exclude V1 holdout buckets 0..19; test 20..29, validation 30..39, train 40..99",
        "prior_document_hashes_excluded": len(excluded),
        "nll_ci_scope": "paired bootstrap over documents, not training seeds",
    }
    write_json(outdir / "manifest.json", manifest)
    torch.save(blocks, outdir / "token_blocks.pt")
    train = capture_samples(model, blocks["train"], args.layers, args.context, args.device)
    validation = capture_samples(model, blocks["validation"], args.layers, args.context, args.device)
    teacher_validation = evaluate_nll(model, blocks["validation"], args.context, args.device)
    # Verify the replacement plumbing with an exact-softmax core before training.
    controls = {}
    for layer in args.layers:
        with replace_attention(model, layer, None):
            control = evaluate_nll(model, blocks["validation"], args.context, args.device)
        error = max(abs(a - b) for a, b in zip(control, teacher_validation))
        controls[str(layer)] = error
        if error > 0.01:
            raise RuntimeError(f"Exact replacement control differs by {error:.5f} NLL; inspect compatibility")
    manifest["exact_replacement_max_document_nll_error"] = controls
    write_json(outdir / "manifest.json", manifest)

    candidates, history = [], []
    config = model.config
    for layer in args.layers:
        for variant in args.variants:
            dimensions = args.feature_dims if variant.startswith("cenn_") else args.feature_dims[:1]
            for features in dimensions:
                torch.manual_seed(args.seed)  # matched initial feature maps
                key = f"layer{layer:02d}_{variant}_f{features}_seed{args.seed}"
                checkpoint = outdir / "checkpoints" / f"{key}.pt"
                core = OptimizedMemory(
                    config.num_attention_heads, config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                    features, variant, args.block_size, args.sink_tokens
                ).to(args.device)
                start = time.perf_counter()
                val_nll, candidate_history, extras = train_candidate(
                    model, layer, core, blocks["train"], train[layer], blocks["validation"],
                    validation[layer], args, checkpoint, key)
                history.extend(candidate_history)
                transfer = evaluate_transfer(core, validation[layer], args.device)
                record = {
                    "candidate": key, "layer": layer, "variant": variant, "feature_dim": features,
                    "seed": args.seed, "validation_nll": val_nll, "scope": "single", **extras,
                    "validation_delta_nll": val_nll - statistics.fmean(teacher_validation),
                    "validation_output_nmse": transfer["output_nmse"],
                    "validation_output_cosine": transfer["output_cosine"],
                    "trainable_parameters": sum(p.numel() for p in core.parameters()),
                    "training_seconds": time.perf_counter() - start,
                    "checkpoint": str(checkpoint.relative_to(outdir)),
                    "uses_local_softmax": variant in ("cenn_partition", "sink_window"),
                    "training_steps": args.steps if variant != "sink_window" else 0,
                    "language_steps": args.lm_steps if variant != "sink_window" else 0,
                }
                candidates.append(record)
                # Checkpoint contains the best validation-NLL state and full reconstruction config.
                save_checkpoint(checkpoint, core, {
                    **record, "model_revision": model_sha, "dataset_revision": data_sha
                })
                write_csv(outdir / "training_history.csv", history)
                write_csv(outdir / "validation_summary.csv", candidates)
                write_json(outdir / "progress.json", {
                    "status": "training", "completed_candidates": candidates
                })
                del core

    # Lock architecture selection BEFORE accessing test metrics.
    winners = {str(layer): min((r for r in candidates if r["layer"] == layer and r["variant"] != "transformer_readout"),
                                key=lambda r: r["validation_nll"])["candidate"]
               for layer in args.layers}
    write_json(outdir / "selection.json", {
        "criterion": "lowest validation NLL at training context", "winners": winners,
        "test_not_used_for_selection": True, "declared_nll_margin": args.nll_margin
    })
    test_rows, document_rows = [], []
    for context in args.test_contexts:
        teacher_nll = evaluate_nll(model, blocks["test"], context, args.device)
        captures = capture_samples(model, blocks["test"], args.layers, context, args.device)
        teacher_mean = statistics.fmean(teacher_nll)
        adapted_observations = {}
        test_rows.append({"candidate": "transformer_original", "variant": "transformer_original",
                          "context": context, "test_nll": teacher_mean,
                          "test_perplexity": math.exp(teacher_mean), "ppl_ratio": 1.0,
                          "delta_nll": 0.0, "quality": "reference",
                          "selected_on_validation": False, "scope": "reference"})
        for record in candidates:
            payload = torch.load(outdir / record["checkpoint"], map_location="cpu", weights_only=True)
            core = OptimizedMemory(**payload["config"]).to(args.device).eval()
            core.load_state_dict(payload["state_dict"])
            layer = record["layer"]
            with replace_attention(model, layer, core):
                values = evaluate_nll(model, blocks["test"], context, args.device)
            if record["variant"] == "transformer_readout":
                adapted_observations[layer] = values
            delta, low, high = paired_interval(values, teacher_nll, args.seed)
            test_mean = statistics.fmean(values)
            row = {**record, "context": context, "test_nll": test_mean,
                   "test_perplexity": math.exp(test_mean),
                   "transformer_perplexity": math.exp(teacher_mean),
                   "delta_nll": delta, "delta_nll_ci_low": low, "delta_nll_ci_high": high,
                   "ppl_ratio": math.exp(delta), "test_documents": len(values),
                   "scored_tokens": len(values) * context,
                   "selected_on_validation": winners[str(layer)] == record["candidate"],
                   "quality": quality_label(delta, low, high, len(values), args.nll_margin),
                   **evaluate_transfer(core, captures[layer], args.device),
                   **gradient_diagnostic(core, captures[layer][0], args.device),
                   **benchmark_kernels(core, captures[layer][0], args.device, dtype, args.compile_kernels),
                   **compare_adapted(values, adapted_observations[layer], len(values), args.seed)}
            row["beats_both_quality"] = len(values) >= 8 and high < 0 and row["beats_adapted_quality"]
            row["quality_preserving_efficiency_win"] = (
                len(values) >= 8 and high <= args.nll_margin and
                row["adapted_delta_nll_ci_high"] <= args.nll_margin and
                row["prefill_speedup"] > 1 and row["decode_speedup"] > 1 and row["state_ratio"] < 1)
            test_rows.append(row)
            for i, (value, ref) in enumerate(zip(values, teacher_nll)):
                document_rows.append({
                    "candidate": record["candidate"], "context": context,
                    "document_hash": hashes["test"][i], "candidate_nll": value,
                    "transformer_nll": ref, "delta_nll": value - ref
                })
            write_csv(outdir / "optimized_memory_summary.csv", test_rows)
            write_csv(outdir / "test_document_nll.csv", document_rows)
            print(json.dumps(row, indent=2), flush=True)
            del core
        if len(args.layers) > 1:
            with torch.no_grad():
                joint_rows, joint_docs = evaluate_joint(
                    model, candidates, winners, outdir, blocks["test"], context,
                    args.device, teacher_nll, args.seed, args.nll_margin)
            test_rows.extend(joint_rows)
            for observation in joint_docs:
                observation["document_hash"] = hashes["test"][observation.pop("document_index")]
            document_rows.extend(joint_docs)
            write_csv(outdir / "optimized_memory_summary.csv", test_rows)
            write_csv(outdir / "test_document_nll.csv", document_rows)
        del captures
    manifest["status"] = "completed"  # Completion does not mean parity was achieved.
    write_json(outdir / "manifest.json", manifest)
    write_json(outdir / "optimized_memory_report.json", {
        **manifest, "candidates": candidates, "validation_winners": winners, "rows": test_rows,
        "interpretation": (
            "Only held-out next-token NLL supports a task-quality comparison. "
            "Joint composition is tested, but it is not a fully replaced model or proof of superiority. "
            "training-seed robustness, or production throughput. Timing is a float32 "
            "PyTorch kernel microbenchmark on one cached document, including feature maps and calibration, "
            "excluding Q/K/V/O. Native-precision and FP32 SDPA controls are both reported."
        ),
    })
    print(f"Completed. Results: {outdir.resolve()}", flush=True)



if __name__ == "__main__":
    main()
