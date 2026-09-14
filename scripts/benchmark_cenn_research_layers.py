#!/usr/bin/env python3
"""Layer-only transfer, replacement NLL, and measured recurrent-kernel benchmarks."""
from __future__ import annotations

import argparse
import contextlib
import csv
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
from torch import nn

from tinycenn_lm.research_layers import VARIANTS, ResearchCeNNLayer, softmax_reference


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, core, metadata):
    temp = Path(path).with_suffix(".tmp")
    torch.save({"config": core.config,
                "state_dict": {k: v.detach().cpu() for k, v in core.state_dict().items()},
                "metadata": metadata}, temp)
    temp.replace(path)


def nmse(target, output):
    return (output - target).square().mean() / target.square().mean().clamp_min(1e-8)


def cosine(target, output):
    return F.cosine_similarity(target, output, dim=-1).mean()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def project_qkv(attention, hidden, position_embeddings, num_heads, num_kv_heads):
    b, t, _ = hidden.shape
    q = attention.q_proj(hidden).view(b, t, num_heads, -1).transpose(1, 2)
    k = attention.k_proj(hidden).view(b, t, num_kv_heads, -1).transpose(1, 2)
    v = attention.v_proj(hidden).view(b, t, num_kv_heads, -1).transpose(1, 2)
    cos, sin = (x.unsqueeze(1) for x in position_embeddings)
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    return q.float(), k.float(), v.float()


class FrozenAttentionReplacement(nn.Module):
    """No-cache Llama attention replacement used only for unpadded benchmark blocks.

    Q/K/V/O and RoPE stay frozen. Returning two elements matches Transformers
    4.57.6 LlamaAttention; integration is tested with a randomly initialized Llama.
    """
    def __init__(self, original, core, num_heads, num_kv_heads):
        super().__init__()
        self.original, self.core = original, core
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.last_output = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if (kwargs.get("past_key_values") is not None
                or kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False)):
            raise ValueError("This benchmark wrapper requires use_cache=False")
        if position_embeddings is None:
            raise ValueError("Llama position_embeddings are required")
        t = hidden_states.shape[1]
        if attention_mask is not None:
            # Full causal unpadded blocks have an entirely valid last query row.
            if (attention_mask.ndim != 4 or attention_mask.shape[-1] != t
                    or bool((attention_mask[..., -1, :] < 0).any())):
                raise ValueError("Only unpadded full causal blocks are supported")
        q, k, v = project_qkv(self.original, hidden_states, position_embeddings,
                              self.num_heads, self.num_kv_heads)
        if self.core is None:
            output = softmax_reference(q, k, v, self.num_heads // self.num_kv_heads)
        else:
            output = self.core(q, k, v)
        self.last_output = output
        flat = output.transpose(1, 2).reshape(hidden_states.shape)
        return self.original.o_proj(flat.to(hidden_states.dtype)), None


@contextlib.contextmanager
def replace_attention(model, index, core):
    original = model.model.layers[index].self_attn
    wrapper = FrozenAttentionReplacement(
        original, core, model.config.num_attention_heads, model.config.num_key_value_heads
    )
    model.model.layers[index].self_attn = wrapper
    try:
        yield wrapper
    finally:
        wrapper.last_output = None
        model.model.layers[index].self_attn = original


def document_split(text):
    """Identical normalized documents always receive the same partition."""
    normalized = " ".join(text.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return digest, "test" if bucket < 10 else "validation" if bucket < 20 else "train"


def collect_documents(rows, tokenizer, counts, length, max_documents=100000):
    """One block per unique document; never pack a document across split boundaries."""
    blocks = {key: [] for key in counts}
    hashes = {key: [] for key in counts}
    seen = set()
    for scanned, row in enumerate(rows, 1):
        if scanned > max_documents:
            break
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        digest, split = document_split(text)
        if digest in seen or len(blocks[split]) >= counts[split]:
            continue
        seen.add(digest)
        tokens = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=length + 1)["input_ids"]
        if len(tokens) < length + 1:
            continue
        blocks[split].append(torch.tensor(tokens, dtype=torch.long))
        hashes[split].append(digest)
        if len(blocks[split]) == counts[split]:
            print(f"data: {split} complete ({counts[split]} documents)", flush=True)
        if all(len(blocks[key]) == count for key, count in counts.items()):
            return blocks, hashes
        if scanned >= max_documents:
            break
    raise RuntimeError(f"Insufficient documents: { {k: len(v) for k, v in blocks.items()} }")


@torch.no_grad()
def capture_samples(model, blocks, layer_indices, context, device):
    samples = {index: [] for index in layer_indices}
    config = model.config
    for number, block in enumerate(blocks, 1):
        ids = block[:context].unsqueeze(0).to(device)
        result = model.model(input_ids=ids, output_hidden_states=True,
                             use_cache=False, return_dict=True)
        positions = torch.arange(context, device=device)[None]
        for index in layer_indices:
            layer = model.model.layers[index]
            hidden = layer.input_layernorm(result.hidden_states[index])
            rope = model.model.rotary_emb(hidden, positions)
            q, k, v = project_qkv(layer.self_attn, hidden, rope,
                                 config.num_attention_heads, config.num_key_value_heads)
            target = softmax_reference(q, k, v, config.num_attention_heads // config.num_key_value_heads)
            samples[index].append(tuple(x.detach().cpu() for x in (q, k, v, target)))
        if number % 16 == 0 or number == len(blocks):
            print(f"captured {number}/{len(blocks)} at context {context}", flush=True)
    return samples


def nll_loss(model, block, context, device):
    """Score exactly context next-token targets; no padding and no capped perplexity."""
    inputs = block[:context].unsqueeze(0).to(device)
    labels = block[1:context + 1].to(device)
    logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
    return F.cross_entropy(logits[0].float(), labels)


@torch.no_grad()
def evaluate_nll(model, blocks, context, device):
    values = [float(nll_loss(model, block, context, device)) for block in blocks]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Non-finite evaluation NLL")
    return values


@torch.no_grad()
def evaluate_transfer(core, samples, device):
    errors, similarities = [], []
    for sample in samples:
        q, k, v, target = (x.to(device) for x in sample)
        prediction = core(q, k, v)
        errors.append(float(nmse(target, prediction)))
        similarities.append(float(cosine(target, prediction)))
    return {"output_nmse": statistics.fmean(errors),
            "output_cosine": statistics.fmean(similarities)}


def fit_transfer(core, train, validation, args, checkpoint, key):
    core.train()
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    # Sampling order is deliberately identical across variants and feature dimensions.
    rng = random.Random(args.seed)
    history = []
    initial = evaluate_transfer(core, validation, args.device)
    best = initial["output_nmse"]
    best_state = {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}
    save_checkpoint(checkpoint, core, {"phase": "transfer", "step": 0, **initial})
    for step in range(1, args.steps + 1):
        q, k, v, target = (x.to(args.device) for x in train[rng.randrange(len(train))])
        optimizer.zero_grad(set_to_none=True)
        prediction = core(q, k, v)
        loss = nmse(target, prediction) + 0.1 * (1.0 - cosine(target, prediction))
        if not torch.isfinite(loss):
            raise RuntimeError(f"{key}: non-finite transfer loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
        # Identical cosine LR schedule for each proposed layer.
        ratio = step / max(args.steps, 1)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * ratio)) / 2)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_transfer(core, validation, args.device)
            history.append({"candidate": key, "phase": "transfer", "step": step,
                            "train_loss": float(loss.detach()), **metrics})
            if metrics["output_nmse"] < best:
                best = metrics["output_nmse"]
                best_state = {name: value.detach().cpu().clone()
                              for name, value in core.state_dict().items()}
                save_checkpoint(checkpoint, core, {"phase": "transfer", "step": step, **metrics})
            print(f"{key} transfer {step}/{args.steps}: {metrics}", flush=True)
    core.load_state_dict(best_state)
    return history


def fit_language_loss(model, layer, core, blocks, samples, validation, args, checkpoint, key):
    """Refine only the new layer against true next-token loss; teacher stays frozen."""
    history = []
    rng = random.Random(args.seed)
    with replace_attention(model, layer, core) as wrapper:
        before = evaluate_nll(model, validation, args.context, args.device)
        best = statistics.fmean(before)
        best_state = {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}
        save_checkpoint(checkpoint, core, {"phase": "lm", "step": 0, "validation_nll": best})
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr * 0.2, weight_decay=1e-4)
        for step in range(1, args.lm_steps + 1):
            i = rng.randrange(len(blocks))
            optimizer.zero_grad(set_to_none=True)
            ce = nll_loss(model, blocks[i], args.context, args.device)
            target = samples[i][3].to(args.device)
            loss = ce + 0.2 * nmse(target, wrapper.last_output)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{key}: non-finite language loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            wrapper.last_output = None
            if step == 1 or step % args.eval_every == 0 or step == args.lm_steps:
                score = statistics.fmean(evaluate_nll(model, validation, args.context, args.device))
                history.append({"candidate": key, "phase": "lm", "step": step,
                                "train_loss": float(loss.detach()), "validation_nll": score})
                if score < best:
                    best = score
                    best_state = {name: value.detach().cpu().clone()
                                  for name, value in core.state_dict().items()}
                    save_checkpoint(checkpoint, core, {"phase": "lm", "step": step,
                                                      "validation_nll": score})
                print(f"{key} language {step}/{args.lm_steps}: val NLL={score:.5f}", flush=True)
        core.load_state_dict(best_state)
    return best, history


def paired_interval(candidate, reference, seed=2026, repeats=2000):
    if len(candidate) != len(reference) or len(candidate) < 2:
        raise ValueError("paired evaluation needs at least two matching documents")
    differences = [a - b for a, b in zip(candidate, reference)]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(differences, k=len(differences)))
                   for _ in range(repeats))
    return statistics.fmean(differences), means[int(0.025 * repeats)], means[int(0.975 * repeats)]


def quality_label(delta, low, high, count, margin=0.02):
    if count < 8:
        return "insufficient_test_documents"
    if high < 0:
        return "lower_nll_on_this_test"
    if low >= -margin and high <= margin:
        return "within_declared_nll_margin"
    if low > margin:
        return "worse_than_declared_margin"
    return "inconclusive"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_kernel(function, device, repeats=10):
    for _ in range(3):
        function()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base_memory = torch.cuda.memory_allocated(device)
    milliseconds = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        milliseconds.append(1000 * (time.perf_counter() - start))
    peak = (torch.cuda.max_memory_allocated(device) - base_memory
            if device.type == "cuda" else None)
    return statistics.median(milliseconds), peak


@torch.no_grad()
def benchmark_kernels(core, sample, device, repeats=10):
    q, k, v, _ = (x.to(device) for x in sample)
    exact = lambda: softmax_reference(q, k, v, core.groups)
    candidate = lambda: core(q, k, v)
    reference_ms, reference_peak = time_kernel(exact, device, repeats)
    candidate_ms, candidate_peak = time_kernel(candidate, device, repeats)
    _, state = core(q[:, :, :-1], k[:, :, :-1], v[:, :, :-1], return_state=True)
    step = lambda: core(q[:, :, -1:], k[:, :, -1:], v[:, :, -1:], state=state)
    # A last-token query may see the entire supplied cache: is_causal=False is intentional.
    exact_step = lambda: F.scaled_dot_product_attention(
        q[:, :, -1:], k.repeat_interleave(core.groups, 1), v.repeat_interleave(core.groups, 1),
        is_causal=False
    )
    step_ms, _ = time_kernel(step, device, repeats)
    exact_step_ms, _ = time_kernel(exact_step, device, repeats)
    # Bytes describe retained decode state, not training activations or parameters.
    t = q.shape[2]
    kv_fp32 = 2 * t * core.num_kv_heads * core.head_dim * 4
    return {
        "prefill_ms": candidate_ms, "transformer_prefill_ms": reference_ms,
        "prefill_speedup": reference_ms / candidate_ms,
        "decode_step_ms": step_ms, "transformer_decode_step_ms": exact_step_ms,
        "decode_speedup": exact_step_ms / step_ms,
        "peak_extra_bytes": candidate_peak, "transformer_peak_extra_bytes": reference_peak,
        "state_bytes_fp32": core.recurrent_state_bytes(),
        "transformer_kv_bytes_fp32": kv_fp32,
        "transformer_kv_bytes_fp16": kv_fp32 // 2,
        "state_vs_transformer_fp32": core.recurrent_state_bytes() / kv_fp32,
        "state_vs_transformer_fp16": core.recurrent_state_bytes() / (kv_fp32 // 2),
        "break_even_tokens_vs_fp16": math.ceil(
            core.recurrent_state_bytes() / (2 * core.num_kv_heads * core.head_dim * 2)
        ),
        "timing_repeats": repeats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--layers", default="18")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--feature-dims", default="64")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--test-contexts", default="256,512")
    parser.add_argument("--train-documents", type=int, default=64)
    parser.add_argument("--validation-documents", type=int, default=12)
    parser.add_argument("--test-documents", type=int, default=24)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lm-steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=9107)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/cenn-research-layers")
    args = parser.parse_args()
    for name in ("layers", "feature_dims", "test_contexts"):
        setattr(args, name, list(dict.fromkeys(int(x) for x in getattr(args, name).split(","))))
    args.variants = list(dict.fromkeys(args.variants.split(",")))
    if not args.variants or set(args.variants) - set(VARIANTS):
        parser.error(f"variants must be selected from {VARIANTS}")
    if min(args.context, *args.test_contexts) < 2:
        parser.error("contexts must be >=2")
    if args.window >= min(args.context, *args.test_contexts):
        parser.error("window must be smaller than every context, so it cannot cover full attention")
    if min(args.train_documents, args.validation_documents, args.test_documents) < 2:
        parser.error("each partition requires at least two documents")
    if min(args.steps, args.lm_steps) < 0 or args.eval_every < 1 or args.lr <= 0:
        parser.error("invalid training budget or learning rate")
    if min(args.feature_dims) < 1 or min(args.layers) < 0 or args.nll_margin <= 0:
        parser.error("invalid feature dimensions, layers, or margin")
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
    blocks, hashes = collect_documents(raw, tokenizer, counts, max_context)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "status": "running", "experiment": "cenn-research-layers",
        "args": {key: str(value) if key == "device" else value for key, value in vars(args).items()},
        "model_revision": model_sha, "dataset_revision": data_sha,
        "source_commit": git, "document_hashes": hashes,
        "unique_train_tokens": args.train_documents * args.context,
        "transfer_token_presentations_per_candidate": args.steps * args.context,
        "lm_token_presentations_per_candidate": args.lm_steps * args.context,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "datasets": datasets.__version__,
        "device": str(args.device), "teacher_dtype": str(dtype), "kernel_dtype": "float32",
        "gpu": torch.cuda.get_device_name(0) if args.device.type == "cuda" else None,
        "scope": "one attention layer replaced at a time; all other layers are original",
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
            for features in args.feature_dims:
                torch.manual_seed(args.seed)  # matched initial feature maps
                key = f"layer{layer:02d}_{variant}_f{features}_seed{args.seed}"
                checkpoint = outdir / "checkpoints" / f"{key}.pt"
                core = ResearchCeNNLayer(
                    config.num_attention_heads, config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                    features, variant, args.window
                ).to(args.device)
                start = time.perf_counter()
                history.extend(fit_transfer(
                    core, train[layer], validation[layer], args, checkpoint, key
                ))
                val_nll, lm_history = fit_language_loss(
                    model, layer, core, blocks["train"], train[layer],
                    blocks["validation"], args, checkpoint, key
                )
                history.extend(lm_history)
                transfer = evaluate_transfer(core, validation[layer], args.device)
                record = {
                    "candidate": key, "layer": layer, "variant": variant, "feature_dim": features,
                    "seed": args.seed, "validation_nll": val_nll,
                    "validation_delta_nll": val_nll - statistics.fmean(teacher_validation),
                    "validation_output_nmse": transfer["output_nmse"],
                    "validation_output_cosine": transfer["output_cosine"],
                    "trainable_parameters": sum(p.numel() for p in core.parameters()),
                    "training_seconds": time.perf_counter() - start,
                    "checkpoint": str(checkpoint.relative_to(outdir)),
                    "uses_local_softmax": variant == "cenn_delta2_window",
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
    winners = {str(layer): min((r for r in candidates if r["layer"] == layer),
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
        test_rows.append({"candidate": "transformer_original", "variant": "transformer_original",
                          "context": context, "test_nll": teacher_mean,
                          "test_perplexity": math.exp(teacher_mean), "ppl_ratio": 1.0,
                          "delta_nll": 0.0, "quality": "reference",
                          "selected_on_validation": False})
        for record in candidates:
            payload = torch.load(outdir / record["checkpoint"], map_location="cpu", weights_only=True)
            core = ResearchCeNNLayer(**payload["config"]).to(args.device).eval()
            core.load_state_dict(payload["state_dict"])
            layer = record["layer"]
            with replace_attention(model, layer, core):
                values = evaluate_nll(model, blocks["test"], context, args.device)
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
                   **benchmark_kernels(core, captures[layer][0], args.device)}
            test_rows.append(row)
            for i, (value, ref) in enumerate(zip(values, teacher_nll)):
                document_rows.append({
                    "candidate": record["candidate"], "context": context,
                    "document_hash": hashes["test"][i], "candidate_nll": value,
                    "transformer_nll": ref, "delta_nll": value - ref
                })
            write_csv(outdir / "research_layer_summary.csv", test_rows)
            write_csv(outdir / "test_document_nll.csv", document_rows)
            print(json.dumps(row, indent=2), flush=True)
            del core
        del captures
    manifest["status"] = "completed"  # Completion does not mean parity was achieved.
    write_json(outdir / "manifest.json", manifest)
    write_json(outdir / "research_layer_report.json", {
        **manifest, "candidates": candidates, "validation_winners": winners, "rows": test_rows,
        "interpretation": (
            "Only held-out next-token NLL supports a task-quality comparison. "
            "This small single-layer benchmark cannot establish full-model superiority, "
            "training-seed robustness, or production throughput. Timing is a float32 "
            "PyTorch kernel microbenchmark on one cached document, excluding Q/K/V/O."
        ),
    })
    print(f"Completed. Results: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
