#!/usr/bin/env python3
"""Benchmark PDelta2-Flash layer ingredients against exact Transformer attention."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from tinycenn_lm.pdelta2_flash import FlashPDelta2Layer
from tinycenn_lm.research_layers import softmax_reference
from scripts.benchmark_cenn_research_layers import (
    benchmark_kernels,
    capture_samples,
    collect_documents,
    evaluate_nll,
    evaluate_transfer,
    fit_language_loss,
    fit_transfer,
    gradient_fidelity,
    paired_interval,
    quality_label,
    replace_attention,
    save_checkpoint,
    time_kernel,
)


def profile(name):
    return {
        "quick": dict(train=32, validation=8, test=16, transfer=80, lm=16, every=20),
        "balanced": dict(train=64, validation=16, test=32, transfer=220, lm=48, every=44),
        "strong": dict(train=128, validation=32, test=64, transfer=500, lm=100, every=50),
    }[name]


def candidate_specs(name):
    specs = [
        dict(name="pdelta2_f96", feature_dim=96, output_gate=False, conv_kernel=1,
             indexed_retrieval=False, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="proven_baseline"),
        dict(name="flash_gate_f96", feature_dim=96, output_gate=True, conv_kernel=1,
             indexed_retrieval=False, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="qwen_style_output_gate"),
        dict(name="flash_conv4_f96", feature_dim=96, output_gate=False, conv_kernel=4,
             indexed_retrieval=False, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="short_causal_value_conv"),
        dict(name="flash_gate_conv4_f96", feature_dim=96, output_gate=True, conv_kernel=4,
             indexed_retrieval=False, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="gate_plus_short_conv"),
        dict(name="flash_indexed_f96", feature_dim=96, output_gate=False, conv_kernel=1,
             indexed_retrieval=True, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="compact_content_index"),
        dict(name="flash_gate_conv4_indexed_f96", feature_dim=96, output_gate=True, conv_kernel=4,
             indexed_retrieval=True, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="full_flash_combo"),
        dict(name="flash_gate_conv4_f64", feature_dim=64, output_gate=True, conv_kernel=4,
             indexed_retrieval=False, block_size=16, index_topk=4, state_dtype="fp16",
             ingredient="lean_speed_candidate"),
    ]
    if name == "quick":
        wanted = {"pdelta2_f96", "flash_gate_conv4_f96", "flash_indexed_f96",
                  "flash_gate_conv4_indexed_f96"}
        return [s for s in specs if s["name"] in wanted]
    return specs


class ExactGatedControl(nn.Module):
    """Exact softmax attention with a tiny trainable head gain, initialized exact."""
    def __init__(self, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.groups = num_heads // num_kv_heads
        self.log_gain = nn.Parameter(torch.zeros(num_heads))

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }

    def forward(self, q, k, v, **_):
        out = softmax_reference(q, k, v, self.groups)
        return out * self.log_gain.clamp(-1, 1).exp()[None, :, None, None]

    def recurrent_state_bytes(self, *_args, **_kwargs):
        return 0


def make_core(spec, model, device):
    heads = model.config.num_attention_heads
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // heads
    return FlashPDelta2Layer(
        heads, kv, dim,
        feature_dim=spec["feature_dim"], chunk_size=32,
        output_gate=spec["output_gate"], conv_kernel=spec["conv_kernel"],
        indexed_retrieval=spec["indexed_retrieval"], block_size=spec["block_size"],
        index_topk=spec["index_topk"], state_dtype=spec["state_dtype"],
    ).to(device)


def transformer_kv_bytes(model, context, bytes_per_element=2):
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    return 2 * context * kv * dim * bytes_per_element


@torch.no_grad()
def compiled_timing(core, sample, device):
    q, k, v, _ = (x.to(device) for x in sample)
    eager_ms, _ = time_kernel(lambda: core(q, k, v), device, repeats=5)
    result = {"eager_prefill_ms": eager_ms, "compiled_prefill_ms": None,
              "compile_speedup": None}
    if not hasattr(torch, "compile"):
        return result
    try:
        compiled = torch.compile(core, mode="reduce-overhead", fullgraph=False)
        # one untimed warmup so compilation is outside the measured region
        compiled(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compiled_ms, _ = time_kernel(lambda: compiled(q, k, v), device, repeats=5)
        result.update(compiled_prefill_ms=compiled_ms, compile_speedup=eager_ms / compiled_ms)
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"[:600]
    return result


def score(model, layer, core, blocks, reference, context, device, margin, seed):
    with replace_attention(model, layer, core):
        values = evaluate_nll(model, blocks, context, device)
    delta, low, high = paired_interval(values, reference, seed=seed, repeats=3000)
    mean = statistics.fmean(values)
    ref_mean = statistics.fmean(reference)
    if isinstance(core, FlashPDelta2Layer):
        state = core.recurrent_state_bytes(context=context)
        ratio = state / transformer_kv_bytes(model, context)
    else:
        state = transformer_kv_bytes(model, context)
        ratio = 1.0
    if high < 0:
        verdict = "strict_quality_win"
    elif delta < 0:
        verdict = "point_quality_win"
    elif low >= -margin and high <= margin and ratio < 0.5:
        verdict = "memory_efficient_parity"
    else:
        verdict = quality_label(delta, low, high, len(values), margin)
    return {
        "candidate_nll": mean,
        "transformer_nll": ref_mean,
        "candidate_ppl": math.exp(mean),
        "transformer_ppl": math.exp(ref_mean),
        "delta_nll": delta,
        "ci95_low": low,
        "ci95_high": high,
        "ppl_ratio": math.exp(delta),
        "state_bytes": state,
        "transformer_kv_bytes_fp16": transformer_kv_bytes(model, context),
        "state_vs_transformer_fp16": ratio,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "balanced", "strong"], default="balanced")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--train-context", type=int, default=256)
    parser.add_argument("--test-contexts", default="256,512,1024,2048")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=9107)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/pdelta2-flash-layer")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    cfg = profile(args.profile)
    contexts = list(dict.fromkeys([args.train_context] + [int(x) for x in args.test_contexts.split(",")]))
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    api = HfApi()
    model_sha = api.model_info(args.base_model).sha
    data_sha = api.dataset_info(args.dataset).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=model_sha)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device.type == "cuda" else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=model_sha, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    model.requires_grad_(False)
    if args.layer >= len(model.model.layers):
        raise ValueError("layer is outside the model")

    stream = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True,
                          revision=data_sha).shuffle(seed=args.dataset_seed, buffer_size=2048)
    counts = {"train": cfg["train"], "validation": cfg["validation"], "test": cfg["test"]}
    blocks, hashes = collect_documents(stream, tokenizer, counts, length=max(contexts))
    train_capture = capture_samples(model, blocks["train"], [args.layer], args.train_context, device)[args.layer]
    val_capture = capture_samples(model, blocks["validation"], [args.layer], args.train_context, device)[args.layer]
    test_capture = capture_samples(model, blocks["test"], [args.layer], args.train_context, device)[args.layer]

    teacher_val_docs = evaluate_nll(model, blocks["validation"], args.train_context, device)
    teacher_val = statistics.fmean(teacher_val_docs)
    with replace_attention(model, args.layer, None):
        exact_control_docs = evaluate_nll(model, blocks["validation"], args.train_context, device)
    exact_error = max(abs(a - b) for a, b in zip(teacher_val_docs, exact_control_docs))
    if exact_error > 0.01:
        raise RuntimeError(f"Exact replacement control mismatch: {exact_error:.6f}")

    fit_args = SimpleNamespace(
        device=device, seed=args.seed, lr=2e-3, steps=cfg["transfer"],
        lm_steps=cfg["lm"], eval_every=cfg["every"], context=args.train_context,
    )

    specs = candidate_specs(args.profile)
    records, histories, cores = [], [], {}
    for spec in specs:
        name = spec["name"]
        print("\n" + "=" * 92 + f"\n{name}", flush=True)
        torch.manual_seed(args.seed)
        core = make_core(spec, model, device)
        checkpoint = outdir / "checkpoints" / f"{name}.pt"
        histories += fit_transfer(core, train_capture, val_capture, fit_args, checkpoint, name)
        val_nll, lm_hist = fit_language_loss(
            model, args.layer, core, blocks["train"], train_capture,
            blocks["validation"], fit_args, checkpoint, name,
        )
        histories += lm_hist
        transfer = evaluate_transfer(core, val_capture, device)
        speed = benchmark_kernels(core.eval(), val_capture[0], device, repeats=5)
        record = {
            **spec,
            "validation_nll": val_nll,
            "validation_delta_nll": val_nll - teacher_val,
            "validation_ppl": math.exp(val_nll),
            "output_nmse": transfer["output_nmse"],
            "output_cosine": transfer["output_cosine"],
            "trainable_parameters": sum(p.numel() for p in core.parameters() if p.requires_grad),
            "state_bytes_256": core.recurrent_state_bytes(context=args.train_context),
            "state_vs_transformer_fp16_256": (
                core.recurrent_state_bytes(context=args.train_context)
                / transformer_kv_bytes(model, args.train_context)
            ),
            "index_pairs_per_token": core.index_pairs_per_token(args.train_context),
            "prefill_ms": speed["prefill_ms"],
            "decode_step_ms": speed["decode_step_ms"],
            "prefill_speedup_vs_exact_reference": speed["prefill_speedup"],
            "decode_speedup_vs_exact_reference": speed["decode_speedup"],
        }
        records.append(record)
        save_checkpoint(checkpoint, core, {**record, "model_revision": model_sha,
                                           "dataset_revision": data_sha})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reload_core = FlashPDelta2Layer(**payload["config"]).to(device).eval()
        reload_core.load_state_dict(payload["state_dict"])
        check = evaluate_transfer(reload_core, val_capture[:1], device)
        if not math.isfinite(check["output_nmse"]):
            raise RuntimeError("checkpoint reload produced non-finite output")
        cores[name] = core.eval()
        pd.DataFrame(records).to_csv(outdir / "validation_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)

    # Harder control: exact Transformer attention with a trainable per-head gain.
    control = ExactGatedControl(
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.hidden_size // model.config.num_attention_heads,
    ).to(device)
    control_checkpoint = outdir / "checkpoints" / "transformer_exact_trainable_control.pt"
    control_val, control_hist = fit_language_loss(
        model, args.layer, control, blocks["train"], train_capture,
        blocks["validation"], fit_args, control_checkpoint, "transformer_exact_trainable_control",
    )
    histories += control_hist

    validation = pd.DataFrame(records).sort_values("validation_nll").reset_index(drop=True)
    quality_winner = str(validation.iloc[0]["name"])
    eligible = validation[validation["validation_nll"] <= teacher_val + args.nll_margin]
    efficient_pool = eligible if len(eligible) else validation
    efficient_winner = str(efficient_pool.sort_values(
        ["decode_step_ms", "validation_nll"]
    ).iloc[0]["name"])
    selection = {
        "quality_winner": quality_winner,
        "efficient_winner": efficient_winner,
        "quality_criterion": "lowest validation NLL among PDelta2-Flash candidates",
        "efficiency_criterion": "lowest decode-step time among candidates within validation NLL margin",
        "test_not_used_for_selection": True,
        "teacher_validation_nll": teacher_val,
        "trainable_exact_control_validation_nll": control_val,
    }
    (outdir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    teacher_test = {c: evaluate_nll(model, blocks["test"], c, device) for c in contexts}
    test_rows = []
    # frozen Transformer reference rows
    for context in contexts:
        ref = statistics.fmean(teacher_test[context])
        test_rows.append({
            "candidate": "transformer_original", "ingredient": "reference", "context": context,
            "candidate_nll": ref, "transformer_nll": ref,
            "candidate_ppl": math.exp(ref), "transformer_ppl": math.exp(ref),
            "delta_nll": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "ppl_ratio": 1.0,
            "state_bytes": transformer_kv_bytes(model, context),
            "transformer_kv_bytes_fp16": transformer_kv_bytes(model, context),
            "state_vs_transformer_fp16": 1.0, "verdict": "reference",
            "selected_quality": False, "selected_efficient": False,
        })

    # all architecture candidates at training context
    for spec in specs:
        name = spec["name"]
        metrics = score(model, args.layer, cores[name], blocks["test"],
                        teacher_test[args.train_context], args.train_context,
                        device, args.nll_margin, args.seed)
        test_rows.append({
            "candidate": name, **spec, "context": args.train_context,
            "selected_quality": name == quality_winner,
            "selected_efficient": name == efficient_winner, **metrics,
        })

    # trainable exact control at training context
    control_metrics = score(model, args.layer, control, blocks["test"],
                            teacher_test[args.train_context], args.train_context,
                            device, args.nll_margin, args.seed)
    test_rows.append({
        "candidate": "transformer_exact_trainable_control",
        "ingredient": "exact_softmax_plus_trainable_head_gain",
        "context": args.train_context,
        "selected_quality": False, "selected_efficient": False, **control_metrics,
    })

    # only validation-selected candidates get expensive long-context opening
    long_names = list(dict.fromkeys([quality_winner, efficient_winner, "pdelta2_f96"]))
    for name in long_names:
        spec = next(x for x in specs if x["name"] == name)
        for context in contexts:
            if context == args.train_context:
                continue
            metrics = score(model, args.layer, cores[name], blocks["test"], teacher_test[context],
                            context, device, args.nll_margin, args.seed)
            test_rows.append({
                "candidate": name, **spec, "context": context,
                "selected_quality": name == quality_winner,
                "selected_efficient": name == efficient_winner, **metrics,
            })

    tests = pd.DataFrame(test_rows).sort_values(["context", "delta_nll"]).reset_index(drop=True)
    winner = cores[quality_winner]
    diagnostics = {
        **gradient_fidelity(winner, test_capture[0], device),
        **benchmark_kernels(winner, test_capture[0], device, repeats=10),
        **compiled_timing(winner, test_capture[0], device),
    }

    validation.to_csv(outdir / "validation_summary.csv", index=False)
    tests.to_csv(outdir / "test_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)
    report = {
        "experiment": "pdelta2-flash-layer-lab",
        "profile": args.profile,
        "base_model": args.base_model,
        "model_revision": model_sha,
        "dataset": args.dataset,
        "dataset_revision": data_sha,
        "document_hashes": hashes,
        "layer": args.layer,
        "train_context": args.train_context,
        "test_contexts": contexts,
        "teacher_validation_nll": teacher_val,
        "exact_softmax_wrapper_max_document_nll_error": exact_error,
        "selection": selection,
        "winner_diagnostics": diagnostics,
        "candidate_specs": specs,
        "win_definition": {
            "strict_quality_win": "paired 95% bootstrap CI for candidate-minus-Transformer NLL is entirely below zero",
            "point_quality_win": "mean candidate-minus-Transformer NLL is below zero but CI overlaps zero",
            "memory_efficient_parity": f"paired CI lies inside +/-{args.nll_margin} nats and state ratio < 0.5",
        },
        "research_clues": {
            "Qwen3-Next": "hybrid Gated DeltaNet / gated attention and short convolution",
            "GLM-5.3-Flash": "three linear-attention layers followed by sparse attention; KDA short conv kernel 4",
            "DeepSeek-V4.1-Flash": "cache compression and cross-layer/index reuse motivate compact indexed summaries",
        },
        "limitations": "One frozen pretrained layer replacement, reference PyTorch kernels, one training seed. A win here is not a whole-model claim.",
    }
    (outdir / "pdelta2_flash_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSELECTION\n", json.dumps(selection, indent=2))
    print("\nVALIDATION\n", validation.to_string(index=False))
    print("\nTEST\n", tests.to_string(index=False))
    print("\nWINNER DIAGNOSTICS\n", json.dumps(diagnostics, indent=2))
    print("Results:", outdir.resolve())


if __name__ == "__main__":
    main()
