#!/usr/bin/env python3
"""Ablate speed/quality ingredients for P-Delta2 against one frozen Transformer layer."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from tinycenn_lm.pdelta2_features import (
    FeaturePDelta2Layer,
    precondition_chunked,
    precondition_reference,
)
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
        "quick": dict(train=32, validation=8, test=12, transfer=80, lm=12, every=20),
        "balanced": dict(train=64, validation=16, test=32, transfer=180, lm=32, every=45),
        "strong": dict(train=128, validation=32, test=64, transfer=450, lm=80, every=75),
    }[name]


def candidate_specs(profile_name):
    specs = [
        dict(name="pdelta2_none_f96", feature_dim=96, retrieval="none", window=1,
             offsets=[0], dual_timescale=False, ingredient="recurrent_only"),
        dict(name="pdelta2_dense32_f96", feature_dim=96, retrieval="dense", window=32,
             offsets=[0], dual_timescale=False, ingredient="dense_local_retrieval"),
        dict(name="pdelta2_dense64_f128", feature_dim=128, retrieval="dense", window=64,
             offsets=[0], dual_timescale=False, ingredient="previous_quality_control"),
        dict(name="pdelta2_dilated32_f96", feature_dim=96, retrieval="dilated", window=1,
             offsets=[0, 1, 2, 4, 8, 16, 32], dual_timescale=False,
             ingredient="sparse_dilated_retrieval"),
        dict(name="pdelta2_dilated32_f128", feature_dim=128, retrieval="dilated", window=1,
             offsets=[0, 1, 2, 4, 8, 16, 32], dual_timescale=False,
             ingredient="sparse_dilated_plus_capacity"),
        dict(name="pdelta2_dual_none_f96", feature_dim=96, retrieval="none", window=1,
             offsets=[0], dual_timescale=True, ingredient="dual_timescale_memory"),
        dict(name="pdelta2_dual_dilated32_f96", feature_dim=96, retrieval="dilated", window=1,
             offsets=[0, 1, 2, 4, 8, 16, 32], dual_timescale=True,
             ingredient="dual_timescale_plus_dilated"),
        dict(name="pdelta2_dual_dilated32_f64", feature_dim=64, retrieval="dilated", window=1,
             offsets=[0, 1, 2, 4, 8, 16, 32], dual_timescale=True,
             ingredient="speed_lean_combo"),
    ]
    if profile_name == "quick":
        wanted = {"pdelta2_none_f96", "pdelta2_dense32_f96", "pdelta2_dilated32_f96",
                  "pdelta2_dual_dilated32_f96"}
        return [s for s in specs if s["name"] in wanted]
    return specs


def make_core(spec, model, device):
    heads = model.config.num_attention_heads
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // heads
    return FeaturePDelta2Layer(
        heads, kv, dim,
        feature_dim=spec["feature_dim"], retrieval=spec["retrieval"], window=spec["window"],
        offsets=spec["offsets"], dual_timescale=spec["dual_timescale"], chunk_size=32,
    ).to(device)


def transformer_kv_bytes(model, context):
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    return 2 * context * kv * dim * 2


@torch.no_grad()
def preconditioner_speed(core, sample, device):
    q, k, v, _ = (x.to(device) for x in sample)
    inner = core.fast
    _, kp, _, _, _ = inner.features(q, k, v)
    curvature = kp.new_ones(kp.shape[0], inner.num_kv_heads, inner.feature_dim)
    alpha, beta, log_x, center = inner.precondition_parameters()
    serial = lambda: precondition_reference(kp, curvature, alpha, beta, log_x, center)
    vector = lambda: precondition_chunked(kp, curvature, alpha, beta, log_x, center, inner.chunk_size)
    serial_ms, _ = time_kernel(serial, device, repeats=5)
    vector_ms, _ = time_kernel(vector, device, repeats=5)
    return {
        "serial_preconditioner_ms": serial_ms,
        "vectorized_preconditioner_ms": vector_ms,
        "preconditioner_speedup": serial_ms / vector_ms,
    }


@torch.no_grad()
def compile_prefill_benchmark(core, sample, device):
    q, k, v, _ = (x.to(device) for x in sample)
    eager_ms, _ = time_kernel(lambda: core(q, k, v), device, repeats=5)
    result = {"eager_prefill_ms": eager_ms, "compiled_prefill_ms": None, "compile_speedup": None}
    if not hasattr(torch, "compile"):
        return result
    try:
        compiled = torch.compile(core, mode="reduce-overhead", fullgraph=False)
        compiled_ms, _ = time_kernel(lambda: compiled(q, k, v), device, repeats=5)
        result.update(compiled_prefill_ms=compiled_ms, compile_speedup=eager_ms / compiled_ms)
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def score(model, layer, core, blocks, reference, context, device, margin, seed):
    with replace_attention(model, layer, core):
        values = evaluate_nll(model, blocks, context, device)
    delta, low, high = paired_interval(values, reference, seed=seed, repeats=2000)
    mean = statistics.fmean(values)
    ref_mean = statistics.fmean(reference)
    ratio = core.recurrent_state_bytes() / transformer_kv_bytes(model, context)
    if high < 0:
        verdict = "strict_quality_win"
    elif delta < 0:
        verdict = "point_quality_win"
    elif low >= -margin and high <= margin and ratio < 0.5:
        verdict = "memory_efficient_parity"
    else:
        verdict = quality_label(delta, low, high, len(values), margin)
    return {
        "candidate_nll": mean, "transformer_nll": ref_mean,
        "candidate_ppl": math.exp(mean), "transformer_ppl": math.exp(ref_mean),
        "ppl_ratio": math.exp(delta), "delta_nll": delta,
        "ci95_low": low, "ci95_high": high,
        "state_bytes": core.recurrent_state_bytes(),
        "transformer_kv_bytes_fp16": transformer_kv_bytes(model, context),
        "state_vs_transformer_fp16": ratio, "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "balanced", "strong"], default="balanced")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--train-context", type=int, default=256)
    parser.add_argument("--test-contexts", default="256,512,1024")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=9107)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/pdelta2-feature-lab")
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
        exact_control = evaluate_nll(model, blocks["validation"], args.train_context, device)
    exact_error = max(abs(a - b) for a, b in zip(teacher_val_docs, exact_control))
    if exact_error > 0.01:
        raise RuntimeError(f"Exact-softmax wrapper mismatch: {exact_error:.6f}")

    fit_args = SimpleNamespace(
        device=device, seed=args.seed, lr=2e-3, steps=cfg["transfer"],
        lm_steps=cfg["lm"], eval_every=cfg["every"], context=args.train_context,
    )
    specs = candidate_specs(args.profile)
    cores, records, histories = {}, [], []
    for spec in specs:
        print("\n" + "=" * 90 + f"\n{spec['name']}", flush=True)
        torch.manual_seed(args.seed)
        core = make_core(spec, model, device)
        checkpoint = outdir / "checkpoints" / f"{spec['name']}.pt"
        histories += fit_transfer(core, train_capture, val_capture, fit_args, checkpoint, spec["name"])
        val_nll, lm_history = fit_language_loss(
            model, args.layer, core, blocks["train"], train_capture, blocks["validation"],
            fit_args, checkpoint, spec["name"],
        )
        histories += lm_history
        transfer = evaluate_transfer(core, val_capture, device)
        speed = benchmark_kernels(core.eval(), val_capture[0], device, repeats=5)
        record = {
            **spec,
            "validation_nll": val_nll, "validation_delta_nll": val_nll - teacher_val,
            "validation_ppl": math.exp(val_nll),
            "output_nmse": transfer["output_nmse"], "output_cosine": transfer["output_cosine"],
            "trainable_parameters": sum(p.numel() for p in core.parameters() if p.requires_grad),
            "retrieval_pairs_per_token": core.retrieval_pairs_per_token(),
            "state_bytes": core.recurrent_state_bytes(),
            "state_vs_transformer_fp16": core.recurrent_state_bytes() / transformer_kv_bytes(model, args.train_context),
            "prefill_ms": speed["prefill_ms"], "decode_step_ms": speed["decode_step_ms"],
            "prefill_speedup_vs_transformer": speed["prefill_speedup"],
            "decode_speedup_vs_transformer": speed["decode_speedup"],
        }
        records.append(record)
        save_checkpoint(checkpoint, core, {**record, "model_revision": model_sha, "dataset_revision": data_sha})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reloaded = FeaturePDelta2Layer(**payload["config"]).to(device).eval()
        reloaded.load_state_dict(payload["state_dict"])
        reload_metrics = evaluate_transfer(reloaded, val_capture[:1], device)
        if not math.isfinite(reload_metrics["output_nmse"]):
            raise RuntimeError("checkpoint reload produced non-finite output")
        cores[spec["name"]] = core.eval()
        pd.DataFrame(records).to_csv(outdir / "validation_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)

    validation = pd.DataFrame(records).sort_values("validation_nll").reset_index(drop=True)
    quality_winner = str(validation.iloc[0]["name"])
    eligible = validation[validation["validation_nll"] <= teacher_val + args.nll_margin]
    efficient_winner = str((eligible if len(eligible) else validation).sort_values(
        ["decode_step_ms", "validation_nll"]
    ).iloc[0]["name"])

    selection = {
        "quality_winner": quality_winner,
        "efficient_winner": efficient_winner,
        "quality_criterion": "lowest validation NLL",
        "efficiency_criterion": "lowest measured decode-step time among validation candidates within NLL margin",
        "test_not_used_for_selection": True,
    }
    (outdir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    teacher_test = {c: evaluate_nll(model, blocks["test"], c, device) for c in contexts}
    test_rows = []
    for spec in specs:
        name = spec["name"]
        metrics = score(model, args.layer, cores[name], blocks["test"], teacher_test[args.train_context],
                        args.train_context, device, args.nll_margin, args.seed)
        test_rows.append({**spec, "context": args.train_context,
                          "selected_quality": name == quality_winner,
                          "selected_efficient": name == efficient_winner, **metrics})

    long_names = list(dict.fromkeys([quality_winner, efficient_winner, "pdelta2_dense64_f128"]))
    for name in long_names:
        if name not in cores:
            continue
        spec = next(x for x in specs if x["name"] == name)
        for context in contexts:
            if context == args.train_context:
                continue
            metrics = score(model, args.layer, cores[name], blocks["test"], teacher_test[context],
                            context, device, args.nll_margin, args.seed)
            test_rows.append({**spec, "context": context,
                              "selected_quality": name == quality_winner,
                              "selected_efficient": name == efficient_winner, **metrics})

    tests = pd.DataFrame(test_rows).sort_values(["context", "delta_nll"]).reset_index(drop=True)
    winner = cores[quality_winner]
    diagnostics = {
        **gradient_fidelity(winner, test_capture[0], device),
        **benchmark_kernels(winner, test_capture[0], device, repeats=10),
        **preconditioner_speed(winner, test_capture[0], device),
        **compile_prefill_benchmark(winner, test_capture[0], device),
    }

    validation.to_csv(outdir / "validation_summary.csv", index=False)
    tests.to_csv(outdir / "test_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)
    report = {
        "experiment": "pdelta2-feature-lab",
        "profile": args.profile,
        "base_model": args.base_model, "model_revision": model_sha,
        "dataset": args.dataset, "dataset_revision": data_sha,
        "document_hashes": hashes,
        "layer": args.layer, "train_context": args.train_context, "test_contexts": contexts,
        "teacher_validation_nll": teacher_val,
        "exact_softmax_wrapper_max_document_nll_error": exact_error,
        "selection": selection,
        "winner_diagnostics": diagnostics,
        "ingredients": {
            "vectorized_preconditioner": "same P-Delta2 curvature recurrence, vectorized inside chunks",
            "dilated_retrieval": "exact causal softmax over offsets 0,1,2,4,8,16,32",
            "dual_timescale": "same total recurrent feature budget split into fast and slow memories with query mixing",
            "compile": "post-training torch.compile timing only; not used for quality selection",
        },
    }
    (outdir / "feature_lab_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSELECTION", json.dumps(selection, indent=2))
    print("\nVALIDATION\n", validation.to_string(index=False))
    print("\nTEST\n", tests.to_string(index=False))
    print("\nDIAGNOSTICS\n", json.dumps(diagnostics, indent=2))
    print("Results:", outdir.resolve())


if __name__ == "__main__":
    main()
