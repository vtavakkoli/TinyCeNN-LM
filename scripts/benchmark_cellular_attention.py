#!/usr/bin/env python3
"""Benchmark causal Cellular Attention replacements against frozen Transformer attention.

The protocol mirrors benchmark_cenn_research_layers.py: document-disjoint
train/validation/test splits, attention-output transfer, next-token refinement,
validation-only architecture selection, and paired held-out NLL intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from tinycenn_lm.cellular_attention import VARIANTS, CellularAttentionLayer
from tinycenn_lm.research_layers import softmax_reference
from scripts.benchmark_cenn_research_layers import (
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
    write_csv,
    write_json,
)


@torch.no_grad()
def benchmark_prefill(core, sample, device, repeats=10):
    """Measure reference-kernel prefill plus sparse score-pair complexity."""
    q, k, v, _ = (x.to(device) for x in sample)
    exact = lambda: softmax_reference(q, k, v, core.groups)
    candidate = lambda: core(q, k, v)
    reference_ms, reference_peak = time_kernel(exact, device, repeats)
    candidate_ms, candidate_peak = time_kernel(candidate, device, repeats)
    context = q.shape[2]
    sparse_pairs = core.max_score_pairs(context)
    dense_pairs = context * (context + 1) // 2
    return {
        "prefill_ms": candidate_ms,
        "transformer_prefill_ms": reference_ms,
        "prefill_speedup": reference_ms / candidate_ms,
        "peak_extra_bytes": candidate_peak,
        "transformer_peak_extra_bytes": reference_peak,
        "sparse_score_pairs_per_head": sparse_pairs,
        "dense_causal_score_pairs_per_head": dense_pairs,
        "score_pair_ratio": sparse_pairs / dense_pairs,
        "cellular_steps": len(core.dilations),
        "max_neighbors_per_step": core.max_neighbors_per_step(),
        "receptive_field_tokens": core.receptive_field_tokens(),
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
    parser.add_argument("--feature-dims", default="32,64")
    parser.add_argument("--dilations", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--shifted-window", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--test-contexts", default="256,512")
    parser.add_argument("--train-documents", type=int, default=64)
    parser.add_argument("--validation-documents", type=int, default=12)
    parser.add_argument("--test-documents", type=int, default=24)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lm-steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=9107)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/cellular-attention")
    args = parser.parse_args()

    for name in ("layers", "feature_dims", "test_contexts", "dilations"):
        setattr(args, name, list(dict.fromkeys(
            int(x.strip()) for x in getattr(args, name).split(",") if x.strip()
        )))
    args.variants = list(dict.fromkeys(
        x.strip() for x in args.variants.split(",") if x.strip()
    ))
    if not args.variants or set(args.variants) - set(VARIANTS):
        parser.error(f"variants must be selected from {VARIANTS}")
    if min(args.context, *args.test_contexts) < 2:
        parser.error("contexts must be >= 2")
    if min(args.train_documents, args.validation_documents, args.test_documents) < 2:
        parser.error("each partition requires at least two documents")
    if min(args.steps, args.lm_steps) < 0 or args.eval_every < 1 or args.lr <= 0:
        parser.error("invalid training budget or learning rate")
    if min(args.feature_dims) < 1 or min(args.layers) < 0 or min(args.dilations) < 1:
        parser.error("invalid feature dimensions, layers, or dilations")
    if args.shifted_window < 2 or args.nll_margin <= 0:
        parser.error("shifted window must be >=2 and margin must be positive")
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
        raise FileExistsError("Use a new output directory; experiments are never overwritten")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ) if args.device.type == "cuda" else torch.float32

    api = HfApi()
    model_sha = api.model_info(args.base_model, revision=args.model_revision).sha
    data_sha = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=model_sha)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=model_sha,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    model.requires_grad_(False)
    if model.config.model_type != "llama":
        raise ValueError("This benchmark supports Llama-family models, including SmolLM2")
    if max(args.layers) >= len(model.model.layers):
        raise ValueError("A requested layer is outside the model")

    counts = {
        "train": args.train_documents,
        "validation": args.validation_documents,
        "test": args.test_documents,
    }
    max_context = max(args.context, *args.test_contexts)
    raw = load_dataset(
        args.dataset,
        args.dataset_config,
        revision=data_sha,
        split="train",
        streaming=True,
    ).shuffle(seed=args.dataset_seed, buffer_size=2048)
    blocks, hashes = collect_documents(raw, tokenizer, counts, max_context)

    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = {
        "status": "running",
        "experiment": "cenn-cellular-attention",
        "args": {
            key: str(value) if key == "device" else value
            for key, value in vars(args).items()
        },
        "model_revision": model_sha,
        "dataset_revision": data_sha,
        "source_commit": git,
        "document_hashes": hashes,
        "unique_train_tokens": args.train_documents * args.context,
        "transfer_token_presentations_per_candidate": args.steps * args.context,
        "lm_token_presentations_per_candidate": args.lm_steps * args.context,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "device": str(args.device),
        "teacher_dtype": str(dtype),
        "kernel_dtype": "float32",
        "gpu": torch.cuda.get_device_name(0) if args.device.type == "cuda" else None,
        "scope": "one attention layer replaced at a time; all other model layers remain original",
        "nll_ci_scope": "paired bootstrap over held-out documents, not training seeds",
        "causality": "all Cellular Attention neighborhoods contain current/past tokens only",
    }
    write_json(outdir / "manifest.json", manifest)
    torch.save(blocks, outdir / "token_blocks.pt")

    train = capture_samples(model, blocks["train"], args.layers, args.context, args.device)
    validation = capture_samples(
        model, blocks["validation"], args.layers, args.context, args.device
    )
    teacher_validation = evaluate_nll(
        model, blocks["validation"], args.context, args.device
    )

    controls = {}
    for layer in args.layers:
        with replace_attention(model, layer, None):
            control = evaluate_nll(
                model, blocks["validation"], args.context, args.device
            )
        error = max(abs(a - b) for a, b in zip(control, teacher_validation))
        controls[str(layer)] = error
        if error > 0.01:
            raise RuntimeError(
                f"Exact replacement control differs by {error:.5f} NLL"
            )
    manifest["exact_replacement_max_document_nll_error"] = controls
    write_json(outdir / "manifest.json", manifest)

    candidates, history = [], []
    config = model.config
    for layer in args.layers:
        for variant in args.variants:
            for features in args.feature_dims:
                torch.manual_seed(args.seed)
                key = f"layer{layer:02d}_{variant}_f{features}_seed{args.seed}"
                checkpoint = outdir / "checkpoints" / f"{key}.pt"
                core = CellularAttentionLayer(
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                    feature_dim=features,
                    variant=variant,
                    dilations=args.dilations,
                    shifted_window=args.shifted_window,
                ).to(args.device)

                start = time.perf_counter()
                history.extend(fit_transfer(
                    core, train[layer], validation[layer], args, checkpoint, key
                ))
                val_nll, lm_history = fit_language_loss(
                    model,
                    layer,
                    core,
                    blocks["train"],
                    train[layer],
                    blocks["validation"],
                    args,
                    checkpoint,
                    key,
                )
                history.extend(lm_history)
                transfer = evaluate_transfer(core, validation[layer], args.device)
                record = {
                    "candidate": key,
                    "layer": layer,
                    "variant": variant,
                    "feature_dim": features,
                    "seed": args.seed,
                    "validation_nll": val_nll,
                    "validation_delta_nll": (
                        val_nll - statistics.fmean(teacher_validation)
                    ),
                    "validation_output_nmse": transfer["output_nmse"],
                    "validation_output_cosine": transfer["output_cosine"],
                    "trainable_parameters": sum(
                        p.numel() for p in core.parameters()
                    ),
                    "training_seconds": time.perf_counter() - start,
                    "checkpoint": str(checkpoint.relative_to(outdir)),
                    "cellular_steps": len(core.dilations),
                    "receptive_field_tokens": core.receptive_field_tokens(),
                    "max_neighbors_per_step": core.max_neighbors_per_step(),
                    "uses_sparse_softmax": True,
                }
                candidates.append(record)
                save_checkpoint(checkpoint, core, {
                    **record,
                    "model_revision": model_sha,
                    "dataset_revision": data_sha,
                })
                write_csv(outdir / "training_history.csv", history)
                write_csv(outdir / "validation_summary.csv", candidates)
                write_json(outdir / "progress.json", {
                    "status": "training",
                    "completed_candidates": candidates,
                })
                del core

    winners = {
        str(layer): min(
            (r for r in candidates if r["layer"] == layer),
            key=lambda r: r["validation_nll"],
        )["candidate"]
        for layer in args.layers
    }
    write_json(outdir / "selection.json", {
        "criterion": "lowest validation NLL at training context",
        "winners": winners,
        "test_not_used_for_selection": True,
        "declared_nll_margin": args.nll_margin,
    })

    test_rows, document_rows = [], []
    for context in args.test_contexts:
        teacher_nll = evaluate_nll(model, blocks["test"], context, args.device)
        captures = capture_samples(
            model, blocks["test"], args.layers, context, args.device
        )
        teacher_mean = statistics.fmean(teacher_nll)
        test_rows.append({
            "candidate": "transformer_original",
            "variant": "transformer_original",
            "context": context,
            "test_nll": teacher_mean,
            "test_perplexity": math.exp(teacher_mean),
            "ppl_ratio": 1.0,
            "delta_nll": 0.0,
            "quality": "reference",
            "selected_on_validation": False,
        })

        for record in candidates:
            payload = torch.load(
                outdir / record["checkpoint"],
                map_location="cpu",
                weights_only=True,
            )
            core = CellularAttentionLayer(**payload["config"]).to(args.device).eval()
            core.load_state_dict(payload["state_dict"])
            layer = record["layer"]
            with replace_attention(model, layer, core):
                values = evaluate_nll(
                    model, blocks["test"], context, args.device
                )

            delta, low, high = paired_interval(
                values, teacher_nll, args.seed
            )
            test_mean = statistics.fmean(values)
            complexity = benchmark_prefill(
                core, captures[layer][0], args.device
            )
            row = {
                **record,
                "context": context,
                "test_nll": test_mean,
                "test_perplexity": math.exp(test_mean),
                "transformer_perplexity": math.exp(teacher_mean),
                "delta_nll": delta,
                "delta_nll_ci_low": low,
                "delta_nll_ci_high": high,
                "ppl_ratio": math.exp(delta),
                "test_documents": len(values),
                "scored_tokens": len(values) * context,
                "selected_on_validation": (
                    winners[str(layer)] == record["candidate"]
                ),
                "quality": quality_label(
                    delta, low, high, len(values), args.nll_margin
                ),
                "strict_quality_win": high < 0,
                **evaluate_transfer(core, captures[layer], args.device),
                **gradient_fidelity(core, captures[layer][0], args.device),
                **complexity,
            }
            test_rows.append(row)
            for i, (value, ref) in enumerate(zip(values, teacher_nll)):
                document_rows.append({
                    "candidate": record["candidate"],
                    "context": context,
                    "document_hash": hashes["test"][i],
                    "candidate_nll": value,
                    "transformer_nll": ref,
                    "delta_nll": value - ref,
                })

            write_csv(outdir / "cellular_attention_summary.csv", test_rows)
            write_csv(outdir / "test_document_nll.csv", document_rows)
            print(json.dumps(row, indent=2), flush=True)
            del core
        del captures

    manifest["status"] = "completed"
    write_json(outdir / "manifest.json", manifest)
    write_json(outdir / "cellular_attention_report.json", {
        **manifest,
        "candidates": candidates,
        "validation_winners": winners,
        "rows": test_rows,
        "interpretation": (
            "A strict held-out NLL win requires the paired 95% interval to be below zero. "
            "Sparse score-pair counts describe the attention core only; they are not end-to-end FLOPs. "
            "This single-layer adaptation experiment does not establish full-model superiority, "
            "multi-seed robustness, or optimized production throughput."
        ),
    })
    print(f"Completed. Results: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
