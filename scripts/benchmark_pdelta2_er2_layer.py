#!/usr/bin/env python3
"""Benchmark PDelta2-ER2 selective compressed residual memory at 512-token training context."""
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
import torch.nn.functional as F
from torch import nn

from tinycenn_lm.pdelta2_er2 import SelectiveCompressedPDelta2Layer
from tinycenn_lm.research_layers import softmax_reference
from scripts.benchmark_cenn_research_layers import (
    benchmark_kernels,
    capture_samples,
    collect_documents,
    cosine,
    evaluate_nll,
    evaluate_transfer,
    fit_language_loss,
    gradient_fidelity,
    nmse,
    paired_interval,
    quality_label,
    replace_attention,
    save_checkpoint,
    time_kernel,
)


def profile(name):
    return {
        "quick": dict(train=24, validation=8, test=12, transfer=100, lm=20, every=20),
        "balanced": dict(train=48, validation=16, test=32, transfer=260, lm=72, every=36),
        "strong": dict(train=96, validation=32, test=64, transfer=550, lm=140, every=35),
    }[name]


def candidate_specs(name):
    specs = [
        dict(
            name="conv4_f96_512",
            feature_dim=96, residual_dim=0, code_rank=0,
            retention_spectrum=False, residual_mode="none",
            error_directed=False, hard_fraction=0.25,
            ingredient="512_context_conv4_baseline",
        ),
        dict(
            name="retention_f96_512_hard",
            feature_dim=96, residual_dim=0, code_rank=0,
            retention_spectrum=True, residual_mode="none",
            error_directed=True, hard_fraction=0.25,
            ingredient="retention_plus_error_directed_512",
        ),
        dict(
            name="raw_residual16_512_hard",
            feature_dim=96, residual_dim=16, code_rank=0,
            retention_spectrum=True, residual_mode="raw",
            error_directed=True, hard_fraction=0.25,
            ingredient="raw_residual16_control",
        ),
        dict(
            name="er2_rank8_select25",
            feature_dim=96, residual_dim=16, code_rank=8,
            retention_spectrum=True, residual_mode="compressed",
            error_directed=True, hard_fraction=0.25,
            ingredient="compressed_rank8_top25",
        ),
        dict(
            name="er2_rank12_select25",
            feature_dim=96, residual_dim=16, code_rank=12,
            retention_spectrum=True, residual_mode="compressed",
            error_directed=True, hard_fraction=0.25,
            ingredient="compressed_rank12_top25",
        ),
        dict(
            name="er2_rank8_select125",
            feature_dim=96, residual_dim=16, code_rank=8,
            retention_spectrum=True, residual_mode="compressed",
            error_directed=True, hard_fraction=0.125,
            ingredient="compressed_rank8_top12_5",
        ),
    ]
    if name == "quick":
        wanted = {"conv4_f96_512", "raw_residual16_512_hard",
                  "er2_rank8_select25", "er2_rank12_select25"}
        return [s for s in specs if s["name"] in wanted]
    return specs


class ExactGatedControl(nn.Module):
    """Exact softmax attention with only a trainable per-head output gain."""
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
    return SelectiveCompressedPDelta2Layer(
        heads, kv, dim,
        feature_dim=spec["feature_dim"],
        residual_dim=spec["residual_dim"],
        code_rank=spec["code_rank"],
        chunk_size=32,
        conv_kernel=4,
        retention_spectrum=spec["retention_spectrum"],
        retention_min=8.0,
        retention_max=4096.0,
        residual_mode=spec["residual_mode"],
        state_dtype="fp16",
    ).to(device)


def transformer_kv_bytes(model, context, bytes_per_element=2):
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    return 2 * context * kv * dim * bytes_per_element


def weighted_auxiliary_objective(aux, residual_mode):
    loss = 0.08 * aux["hard_teacher_nmse"]
    if residual_mode != "none":
        loss = loss + 0.25 * aux["residual_code_nmse"]
    if residual_mode == "compressed":
        loss = (
            loss
            + 0.08 * aux["residual_reconstruction_nmse"]
            + 0.03 * aux["gate_bce"]
            + 0.01 * aux["orthogonality"]
        )
    return loss


@torch.no_grad()
def evaluate_auxiliary(core, samples, hard_fraction, device):
    rows = []
    for sample in samples:
        q, k, v, teacher = (x.to(device) for x in sample)
        aux = core.auxiliary_losses(
            q, k, v, teacher,
            hard_fraction=hard_fraction,
            hard_boost=3.0,
        )
        rows.append({
            key: float(value.detach())
            for key, value in aux.items()
            if torch.is_tensor(value) and value.numel() == 1
        })
    keys = sorted({key for row in rows for key in row})
    return {key: statistics.fmean(row[key] for row in rows if key in row) for key in keys}


def fit_transfer(core, train, validation, spec, args, checkpoint):
    core.train()
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    initial = evaluate_transfer(core, validation, args.device)
    best = initial["output_nmse"]
    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    save_checkpoint(checkpoint, core, {"phase": "transfer", "step": 0, **initial})
    history.append({"candidate": spec["name"], "phase": "transfer", "step": 0,
                    "train_loss": None, **initial})

    for step in range(1, args.steps + 1):
        q, k, v, target = (x.to(args.device) for x in train[rng.randrange(len(train))])
        optimizer.zero_grad(set_to_none=True)
        if spec["error_directed"]:
            output, _, _ = core.components(q, k, v)
            aux = core.auxiliary_losses(
                q, k, v, target,
                hard_fraction=spec["hard_fraction"],
                hard_boost=3.0,
            )
            loss = nmse(target, output) + 0.1 * (1.0 - cosine(target, output))
            loss = loss + weighted_auxiliary_objective(aux, spec["residual_mode"])
        else:
            output = core(q, k, v)
            aux = None
            loss = nmse(target, output) + 0.1 * (1.0 - cosine(target, output))

        if not torch.isfinite(loss):
            raise RuntimeError(f"{spec['name']}: non-finite transfer loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
        ratio = step / max(args.steps, 1)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * ratio)) / 2)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_transfer(core, validation, args.device)
            row = {"candidate": spec["name"], "phase": "transfer", "step": step,
                   "train_loss": float(loss.detach()), **metrics}
            if aux is not None:
                for key, value in aux.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        row[f"train_{key}"] = float(value.detach())
            history.append(row)
            if metrics["output_nmse"] < best:
                best = metrics["output_nmse"]
                best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
                save_checkpoint(checkpoint, core, {"phase": "transfer", "step": step, **metrics})
            print(f"{spec['name']} transfer {step}/{args.steps}: {metrics}", flush=True)

    core.load_state_dict(best_state)
    return history


def fit_language(model, layer, core, blocks, samples, validation, spec, args, checkpoint):
    history = []
    rng = random.Random(args.seed)
    with replace_attention(model, layer, core) as wrapper:
        before = evaluate_nll(model, validation, args.context, args.device)
        best = statistics.fmean(before)
        best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        history.append({"candidate": spec["name"], "phase": "lm", "step": 0,
                        "train_loss": None, "validation_nll": best})
        save_checkpoint(checkpoint, core, {"phase": "lm", "step": 0, "validation_nll": best})
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr * 0.2, weight_decay=1e-4)

        for step in range(1, args.lm_steps + 1):
            i = rng.randrange(len(blocks))
            optimizer.zero_grad(set_to_none=True)
            inputs = blocks[i][:args.context].unsqueeze(0).to(args.device)
            labels = blocks[i][1:args.context + 1].to(args.device)
            logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
            ce = F.cross_entropy(logits[0].float(), labels)

            q, k, v, teacher = (x.to(args.device) for x in samples[i])
            if spec["error_directed"]:
                aux = core.auxiliary_losses(
                    q, k, v, teacher,
                    hard_fraction=spec["hard_fraction"],
                    hard_boost=3.0,
                )
                loss = 2.0 * ce + weighted_auxiliary_objective(aux, spec["residual_mode"])
            else:
                aux = None
                loss = ce + 0.2 * nmse(teacher, wrapper.last_output)

            if not torch.isfinite(loss):
                raise RuntimeError(f"{spec['name']}: non-finite language loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            wrapper.last_output = None

            if step == 1 or step % args.eval_every == 0 or step == args.lm_steps:
                score = statistics.fmean(evaluate_nll(
                    model, validation, args.context, args.device
                ))
                row = {
                    "candidate": spec["name"], "phase": "lm", "step": step,
                    "train_loss": float(loss.detach()), "train_ce": float(ce.detach()),
                    "validation_nll": score,
                }
                if aux is not None:
                    for key, value in aux.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            row[f"train_{key}"] = float(value.detach())
                history.append(row)
                if score < best:
                    best = score
                    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
                    save_checkpoint(
                        checkpoint, core,
                        {"phase": "lm", "step": step, "validation_nll": score},
                    )
                print(
                    f"{spec['name']} language {step}/{args.lm_steps}: val NLL={score:.6f}",
                    flush=True,
                )
        core.load_state_dict(best_state)
    return best, history


@torch.no_grad()
def compiled_timing(core, sample, device):
    q, k, v, _ = (x.to(device) for x in sample)
    eager_ms, _ = time_kernel(lambda: core(q, k, v), device, repeats=5)
    result = {
        "eager_prefill_ms": eager_ms,
        "compiled_prefill_ms": None,
        "compile_speedup": None,
    }
    if not hasattr(torch, "compile"):
        return result
    try:
        compiled = torch.compile(core, mode="reduce-overhead", fullgraph=False)
        compiled(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compiled_ms, _ = time_kernel(lambda: compiled(q, k, v), device, repeats=5)
        result.update(
            compiled_prefill_ms=compiled_ms,
            compile_speedup=eager_ms / compiled_ms,
        )
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"[:600]
    return result


def score(model, layer, core, blocks, reference, context, device, margin, seed):
    with replace_attention(model, layer, core):
        values = evaluate_nll(model, blocks, context, device)
    delta, low, high = paired_interval(values, reference, seed=seed, repeats=3000)
    mean = statistics.fmean(values)
    ref_mean = statistics.fmean(reference)
    if isinstance(core, SelectiveCompressedPDelta2Layer):
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
        "test_documents": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "balanced", "strong"], default="balanced")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--train-context", type=int, default=512)
    parser.add_argument("--test-contexts", default="512,1024,2048,4096")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=9107)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/pdelta2-er2-layer")
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
    contexts = list(dict.fromkeys(
        [args.train_context] + [int(x) for x in args.test_contexts.split(",")]
    ))
    if min(contexts) < 2 or args.train_context != 512:
        raise ValueError("ER2 experiment is designed for train-context=512")
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    api = HfApi()
    model_sha = api.model_info(args.base_model).sha
    data_sha = api.dataset_info(args.dataset).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=model_sha)
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=model_sha,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    if args.layer >= len(model.model.layers):
        raise ValueError("layer is outside the model")

    stream = load_dataset(
        args.dataset,
        args.dataset_config,
        split="train",
        streaming=True,
        revision=data_sha,
    ).shuffle(seed=args.dataset_seed, buffer_size=4096)
    counts = {
        "train": cfg["train"],
        "validation": cfg["validation"],
        "test": cfg["test"],
    }
    blocks, hashes = collect_documents(
        stream, tokenizer, counts, length=max(contexts), max_documents=200000
    )
    train_capture = capture_samples(
        model, blocks["train"], [args.layer], args.train_context, device
    )[args.layer]
    val_capture = capture_samples(
        model, blocks["validation"], [args.layer], args.train_context, device
    )[args.layer]

    teacher_val_docs = evaluate_nll(
        model, blocks["validation"], args.train_context, device
    )
    teacher_val = statistics.fmean(teacher_val_docs)
    with replace_attention(model, args.layer, None):
        exact_control_docs = evaluate_nll(
            model, blocks["validation"], args.train_context, device
        )
    exact_error = max(abs(a - b) for a, b in zip(teacher_val_docs, exact_control_docs))
    if exact_error > 0.01:
        raise RuntimeError(f"Exact replacement control mismatch: {exact_error:.6f}")

    fit_args = SimpleNamespace(
        device=device,
        seed=args.seed,
        lr=2e-3,
        steps=cfg["transfer"],
        lm_steps=cfg["lm"],
        eval_every=cfg["every"],
        context=args.train_context,
    )

    specs = candidate_specs(args.profile)
    records, histories, cores = [], [], {}
    for spec in specs:
        name = spec["name"]
        print("\n" + "=" * 96 + f"\n{name}", flush=True)
        torch.manual_seed(args.seed)
        core = make_core(spec, model, device)
        checkpoint = outdir / "checkpoints" / f"{name}.pt"

        histories += fit_transfer(
            core, train_capture, val_capture, spec, fit_args, checkpoint
        )
        val_nll, lm_hist = fit_language(
            model,
            args.layer,
            core,
            blocks["train"],
            train_capture,
            blocks["validation"],
            spec,
            fit_args,
            checkpoint,
        )
        histories += lm_hist

        transfer = evaluate_transfer(core, val_capture, device)
        aux = evaluate_auxiliary(
            core, val_capture, spec["hard_fraction"], device
        )
        speed = benchmark_kernels(core.eval(), val_capture[0], device, repeats=5)
        retention = core.retention_statistics()
        record = {
            **spec,
            "validation_nll": val_nll,
            "validation_delta_nll": val_nll - teacher_val,
            "validation_ppl": math.exp(val_nll),
            "output_nmse": transfer["output_nmse"],
            "output_cosine": transfer["output_cosine"],
            "trainable_parameters": sum(
                p.numel() for p in core.parameters() if p.requires_grad
            ),
            "state_bytes_512": core.recurrent_state_bytes(),
            "state_vs_transformer_fp16_512": (
                core.recurrent_state_bytes()
                / transformer_kv_bytes(model, args.train_context)
            ),
            "prefill_ms": speed["prefill_ms"],
            "decode_step_ms": speed["decode_step_ms"],
            **{f"aux_{k}": v for k, v in aux.items()},
            **retention,
        }
        records.append(record)
        save_checkpoint(
            checkpoint,
            core,
            {**record, "model_revision": model_sha, "dataset_revision": data_sha},
        )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        clone = SelectiveCompressedPDelta2Layer(**payload["config"]).to(device).eval()
        clone.load_state_dict(payload["state_dict"])
        check = evaluate_transfer(clone, val_capture[:1], device)
        if not math.isfinite(check["output_nmse"]):
            raise RuntimeError("checkpoint reload produced non-finite output")
        cores[name] = core.eval()

        pd.DataFrame(records).to_csv(outdir / "validation_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)

    control = ExactGatedControl(
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.hidden_size // model.config.num_attention_heads,
    ).to(device)
    control_checkpoint = outdir / "checkpoints" / "transformer_exact_trainable_control.pt"
    control_val, control_hist = fit_language_loss(
        model,
        args.layer,
        control,
        blocks["train"],
        train_capture,
        blocks["validation"],
        fit_args,
        control_checkpoint,
        "transformer_exact_trainable_control",
    )
    histories += control_hist

    validation = pd.DataFrame(records).sort_values("validation_nll").reset_index(drop=True)
    quality_winner = str(validation.iloc[0]["name"])
    eligible = validation[
        validation["validation_nll"] <= teacher_val + args.nll_margin
    ]
    efficient_pool = eligible if len(eligible) else validation
    efficient_winner = str(
        efficient_pool.sort_values(["decode_step_ms", "validation_nll"]).iloc[0]["name"]
    )
    selection = {
        "quality_winner": quality_winner,
        "efficient_winner": efficient_winner,
        "quality_criterion": "lowest validation NLL",
        "efficiency_criterion": "lowest decode-step time among candidates inside validation NLL margin",
        "test_not_used_for_selection": True,
        "teacher_validation_nll": teacher_val,
        "trainable_exact_control_validation_nll": control_val,
        "train_context": args.train_context,
    }
    (outdir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    teacher_test = {
        context: evaluate_nll(model, blocks["test"], context, device)
        for context in contexts
    }
    test_rows = []

    long_names = {
        quality_winner,
        efficient_winner,
        "conv4_f96_512",
        "raw_residual16_512_hard",
    }
    for name, core in cores.items():
        eval_contexts = contexts if name in long_names else [args.train_context]
        for context in eval_contexts:
            row = score(
                model,
                args.layer,
                core,
                blocks["test"],
                teacher_test[context],
                context,
                device,
                args.nll_margin,
                args.seed,
            )
            test_rows.append({
                "candidate": name,
                "context": context,
                "selected_quality": name == quality_winner,
                "selected_efficiency": name == efficient_winner,
                **row,
            })

    for context in contexts:
        row = score(
            model,
            args.layer,
            control,
            blocks["test"],
            teacher_test[context],
            context,
            device,
            args.nll_margin,
            args.seed,
        )
        test_rows.append({
            "candidate": "transformer_exact_trainable_control",
            "context": context,
            "selected_quality": False,
            "selected_efficiency": False,
            **row,
        })

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(outdir / "test_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)
    validation.to_csv(outdir / "validation_summary.csv", index=False)

    baseline_val = float(
        validation.loc[validation["name"] == "conv4_f96_512", "validation_nll"].iloc[0]
    )
    effects = validation[[
        "name", "ingredient", "validation_nll", "validation_delta_nll",
        "state_vs_transformer_fp16_512", "decode_step_ms",
        "aux_hard_teacher_nmse", "aux_residual_code_nmse",
        "aux_residual_reconstruction_nmse", "aux_gate_bce",
    ]].copy()
    effects["validation_improvement_vs_conv4"] = baseline_val - effects["validation_nll"]
    effects.to_csv(outdir / "ingredient_effects.csv", index=False)

    compressed = validation[
        validation["residual_mode"] == "compressed"
    ][[
        "name", "code_rank", "hard_fraction",
        "aux_residual_code_nmse", "aux_residual_reconstruction_nmse",
        "aux_gate_bce", "predicted_gate_mean", "residual_gain_abs_mean",
    ]].copy()
    compressed.to_csv(outdir / "compressed_residual_summary.csv", index=False)

    winner = cores[quality_winner]
    diagnostics = {}
    diagnostics.update(gradient_fidelity(winner, val_capture[0], device))
    diagnostics.update(benchmark_kernels(winner, val_capture[0], device, repeats=10))
    diagnostics.update(compiled_timing(winner, val_capture[0], device))
    diagnostics.update(evaluate_auxiliary(
        winner,
        val_capture,
        next(s["hard_fraction"] for s in specs if s["name"] == quality_winner),
        device,
    ))
    diagnostics.update(winner.retention_statistics())

    strict_wins = [
        row for row in test_rows
        if row["candidate"] != "transformer_exact_trainable_control"
        and row["verdict"] == "strict_quality_win"
    ]
    report = {
        "experiment": "pdelta2-er2-long-context-layer-lab",
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
        "strict_quality_wins": strict_wins,
        "training_design": {
            "main_change_1": "train at context 512 instead of 256",
            "main_change_2": "Residual16 only; no larger residual state",
            "main_change_3": "compressed low-rank teacher-error code plus learned token/head difficulty gate",
            "hard_target": "teacher attention minus detached Conv4-PDelta base output",
            "gate_target": "per-head top-error teacher-divergence positions",
            "long_context": "held-out 512/1024/2048/4096 evaluation",
        },
        "win_definition": {
            "strict_quality_win": "paired 95% bootstrap CI for candidate-minus-Transformer NLL entirely below zero",
            "point_quality_win": "mean delta NLL below zero but CI overlaps zero",
            "memory_efficient_parity": "paired CI inside +/-0.02 nats and persistent state ratio below 0.5",
        },
        "limitations": (
            "One pretrained attention-layer replacement and one seed. "
            "A win is not a whole-model superiority claim."
        ),
    }
    (outdir / "pdelta2_er2_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("\nSelection:", json.dumps(selection, indent=2), flush=True)
    print("\nHeld-out results:", flush=True)
    print(test_df.to_string(index=False), flush=True)
    if strict_wins:
        print("\nSTRICT TRANSFORMER QUALITY WIN DETECTED:", flush=True)
        for row in strict_wins:
            print(row, flush=True)
    else:
        print("\nNo strict Transformer quality win in this run.", flush=True)
    print(f"\nResults written to {outdir}", flush=True)


if __name__ == "__main__":
    main()
