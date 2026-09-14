#!/usr/bin/env python3
"""Benchmark PDelta2-ER retention, residual memory and teacher-error training."""
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

from tinycenn_lm.pdelta2_er import ErrorResidualPDelta2Layer, teacher_error_weights
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
        "quick": dict(train=24, validation=8, test=12, transfer=60, lm=12, every=12),
        "balanced": dict(train=64, validation=24, test=40, transfer=240, lm=64, every=40),
        "strong": dict(train=128, validation=40, test=80, transfer=600, lm=140, every=50),
    }[name]


def candidate_specs(name):
    specs = [
        dict(name="conv4_f96_standard", feature_dim=96, residual_dim=0,
             retention_spectrum=False, error_directed=False,
             ingredient="previous_conv4_baseline"),
        dict(name="conv4_f96_hard", feature_dim=96, residual_dim=0,
             retention_spectrum=False, error_directed=True,
             ingredient="teacher_error_training_only"),
        dict(name="retention_f96_hard", feature_dim=96, residual_dim=0,
             retention_spectrum=True, error_directed=True,
             ingredient="per_feature_retention_plus_teacher_error"),
        dict(name="residual16_f96_hard", feature_dim=96, residual_dim=16,
             retention_spectrum=False, error_directed=True,
             ingredient="small_error_memory_plus_teacher_error"),
        dict(name="retention_residual16_f96_hard", feature_dim=96, residual_dim=16,
             retention_spectrum=True, error_directed=True,
             ingredient="all_three_lean"),
        dict(name="retention_residual24_f96_hard", feature_dim=96, residual_dim=24,
             retention_spectrum=True, error_directed=True,
             ingredient="all_three_quality"),
    ]
    if name == "quick":
        wanted = {
            "conv4_f96_standard", "conv4_f96_hard",
            "retention_f96_hard", "retention_residual16_f96_hard",
        }
        return [s for s in specs if s["name"] in wanted]
    return specs


class ExactGatedControl(nn.Module):
    """Exact Transformer attention with a trainable head gain, initialized exact."""
    def __init__(self, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.groups = num_heads // num_kv_heads
        self.log_gain = nn.Parameter(torch.zeros(num_heads))

    @property
    def config(self):
        return dict(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim)

    def forward(self, q, k, v, **_):
        out = softmax_reference(q, k, v, self.groups)
        return out * self.log_gain.clamp(-1, 1).exp()[None, :, None, None]

    def recurrent_state_bytes(self, *_args, **_kwargs):
        return 0


def make_core(spec, model, device):
    heads = model.config.num_attention_heads
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // heads
    return ErrorResidualPDelta2Layer(
        heads, kv, dim,
        feature_dim=spec["feature_dim"], residual_dim=spec["residual_dim"],
        chunk_size=32, conv_kernel=4,
        retention_spectrum=spec["retention_spectrum"],
        retention_min=8.0, retention_max=2048.0,
        state_dtype="fp16",
    ).to(device)


def transformer_kv_bytes(model, context, bytes_per_element=2):
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    return 2 * context * kv * dim * bytes_per_element


def hard_nmse(prediction, target, error_directed, hard_fraction=0.25, hard_boost=3.0):
    if not error_directed:
        return nmse(target, prediction)
    weights = teacher_error_weights(prediction, target, hard_fraction, hard_boost)
    token_error = (prediction - target).square().mean(dim=(-1, 1))
    numerator = (token_error * weights).mean()
    denominator = target.square().mean().clamp_min(1e-8)
    return numerator / denominator


def residual_objective(components, target):
    raw = components["residual_raw"]
    if raw is None:
        return target.new_zeros(())
    residual_target = target - components["base"].detach()
    return nmse(residual_target, raw) + 0.05 * (1.0 - cosine(residual_target, raw))


@torch.no_grad()
def focus_diagnostics(core, sample, device):
    q, k, v, target = (x.to(device) for x in sample)
    _, _, parts = core.components(q, k, v)
    prediction = parts["output"]
    error = (prediction - target).square().mean(dim=(-1, 1)).reshape(-1)
    threshold = torch.quantile(error, 0.75)
    hard = error[error >= threshold]
    easy = error[error < threshold]
    result = {
        "token_error_mean": float(error.mean()),
        "hard_quartile_error_mean": float(hard.mean()),
        "easy_75pct_error_mean": float(easy.mean()) if easy.numel() else 0.0,
        "hard_to_easy_error_ratio": float(hard.mean() / easy.mean().clamp_min(1e-12))
            if easy.numel() else None,
    }
    if parts["residual_raw"] is not None:
        residual_target = target - parts["base"]
        result["residual_target_nmse"] = float(nmse(residual_target, parts["residual_raw"]))
        result["residual_target_cosine"] = float(cosine(residual_target, parts["residual_raw"]))
        result["residual_gain_abs_mean"] = float(core.residual_gain.detach().abs().mean())
    return result


def fit_transfer_er(core, train, validation, args, checkpoint, key, error_directed):
    core.train()
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    initial = evaluate_transfer(core, validation, args.device)
    history.append({"candidate": key, "phase": "transfer", "step": 0,
                    "train_loss": None, **initial})
    best = initial["output_nmse"]
    best_state = {n: x.detach().cpu().clone() for n, x in core.state_dict().items()}
    save_checkpoint(checkpoint, core, {"phase": "transfer", "step": 0, **initial})

    for step in range(1, args.steps + 1):
        q, k, v, target = (x.to(args.device) for x in train[rng.randrange(len(train))])
        optimizer.zero_grad(set_to_none=True)
        prediction, _, parts = core.components(q, k, v)
        transfer = hard_nmse(prediction, target, error_directed)
        shape = 0.1 * (1.0 - cosine(target, prediction))
        residual = residual_objective(parts, target)
        loss = transfer + shape + (0.35 * residual if core.residual is not None else 0.0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{key}: non-finite transfer loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
        ratio = step / max(args.steps, 1)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * ratio)) / 2)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_transfer(core, validation, args.device)
            row = {"candidate": key, "phase": "transfer", "step": step,
                   "train_loss": float(loss.detach()), **metrics}
            if core.residual is not None:
                row["residual_gain_abs_mean"] = float(core.residual_gain.detach().abs().mean())
            history.append(row)
            if metrics["output_nmse"] < best:
                best = metrics["output_nmse"]
                best_state = {n: x.detach().cpu().clone() for n, x in core.state_dict().items()}
                save_checkpoint(checkpoint, core, {"phase": "transfer", "step": step, **metrics})
            print(f"{key} transfer {step}/{args.steps}: {metrics}", flush=True)
    core.load_state_dict(best_state)
    return history


def fit_language_er(model, layer, core, blocks, samples, validation, args,
                    checkpoint, key, error_directed):
    history = []
    rng = random.Random(args.seed)
    with replace_attention(model, layer, core) as wrapper:
        before = evaluate_nll(model, validation, args.context, args.device)
        best = statistics.fmean(before)
        history.append({"candidate": key, "phase": "lm", "step": 0,
                        "train_loss": None, "validation_nll": best})
        best_state = {n: x.detach().cpu().clone() for n, x in core.state_dict().items()}
        save_checkpoint(checkpoint, core, {"phase": "lm", "step": 0,
                                          "validation_nll": best})
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr * 0.2, weight_decay=1e-4)
        for step in range(1, args.lm_steps + 1):
            i = rng.randrange(len(blocks))
            optimizer.zero_grad(set_to_none=True)
            ce = _nll_loss(model, blocks[i], args.context, args.device)
            q, k, v, target = (x.to(args.device) for x in samples[i])
            prediction, _, parts = core.components(q, k, v)
            if error_directed:
                transfer = hard_nmse(prediction, target, True)
                residual = residual_objective(parts, target)
                # Stronger LM emphasis than previous experiments; teacher losses act as stabilizers.
                loss = 2.0 * ce + 0.08 * transfer
                if core.residual is not None:
                    loss = loss + 0.25 * residual
            else:
                transfer = nmse(target, prediction)
                residual = target.new_zeros(())
                loss = ce + 0.2 * transfer
            if not torch.isfinite(loss):
                raise RuntimeError(f"{key}: non-finite language loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            wrapper.last_output = None
            if step == 1 or step % args.eval_every == 0 or step == args.lm_steps:
                score = statistics.fmean(evaluate_nll(model, validation, args.context, args.device))
                row = {"candidate": key, "phase": "lm", "step": step,
                       "train_loss": float(loss.detach()), "ce": float(ce.detach()),
                       "teacher_transfer": float(transfer.detach()),
                       "residual_loss": float(residual.detach()),
                       "validation_nll": score}
                history.append(row)
                if score < best:
                    best = score
                    best_state = {n: x.detach().cpu().clone() for n, x in core.state_dict().items()}
                    save_checkpoint(checkpoint, core, {"phase": "lm", "step": step,
                                                      "validation_nll": score})
                print(f"{key} language {step}/{args.lm_steps}: val NLL={score:.6f}", flush=True)
        core.load_state_dict(best_state)
    return best, history


def _nll_loss(model, block, context, device):
    inputs = block[:context].unsqueeze(0).to(device)
    labels = block[1:context + 1].to(device)
    logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
    return torch.nn.functional.cross_entropy(logits[0].float(), labels)


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
        compiled(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compiled_ms, _ = time_kernel(lambda: compiled(q, k, v), device, repeats=5)
        result.update(compiled_prefill_ms=compiled_ms,
                      compile_speedup=eager_ms / compiled_ms)
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"[:600]
    return result


def score(model, layer, core, blocks, reference, context, device, margin, seed):
    with replace_attention(model, layer, core):
        values = evaluate_nll(model, blocks, context, device)
    delta, low, high = paired_interval(values, reference, seed=seed, repeats=4000)
    mean = statistics.fmean(values)
    ref_mean = statistics.fmean(reference)
    state = core.recurrent_state_bytes(context=context)
    ratio = state / transformer_kv_bytes(model, context)
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
        "delta_nll": delta, "ci95_low": low, "ci95_high": high,
        "ppl_ratio": math.exp(delta), "state_bytes": state,
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
    parser.add_argument("--test-contexts", default="256,512,1024,2048")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=9107)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/pdelta2-er-layer")
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
        exact_docs = evaluate_nll(model, blocks["validation"], args.train_context, device)
    exact_error = max(abs(a - b) for a, b in zip(teacher_val_docs, exact_docs))
    if exact_error > 0.01:
        raise RuntimeError(f"Exact replacement control mismatch: {exact_error:.6f}")

    fit_args = SimpleNamespace(
        device=device, seed=args.seed, lr=2e-3, steps=cfg["transfer"],
        lm_steps=cfg["lm"], eval_every=cfg["every"], context=args.train_context,
    )

    specs = candidate_specs(args.profile)
    records, histories, cores = [], [], {}
    retention_records, focus_records = [], []
    for spec in specs:
        name = spec["name"]
        print("\n" + "=" * 92 + f"\n{name}", flush=True)
        torch.manual_seed(args.seed)
        core = make_core(spec, model, device)
        checkpoint = outdir / "checkpoints" / f"{name}.pt"
        histories += fit_transfer_er(
            core, train_capture, val_capture, fit_args, checkpoint, name,
            spec["error_directed"],
        )
        val_nll, lm_hist = fit_language_er(
            model, args.layer, core, blocks["train"], train_capture,
            blocks["validation"], fit_args, checkpoint, name, spec["error_directed"],
        )
        histories += lm_hist
        transfer = evaluate_transfer(core, val_capture, device)
        speed = benchmark_kernels(core.eval(), val_capture[0], device, repeats=5)
        retention = core.retention_statistics()
        focus = focus_diagnostics(core, val_capture[0], device)
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
            "prefill_ms": speed["prefill_ms"], "decode_step_ms": speed["decode_step_ms"],
            "prefill_speedup_vs_exact_reference": speed["prefill_speedup"],
            "decode_speedup_vs_exact_reference": speed["decode_speedup"],
            **retention,
            **focus,
        }
        records.append(record)
        retention_records.append({"candidate": name, **retention})
        focus_records.append({"candidate": name, **focus})
        save_checkpoint(checkpoint, core, {**record, "model_revision": model_sha,
                                           "dataset_revision": data_sha})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reload_core = ErrorResidualPDelta2Layer(**payload["config"]).to(device).eval()
        reload_core.load_state_dict(payload["state_dict"])
        check = evaluate_transfer(reload_core, val_capture[:1], device)
        if not math.isfinite(check["output_nmse"]):
            raise RuntimeError("checkpoint reload produced non-finite output")
        cores[name] = core.eval()
        pd.DataFrame(records).to_csv(outdir / "validation_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)

    control = ExactGatedControl(
        model.config.num_attention_heads, model.config.num_key_value_heads,
        model.config.hidden_size // model.config.num_attention_heads,
    ).to(device)
    control_checkpoint = outdir / "checkpoints" / "transformer_exact_trainable_control.pt"
    control_val, control_hist = fit_language_loss(
        model, args.layer, control, blocks["train"], train_capture,
        blocks["validation"], fit_args, control_checkpoint,
        "transformer_exact_trainable_control",
    )
    histories += control_hist

    validation = pd.DataFrame(records).sort_values("validation_nll").reset_index(drop=True)
    quality_winner = str(validation.iloc[0]["name"])
    eligible = validation[validation["validation_nll"] <= teacher_val + args.nll_margin]
    pool = eligible if len(eligible) else validation
    efficient_winner = str(pool.sort_values(["decode_step_ms", "validation_nll"]).iloc[0]["name"])
    selection = {
        "quality_winner": quality_winner,
        "efficient_winner": efficient_winner,
        "quality_criterion": "lowest validation NLL",
        "efficiency_criterion": "lowest decode-step time among candidates inside validation NLL margin",
        "test_not_used_for_selection": True,
        "teacher_validation_nll": teacher_val,
        "trainable_exact_control_validation_nll": control_val,
    }
    (outdir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    teacher_test = {c: evaluate_nll(model, blocks["test"], c, device) for c in contexts}
    test_rows = []
    for spec in specs:
        name = spec["name"]
        metrics = score(model, args.layer, cores[name], blocks["test"],
                        teacher_test[args.train_context], args.train_context,
                        device, args.nll_margin, args.seed)
        test_rows.append({**spec, "context": args.train_context,
                          "selected_quality": name == quality_winner,
                          "selected_efficient": name == efficient_winner, **metrics})

    long_names = list(dict.fromkeys([
        quality_winner, efficient_winner, "conv4_f96_standard",
        "retention_residual24_f96_hard",
    ]))
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

    # Exact-attention trainable control: quality reference, not a memory competitor.
    with replace_attention(model, args.layer, control):
        control_docs = evaluate_nll(model, blocks["test"], args.train_context, device)
    cdelta, clow, chigh = paired_interval(
        control_docs, teacher_test[args.train_context], seed=args.seed, repeats=4000
    )
    test_rows.append({
        "name": "transformer_exact_trainable_control", "ingredient": "exact_attention_adaptation_control",
        "context": args.train_context, "selected_quality": False, "selected_efficient": False,
        "candidate_nll": statistics.fmean(control_docs),
        "transformer_nll": statistics.fmean(teacher_test[args.train_context]),
        "candidate_ppl": math.exp(statistics.fmean(control_docs)),
        "transformer_ppl": math.exp(statistics.fmean(teacher_test[args.train_context])),
        "delta_nll": cdelta, "ci95_low": clow, "ci95_high": chigh,
        "ppl_ratio": math.exp(cdelta), "state_bytes": transformer_kv_bytes(model, args.train_context),
        "transformer_kv_bytes_fp16": transformer_kv_bytes(model, args.train_context),
        "state_vs_transformer_fp16": 1.0,
        "verdict": "strict_quality_win" if chigh < 0 else (
            "point_quality_win" if cdelta < 0 else quality_label(cdelta, clow, chigh, len(control_docs), args.nll_margin)
        ),
    })

    tests = pd.DataFrame(test_rows).sort_values(["context", "delta_nll"]).reset_index(drop=True)
    baseline_row = tests[(tests["context"] == args.train_context) &
                         (tests["name"] == "conv4_f96_standard")].iloc[0]
    effects = []
    for _, row in tests[tests["context"] == args.train_context].iterrows():
        if row["name"] == "transformer_exact_trainable_control":
            continue
        effects.append({
            "candidate": row["name"],
            "delta_nll_vs_transformer": row["delta_nll"],
            "delta_nll_improvement_vs_conv4_baseline": baseline_row["delta_nll"] - row["delta_nll"],
            "ppl_ratio": row["ppl_ratio"],
            "state_vs_transformer_fp16": row["state_vs_transformer_fp16"],
        })

    winner = cores[quality_winner]
    diagnostics = {
        **gradient_fidelity(winner, test_capture[0], device),
        **benchmark_kernels(winner, test_capture[0], device, repeats=10),
        **compiled_timing(winner, test_capture[0], device),
        **focus_diagnostics(winner, test_capture[0], device),
        **winner.retention_statistics(),
    }

    validation.to_csv(outdir / "validation_summary.csv", index=False)
    tests.to_csv(outdir / "test_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)
    pd.DataFrame(retention_records).to_csv(outdir / "retention_summary.csv", index=False)
    pd.DataFrame(focus_records).to_csv(outdir / "teacher_error_focus.csv", index=False)
    pd.DataFrame(effects).to_csv(outdir / "ingredient_effects.csv", index=False)

    report = {
        "experiment": "pdelta2-er-layer-lab",
        "profile": args.profile,
        "base_model": args.base_model, "model_revision": model_sha,
        "dataset": args.dataset, "dataset_revision": data_sha,
        "document_hashes": hashes,
        "layer": args.layer, "train_context": args.train_context, "test_contexts": contexts,
        "teacher_validation_nll": teacher_val,
        "exact_softmax_wrapper_max_document_nll_error": exact_error,
        "selection": selection,
        "winner_diagnostics": diagnostics,
        "candidate_specs": specs,
        "training_design": {
            "teacher_error_directed": "top 25% current teacher-divergence tokens receive 4x raw weight, normalized to mean 1",
            "lm_objective": "2.0*next-token CE + 0.08*hard teacher NMSE + 0.25*residual-target loss when residual memory exists",
            "residual_target": "exact Transformer attention output minus detached main PDelta2 output",
            "retention": "learnable feature channels initialized with log-spaced 8..2048 token half-lives",
        },
        "win_definition": {
            "strict_quality_win": "paired 95% bootstrap CI for candidate-minus-Transformer NLL entirely below zero",
            "point_quality_win": "mean delta NLL below zero but CI overlaps zero",
            "memory_efficient_parity": "paired CI inside +/-0.02 nats and persistent state ratio below 0.5",
        },
        "limitations": "One pretrained attention-layer replacement, one seed, reference PyTorch kernels. Whole-model superiority requires multi-layer training and independent seeds.",
    }
    (outdir / "pdelta2_er_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSELECTION\n", json.dumps(selection, indent=2), flush=True)
    print("\nVALIDATION\n", validation.to_string(index=False), flush=True)
    print("\nTEST\n", tests.to_string(index=False), flush=True)
    print("\nINGREDIENT EFFECTS\n", pd.DataFrame(effects).to_string(index=False), flush=True)
    print("\nDIAGNOSTICS\n", json.dumps(diagnostics, indent=2), flush=True)
    print("Results:", outdir.resolve(), flush=True)


if __name__ == "__main__":
    main()
