#!/usr/bin/env python3
"""Preconditioned Delta2 replacement screen against a frozen Transformer layer.

Independent research adaptation combining DeltaNet-2 erase/write separation,
diagonal curvature-aware write-key preconditioning, and optional bounded local
attention. Candidate selection is validation-only; test scoring happens after
the winner is locked.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.research_layers import ResearchCeNNLayer, delta_recurrence, local_window_attention
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
)


@dataclass
class PDeltaState:
    memory: Tensor
    curvature: Tensor
    keys: Tensor | None = None
    values: Tensor | None = None


class PreconditionedDelta2Layer(nn.Module):
    """Curvature-aware Delta2 reference layer with optional bounded local attention."""

    def __init__(self, num_heads, num_kv_heads, head_dim, feature_dim=64, window=0, chunk_size=32):
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if min(num_heads, num_kv_heads, head_dim, feature_dim) < 1 or window < 0:
            raise ValueError("invalid dimensions")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.groups = num_heads // num_kv_heads
        self.window = window
        self.chunk_size = chunk_size
        self.variant = "cenn_pdelta2_window" if window else "cenn_pdelta2"

        base = torch.zeros(num_kv_heads, feature_dim, head_dim)
        for h in range(num_kv_heads):
            if feature_dim == head_dim:
                base[h] = torch.eye(head_dim)
            else:
                nn.init.orthogonal_(base[h])
        self.wk = nn.Parameter(base.clone())
        self.wq = nn.Parameter(base.repeat_interleave(self.groups, dim=0).clone())

        self.forget_w = nn.Parameter(torch.zeros(num_kv_heads, feature_dim, head_dim))
        self.forget_b = nn.Parameter(torch.full((num_kv_heads, feature_dim), math.log(0.04 / 0.96)))
        self.erase_w = nn.Parameter(torch.zeros(num_kv_heads, feature_dim, head_dim))
        self.erase_b = nn.Parameter(torch.full((num_kv_heads, feature_dim), -1.0))
        self.write_w = nn.Parameter(torch.zeros(num_kv_heads, head_dim, head_dim))
        self.write_b = nn.Parameter(torch.full((num_kv_heads, head_dim), -1.0))

        # Diagonal curvature state A_t. Initial alpha≈0.995, beta≈0.12.
        self.pre_log_decay = nn.Parameter(torch.full((num_kv_heads, feature_dim), math.log(0.995)))
        self.pre_gain_logit = nn.Parameter(torch.full((num_kv_heads, feature_dim), math.log(0.12 / 0.88)))
        self.pre_range_raw = nn.Parameter(torch.zeros(num_kv_heads, 1))
        self.pre_center = nn.Parameter(torch.zeros(num_kv_heads, 1))

        self.log_gain = nn.Parameter(torch.zeros(num_heads))
        if window:
            self.mix_w = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.mix_b = nn.Parameter(torch.full((num_heads,), -0.5))

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "variant": self.variant,
            "window": self.window,
            "chunk_size": self.chunk_size,
        }

    @staticmethod
    def _project(x, weight, bias):
        return torch.einsum("bhtd,hfd->bhtf", x, weight) + bias[None, :, None]

    def features(self, q, k, v):
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        qp = F.normalize(torch.einsum("bhtd,hfd->bhtf", qn, self.wq), dim=-1)
        kp = F.normalize(torch.einsum("bhtd,hfd->bhtf", kn, self.wk), dim=-1)
        log_decay = -0.25 * self._project(kn, self.forget_w, self.forget_b).sigmoid()
        erase = kp * self._project(kn, self.erase_w, self.erase_b).sigmoid()
        write_gate = self._project(F.normalize(v, dim=-1), self.write_w, self.write_b).sigmoid()
        return qp, kp, v * write_gate, erase, log_decay

    def precondition_keys(self, kp, curvature):
        # Stable bounded diagonal preconditioner inspired by Preconditioned DeltaNet.
        alpha = self.pre_log_decay.clamp(math.log(0.98), math.log(0.9999)).exp()
        beta = self.pre_gain_logit.sigmoid()
        log_x = math.log(2.0) + self.pre_range_raw.sigmoid() * (math.log(8.0) - math.log(2.0))
        writes = []
        for t in range(kp.shape[2]):
            kt = kp[:, :, t]
            r = (curvature + 1e-4).log() - self.pre_center[None]
            s = r / (1.0 + r.abs())
            scale = torch.exp(-log_x[None] * s)
            numerator = scale * kt
            denominator = 1.0 + (kt * numerator).sum(-1, keepdim=True)
            writes.append(numerator / denominator.clamp_min(1e-4))
            curvature = alpha[None] * curvature + beta[None] * kt.square()
        return torch.stack(writes, dim=2), curvature

    def forward(self, q, k, v, state=None, return_state=False, implementation="chunk"):
        if implementation != "chunk":
            raise ValueError("P-Delta2 uses the bounded chunk recurrence")
        q, k, v = (x.to(self.wq.dtype) for x in (q, k, v))
        batch = q.shape[0]
        if state is None:
            state = PDeltaState(
                memory=q.new_zeros(batch, self.num_kv_heads, self.feature_dim, self.head_dim),
                curvature=q.new_ones(batch, self.num_kv_heads, self.feature_dim),
            )

        qp, kp, z, erase, log_decay = self.features(q, k, v)
        kpre, curvature = self.precondition_keys(kp, state.curvature)
        output, memory = delta_recurrence(
            qp, kpre, z, erase, log_decay, state.memory, self.groups, self.chunk_size
        )
        output = output * self.log_gain.clamp(-4, 4).exp()[None, :, None, None]
        new_state = PDeltaState(memory=memory, curvature=curvature)

        if self.window:
            keys = k if state.keys is None else torch.cat((state.keys, k), dim=2)
            values = v if state.values is None else torch.cat((state.values, v), dim=2)
            local = local_window_attention(q, keys, values, self.window, self.groups)
            mix = (
                torch.einsum("bhtd,hd->bht", F.normalize(q, dim=-1), self.mix_w)
                + self.mix_b[None, :, None]
            ).sigmoid().unsqueeze(-1)
            output = mix * local + (1.0 - mix) * output
            keep = self.window - 1
            new_state.keys = keys[:, :, -keep:].clone() if keep else None
            new_state.values = values[:, :, -keep:].clone() if keep else None

        return (output, new_state) if return_state else output

    def recurrent_state_bytes(self, batch_size=1):
        elements = self.num_kv_heads * self.feature_dim * self.head_dim
        elements += self.num_kv_heads * self.feature_dim
        if self.window:
            elements += 2 * (self.window - 1) * self.num_kv_heads * self.head_dim
        return batch_size * elements * self.wq.element_size()


def sanity_check(device):
    torch.manual_seed(7)
    core = PreconditionedDelta2Layer(4, 2, 16, 24).to(device)
    q = torch.randn(2, 4, 40, 16, device=device, requires_grad=True)
    k = torch.randn(2, 2, 40, 16, device=device, requires_grad=True)
    v = torch.randn(2, 2, 40, 16, device=device, requires_grad=True)
    out = core(q, k, v)
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite forward")
    out.square().mean().backward()
    if not all(torch.isfinite(x.grad).all() for x in (q, k, v)):
        raise RuntimeError("non-finite gradients")
    with torch.no_grad():
        base = core(q.detach(), k.detach(), v.detach())
        k2, v2 = k.detach().clone(), v.detach().clone()
        k2[:, :, 25:] += 100 * torch.randn_like(k2[:, :, 25:])
        v2[:, :, 25:] += 100 * torch.randn_like(v2[:, :, 25:])
        perturbed = core(q.detach(), k2, v2)
        causal_error = float((base[:, :, :25] - perturbed[:, :, :25]).abs().max())
        full = core(q.detach()[:, :, :32], k.detach()[:, :, :32], v.detach()[:, :, :32])
        first, state = core(
            q.detach()[:, :, :16], k.detach()[:, :, :16], v.detach()[:, :, :16], return_state=True
        )
        second, _ = core(
            q.detach()[:, :, 16:32], k.detach()[:, :, 16:32], v.detach()[:, :, 16:32],
            state=state, return_state=True
        )
        streaming_error = float((full - torch.cat((first, second), dim=2)).abs().max())
    if causal_error >= 5e-5 or streaming_error >= 1e-4:
        raise RuntimeError(f"sanity failed: causal={causal_error}, streaming={streaming_error}")
    return {"causal_max_error": causal_error, "streaming_max_error": streaming_error}


def profile(name):
    return {
        "quick": dict(train=32, validation=8, test=12, transfer=120, lm=16, every=40),
        "balanced": dict(train=64, validation=12, test=24, transfer=320, lm=48, every=80),
        "strong": dict(train=128, validation=24, test=48, transfer=700, lm=100, every=100),
    }[name]


def candidate_specs(profile_name):
    specs = [
        dict(name="delta2_f64", kind="delta2", feature_dim=64, window=0),
        dict(name="pdelta2_f64", kind="pdelta2", feature_dim=64, window=0),
        dict(name="pdelta2_f96", kind="pdelta2", feature_dim=96, window=0),
        dict(name="pdelta2_f128", kind="pdelta2", feature_dim=128, window=0),
        dict(name="pdelta2_w32_f96", kind="pdelta2", feature_dim=96, window=32),
        dict(name="pdelta2_w64_f128", kind="pdelta2", feature_dim=128, window=64),
        dict(name="delta2_w32_f64", kind="delta2", feature_dim=64, window=32),
    ]
    return specs[:4] + [specs[4]] if profile_name == "quick" else specs


def make_core(spec, model, device):
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    if spec["kind"] == "delta2":
        variant = "cenn_delta2_window" if spec["window"] else "cenn_delta2"
        return ResearchCeNNLayer(
            num_heads, num_kv_heads, head_dim,
            feature_dim=spec["feature_dim"], variant=variant,
            window=max(1, spec["window"]), chunk_size=32,
        ).to(device)
    return PreconditionedDelta2Layer(
        num_heads, num_kv_heads, head_dim,
        feature_dim=spec["feature_dim"], window=spec["window"], chunk_size=32,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "balanced", "strong"], default="balanced")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--train-context", type=int, default=256)
    parser.add_argument("--test-contexts", default="256,512,1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/pdelta2-beat-transformer")
    parser.add_argument("--stress-refinement", action=argparse.BooleanOptionalAction, default=True)
    a = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = sanity_check(device)
    print("SANITY:", checks, flush=True)

    cfg = profile(a.profile)
    contexts = [int(x) for x in a.test_contexts.split(",")]
    if a.train_context not in contexts:
        contexts.insert(0, a.train_context)

    api = HfApi()
    model_sha = api.model_info(a.base_model).sha
    data_sha = api.dataset_info(a.dataset).sha
    tokenizer = AutoTokenizer.from_pretrained(a.base_model, revision=model_sha)
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, revision=model_sha, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    model.requires_grad_(False)

    rows = load_dataset(
        a.dataset, a.dataset_config, split="train", streaming=True, revision=data_sha
    )
    counts = {"train": cfg["train"], "validation": cfg["validation"], "test": cfg["test"]}
    blocks, _hashes = collect_documents(rows, tokenizer, counts, length=max(contexts))

    train_capture = capture_samples(model, blocks["train"], [a.layer], a.train_context, device)[a.layer]
    val_capture = capture_samples(model, blocks["validation"], [a.layer], a.train_context, device)[a.layer]
    test_capture = capture_samples(model, blocks["test"], [a.layer], a.train_context, device)[a.layer]
    teacher_test = {c: evaluate_nll(model, blocks["test"], c, device) for c in contexts}
    teacher_val = statistics.fmean(evaluate_nll(model, blocks["validation"], a.train_context, device))

    fit_args = SimpleNamespace(
        device=device, seed=a.seed, lr=3e-3, steps=cfg["transfer"],
        lm_steps=cfg["lm"], eval_every=cfg["every"], context=a.train_context,
    )
    outdir = Path(a.output_dir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    records, histories, cores = [], [], {}
    specs = candidate_specs(a.profile)

    for spec in specs:
        name = spec["name"]
        print("\n" + "=" * 90 + f"\n{name}", flush=True)
        torch.manual_seed(a.seed)
        core = make_core(spec, model, device)
        checkpoint = outdir / "checkpoints" / f"{name}.pt"
        histories += fit_transfer(core, train_capture, val_capture, fit_args, checkpoint, name)
        best_val, lm_hist = fit_language_loss(
            model, a.layer, core, blocks["train"], train_capture, blocks["validation"],
            fit_args, checkpoint, name,
        )
        histories += lm_hist
        transfer = evaluate_transfer(core, val_capture, device)
        records.append({
            **spec,
            "validation_nll": best_val,
            "validation_ppl": math.exp(best_val),
            "output_nmse": transfer["output_nmse"],
            "output_cosine": transfer["output_cosine"],
            "trainable_parameters": sum(p.numel() for p in core.parameters() if p.requires_grad),
        })
        cores[name] = core

    validation = pd.DataFrame(records).sort_values("validation_nll").reset_index(drop=True)
    winner_name = str(validation.iloc[0]["name"])

    if a.stress_refinement and float(validation.iloc[0]["validation_nll"]) >= teacher_val:
        p_rows = validation[validation["kind"] == "pdelta2"]
        stress_name = str(p_rows.iloc[0]["name"])
        stress_core = cores[stress_name]
        stress_args = SimpleNamespace(**vars(fit_args))
        stress_args.lm_steps = cfg["lm"]
        stress_args.eval_every = max(8, min(cfg["every"], max(1, cfg["lm"] // 2)))
        stress_val, stress_hist = fit_language_loss(
            model, a.layer, stress_core, blocks["train"], train_capture, blocks["validation"],
            stress_args, outdir / "checkpoints" / f"{stress_name}-stress.pt", stress_name + "_stress",
        )
        histories += stress_hist
        idx = validation.index[validation["name"] == stress_name][0]
        if stress_val < float(validation.loc[idx, "validation_nll"]):
            validation.loc[idx, "validation_nll"] = stress_val
            validation.loc[idx, "validation_ppl"] = math.exp(stress_val)
        validation = validation.sort_values("validation_nll").reset_index(drop=True)
        winner_name = str(validation.iloc[0]["name"])

    # Winner is locked before test metrics.
    winner = cores[winner_name]
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    def transformer_kv_bytes(context):
        return 2 * context * num_kv_heads * head_dim * 2

    def score(core, name, context):
        with replace_attention(model, a.layer, core):
            candidate = evaluate_nll(model, blocks["test"], context, device)
        reference = teacher_test[context]
        delta, low, high = paired_interval(candidate, reference, seed=2026, repeats=2000)
        ratio = core.recurrent_state_bytes() / transformer_kv_bytes(context)
        if high < 0:
            verdict = "strict_quality_win"
        elif delta < 0:
            verdict = "point_quality_win"
        elif low >= -a.nll_margin and high <= a.nll_margin and ratio < 0.5:
            verdict = "memory_efficient_parity"
        else:
            verdict = quality_label(delta, low, high, len(candidate), a.nll_margin)
        return {
            "candidate": name,
            "context": context,
            "candidate_nll": statistics.fmean(candidate),
            "transformer_nll": statistics.fmean(reference),
            "delta_nll": delta,
            "ci95_low": low,
            "ci95_high": high,
            "candidate_ppl": math.exp(statistics.fmean(candidate)),
            "transformer_ppl": math.exp(statistics.fmean(reference)),
            "ppl_ratio": math.exp(delta),
            "state_bytes": core.recurrent_state_bytes(),
            "transformer_kv_bytes_fp16": transformer_kv_bytes(context),
            "state_vs_transformer_fp16": ratio,
            "verdict": verdict,
            "selected_on_validation": name == winner_name,
        }

    test_rows = [score(cores[s["name"]], s["name"], a.train_context) for s in specs]
    long_names = [winner_name] + ([] if winner_name == "delta2_f64" else ["delta2_f64"])
    for name in long_names:
        for context in contexts:
            if context != a.train_context:
                test_rows.append(score(cores[name], name, context))

    tests = pd.DataFrame(test_rows).sort_values(["context", "delta_nll"]).reset_index(drop=True)
    diagnostics = {
        **gradient_fidelity(winner, test_capture[0], device),
        **benchmark_kernels(winner, test_capture[0], device, repeats=10),
    }

    validation.to_csv(outdir / "validation_summary.csv", index=False)
    tests.to_csv(outdir / "test_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(outdir / "training_history.csv", index=False)
    report = {
        "profile": a.profile,
        "base_model": a.base_model,
        "model_revision": model_sha,
        "dataset": a.dataset,
        "dataset_revision": data_sha,
        "target_layer": a.layer,
        "train_context": a.train_context,
        "test_contexts": contexts,
        "winner_selected_on_validation": winner_name,
        "teacher_validation_nll": teacher_val,
        "sanity": checks,
        "winner_diagnostics": diagnostics,
        "candidate_specs": specs,
        "win_definition": {
            "strict_quality_win": "paired bootstrap 95% CI of candidate-minus-Transformer NLL < 0",
            "memory_efficient_parity": f"paired CI inside +/-{a.nll_margin} nats and state ratio < 0.5",
        },
    }
    (outdir / "pdelta2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nVALIDATION")
    print(validation.to_string(index=False))
    print("\nTEST")
    print(tests.to_string(index=False))
    print("\nWINNER:", winner_name)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
