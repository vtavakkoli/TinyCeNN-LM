#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_cennmixer_v4 import (
    CeNNMixerV4Config,
    clone_cenn_state_v4,
    direct_mixer_losses_v4,
    finalize_cenn_only_v4,
    freeze_all_except_cenn_v4,
    install_cenn_mixer_v4,
    load_cenn_state_v4,
    reset_stream_state_v4,
)
from tinycenn_lm.qwen35_cennmixer_v4_train import (
    causal_ce,
    context_lengths,
    forward_kl,
    generation_health,
    generation_suite,
    hidden_losses,
    load_data,
    on_policy_distill_loss,
    probe_contexts,
    quality_key,
    quality_ok,
    quality_targets,
    quality_violation,
    reverse_kl,
    select_positions,
    top1_margin_loss,
    topk_rank_loss,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Direct single-layer CeNNMixer-v4 distillation: no alpha blending"
    )
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--layer", type=int, default=0)

    # 1024 is the real training context, not a late curriculum stage.
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--context-lengths", default="256,512,1024")
    p.add_argument("--train-blocks", type=int, default=768)
    p.add_argument("--val-blocks", type=int, default=4)

    p.add_argument("--groups", type=int, default=24)
    p.add_argument("--cell-dim", type=int, default=32)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--assoc-heads", type=int, default=16)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--conv-kernel", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-updates", type=int, default=50)
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--min-updates", type=int, default=200)
    p.add_argument("--probe-every", type=int, default=50)
    p.add_argument("--logit-tokens", type=int, default=128)
    p.add_argument("--topk", type=int, default=32)

    p.add_argument("--on-policy-every", type=int, default=100)
    p.add_argument("--on-policy-tokens", type=int, default=16)

    p.add_argument("--min-top1", type=float, default=0.97)
    p.add_argument("--max-kl", type=float, default=0.03)
    p.add_argument("--max-hidden-mse", type=float, default=0.05)
    p.add_argument("--max-mixer-mse", type=float, default=0.12)
    p.add_argument("--max-mixer-cosine", type=float, default=0.07)
    p.add_argument("--max-mixer-delta", type=float, default=0.20)
    p.add_argument("--max-mixer-multiscale", type=float, default=0.12)
    p.add_argument("--max-ce-gap", type=float, default=0.05)

    p.add_argument("--output-dir", default="results/cennmixer_v4_direct_1024")
    p.add_argument("--seed", type=int, default=8621)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_for(device):
    if device.type != "cuda":
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    except TypeError:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def lr_at(step, a):
    if step <= a.warmup_updates:
        return a.lr * max(step, 1) / max(a.warmup_updates, 1)
    progress = (step - a.warmup_updates) / max(a.updates - a.warmup_updates, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return a.min_lr + (a.lr - a.min_lr) * cosine


def main():
    a = parse_args()
    seed_all(a.seed)

    if a.seq_len < 1024:
        raise ValueError("This direct-quality runner is intentionally configured for >=1024 token training.")
    if a.updates <= 0 or a.min_updates < 0 or a.probe_every <= 0:
        raise ValueError("updates/probe settings must be positive")
    if a.min_updates > a.updates:
        raise ValueError("min-updates cannot exceed updates")

    lengths = context_lengths(a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_for(device)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("DEVICE", device, "dtype", dtype, flush=True)
    print("DIRECT MODE: alpha removed; compact mixer is active from update 1", flush=True)
    print("TRAIN CONTEXT", a.seq_len, "EVAL CONTEXTS", lengths, flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        a.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(
        a.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    for p in teacher.parameters():
        p.requires_grad_(False)

    if not 0 <= a.layer < len(student.model.layers):
        raise ValueError(f"Layer index out of range: {a.layer}")

    layer_kind = getattr(student.model.layers[a.layer], "block_type", "unknown")
    if layer_kind != "linear_attention":
        raise ValueError(
            f"Layer {a.layer} is {layer_kind!r}; this experiment is designed for one Qwen3.5 "
            "Gated-Delta linear-attention layer. Choose a linear_attention layer."
        )

    cfg = CeNNMixerV4Config(
        hidden_size=int(student.config.hidden_size),
        groups=a.groups,
        cell_dim=a.cell_dim,
        graph_steps=a.graph_steps,
        assoc_heads=a.assoc_heads,
        key_dim=a.key_dim,
        value_dim=a.value_dim,
        conv_kernel=a.conv_kernel,
    )

    wrapper = install_cenn_mixer_v4(student, a.layer, cfg)
    print("TEACHER-ALIGNED INIT", json.dumps(wrapper.init_report), flush=True)
    if not wrapper.init_report.get("used"):
        raise RuntimeError(
            "Teacher-aligned initialization was not available. Refusing to start a 1024-token "
            "quality run from a random recurrent state: " + json.dumps(wrapper.init_report)
        )

    train, validation = load_data(tok, a)

    trainable = freeze_all_except_cenn_v4(student)
    cenn_params = sum(p.numel() for p in trainable)

    teacher_layer = teacher.model.layers[a.layer]
    teacher_mixer = teacher_layer.linear_attn
    replaced_params = sum(p.numel() for p in teacher_mixer.parameters())

    print("CeNN-v4 trainable params", f"{cenn_params:,}", flush=True)
    print("Native Qwen mixer params", f"{replaced_params:,}", flush=True)
    print("Mixer parameter reduction", f"{100 * (1 - cenn_params / max(replaced_params, 1)):.2f}%", flush=True)
    print("Runtime associative state floats", a.assoc_heads * a.key_dim * a.value_dim, flush=True)

    # The replaced layer is early, so gradients must traverse the frozen suffix.
    # Checkpointing trades compute for much lower activation memory at 1024 tokens.
    try:
        student.gradient_checkpointing_enable()
        print("Gradient checkpointing: enabled", flush=True)
    except Exception as e:
        print("Gradient checkpointing unavailable:", repr(e), flush=True)

    if hasattr(student.config, "use_cache"):
        student.config.use_cache = False

    optimizer = torch.optim.AdamW(
        trainable,
        lr=a.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    rng = random.Random(a.seed + 2026)

    history = []

    def evaluate(with_generation=False, local_teacher=True):
        metrics, per_context = probe_contexts(
            student,
            teacher,
            validation,
            a.layer,
            device,
            local_teacher=local_teacher,
            logit_tokens=a.logit_tokens,
        )
        gens = []
        if with_generation:
            gens = generation_suite(student, teacher, tok, device, lengths)
            metrics["generation_ok"] = float(generation_health(gens))
        else:
            # Numeric model selection is intentionally cheap.  If all numeric
            # gates pass we immediately run the real generation gate.
            metrics["generation_ok"] = 1.0
        return metrics, per_context, gens

    print("\n=== INITIAL DIRECT STUDENT ===", flush=True)
    initial, initial_contexts, initial_gens = evaluate(with_generation=True)
    print("TARGETS", json.dumps(quality_targets(a)), flush=True)
    print("INITIAL", json.dumps(initial), flush=True)

    best_metrics = dict(initial)
    best_state = clone_cenn_state_v4(student)
    best_key = quality_key(best_metrics, a)
    best_step = 0

    torch.save(
        {
            "state": best_state,
            "config": cfg.to_dict(),
            "layer": a.layer,
            "layer_kind": layer_kind,
            "base_model": a.base_model,
            "step": 0,
            "metrics": best_metrics,
            "teacher_aligned_init": wrapper.init_report,
        },
        out / "cennmixer_v4_best.pt",
    )

    stop_early = False

    for step in range(1, a.updates + 1):
        student.train()
        reset_stream_state_v4(student)

        ids = train[rng.randrange(len(train))].to(device)
        x, y = ids[:, :-1], ids[:, 1:]

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, a)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            to = teacher(
                input_ids=x,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        so = student(
            input_ids=x,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        loc = direct_mixer_losses_v4(student)
        hm, dm = hidden_losses(so.hidden_states, to.hidden_states, a.layer)

        sl, sy = select_positions(so.logits, y, a.logit_tokens)
        tl, _ = select_positions(to.logits, None, a.logit_tokens)

        fkl = forward_kl(sl, tl)
        rkl = reverse_kl(sl, tl)
        rank = topk_rank_loss(sl, tl, a.topk)
        margin = top1_margin_loss(sl, tl)
        ce = causal_ce(sl, sy)

        local_loss = (
            0.42 * loc["mse"]
            + 0.12 * loc["cosine"]
            + 0.16 * loc["delta"]
            + 0.12 * loc["multiscale"]
            + 0.06 * loc["rms"]
            + 0.12 * loc["tail"]
        )
        global_loss = (
            0.30 * fkl
            + 0.20 * rkl
            + 0.18 * rank
            + 0.15 * margin
            + 0.10 * hm
            + 0.05 * dm
            + 0.02 * ce
        )

        # Early training is dominated by direct function approximation.  As the
        # mixer converges, weight moves toward end-to-end behavior automatically.
        mixer_level = min(max(float(loc["mse"].detach()), 0.0), 1.0)
        local_weight = 0.62 + 0.23 * mixer_level
        loss = local_weight * local_loss + (1.0 - local_weight) * global_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")

        train_row = {
            "step": step,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": float(loss.detach()),
            "local_weight": local_weight,
            "train_mixer_mse": float(loc["mse"].detach()),
            "train_mixer_cosine": float(loc["cosine"].detach()),
            "train_mixer_delta": float(loc["delta"].detach()),
            "train_mixer_multiscale": float(loc["multiscale"].detach()),
            "train_mixer_rms": float(loc["rms"].detach()),
            "train_mixer_tail": float(loc["tail"].detach()),
            "train_forward_kl": float(fkl.detach()),
            "train_reverse_kl": float(rkl.detach()),
            "train_rank_loss": float(rank.detach()),
            "train_margin_loss": float(margin.detach()),
            "train_hidden_mse": float(hm.detach()),
            "train_delta_hidden_mse": float(dm.detach()),
            "train_ce": float(ce.detach()),
            "train_on_policy_loss": 0.0,
            "train_on_policy_scale": 0.0,
        }

        loss.backward()

        # Free the 1024-token offline graph before occasional on-policy feedback.
        del so, to, sl, tl

        if a.on_policy_every > 0 and step % a.on_policy_every == 0:
            prefix_len = min(256, x.shape[1])
            prefix = x[:, :prefix_len].detach()
            onp = on_policy_distill_loss(
                student,
                teacher,
                prefix,
                device,
                max_new_tokens=a.on_policy_tokens,
                logit_tokens=min(64, a.logit_tokens),
            )
            raw = float(onp.detach())
            scale = min(1.0, 0.20 / max(raw, 1e-8))
            (0.10 * onp * scale).backward()
            train_row["train_on_policy_loss"] = raw
            train_row["train_on_policy_scale"] = scale

        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()

        should_probe = step == 1 or step % a.probe_every == 0 or step == a.updates
        if not should_probe:
            continue

        vm, per_context, _ = evaluate(with_generation=False)
        row = {**train_row, **vm}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

        print("PROBE", json.dumps(row), flush=True)

        key = quality_key(vm, a)
        if key < best_key:
            best_key = key
            best_metrics = dict(vm)
            best_state = clone_cenn_state_v4(student)
            best_step = step
            torch.save(
                {
                    "state": best_state,
                    "config": cfg.to_dict(),
                    "layer": a.layer,
                    "layer_kind": layer_kind,
                    "base_model": a.base_model,
                    "step": best_step,
                    "metrics": best_metrics,
                    "teacher_aligned_init": wrapper.init_report,
                },
                out / "cennmixer_v4_best.pt",
            )
            print(
                "✓ NEW BEST",
                json.dumps({
                    "step": best_step,
                    "violation": quality_violation(best_metrics, a),
                    "numeric_pass": quality_ok(best_metrics, a),
                }),
                flush=True,
            )

        # Only pay for generation when numerical convergence is already good.
        if step >= a.min_updates and quality_ok(vm, a):
            gated, gated_contexts, gated_gens = evaluate(with_generation=True)
            print("GENERATION GATE", json.dumps(gated), flush=True)
            if quality_ok(gated, a):
                best_metrics = dict(gated)
                best_state = clone_cenn_state_v4(student)
                best_step = step
                stop_early = True
                break

    load_cenn_state_v4(student, best_state)
    reset_stream_state_v4(student)

    print("\n=== BEST DIRECT WRAPPER ===", flush=True)
    best_probe, best_contexts, best_gens = evaluate(with_generation=True)
    print("BEST STEP", best_step, flush=True)
    print("BEST METRICS", json.dumps(best_probe), flush=True)

    # Remove the frozen native mixer entirely and verify that metrics do not
    # change because training has already been 100% compact from step 1.
    finalize_cenn_only_v4(student)
    reset_stream_state_v4(student)
    final_probe, final_contexts = probe_contexts(
        student,
        teacher,
        validation,
        a.layer,
        device,
        local_teacher=False,
        logit_tokens=a.logit_tokens,
    )
    final_gens = generation_suite(student, teacher, tok, device, lengths)
    final_probe["generation_ok"] = float(generation_health(final_gens))

    strict_final = (
        final_probe["generation_ok"] == 1
        and abs(final_probe["ce_gap"]) <= a.max_ce_gap
        and final_probe["top1"] >= a.min_top1
        and final_probe["kl"] <= a.max_kl
        and final_probe["hidden_mse"] <= a.max_hidden_mse
    )

    torch.save(
        {
            "state": clone_cenn_state_v4(student),
            "config": cfg.to_dict(),
            "layer": a.layer,
            "layer_kind": layer_kind,
            "base_model": a.base_model,
            "step": best_step,
            "cenn_only": True,
            "metrics": final_probe,
            "teacher_aligned_init": wrapper.init_report,
        },
        out / "cennmixer_v4_final_cenn_only.pt",
    )

    (out / "generation_best.json").write_text(json.dumps(best_gens, indent=2), encoding="utf-8")
    (out / "generation_final.json").write_text(json.dumps(final_gens, indent=2), encoding="utf-8")

    report = {
        "architecture": "CeNNMixer-v4 Direct DeltaCell: contractive CeNN correction + spectrally initialized compact Gated-Delta memory",
        "training_mode": "direct compact mixer from update 1; no alpha interpolation",
        "base_model": a.base_model,
        "layer": a.layer,
        "layer_kind": layer_kind,
        "train_context": a.seq_len,
        "evaluated_context_lengths": lengths,
        "teacher_aligned_init": wrapper.init_report,
        "config": cfg.to_dict(),
        "cenn_params": cenn_params,
        "replaced_qwen_mixer_params": replaced_params,
        "mixer_param_reduction_pct": 100 * (1 - cenn_params / max(replaced_params, 1)),
        "runtime_associative_state_floats": a.assoc_heads * a.key_dim * a.value_dim,
        "best_step": best_step,
        "stopped_early": stop_early,
        "best_direct_probe": best_probe,
        "best_direct_per_context": best_contexts,
        "final_cenn_only_probe": final_probe,
        "final_cenn_only_per_context": final_contexts,
        "strict_quality_gate": strict_final,
        "targets": quality_targets(a),
        "args": vars(a),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFINAL", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
