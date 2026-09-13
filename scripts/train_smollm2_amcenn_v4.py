#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.smollm2_amcenn import DEFAULT_SMOLLM2
from tinycenn_lm.smollm2_amcenn_v4 import (
    AdaptiveHybridAMCeNNAttentionV4,
    SmolAMCeNNV4Config,
    freeze_for_v4_calibration,
    replace_all_attention_v4,
    replace_attention_layers_v4,
    save_smollm2_amcenn_v4,
    v4_attention_stats,
    v4_global_parameter_groups,
    v4_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SmolLM2 AM-CeNN adaptive hybrid v4")
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--output-dir", default="checkpoints/smollm2-amcenn-adaptive-v4")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)

    p.add_argument("--easy-feature-dim", type=int, default=256)
    p.add_argument("--medium-feature-dim", type=int, default=320)
    p.add_argument("--hard-feature-dim", type=int, default=384)
    p.add_argument("--critical-feature-dim", type=int, default=512)
    p.add_argument("--easy-window", type=int, default=32)
    p.add_argument("--medium-window", type=int, default=48)
    p.add_argument("--hard-window", type=int, default=64)
    p.add_argument("--critical-window", type=int, default=96)
    p.add_argument("--anchor-tokens", type=int, default=8)
    p.add_argument("--gate-init", type=float, default=0.03)

    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--calibration-steps", type=int, default=20)
    p.add_argument("--calibration-lr", type=float, default=8e-5)
    p.add_argument("--calibration-eval-batches", type=int, default=2)
    p.add_argument("--calibration-eval-every", type=int, default=5)
    p.add_argument("--feature-reg-weight", type=float, default=1e-6)
    p.add_argument("--gate-weight-reg", type=float, default=2e-6)

    p.add_argument("--final-max-tokens", type=int, default=500_000)
    p.add_argument("--final-grad-accum", type=int, default=8)
    p.add_argument("--memory-lr", type=float, default=4e-5)
    p.add_argument("--qkvo-lr", type=float, default=2e-6)
    p.add_argument("--warmup-updates", type=int, default=25)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=0.6)
    p.add_argument("--kl-weight", type=float, default=1.2)
    p.add_argument("--hidden-weight", type=float, default=0.30)
    p.add_argument("--repetition-weight", type=float, default=0.10)
    p.add_argument("--recent-window", type=int, default=8)
    p.add_argument("--final-eval-batches", type=int, default=8)
    p.add_argument("--max-runtime-minutes", type=float, default=55.0)
    p.add_argument("--shuffle-buffer", type=int, default=2048)
    p.add_argument("--seed", type=int, default=97)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_value(value) -> float:
    if torch.is_tensor(value):
        return value.detach().float().item()
    return float(value)


def token_blocks(dataset, tokenizer, text_field: str, context_length: int):
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    buffer: list[int] = []
    for row in dataset:
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos)
        while len(buffer) >= context_length:
            yield torch.tensor(buffer[:context_length], dtype=torch.long)
            del buffer[:context_length]


def batches(blocks, batch_size: int):
    pending = []
    for block in blocks:
        pending.append(block)
        if len(pending) == batch_size:
            yield torch.stack(pending)
            pending.clear()


def make_batch_iter(args, tokenizer, seed: int):
    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=args.shuffle_buffer)
    return iter(batches(token_blocks(raw, tokenizer, args.text_field, args.context_length), args.batch_size))


def make_optimizer(params, lr: float, device: torch.device):
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
    except Exception:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def make_group_optimizer(groups, device: torch.device):
    try:
        return torch.optim.AdamW(groups, fused=(device.type == "cuda"))
    except Exception:
        return torch.optim.AdamW(groups)


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(v) for v in value)
    if isinstance(value, list):
        return [_detach_tree(v) for v in value]
    return value


def capture_teacher_attention_io(teacher, ids: torch.Tensor, layer_indices: list[int], amp):
    captures: dict[int, dict] = {idx: {} for idx in layer_indices}
    handles = []
    for idx in layer_indices:
        module = teacher.model.layers[idx].self_attn

        def pre_hook(mod, args, kwargs, layer_idx=idx):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("teacher attention hidden_states not found")
            captures[layer_idx]["hidden"] = hidden.detach()
            for key in ("position_embeddings", "position_ids", "attention_mask", "cache_position"):
                if key in kwargs and kwargs[key] is not None:
                    captures[layer_idx][key] = _detach_tree(kwargs[key])

        def post_hook(mod, args, kwargs, output, layer_idx=idx):
            target = output[0] if isinstance(output, (tuple, list)) else output
            captures[layer_idx]["target"] = target.detach()

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    try:
        with torch.no_grad(), amp():
            teacher(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captures


def attention_alignment_loss(pred: torch.Tensor, target: torch.Tensor):
    pred32 = pred.float()
    target32 = target.float()
    mse = (pred32 - target32).square().mean()
    power = target32.square().mean().clamp_min(1e-5)
    nmse = mse / power
    cosine = 1.0 - F.cosine_similarity(pred32, target32, dim=-1).mean()
    return nmse + 0.25 * cosine, nmse.detach(), cosine.detach()


def calibration_regularizer(model, group: list[int], gate_weight_reg: float, feature_reg: float):
    terms = []
    selected = set(group)
    for m in model.modules():
        if not isinstance(m, AdaptiveHybridAMCeNNAttentionV4) or m.layer_idx not in selected:
            continue
        if m.features.delta_projection is not None:
            terms.append(feature_reg * m.features.delta_projection.float().square().mean())
        terms.append(gate_weight_reg * m.gate_proj.weight.float().square().mean())
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).sum()


def snapshot_layers(model, layer_indices: list[int]):
    return {
        idx: {k: v.detach().cpu().clone() for k, v in model.model.layers[idx].self_attn.state_dict().items()}
        for idx in layer_indices
    }


def restore_layers(model, snapshot) -> None:
    for idx, state in snapshot.items():
        model.model.layers[idx].self_attn.load_state_dict(state, strict=True)


def evaluate_calibration(model, fixed_captures, group: list[int], amp):
    losses, nmses, cosines = [], [], []
    with torch.no_grad():
        for captures in fixed_captures:
            with amp():
                for idx in group:
                    c = captures[idx]
                    kwargs = {"use_cache": False}
                    for key in ("position_embeddings", "position_ids", "attention_mask", "cache_position"):
                        if key in c:
                            kwargs[key] = c[key]
                    pred = model.model.layers[idx].self_attn(c["hidden"], **kwargs)[0]
                    align, nmse, cosine = attention_alignment_loss(pred, c["target"])
                    losses.append(align.float())
                    nmses.append(nmse.float())
                    cosines.append(cosine.float())
    return {
        "alignment": scalar_value(torch.stack(losses).mean()),
        "nmse": scalar_value(torch.stack(nmses).mean()),
        "cosine": scalar_value(torch.stack(cosines).mean()),
    }


def distillation_kl(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean() * (temperature ** 2)


def recent_token_excess_loss(student_logits, teacher_logits, ids: torch.Tensor, window: int) -> torch.Tensor:
    """Penalize only repeated-token probability mass above the teacher.

    This is safer than a generic repetition penalty: if the teacher itself wants a
    recent token, the student is not penalized for matching it.
    """
    if window <= 0 or ids.shape[1] < 3:
        return student_logits.float().new_zeros(())
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    seq_len = ids.shape[1]
    terms = []
    max_offset = min(window, seq_len - 1)
    for offset in range(max_offset):
        # logit position j predicts token j+1; ids[j-offset] is a recent history token.
        j0 = offset
        j1 = seq_len - 1
        if j0 >= j1:
            continue
        candidate_ids = ids[:, : j1 - j0]
        s_slice = s_logp[:, j0:j1]
        t_slice = t_logp[:, j0:j1]
        s_recent = s_slice.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
        t_recent = t_slice.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
        terms.append(F.relu(s_recent - t_recent).mean())
    if not terms:
        return student_logits.float().new_zeros(())
    return torch.stack(terms).mean()


def representation_loss(student_hidden, teacher_hidden) -> torch.Tensor:
    last_idx = min(len(student_hidden), len(teacher_hidden)) - 1
    if last_idx < 1:
        return student_hidden[-1].float().new_zeros(())
    indices = sorted({max(1, min(last_idx, round(last_idx * f))) for f in (0.2, 0.4, 0.6, 0.8, 1.0)})
    terms = []
    for idx in indices:
        s = student_hidden[idx].float()
        t = teacher_hidden[idx].float()
        cosine = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        nmse = (s - t).square().mean() / t.square().mean().clamp_min(1e-5)
        terms.append(cosine + 0.25 * nmse)
    return torch.stack(terms).mean()


def evaluate_student_teacher(student, teacher, eval_batches, device, amp, temperature: float) -> dict:
    student.eval()
    teacher.eval()
    teacher_losses, student_losses, kls, hiddens = [], [], [], []
    with torch.no_grad():
        for cpu_ids in eval_batches:
            ids = cpu_ids.to(device, non_blocking=True)
            with amp():
                t_out = teacher(input_ids=ids, labels=ids, use_cache=False, output_hidden_states=True, return_dict=True)
                s_out = student(input_ids=ids, labels=ids, use_cache=False, output_hidden_states=True, return_dict=True)
            teacher_losses.append(t_out.loss.detach().float())
            student_losses.append(s_out.loss.detach().float())
            kls.append(distillation_kl(s_out.logits, t_out.logits, temperature).detach().float())
            hiddens.append(representation_loss(s_out.hidden_states, t_out.hidden_states).detach().float())
    teacher_ce = scalar_value(torch.stack(teacher_losses).mean())
    student_ce = scalar_value(torch.stack(student_losses).mean())
    return {
        "teacher_ce": teacher_ce,
        "student_ce": student_ce,
        "teacher_perplexity": math.exp(min(teacher_ce, 20.0)),
        "student_perplexity": math.exp(min(student_ce, 20.0)),
        "ce_gap": student_ce - teacher_ce,
        "distillation_kl": scalar_value(torch.stack(kls).mean()),
        "hidden_alignment": scalar_value(torch.stack(hiddens).mean()),
        "eval_tokens": sum(int(ids.numel()) for ids in eval_batches),
    }


def structural_assertions(model) -> None:
    layers = model.model.layers
    if not all(isinstance(layer.self_attn, AdaptiveHybridAMCeNNAttentionV4) for layer in layers):
        raise RuntimeError("not every attention layer is AM-CeNN adaptive v4")


def main() -> None:
    args = parse_args()
    if max(args.easy_window, args.medium_window, args.hard_window, args.critical_window) > args.context_length:
        raise ValueError("v4 local windows cannot exceed context_length")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    student.config.use_cache = False
    config = SmolAMCeNNV4Config(
        easy_feature_dim=args.easy_feature_dim,
        medium_feature_dim=args.medium_feature_dim,
        hard_feature_dim=args.hard_feature_dim,
        critical_feature_dim=args.critical_feature_dim,
        easy_window=args.easy_window,
        medium_window=args.medium_window,
        hard_window=args.hard_window,
        critical_window=args.critical_window,
        anchor_tokens=args.anchor_tokens,
        feature_seed=4040 + args.seed,
        gate_init=args.gate_init,
    )
    config.validate(student.config)

    train_iter = make_batch_iter(args, tokenizer, args.seed)
    cal_eval_iter = make_batch_iter(args, tokenizer, args.seed + 100_003)
    cal_eval_ids = [next(cal_eval_iter).to(device, non_blocking=True) for _ in range(args.calibration_eval_batches)]
    global_eval_iter = make_batch_iter(args, tokenizer, args.seed + 200_003)
    global_eval_batches = [next(global_eval_iter).cpu() for _ in range(args.final_eval_batches)]

    started = time.perf_counter()
    num_layers = int(student.config.num_hidden_layers)
    groups = [list(range(i, min(i + args.group_size, num_layers))) for i in range(0, num_layers, args.group_size)]
    calibration_reports = []
    calibration_tokens = 0

    print("\n=== Phase 1: layer-adaptive exact/AM-CeNN calibration ===")
    for stage_id, group in enumerate(groups, start=1):
        replace_attention_layers_v4(student, config, group)
        trainable = freeze_for_v4_calibration(student, group)
        optimizer = make_optimizer(trainable, args.calibration_lr, device)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
        fixed_captures = [capture_teacher_attention_io(teacher, ids, group, amp) for ids in cal_eval_ids]
        initial_eval = evaluate_calibration(student, fixed_captures, group, amp)
        best_eval = dict(initial_eval)
        best_step = 0
        best_state = snapshot_layers(student, group)

        profiles = [student.model.layers[idx].self_attn.profile.to_dict() for idx in group]
        for step in range(1, args.calibration_steps + 1):
            ids = next(train_iter).to(device, non_blocking=True)
            captures = capture_teacher_attention_io(teacher, ids, group, amp)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            with amp():
                for idx in group:
                    c = captures[idx]
                    kwargs = {"use_cache": False}
                    for key in ("position_embeddings", "position_ids", "attention_mask", "cache_position"):
                        if key in c:
                            kwargs[key] = c[key]
                    pred = student.model.layers[idx].self_attn(c["hidden"], **kwargs)[0]
                    align, _, _ = attention_alignment_loss(pred, c["target"])
                    losses.append(align)
                align_loss = torch.stack(losses).mean()
                reg = calibration_regularizer(student, group, args.gate_weight_reg, args.feature_reg_weight)
                loss = align_loss + reg
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite v4 calibration loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            calibration_tokens += ids.numel()

            if step == 1 or step % args.calibration_eval_every == 0 or step == args.calibration_steps:
                metrics = evaluate_calibration(student, fixed_captures, group, amp)
                if metrics["alignment"] < best_eval["alignment"]:
                    best_eval = dict(metrics)
                    best_step = step
                    best_state = snapshot_layers(student, group)
                gate = sum(float(student.model.layers[idx].self_attn.last_mean_gate) for idx in group) / len(group)
                p0 = profiles[0]
                print(
                    f"stage={stage_id}/{len(groups)} layers={group[0]}-{group[-1]} "
                    f"tier={p0['tier']} win={p0['local_window']} feat={p0['feature_dim']} "
                    f"step={step}/{args.calibration_steps} train_align={scalar_value(align_loss):.4f} "
                    f"val_align={metrics['alignment']:.4f} val_nmse={metrics['nmse']:.4f} gate={gate:.4f}"
                )
        restore_layers(student, best_state)
        calibration_reports.append({
            "stage": stage_id,
            "layers": group,
            "profiles": profiles,
            "initial_validation": initial_eval,
            "best_validation": best_eval,
            "best_step": best_step,
        })
        print(f"stage={stage_id}/{len(groups)} restored best step={best_step} val_align={best_eval['alignment']:.4f}")

    replace_all_attention_v4(student, config)
    structural_assertions(student)

    print("\n=== Phase 2: protected global distillation + teacher-aware anti-repetition ===")
    optimizer_groups, trainable = v4_global_parameter_groups(
        student, memory_lr=args.memory_lr, qkvo_lr=args.qkvo_lr
    )
    optimizer = make_group_optimizer(optimizer_groups, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    expected_updates = max(1, math.ceil(args.final_max_tokens / max(1, args.context_length * args.batch_size * args.final_grad_accum)))

    def lr_lambda(update: int) -> float:
        if update < args.warmup_updates:
            return max(0.05, (update + 1) / max(1, args.warmup_updates))
        progress = min(1.0, (update - args.warmup_updates) / max(1, expected_updates - args.warmup_updates))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    params = v4_parameter_summary(student)
    print("parameter summary:", json.dumps(params, indent=2))
    student.train()
    optimizer.zero_grad(set_to_none=True)

    final_tokens = micro = updates = 0
    stop_reason = "token_budget"
    last = {"ce": float("nan"), "kl": float("nan"), "hidden": float("nan"), "repeat": float("nan"), "total": float("nan")}

    while final_tokens < args.final_max_tokens:
        if (time.perf_counter() - started) / 60.0 >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            break
        ids = next(train_iter).to(device, non_blocking=True)
        with torch.no_grad(), amp():
            t_out = teacher(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
        with amp():
            s_out = student(input_ids=ids, labels=ids, use_cache=False, output_hidden_states=True, return_dict=True)
            kl = distillation_kl(s_out.logits, t_out.logits, args.temperature)
            hidden = representation_loss(s_out.hidden_states, t_out.hidden_states)
            repeat = recent_token_excess_loss(s_out.logits, t_out.logits, ids, args.recent_window)
            loss = (
                args.ce_weight * s_out.loss
                + args.kl_weight * kl
                + args.hidden_weight * hidden
                + args.repetition_weight * repeat
            )
            scaled = loss / args.final_grad_accum
        if not torch.isfinite(scaled):
            raise RuntimeError("non-finite v4 global loss")
        scaler.scale(scaled).backward()
        micro += 1
        final_tokens += ids.numel()
        last = {
            "ce": scalar_value(s_out.loss),
            "kl": scalar_value(kl),
            "hidden": scalar_value(hidden),
            "repeat": scalar_value(repeat),
            "total": scalar_value(loss),
        }
        del t_out, s_out
        if micro % args.final_grad_accum:
            continue
        scaler.unscale_(optimizer)
        grad_norm = scalar_value(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        updates += 1
        if updates == 1 or updates % args.log_every == 0:
            stats = v4_attention_stats(student)
            print(
                f"global update={updates} tokens={final_tokens:,} ce={last['ce']:.4f} "
                f"kl={last['kl']:.4f} hidden={last['hidden']:.4f} repeat={last['repeat']:.4f} "
                f"gate={stats['mean_global_gate']:.4f} grad={grad_norm:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e}"
            )

    elapsed = time.perf_counter() - started
    final_eval = evaluate_student_teacher(student, teacher, global_eval_batches, device, amp, args.temperature)
    attention_stats = v4_attention_stats(student)
    report = {
        "status": "trained",
        "architecture": "smollm2-amcenn-adaptive-v4",
        "method_version": 4,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "evaluation_performed": True,
        "stop_reason": stop_reason,
        "context_length": args.context_length,
        "anchor_tokens": args.anchor_tokens,
        "layer_profile": {
            "easy_window": args.easy_window, "medium_window": args.medium_window,
            "hard_window": args.hard_window, "critical_window": args.critical_window,
            "easy_feature_dim": args.easy_feature_dim, "medium_feature_dim": args.medium_feature_dim,
            "hard_feature_dim": args.hard_feature_dim, "critical_feature_dim": args.critical_feature_dim,
        },
        "calibration_steps_per_group": args.calibration_steps,
        "calibration_tokens": calibration_tokens,
        "calibration_stages": calibration_reports,
        "final_seen_tokens": final_tokens,
        "final_updates": updates,
        "parameters": params,
        "last_training_ce": last["ce"],
        "last_distillation_kl": last["kl"],
        "last_hidden_alignment": last["hidden"],
        "last_teacher_aware_repeat_loss": last["repeat"],
        "attention_stats": attention_stats,
        "evaluation": final_eval,
        "elapsed_minutes": elapsed / 60.0,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0,
    }

    output_dir = Path(args.output_dir)
    save_smollm2_amcenn_v4(student, output_dir, config=config, base_model=args.base_model, extra_metadata=report)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "smollm2_amcenn_v4_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
