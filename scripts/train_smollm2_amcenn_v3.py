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
from tinycenn_lm.smollm2_amcenn_v3 import (
    HybridLocalAMCeNNAttention,
    SmolAMCeNNV3Config,
    freeze_for_v3_calibration,
    replace_all_attention_v3,
    replace_attention_layers_v3,
    save_smollm2_amcenn_v3,
    v3_attention_stats,
    v3_global_parameter_groups,
    v3_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SmolLM2 AM-CeNN hybrid v3: exact-local attention + recurrent global memory"
    )
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--output-dir", default="checkpoints/smollm2-amcenn-hybrid-v3")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--local-window", type=int, default=32)
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--calibration-steps", type=int, default=20)
    p.add_argument("--calibration-lr", type=float, default=1e-4)
    p.add_argument("--calibration-eval-batches", type=int, default=2)
    p.add_argument("--calibration-eval-every", type=int, default=5)
    p.add_argument("--global-gate-init", type=float, default=0.05)
    p.add_argument("--global-gate-cap", type=float, default=0.35)
    p.add_argument("--feature-reg-weight", type=float, default=1e-6)
    p.add_argument("--gate-reg-weight", type=float, default=1e-4)
    p.add_argument("--final-max-tokens", type=int, default=500_000)
    p.add_argument("--final-grad-accum", type=int, default=8)
    p.add_argument("--final-lr", type=float, default=5e-5)
    p.add_argument("--qkvo-lr", type=float, default=3e-6)
    p.add_argument("--warmup-updates", type=int, default=25)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.25)
    p.add_argument("--cosine-weight", type=float, default=0.25)
    p.add_argument("--final-eval-batches", type=int, default=8)
    p.add_argument("--max-runtime-minutes", type=float, default=60.0)
    p.add_argument("--shuffle-buffer", type=int, default=2048)
    p.add_argument("--seed", type=int, default=83)
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
    return iter(
        batches(
            token_blocks(raw, tokenizer, args.text_field, args.context_length),
            args.batch_size,
        )
    )


def make_optimizer(params, lr: float, device: torch.device):
    try:
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=0.01,
            fused=(device.type == "cuda"),
        )
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

    for idx in layer_indices:
        if "hidden" not in captures[idx] or "target" not in captures[idx]:
            raise RuntimeError(f"failed to capture teacher attention tensors for layer {idx}")
    return captures


def attention_alignment_loss(pred: torch.Tensor, target: torch.Tensor, cosine_weight: float):
    pred32 = pred.float()
    target32 = target.float()
    mse = (pred32 - target32).square().mean()
    power = target32.square().mean().clamp_min(1e-5)
    nmse = mse / power
    cosine = 1.0 - F.cosine_similarity(pred32, target32, dim=-1).mean()
    return nmse + cosine_weight * cosine, nmse.detach(), cosine.detach()


def feature_delta_regularizer(model, layer_indices: list[int]) -> torch.Tensor:
    terms = []
    selected = set(layer_indices)
    for module in model.modules():
        if isinstance(module, HybridLocalAMCeNNAttention) and module.layer_idx in selected:
            delta = module.features.delta_projection
            if delta is not None:
                terms.append(delta.float().square().mean())
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


def gate_regularizer(model, layer_indices: list[int]) -> torch.Tensor:
    gates = []
    selected = set(layer_indices)
    for module in model.modules():
        if isinstance(module, HybridLocalAMCeNNAttention) and module.layer_idx in selected:
            gates.append(torch.sigmoid(module.global_gate_logit.float()).square().mean())
    if not gates:
        return next(model.parameters()).new_zeros(())
    return torch.stack(gates).mean()


def clamp_global_gates(model, cap: float) -> None:
    max_logit = math.log(cap / (1.0 - cap))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HybridLocalAMCeNNAttention):
                module.global_gate_logit.clamp_(max=max_logit)


def snapshot_layers(model, layer_indices: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    result = {}
    for idx in layer_indices:
        module = model.model.layers[idx].self_attn
        result[idx] = {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
    return result


def restore_layers(model, snapshot: dict[int, dict[str, torch.Tensor]]) -> None:
    for idx, state in snapshot.items():
        model.model.layers[idx].self_attn.load_state_dict(state, strict=True)


def evaluate_calibration(model, fixed_captures, group: list[int], amp, cosine_weight: float):
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
                    align, nmse, cosine = attention_alignment_loss(
                        pred, c["target"], cosine_weight
                    )
                    losses.append(align.detach().float())
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
    per_token = F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean() * (temperature ** 2)


def representation_loss(student_hidden, teacher_hidden) -> torch.Tensor:
    last_idx = min(len(student_hidden), len(teacher_hidden)) - 1
    if last_idx < 1:
        return student_hidden[-1].float().new_zeros(())
    indices = sorted(
        {
            max(1, min(last_idx, round(last_idx * fraction)))
            for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)
        }
    )
    terms = []
    for idx in indices:
        s = student_hidden[idx].float()
        t = teacher_hidden[idx].float()
        cosine = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        nmse = (s - t).square().mean() / t.square().mean().clamp_min(1e-5)
        terms.append(cosine + 0.25 * nmse)
    return torch.stack(terms).mean()


def structural_assertions(model) -> None:
    layers = model.model.layers
    if not all(isinstance(layer.self_attn, HybridLocalAMCeNNAttention) for layer in layers):
        raise RuntimeError("not every attention layer is AM-CeNN hybrid v3")
    if any("LlamaAttention" in module.__class__.__name__ for module in model.modules()):
        raise RuntimeError("Transformer full-attention module remains in final student")


def evaluate_student_teacher(
    student,
    teacher,
    eval_batches: list[torch.Tensor],
    *,
    device: torch.device,
    amp,
    temperature: float,
) -> dict:
    student.eval()
    teacher.eval()
    teacher_losses, student_losses, kls, hiddens = [], [], [], []
    with torch.no_grad():
        for cpu_ids in eval_batches:
            ids = cpu_ids.to(device, non_blocking=True)
            with amp():
                t_out = teacher(
                    input_ids=ids,
                    labels=ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                s_out = student(
                    input_ids=ids,
                    labels=ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                teacher_losses.append(t_out.loss.detach().float())
                student_losses.append(s_out.loss.detach().float())
                kls.append(distillation_kl(s_out.logits, t_out.logits, temperature).detach().float())
                hiddens.append(
                    representation_loss(s_out.hidden_states, t_out.hidden_states).detach().float()
                )

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


def main() -> None:
    args = parse_args()
    if args.local_window > args.context_length:
        raise ValueError("local_window cannot exceed context_length")
    if not 0.0 < args.global_gate_cap < 1.0:
        raise ValueError("global_gate_cap must be in (0, 1)")
    if args.global_gate_init > args.global_gate_cap:
        raise ValueError("global_gate_init cannot exceed global_gate_cap")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    amp = (
        (lambda: torch.autocast("cuda", dtype=dtype))
        if device.type == "cuda"
        else nullcontext
    )
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
    config = SmolAMCeNNV3Config(
        feature_dim=args.feature_dim,
        local_window=args.local_window,
        feature_seed=3030 + args.seed,
        antithetic_features=True,
        learnable_feature_correction=True,
        global_gate_init=args.global_gate_init,
    )

    train_iter = make_batch_iter(args, tokenizer, args.seed)
    calibration_eval_iter = make_batch_iter(args, tokenizer, args.seed + 100_003)
    calibration_eval_ids = [
        next(calibration_eval_iter).to(device, non_blocking=True)
        for _ in range(args.calibration_eval_batches)
    ]
    global_eval_iter = make_batch_iter(args, tokenizer, args.seed + 200_003)
    global_eval_batches = [
        next(global_eval_iter).cpu()
        for _ in range(args.final_eval_batches)
    ]

    started = time.perf_counter()
    num_layers = int(student.config.num_hidden_layers)
    groups = [
        list(range(i, min(i + args.group_size, num_layers)))
        for i in range(0, num_layers, args.group_size)
    ]
    calibration_reports = []
    calibration_tokens = 0

    print("\n=== Phase 1: conservative layerwise hybrid calibration ===")
    for stage_id, group in enumerate(groups, start=1):
        replace_attention_layers_v3(student, config, group)
        trainable = freeze_for_v3_calibration(student, group)
        optimizer = make_optimizer(trainable, args.calibration_lr, device)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and dtype == torch.float16)
        )

        fixed_captures = [
            capture_teacher_attention_io(teacher, ids, group, amp)
            for ids in calibration_eval_ids
        ]
        initial_eval = evaluate_calibration(
            student, fixed_captures, group, amp, args.cosine_weight
        )
        best_eval = dict(initial_eval)
        best_step = 0
        best_state = snapshot_layers(student, group)

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
                    align, _, _ = attention_alignment_loss(
                        pred, c["target"], args.cosine_weight
                    )
                    losses.append(align)
                align_loss = torch.stack(losses).mean()
                reg = feature_delta_regularizer(student, group)
                gate_reg = gate_regularizer(student, group)
                loss = (
                    align_loss
                    + args.feature_reg_weight * reg
                    + args.gate_reg_weight * gate_reg
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite v3 calibration loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            clamp_global_gates(student, args.global_gate_cap)
            calibration_tokens += ids.numel()

            if (
                step == 1
                or step % args.calibration_eval_every == 0
                or step == args.calibration_steps
            ):
                metrics = evaluate_calibration(
                    student, fixed_captures, group, amp, args.cosine_weight
                )
                if metrics["alignment"] < best_eval["alignment"]:
                    best_eval = dict(metrics)
                    best_step = step
                    best_state = snapshot_layers(student, group)
                gate = scalar_value(
                    torch.stack(
                        [
                            torch.sigmoid(
                                student.model.layers[idx]
                                .self_attn.global_gate_logit.detach()
                                .float()
                            ).mean()
                            for idx in group
                        ]
                    ).mean()
                )
                print(
                    f"stage={stage_id}/{len(groups)} layers={group[0]}-{group[-1]} "
                    f"step={step}/{args.calibration_steps} "
                    f"train_align={scalar_value(align_loss):.4f} "
                    f"val_align={metrics['alignment']:.4f} "
                    f"val_nmse={metrics['nmse']:.4f} gate={gate:.4f}"
                )

        restore_layers(student, best_state)
        calibration_reports.append(
            {
                "stage": stage_id,
                "layers": group,
                "initial_validation": initial_eval,
                "best_validation": best_eval,
                "best_step": best_step,
            }
        )
        print(
            f"stage={stage_id}/{len(groups)} restored best step={best_step} "
            f"val_align={best_eval['alignment']:.4f}"
        )

    replace_all_attention_v3(student, config)
    structural_assertions(student)

    print("\n=== Phase 2: protected global CE + teacher distillation ===")
    optimizer_groups, trainable = v3_global_parameter_groups(
        student,
        main_lr=args.final_lr,
        qkvo_lr=args.qkvo_lr,
    )
    optimizer = make_group_optimizer(optimizer_groups, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and dtype == torch.float16)
    )

    expected_updates = max(
        1,
        math.ceil(
            args.final_max_tokens
            / max(1, args.context_length * args.batch_size * args.final_grad_accum)
        ),
    )

    def lr_lambda(update: int) -> float:
        if update < args.warmup_updates:
            return max(0.05, (update + 1) / max(1, args.warmup_updates))
        progress = min(
            1.0,
            (update - args.warmup_updates)
            / max(1, expected_updates - args.warmup_updates),
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    params = v3_parameter_summary(student)
    print("parameter summary:", json.dumps(params, indent=2))
    student.train()
    optimizer.zero_grad(set_to_none=True)

    final_tokens = 0
    micro = 0
    updates = 0
    stop_reason = "token_budget"
    last = {
        "ce": float("nan"),
        "kl": float("nan"),
        "hidden": float("nan"),
        "total": float("nan"),
    }

    while final_tokens < args.final_max_tokens:
        if (time.perf_counter() - started) / 60.0 >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            break

        ids = next(train_iter).to(device, non_blocking=True)
        with torch.no_grad(), amp():
            t_out = teacher(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        with amp():
            s_out = student(
                input_ids=ids,
                labels=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            kl = distillation_kl(s_out.logits, t_out.logits, args.temperature)
            hidden = representation_loss(s_out.hidden_states, t_out.hidden_states)
            loss = (
                args.ce_weight * s_out.loss
                + args.kl_weight * kl
                + args.hidden_weight * hidden
            )
            scaled = loss / args.final_grad_accum

        if not torch.isfinite(scaled):
            raise RuntimeError("non-finite v3 global distillation loss")
        scaler.scale(scaled).backward()
        micro += 1
        final_tokens += ids.numel()
        last = {
            "ce": scalar_value(s_out.loss),
            "kl": scalar_value(kl),
            "hidden": scalar_value(hidden),
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
        clamp_global_gates(student, args.global_gate_cap)
        updates += 1

        if updates == 1 or updates % args.log_every == 0:
            gate_stats = v3_attention_stats(student)
            main_lr = optimizer.param_groups[0]["lr"]
            qkvo_lr = optimizer.param_groups[1]["lr"]
            print(
                f"global update={updates} tokens={final_tokens:,} "
                f"ce={last['ce']:.4f} kl={last['kl']:.4f} "
                f"hidden={last['hidden']:.4f} gate={gate_stats['mean_global_gate']:.4f} "
                f"grad={grad_norm:.3f} lr={main_lr:.2e}/{qkvo_lr:.2e}"
            )

    elapsed = time.perf_counter() - started
    final_eval = evaluate_student_teacher(
        student,
        teacher,
        global_eval_batches,
        device=device,
        amp=amp,
        temperature=args.temperature,
    )
    gate_stats = v3_attention_stats(student)
    report = {
        "status": "trained",
        "architecture": "smollm2-amcenn-hybrid-v3",
        "method_version": 3,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "evaluation_performed": True,
        "stop_reason": stop_reason,
        "context_length": args.context_length,
        "local_window": args.local_window,
        "feature_dim": args.feature_dim,
        "global_gate_init": args.global_gate_init,
        "global_gate_cap": args.global_gate_cap,
        "progressive_group_size": args.group_size,
        "calibration_steps_per_group": args.calibration_steps,
        "calibration_tokens": calibration_tokens,
        "calibration_stages": calibration_reports,
        "final_seen_tokens": final_tokens,
        "final_updates": updates,
        "parameters": params,
        "last_training_ce": last["ce"],
        "last_distillation_kl": last["kl"],
        "last_hidden_alignment": last["hidden"],
        "gate_stats": gate_stats,
        "evaluation": final_eval,
        "elapsed_minutes": elapsed / 60.0,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        ),
    }

    output_dir = Path(args.output_dir)
    save_smollm2_amcenn_v3(
        student,
        output_dir,
        config=config,
        base_model=args.base_model,
        extra_metadata=report,
    )
    tokenizer.save_pretrained(output_dir)
    (output_dir / "smollm2_amcenn_v3_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
