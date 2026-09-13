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

from tinycenn_lm.smollm2_amcenn import ShardedTop2LlamaMLP, amcenn_router_stats
from tinycenn_lm.smollm2_amcenn_v2 import (
    DEFAULT_SMOLLM2,
    AMCeNNAttentionV2,
    SmolAMCeNNV2Config,
    convert_all_ffns_to_sharded_top2,
    freeze_for_global_training,
    freeze_for_group_calibration,
    replace_attention_layers,
    save_smollm2_amcenn_v2,
    v2_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progressive layerwise distillation for SmolLM2 AM-CeNN v2")
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--output-dir", default="checkpoints/smollm2-amcenn-top2-v2")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--calibration-steps", type=int, default=30)
    p.add_argument("--calibration-lr", type=float, default=3e-4)
    p.add_argument("--final-max-tokens", type=int, default=500_000)
    p.add_argument("--final-grad-accum", type=int, default=8)
    p.add_argument("--final-lr", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=0.25)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.5)
    p.add_argument("--cosine-weight", type=float, default=0.25)
    p.add_argument("--feature-reg-weight", type=float, default=1e-6)
    p.add_argument("--router-aux-weight", type=float, default=1e-4)
    p.add_argument("--router-z-weight", type=float, default=1e-5)
    p.add_argument("--max-runtime-minutes", type=float, default=60.0)
    p.add_argument("--shuffle-buffer", type=int, default=2048)
    p.add_argument("--seed", type=int, default=73)
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


def token_blocks(dataset, tokenizer, text_field: str, context_length: int):
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    buffer: list[int] = []
    for row in dataset:
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
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


def make_optimizer(params, lr: float, device: torch.device):
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
    except Exception:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


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
        # Use no_grad rather than inference_mode: captured teacher activations are
        # later consumed as constants by trainable student modules. Inference tensors
        # cannot be saved for backward by those student operations.
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
    target_power = target32.square().mean().clamp_min(1e-5)
    nmse = mse / target_power
    cosine = 1.0 - F.cosine_similarity(pred32, target32, dim=-1).mean()
    return nmse + cosine_weight * cosine, nmse.detach(), cosine.detach()


def feature_delta_regularizer(model, layer_indices: list[int]) -> torch.Tensor:
    terms = []
    selected = set(layer_indices)
    for module in model.modules():
        if isinstance(module, AMCeNNAttentionV2) and module.layer_idx in selected:
            delta = module.features.delta_projection
            if delta is not None:
                terms.append(delta.float().square().mean())
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


def distillation_kl(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    per_token = F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean() * (temperature**2)


def representation_loss(student_hidden, teacher_hidden) -> torch.Tensor:
    # embedding is index 0; use five evenly distributed decoder checkpoints.
    indices = (6, 12, 18, 24, 30)
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
    if not all(isinstance(layer.self_attn, AMCeNNAttentionV2) for layer in layers):
        raise RuntimeError("not every attention layer is AM-CeNN v2")
    if not all(isinstance(layer.mlp, ShardedTop2LlamaMLP) for layer in layers):
        raise RuntimeError("not every FFN is sharded Top-2")
    if any("LlamaAttention" in m.__class__.__name__ for m in model.modules()):
        raise RuntimeError("Transformer self-attention remains in final student")


def main() -> None:
    args = parse_args()
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
    config = SmolAMCeNNV2Config(
        feature_dim=args.feature_dim,
        num_shards=8,
        top_k=2,
        feature_seed=2026 + args.seed,
        antithetic_features=True,
        learnable_feature_correction=True,
    )
    convert_all_ffns_to_sharded_top2(student, config)

    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    batch_iter = iter(batches(token_blocks(raw, tokenizer, args.text_field, args.context_length), args.batch_size))

    start = time.perf_counter()
    num_layers = int(student.config.num_hidden_layers)
    groups = [list(range(i, min(i + args.group_size, num_layers))) for i in range(0, num_layers, args.group_size)]
    stage_reports = []
    calibration_tokens = 0

    print("\n=== Phase 1: direct teacher-attention calibration ===")
    for stage_id, group in enumerate(groups, start=1):
        replace_attention_layers(student, config, group)
        trainable = freeze_for_group_calibration(student, group)
        optimizer = make_optimizer(trainable, args.calibration_lr, device)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
        first_loss = None
        last_loss = last_nmse = last_cos = float("nan")
        completed_steps = 0

        for step in range(1, args.calibration_steps + 1):
            if (time.perf_counter() - start) / 60.0 >= args.max_runtime_minutes * 0.55:
                print("calibration time budget reached; moving to remaining groups/finalization")
                break
            ids = next(batch_iter).to(device, non_blocking=True)
            captures = capture_teacher_attention_io(teacher, ids, group, amp)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            nmses = []
            cosines = []
            with amp():
                for idx in group:
                    c = captures[idx]
                    kwargs = {"use_cache": False}
                    if "position_embeddings" in c:
                        kwargs["position_embeddings"] = c["position_embeddings"]
                    pred = student.model.layers[idx].self_attn(c["hidden"], **kwargs)[0]
                    align, nmse, cosine = attention_alignment_loss(pred, c["target"], args.cosine_weight)
                    losses.append(align)
                    nmses.append(nmse)
                    cosines.append(cosine)
                align_loss = torch.stack(losses).mean()
                reg = feature_delta_regularizer(student, group)
                loss = align_loss + args.feature_reg_weight * reg
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite layerwise calibration loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            completed_steps = step
            calibration_tokens += ids.numel()
            last_loss = float(loss.detach().float())
            last_nmse = float(torch.stack(nmses).mean().float())
            last_cos = float(torch.stack(cosines).mean().float())
            if first_loss is None:
                first_loss = last_loss
            if step == 1 or step % args.log_every == 0 or step == args.calibration_steps:
                print(
                    f"stage={stage_id}/{len(groups)} layers={group[0]}-{group[-1]} "
                    f"step={step}/{args.calibration_steps} align={last_loss:.4f} "
                    f"nmse={last_nmse:.4f} cosine={last_cos:.4f}"
                )

        stage_reports.append({
            "stage": stage_id,
            "layers": group,
            "steps": completed_steps,
            "first_alignment_loss": first_loss,
            "last_alignment_loss": last_loss,
            "last_nmse": last_nmse,
            "last_cosine_distance": last_cos,
        })

    # Ensure all layers exist even if the calibration phase hit its time slice.
    replace_attention_layers(student, config, range(num_layers))
    structural_assertions(student)

    print("\n=== Phase 2: global CE + KL + hidden-state distillation ===")
    trainable = freeze_for_global_training(student, train_router=True)
    params = v2_parameter_summary(student)
    print("parameter summary:", json.dumps(params, indent=2))
    optimizer = make_optimizer(trainable, args.final_lr, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    student.train()
    optimizer.zero_grad(set_to_none=True)

    final_tokens = 0
    micro = 0
    updates = 0
    last = {"ce": float("nan"), "kl": float("nan"), "hidden": float("nan"), "total": float("nan")}
    stop_reason = "token_budget"

    while final_tokens < args.final_max_tokens:
        if (time.perf_counter() - start) / 60.0 >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            break
        ids = next(batch_iter).to(device, non_blocking=True)
        # Teacher outputs participate as constants in losses that backpropagate through
        # the student. no_grad keeps them autograd-safe while avoiding teacher grads.
        with torch.no_grad(), amp():
            t_out = teacher(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
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
            router = amcenn_router_stats(student)
            loss = (
                args.ce_weight * s_out.loss
                + args.kl_weight * kl
                + args.hidden_weight * hidden
                + args.router_aux_weight * router["load_balance"]
                + args.router_z_weight * router["z_loss"]
            )
            scaled = loss / args.final_grad_accum
        if not torch.isfinite(scaled):
            raise RuntimeError("non-finite global distillation loss")
        scaler.scale(scaled).backward()
        micro += 1
        final_tokens += ids.numel()
        last = {
            "ce": float(s_out.loss.detach().float()),
            "kl": float(kl.detach().float()),
            "hidden": float(hidden.detach().float()),
            "total": float(loss.detach().float()),
        }
        del t_out, s_out
        if micro % args.final_grad_accum:
            continue

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if updates == 1 or updates % args.log_every == 0:
            router = amcenn_router_stats(student)
            print(
                f"global update={updates} tokens={final_tokens:,} ce={last['ce']:.4f} "
                f"kl={last['kl']:.4f} hidden={last['hidden']:.4f} "
                f"route_mix={float(router['route_mix']):.5f} grad={grad_norm:.3f}"
            )

    elapsed = time.perf_counter() - start
    student.eval()
    with torch.inference_mode():
        probe = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
        _ = student(input_ids=probe, use_cache=False)
    router = amcenn_router_stats(student)
    report = {
        "status": "trained",
        "architecture": "smollm2-amcenn-top2-v2",
        "method_version": 2,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "evaluation_performed": False,
        "stop_reason": stop_reason,
        "context_length": args.context_length,
        "feature_dim": args.feature_dim,
        "antithetic_features": True,
        "learnable_feature_correction": True,
        "progressive_group_size": args.group_size,
        "calibration_steps_per_group": args.calibration_steps,
        "calibration_tokens": calibration_tokens,
        "calibration_stages": stage_reports,
        "final_seen_tokens": final_tokens,
        "final_updates": updates,
        "parameters": params,
        "last_training_ce": last["ce"],
        "last_distillation_kl": last["kl"],
        "last_hidden_alignment": last["hidden"],
        "mean_route_mix": float(router["route_mix"]),
        "mean_router_entropy": float(router["entropy"]),
        "elapsed_minutes": elapsed / 60.0,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
    }

    output_dir = Path(args.output_dir)
    save_smollm2_amcenn_v2(
        student,
        output_dir,
        config=config,
        base_model=args.base_model,
        extra_metadata=report,
    )
    tokenizer.save_pretrained(output_dir)
    (output_dir / "smollm2_amcenn_v2_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()