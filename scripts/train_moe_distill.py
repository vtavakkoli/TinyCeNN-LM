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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm import DEFAULT_BASE_MODEL
from tinycenn_lm.distill_utils import (
    batch_blocks,
    buffered_shuffle,
    collect_eval_batches,
    combined_loss,
    evaluation_fingerprint,
    partition_rows,
    token_blocks,
)
from tinycenn_lm.moe import (
    MoECeNNConfig,
    freeze_moe_student_interfaces,
    moe_router_stats,
    replace_transformer_with_moe_cenn,
    save_moe_cenn_student,
    warmstart_moe_from_plain_cenn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 8-expert Top-2 MoE-CeNN student by teacher distillation")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--warmstart-plain-dir", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="checkpoints/cenn-moe-top2")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=30_000_000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--dilations", default="1,2,4,8,16,32,64")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--router-noise-std", type=float, default=1e-3)
    p.add_argument("--router-aux-weight", type=float, default=0.01)
    p.add_argument("--router-z-weight", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.25)
    p.add_argument("--kl-chunk-rows", type=int, default=256)
    p.add_argument("--shuffle-buffer", type=int, default=4096)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmstart-ce-tolerance", type=float, default=0.02)
    p.add_argument("--target-gap-recovery", type=float, default=0.90)
    p.add_argument("--health-min-relative-ce-improvement", type=float, default=0.002)
    p.add_argument("--no-compile", action="store_true")
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


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step <= warmup:
        return max(step, 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def eval_moe(teacher, student, batches, device, dtype, args) -> dict:
    teacher.eval(); student.eval()
    sums = {k: 0.0 for k in ("student_ce", "teacher_ce", "kl", "hidden", "total", "router_aux", "router_z", "router_entropy")}
    expert_fraction = torch.zeros(args.num_experts, dtype=torch.float64)
    probability_fraction = torch.zeros(args.num_experts, dtype=torch.float64)
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    with torch.inference_mode():
        for cpu_ids in batches:
            ids = cpu_ids.to(device, non_blocking=True)
            with amp():
                t = teacher(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
                s = student(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
                distill, parts = combined_loss(
                    s, t,
                    temperature=args.temperature,
                    kl_chunk_rows=args.kl_chunk_rows,
                    ce_weight=args.ce_weight,
                    kl_weight=args.kl_weight,
                    hidden_weight=args.hidden_weight,
                )
                router = moe_router_stats(student)
                total = distill + args.router_aux_weight * router["load_balance"] + args.router_z_weight * router["z_loss"]
            sums["student_ce"] += float(s.loss.detach().float())
            sums["teacher_ce"] += float(t.loss.detach().float())
            sums["kl"] += parts["kl"]
            sums["hidden"] += parts["hidden"]
            sums["total"] += float(total.detach().float())
            sums["router_aux"] += float(router["load_balance"].detach().float())
            sums["router_z"] += float(router["z_loss"].detach().float())
            sums["router_entropy"] += float(router["entropy"].detach().float())
            expert_fraction += router["expert_fraction"].detach().float().cpu().double()
            probability_fraction += router["probability_fraction"].detach().float().cpu().double()
    n = len(batches)
    for key in sums:
        sums[key] /= max(n, 1)
    expert_fraction /= max(n, 1)
    probability_fraction /= max(n, 1)
    sums["student_ppl"] = math.exp(min(sums["student_ce"], 30.0))
    sums["teacher_ppl"] = math.exp(min(sums["teacher_ce"], 30.0))
    sums["expert_fraction"] = expert_fraction.tolist()
    sums["probability_fraction"] = probability_fraction.tolist()
    sums["expert_min_fraction"] = float(expert_fraction.min())
    sums["expert_max_fraction"] = float(expert_fraction.max())
    return sums


def save_checkpoint(model, tokenizer, directory, config, args, tokens, source):
    save_moe_cenn_student(
        model, directory, config=config, base_model=args.base_model,
        extra_metadata={"distilled": True, "tokens": tokens, "warmstart_plain": str(source)},
    )
    tokenizer.save_pretrained(directory)


def main() -> None:
    args = parse_args()
    if args.num_experts != 8 or args.top_k != 2:
        print(f"NOTE: requested experts={args.num_experts} top_k={args.top_k}; canonical experiment is 8/2")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    print(f"device={device} dtype={dtype}")

    warmstart = Path(args.warmstart_plain_dir).expanduser().resolve()
    previous_report_path = warmstart / "distillation_report.json"
    if not previous_report_path.exists():
        raise FileNotFoundError("warm-start checkpoint must include distillation_report.json")
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(warmstart, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"attn_implementation": "sdpa"}
    if device.type == "cuda": load_kwargs["dtype"] = dtype

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device).eval()
    teacher.config.use_cache = False
    for p in teacher.parameters(): p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    cfg = MoECeNNConfig(
        hidden_size=int(student.config.hidden_size), kernel_size=args.kernel_size,
        expansion=args.expansion, steps=args.steps,
        dilations=tuple(int(x) for x in args.dilations.split(",") if x.strip()),
        rms_norm_eps=float(getattr(student.config, "rms_norm_eps", 1e-5)),
        num_experts=args.num_experts, top_k=args.top_k, router_noise_std=args.router_noise_std,
    )
    replace_transformer_with_moe_cenn(student, cfg)
    warmstart_moe_from_plain_cenn(student, warmstart)
    freeze_moe_student_interfaces(student)
    student.to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"MoE-CeNN params total={total_params:,} trainable={trainable_params:,} experts={cfg.num_experts} top_k={cfg.top_k}")

    eval_raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    eval_rows = partition_rows(eval_raw, args.text_field, validation=True)
    eval_batches = collect_eval_batches(eval_rows, tokenizer, args.text_field, args.context_length, args.eval_batch_size, args.eval_batches)
    fingerprint = evaluation_fingerprint(eval_batches)
    eval_tokens = sum(x.numel() for x in eval_batches)
    expected_fp = previous_report.get("evaluation", {}).get("fingerprint_sha256")
    if expected_fp and fingerprint != expected_fp:
        raise RuntimeError(f"benchmark fingerprint differs from plain CeNN baseline: {fingerprint} != {expected_fp}")

    start_metrics = eval_moe(teacher, student, eval_batches, device, dtype, args)
    plain_ce = float(previous_report.get("best", {}).get("student_ce", start_metrics["student_ce"]))
    warmstart_delta = abs(start_metrics["student_ce"] - plain_ce)
    print(f"function-preserving warm start: MoE CE={start_metrics['student_ce']:.6f}, plain CE={plain_ce:.6f}, delta={warmstart_delta:.6f}")
    if warmstart_delta > args.warmstart_ce_tolerance:
        raise RuntimeError("MoE warm start failed parity tolerance")

    cold_ce = float(previous_report.get("cold_initial", {}).get("student_ce", start_metrics["student_ce"]))
    teacher_ce_reference = float(previous_report.get("cold_initial", {}).get("teacher_ce", start_metrics["teacher_ce"]))
    previous_tokens = int(previous_report.get("cumulative_training_tokens", previous_report.get("seen_tokens", 0)))

    train_raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    train_rows = buffered_shuffle(partition_rows(train_raw, args.text_field, validation=False), buffer_size=args.shuffle_buffer, seed=args.seed + 101)
    batches = batch_blocks(token_blocks(train_rows, tokenizer, args.text_field, args.context_length), args.batch_size)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    tokens_per_update = args.batch_size * args.context_length * args.grad_accum
    total_updates = math.ceil(args.max_tokens / tokens_per_update)
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))

    output_dir = Path(args.output_dir); best_dir = Path(str(output_dir) + "-best")
    save_checkpoint(student, tokenizer, best_dir, cfg, args, 0, warmstart)
    best = start_metrics; best_update = 0; seen_tokens = 0; micro = 0; update = 0
    history = [{"update": 0, "tokens_this_run": 0, **start_metrics}]
    start_time = time.perf_counter(); student.train(); optimizer.zero_grad(set_to_none=True)

    while update < total_updates:
        ids = next(batches).to(device, non_blocking=True)
        with torch.no_grad():
            with amp():
                t = teacher(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
        with amp():
            s = student(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
            distill, parts = combined_loss(
                s, t, temperature=args.temperature, kl_chunk_rows=args.kl_chunk_rows,
                ce_weight=args.ce_weight, kl_weight=args.kl_weight, hidden_weight=args.hidden_weight,
            )
            router = moe_router_stats(student)
            loss = distill + args.router_aux_weight * router["load_balance"] + args.router_z_weight * router["z_loss"]
            scaled = loss / args.grad_accum
        if not torch.isfinite(scaled): raise RuntimeError("non-finite MoE distillation loss")
        scaler.scale(scaled).backward(); micro += 1; seen_tokens += ids.numel()
        if micro % args.grad_accum: continue
        update += 1
        lr_mult = cosine_lr(update, total_updates, warmup_updates)
        for group in optimizer.param_groups: group["lr"] = args.learning_rate * lr_mult
        scaler.unscale_(optimizer)
        grad = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(grad): raise RuntimeError("non-finite gradient norm")
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if update == 1 or update % args.log_every == 0:
            print(f"update={update}/{total_updates} tokens={seen_tokens:,} ce={parts['ce']:.4f} kl={parts['kl']:.4f} hidden={parts['hidden']:.4f} router_aux={float(router['load_balance'].detach()):.3f} entropy={float(router['entropy'].detach()):.3f} grad={grad:.3f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if update == total_updates or update % args.eval_every == 0:
            metrics = eval_moe(teacher, student, eval_batches, device, dtype, args)
            history.append({"update": update, "tokens_this_run": seen_tokens, **metrics})
            print(f"eval={update}: CE={metrics['student_ce']:.4f} PPL={metrics['student_ppl']:.2f} teacher={metrics['teacher_ce']:.4f} expert_range={metrics['expert_min_fraction']:.3f}-{metrics['expert_max_fraction']:.3f}")
            if metrics["student_ce"] < best["student_ce"]:
                best = metrics; best_update = update
                save_checkpoint(student, tokenizer, best_dir, cfg, args, seen_tokens, warmstart)
            student.train()

    final = eval_moe(teacher, student, eval_batches, device, dtype, args)
    save_checkpoint(student, tokenizer, output_dir, cfg, args, seen_tokens, warmstart)

    cold_gap = cold_ce - teacher_ce_reference
    best_gap = best["student_ce"] - best["teacher_ce"]
    recovery = (cold_gap - best_gap) / max(cold_gap, 1e-12)
    improvement = (start_metrics["student_ce"] - best["student_ce"]) / max(start_metrics["student_ce"], 1e-12)
    status = "target_reached" if recovery >= args.target_gap_recovery else ("healthy_progress" if improvement >= args.health_min_relative_ce_improvement else "warning_no_improvement")
    report = {
        "status": status,
        "benchmark_protocol": "rigorous-v2-moe-top2",
        "architecture": "moe-cenn-top2-replacement",
        "transformer_layers_remaining": 0,
        "base_model_teacher": args.base_model,
        "dataset": args.dataset, "dataset_config": args.dataset_config, "dataset_split": args.split, "text_field": args.text_field,
        "evaluation": {"batches": len(eval_batches), "batch_size": args.eval_batch_size, "tokens": eval_tokens, "fingerprint_sha256": fingerprint, "eval_every_updates": args.eval_every},
        "training_shuffle": {"method": "deterministic bounded-memory buffered shuffle", "buffer_size": args.shuffle_buffer, "seed": args.seed + 101},
        "context_length": args.context_length,
        "previous_plain_training_tokens": previous_tokens,
        "moe_training_tokens": seen_tokens,
        "effective_cumulative_tokens": previous_tokens + seen_tokens,
        "parameters": {"total": total_params, "trainable": trainable_params, "trainable_percent": 100*trainable_params/max(total_params,1)},
        "moe": cfg.to_dict(),
        "active_experts_per_token": cfg.top_k,
        "distillation": {"temperature": args.temperature, "ce_weight": args.ce_weight, "kl_weight": args.kl_weight, "hidden_weight": args.hidden_weight, "router_aux_weight": args.router_aux_weight, "router_z_weight": args.router_z_weight, "learning_rate": args.learning_rate},
        "warmstart": {"plain_dir": str(warmstart), "plain_best_ce": plain_ce, "moe_start_ce": start_metrics["student_ce"], "ce_absolute_delta": warmstart_delta},
        "run_start": start_metrics, "best": best, "best_update_this_run": best_update, "final": final,
        "run_relative_student_ce_improvement": improvement,
        "teacher_gap_recovery_fraction": recovery,
        "target_gap_recovery_fraction": args.target_gap_recovery,
        "target_gap_recovery_reached": recovery >= args.target_gap_recovery,
        "elapsed_seconds": time.perf_counter() - start_time,
        "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0.0,
        "eval_history": history,
    }
    text = json.dumps(report, indent=2)
    output_dir.mkdir(parents=True, exist_ok=True); (output_dir/"moe_distillation_report.json").write_text(text)
    (best_dir/"moe_distillation_report.json").write_text(text)
    print("="*88)
    print(f"MOE-CENN: {status.upper()} | CE {start_metrics['student_ce']:.4f} -> {best['student_ce']:.4f} | teacher={best['teacher_ce']:.4f} | gap recovery={100*recovery:.2f}%")
    print(f"best checkpoint: {best_dir}")
    print("="*88)


if __name__ == "__main__":
    main()
