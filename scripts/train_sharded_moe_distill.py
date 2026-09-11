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
from tinycenn_lm.sharded_moe import (
    ShardedMoECeNNConfig,
    freeze_sharded_moe_interfaces,
    replace_transformer_with_sharded_moe_cenn,
    save_sharded_moe_student,
    sharded_router_stats,
    warmstart_sharded_moe_from_plain_cenn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train parameter-neutral 8-shard Top-2 routed CeNN FFN by teacher distillation"
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--warmstart-plain-dir", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="checkpoints/cenn-sharded-moe-top2")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=20_000_000)
    p.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=50.0,
        help="Training-loop wall-clock cap; leaves margin for evaluation/upload in a 1-hour Colab budget",
    )
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--dilations", default="1,2,4,8,16,32,64")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--router-noise-std", type=float, default=1e-3)
    p.add_argument("--router-aux-weight", type=float, default=0.01)
    p.add_argument("--router-z-weight", type=float, default=1e-3)
    p.add_argument("--route-mix-l2-weight", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.25)
    p.add_argument("--kl-chunk-rows", type=int, default=256)
    p.add_argument("--shuffle-buffer", type=int, default=4096)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmstart-ce-tolerance", type=float, default=0.01)
    p.add_argument("--target-gap-recovery", type=float, default=0.90)
    p.add_argument("--health-min-relative-ce-improvement", type=float, default=0.001)
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


def eval_model(teacher, student, batches, device, dtype, args) -> dict:
    teacher.eval()
    student.eval()
    keys = ("student_ce", "teacher_ce", "kl", "hidden", "total", "router_aux", "router_z", "router_entropy")
    sums = {key: 0.0 for key in keys}
    shard_fraction = torch.zeros(args.num_shards, dtype=torch.float64)
    probability_fraction = torch.zeros(args.num_shards, dtype=torch.float64)
    route_mix = 0.0
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext

    with torch.inference_mode():
        for cpu_ids in batches:
            ids = cpu_ids.to(device, non_blocking=True)
            with amp():
                teacher_out = teacher(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
                student_out = student(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
                distill, parts = combined_loss(
                    student_out,
                    teacher_out,
                    temperature=args.temperature,
                    kl_chunk_rows=args.kl_chunk_rows,
                    ce_weight=args.ce_weight,
                    kl_weight=args.kl_weight,
                    hidden_weight=args.hidden_weight,
                )
                router = sharded_router_stats(student)
                total = (
                    distill
                    + args.router_aux_weight * router["load_balance"]
                    + args.router_z_weight * router["z_loss"]
                    + args.route_mix_l2_weight * router["route_mix"].float().pow(2)
                )
            sums["student_ce"] += float(student_out.loss.detach().float())
            sums["teacher_ce"] += float(teacher_out.loss.detach().float())
            sums["kl"] += parts["kl"]
            sums["hidden"] += parts["hidden"]
            sums["total"] += float(total.detach().float())
            sums["router_aux"] += float(router["load_balance"].detach().float())
            sums["router_z"] += float(router["z_loss"].detach().float())
            sums["router_entropy"] += float(router["entropy"].detach().float())
            shard_fraction += router["shard_fraction"].detach().float().cpu().double()
            probability_fraction += router["probability_fraction"].detach().float().cpu().double()
            route_mix += float(router["route_mix"].detach().float())

    n = max(len(batches), 1)
    for key in sums:
        sums[key] /= n
    shard_fraction /= n
    probability_fraction /= n
    route_mix /= n
    sums["student_ppl"] = math.exp(min(sums["student_ce"], 30.0))
    sums["teacher_ppl"] = math.exp(min(sums["teacher_ce"], 30.0))
    sums["shard_fraction"] = shard_fraction.tolist()
    sums["probability_fraction"] = probability_fraction.tolist()
    sums["shard_min_fraction"] = float(shard_fraction.min())
    sums["shard_max_fraction"] = float(shard_fraction.max())
    sums["route_mix"] = route_mix
    return sums


def save_checkpoint(model, tokenizer, directory, config, args, tokens, source) -> None:
    save_sharded_moe_student(
        model,
        directory,
        config=config,
        base_model=args.base_model,
        extra_metadata={
            "distilled": True,
            "tokens_this_run": tokens,
            "warmstart_plain": str(source),
            "parameter_neutral_ffn": True,
        },
    )
    tokenizer.save_pretrained(directory)


def main() -> None:
    args = parse_args()
    if args.num_shards != 8 or args.top_k != 2:
        print(f"NOTE: canonical experiment is 8 shards / Top-2; requested {args.num_shards}/{args.top_k}")
    if args.max_runtime_minutes <= 0:
        raise ValueError("max-runtime-minutes must be > 0")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"device={device} dtype={dtype}")

    warmstart = Path(args.warmstart_plain_dir).expanduser().resolve()
    previous_report_path = warmstart / "distillation_report.json"
    if not previous_report_path.exists():
        raise FileNotFoundError("warm-start checkpoint must include distillation_report.json")
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(warmstart, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if device.type == "cuda":
        load_kwargs["dtype"] = dtype
    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device).eval()
    teacher.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    config = ShardedMoECeNNConfig(
        hidden_size=int(student.config.hidden_size),
        kernel_size=args.kernel_size,
        expansion=args.expansion,
        steps=args.steps,
        dilations=tuple(int(x) for x in args.dilations.split(",") if x.strip()),
        rms_norm_eps=float(getattr(student.config, "rms_norm_eps", 1e-5)),
        num_shards=args.num_shards,
        top_k=args.top_k,
        router_noise_std=args.router_noise_std,
    )
    replace_transformer_with_sharded_moe_cenn(student, config)
    warmstart_sharded_moe_from_plain_cenn(student, warmstart)
    freeze_sharded_moe_interfaces(student)
    student.to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    plain_trainable = int(previous_report.get("parameters", {}).get("trainable", 480_192))
    overhead = trainable_params - plain_trainable
    overhead_pct = 100.0 * overhead / max(plain_trainable, 1)
    print(
        f"Sharded MoE-CeNN: total={total_params:,} trainable={trainable_params:,} "
        f"plain_trainable={plain_trainable:,} overhead={overhead:+,} ({overhead_pct:.3f}%) "
        f"shards={config.num_shards} shard_inner={config.shard_inner} top_k={config.top_k}"
    )
    # Guard against accidentally returning to 8 full experts.
    if trainable_params > plain_trainable * 1.02:
        raise RuntimeError("parameter budget violated: sharded model must stay within 2% of plain CeNN")

    eval_raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    eval_rows = partition_rows(eval_raw, args.text_field, validation=True)
    eval_batches = collect_eval_batches(
        eval_rows,
        tokenizer,
        args.text_field,
        args.context_length,
        args.eval_batch_size,
        args.eval_batches,
    )
    fingerprint = evaluation_fingerprint(eval_batches)
    eval_tokens = sum(batch.numel() for batch in eval_batches)
    expected_fp = previous_report.get("evaluation", {}).get("fingerprint_sha256")
    if expected_fp and fingerprint != expected_fp:
        raise RuntimeError(f"benchmark fingerprint differs from plain CeNN baseline: {fingerprint} != {expected_fp}")

    start_metrics = eval_model(teacher, student, eval_batches, device, dtype, args)
    plain_ce = float(previous_report.get("best", {}).get("student_ce", start_metrics["student_ce"]))
    warmstart_delta = abs(start_metrics["student_ce"] - plain_ce)
    print(
        f"exact dense-FFN shard warm start: sharded CE={start_metrics['student_ce']:.6f}, "
        f"plain CE={plain_ce:.6f}, delta={warmstart_delta:.6f}, route_mix={start_metrics['route_mix']:.6f}"
    )
    if warmstart_delta > args.warmstart_ce_tolerance:
        raise RuntimeError("sharded warm start failed parity tolerance")

    cold_ce = float(previous_report.get("cold_initial", {}).get("student_ce", start_metrics["student_ce"]))
    teacher_ce_reference = float(previous_report.get("cold_initial", {}).get("teacher_ce", start_metrics["teacher_ce"]))
    previous_tokens = int(previous_report.get("cumulative_training_tokens", previous_report.get("seen_tokens", 0)))

    train_raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    train_rows = buffered_shuffle(
        partition_rows(train_raw, args.text_field, validation=False),
        buffer_size=args.shuffle_buffer,
        seed=args.seed + 211,
    )
    train_batches = batch_blocks(
        token_blocks(train_rows, tokenizer, args.text_field, args.context_length),
        args.batch_size,
    )

    trainable = [p for p in student.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    tokens_per_update = args.batch_size * args.context_length * args.grad_accum
    total_updates = math.ceil(args.max_tokens / tokens_per_update)
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))

    output_dir = Path(args.output_dir)
    best_dir = Path(str(output_dir) + "-best")
    save_checkpoint(student, tokenizer, best_dir, config, args, 0, warmstart)
    best = start_metrics
    best_update = 0
    seen_tokens = 0
    micro = 0
    update = 0
    history = [{"update": 0, "tokens_this_run": 0, **start_metrics}]
    train_start = time.perf_counter()
    stop_reason = "token_budget"
    student.train()
    optimizer.zero_grad(set_to_none=True)

    while update < total_updates:
        elapsed_min = (time.perf_counter() - train_start) / 60.0
        if elapsed_min >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            print(f"runtime budget reached at {elapsed_min:.1f} min; stopping training cleanly")
            break

        ids = next(train_batches).to(device, non_blocking=True)
        with torch.no_grad():
            with amp():
                teacher_out = teacher(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
        with amp():
            student_out = student(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
            distill, parts = combined_loss(
                student_out,
                teacher_out,
                temperature=args.temperature,
                kl_chunk_rows=args.kl_chunk_rows,
                ce_weight=args.ce_weight,
                kl_weight=args.kl_weight,
                hidden_weight=args.hidden_weight,
            )
            router = sharded_router_stats(student)
            loss = (
                distill
                + args.router_aux_weight * router["load_balance"]
                + args.router_z_weight * router["z_loss"]
                + args.route_mix_l2_weight * router["route_mix"].float().pow(2)
            )
            scaled = loss / args.grad_accum

        if not torch.isfinite(scaled):
            raise RuntimeError("non-finite sharded MoE distillation loss")
        scaler.scale(scaled).backward()
        micro += 1
        seen_tokens += ids.numel()
        if micro % args.grad_accum:
            continue

        update += 1
        lr_mult = cosine_lr(update, total_updates, warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_mult
        scaler.unscale_(optimizer)
        grad = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(grad):
            raise RuntimeError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if update == 1 or update % args.log_every == 0:
            print(
                f"update={update}/{total_updates} tokens={seen_tokens:,} ce={parts['ce']:.4f} "
                f"kl={parts['kl']:.4f} hidden={parts['hidden']:.4f} "
                f"route_mix={float(router['route_mix'].detach()):.4f} "
                f"entropy={float(router['entropy'].detach()):.3f} grad={grad:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={elapsed_min:.1f}m"
            )

        if update == total_updates or update % args.eval_every == 0:
            metrics = eval_model(teacher, student, eval_batches, device, dtype, args)
            history.append({"update": update, "tokens_this_run": seen_tokens, **metrics})
            print(
                f"eval={update}: CE={metrics['student_ce']:.4f} PPL={metrics['student_ppl']:.2f} "
                f"teacher={metrics['teacher_ce']:.4f} route_mix={metrics['route_mix']:.4f} "
                f"shard_range={metrics['shard_min_fraction']:.3f}-{metrics['shard_max_fraction']:.3f}"
            )
            if metrics["student_ce"] < best["student_ce"]:
                best = metrics
                best_update = update
                save_checkpoint(student, tokenizer, best_dir, config, args, seen_tokens, warmstart)
            student.train()

    # Always finish with the rigorous validation set, including runtime-stop cases.
    final = eval_model(teacher, student, eval_batches, device, dtype, args)
    history.append({"update": update, "tokens_this_run": seen_tokens, **final})
    if final["student_ce"] < best["student_ce"]:
        best = final
        best_update = update
        save_checkpoint(student, tokenizer, best_dir, config, args, seen_tokens, warmstart)
    save_checkpoint(student, tokenizer, output_dir, config, args, seen_tokens, warmstart)

    cold_gap = cold_ce - teacher_ce_reference
    best_gap = best["student_ce"] - best["teacher_ce"]
    recovery = (cold_gap - best_gap) / max(cold_gap, 1e-12) if cold_gap > 0 else 0.0
    relative_improvement = (start_metrics["student_ce"] - best["student_ce"]) / max(start_metrics["student_ce"], 1e-12)
    if recovery >= args.target_gap_recovery:
        status = "target_reached"
    elif relative_improvement >= args.health_min_relative_ce_improvement:
        status = "healthy_progress"
    else:
        status = "warning_no_improvement"

    report = {
        "status": status,
        "benchmark_protocol": "rigorous-v2-sharded-moe",
        "architecture": "sharded-moe-cenn-top2-replacement",
        "transformer_layers_remaining": 0,
        "base_model_teacher": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.split,
        "text_field": args.text_field,
        "evaluation": {
            "batches": len(eval_batches),
            "batch_size": args.eval_batch_size,
            "tokens": eval_tokens,
            "fingerprint_sha256": fingerprint,
            "eval_every_updates": args.eval_every,
        },
        "context_length": args.context_length,
        "requested_tokens": args.max_tokens,
        "seen_tokens_this_run": seen_tokens,
        "previous_training_tokens": previous_tokens,
        "cumulative_training_tokens": previous_tokens + seen_tokens,
        "updates_completed": update,
        "stop_reason": stop_reason,
        "max_runtime_minutes": args.max_runtime_minutes,
        "num_shards": config.num_shards,
        "top_k": config.top_k,
        "dense_inner": config.dense_inner,
        "shard_inner": config.shard_inner,
        "cenn_steps": config.steps,
        "cenn_dilations": list(config.dilations),
        "cenn_receptive_field": 1 + (config.kernel_size - 1) * sum(config.dilations[: config.steps]),
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "plain_cenn_trainable": plain_trainable,
            "extra_over_plain": overhead,
            "extra_over_plain_percent": overhead_pct,
        },
        "distillation": {
            "temperature": args.temperature,
            "ce_weight": args.ce_weight,
            "kl_weight": args.kl_weight,
            "hidden_weight": args.hidden_weight,
            "router_aux_weight": args.router_aux_weight,
            "router_z_weight": args.router_z_weight,
            "route_mix_l2_weight": args.route_mix_l2_weight,
            "learning_rate": args.learning_rate,
        },
        "plain_baseline_ce": plain_ce,
        "warmstart_ce_delta": warmstart_delta,
        "run_start": start_metrics,
        "best": best,
        "best_update": best_update,
        "final": final,
        "relative_student_ce_improvement": relative_improvement,
        "teacher_gap_recovery_fraction": recovery,
        "target_gap_recovery_fraction": args.target_gap_recovery,
        "target_gap_recovery_reached": recovery >= args.target_gap_recovery,
        "elapsed_training_seconds": time.perf_counter() - train_start,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "eval_history": history,
    }
    report_text = json.dumps(report, indent=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sharded_moe_distillation_report.json").write_text(report_text, encoding="utf-8")
    (best_dir / "sharded_moe_distillation_report.json").write_text(report_text, encoding="utf-8")

    print("=" * 92)
    print(
        f"SHARDED MOE: {status.upper()} | CE {start_metrics['student_ce']:.4f} -> {best['student_ce']:.4f} "
        f"| teacher={best['teacher_ce']:.4f} | gap recovery={100*recovery:.2f}% "
        f"| trainable={trainable_params:,} (+{overhead_pct:.3f}% vs plain) | stop={stop_reason}"
    )
    print(f"best checkpoint: {best_dir}")
    print("=" * 92)


if __name__ == "__main__":
    main()
