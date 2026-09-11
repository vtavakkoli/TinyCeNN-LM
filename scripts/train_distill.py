#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm import (
    DEFAULT_BASE_MODEL,
    CeNNConfig,
    freeze_student_interfaces,
    replace_transformer_with_cenn,
    save_cenn_student,
    student_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill Tiny-LLM Transformer into a CeNN-only student")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="checkpoints/cenn-student-distill")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--dilations", default="1,2,4,8,16,32,64")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.25)
    p.add_argument("--kl-chunk-rows", type=int, default=256)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--health-min-relative-ce-improvement", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def holdout_bucket(text: str, buckets: int = 1000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def partition_rows(dataset: Iterable[dict], text_field: str, *, validation: bool) -> Iterator[dict]:
    """Deterministic document-level split: 99% train / 1% validation."""
    for example in dataset:
        text = example.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        is_validation = holdout_bucket(text) >= 990
        if is_validation == validation:
            yield example


def token_blocks(rows: Iterable[dict], tokenizer, text_field: str, block_size: int) -> Iterator[torch.Tensor]:
    buffer: list[int] = []
    offset = 0
    eos = tokenizer.eos_token_id
    for example in rows:
        ids = tokenizer(example[text_field], add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids.append(eos)
        buffer.extend(ids)
        while len(buffer) - offset >= block_size:
            yield torch.tensor(buffer[offset : offset + block_size], dtype=torch.long)
            offset += block_size
        if offset > 1_000_000:
            buffer = buffer[offset:]
            offset = 0


def batch_blocks(blocks: Iterator[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    batch: list[torch.Tensor] = []
    for block in blocks:
        batch.append(block)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch.clear()


def collect_eval_batches(rows, tokenizer, text_field, block_size, batch_size, count):
    batches = batch_blocks(token_blocks(rows, tokenizer, text_field, block_size), batch_size)
    out = []
    for _ in range(count):
        try:
            out.append(next(batches))
        except StopIteration:
            break
    if not out:
        raise RuntimeError("could not build held-out evaluation batches")
    return out


def chunked_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, chunk_rows: int) -> torch.Tensor:
    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    total = s.new_zeros((), dtype=torch.float32)
    rows = s.shape[0]
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        s_chunk = (s[start:end].float() / temperature)
        t_chunk = (t[start:end].float() / temperature)
        total = total + F.kl_div(
            F.log_softmax(s_chunk, dim=-1),
            F.softmax(t_chunk, dim=-1),
            reduction="sum",
        )
    return total * (temperature * temperature) / max(rows, 1)


def hidden_cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1)).mean()


def combined_loss(student_out, teacher_out, args) -> tuple[torch.Tensor, dict[str, float]]:
    ce = student_out.loss.float()
    kl = chunked_kl(student_out.logits, teacher_out.logits, args.temperature, args.kl_chunk_rows)
    hidden = hidden_cosine_loss(student_out.hidden_states[-1], teacher_out.hidden_states[-1])
    total = args.ce_weight * ce + args.kl_weight * kl + args.hidden_weight * hidden
    return total, {
        "ce": float(ce.detach()),
        "kl": float(kl.detach()),
        "hidden": float(hidden.detach()),
        "total": float(total.detach()),
    }


@torch.inference_mode()
def evaluate(teacher, student, batches, device, dtype, args) -> dict[str, float]:
    teacher.eval()
    student.eval()
    sums = {"student_ce": 0.0, "teacher_ce": 0.0, "kl": 0.0, "hidden": 0.0, "total": 0.0}
    n_batches = 0
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    for cpu_ids in batches:
        ids = cpu_ids.to(device, non_blocking=True)
        with amp():
            teacher_out = teacher(
                input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False
            )
            student_out = student(
                input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False
            )
            total, parts = combined_loss(student_out, teacher_out, args)
        sums["student_ce"] += float(student_out.loss.detach().float())
        sums["teacher_ce"] += float(teacher_out.loss.detach().float())
        sums["kl"] += parts["kl"]
        sums["hidden"] += parts["hidden"]
        sums["total"] += float(total.detach().float())
        n_batches += 1
    for key in sums:
        sums[key] /= max(n_batches, 1)
    sums["student_ppl"] = math.exp(min(sums["student_ce"], 30.0))
    sums["teacher_ppl"] = math.exp(min(sums["teacher_ce"], 30.0))
    return sums


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if device.type == "cuda":
        load_kwargs["dtype"] = dtype

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device)
    teacher.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="sdpa", dtype=torch.float32
    )
    dilations = tuple(int(x) for x in args.dilations.split(",") if x.strip())
    cenn_config = CeNNConfig(
        hidden_size=student.config.hidden_size,
        kernel_size=args.kernel_size,
        expansion=args.expansion,
        steps=args.steps,
        dilations=dilations,
        rms_norm_eps=float(getattr(student.config, "rms_norm_eps", 1e-5)),
    )
    replace_transformer_with_cenn(student, cenn_config, (0,))
    freeze_student_interfaces(student)
    student.to(device)
    summary = student_parameter_summary(student)
    print(
        f"student parameters: total={summary['total']:,} trainable={summary['trainable']:,} "
        f"({summary['trainable_percent']:.2f}%)"
    )
    replacement = next(m for m in student.modules() if m.__class__.__name__ == "CeNNReplacementLayer")
    print(f"Transformer removed: CeNN steps={args.steps}, receptive_field={replacement.cenn.receptive_field}")

    eval_raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    eval_rows = partition_rows(eval_raw, args.text_field, validation=True)
    eval_batches = collect_eval_batches(
        eval_rows, tokenizer, args.text_field, args.context_length, args.eval_batch_size, args.eval_batches
    )
    initial = evaluate(teacher, student, eval_batches, device, dtype, args)
    print(
        f"initial student_ce={initial['student_ce']:.4f} ppl={initial['student_ppl']:.2f} | "
        f"teacher_ce={initial['teacher_ce']:.4f} ppl={initial['teacher_ppl']:.2f} | "
        f"KL={initial['kl']:.4f} hidden={initial['hidden']:.4f}"
    )

    train_raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    train_rows = partition_rows(train_raw, args.text_field, validation=False)
    batches = batch_blocks(
        token_blocks(train_rows, tokenizer, args.text_field, args.context_length), args.batch_size
    )

    trainable = [p for p in student.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable, lr=args.learning_rate, weight_decay=args.weight_decay, fused=(device.type == "cuda")
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    tokens_per_update = args.batch_size * args.context_length * args.grad_accum
    total_updates = max(1, math.ceil(args.max_tokens / tokens_per_update))
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext

    train_student = student
    if not args.no_compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            train_student = torch.compile(student, mode="reduce-overhead", dynamic=False)
            print("torch.compile enabled for student")
        except Exception as exc:
            print(f"torch.compile unavailable, eager mode: {exc}")

    output_dir = Path(args.output_dir)
    best_dir = Path(str(output_dir) + "-best")
    history = [{"update": 0, "tokens": 0, **initial}]
    best = initial
    best_update = 0
    seen_tokens = 0
    update = 0
    micro = 0
    running = {"ce": 0.0, "kl": 0.0, "hidden": 0.0, "total": 0.0}
    running_n = 0
    start = time.perf_counter()
    window_start = start
    optimizer.zero_grad(set_to_none=True)
    diverged = False

    student.train()
    while update < total_updates:
        ids = next(batches).to(device, non_blocking=True)
        with torch.inference_mode():
            with amp():
                teacher_out = teacher(
                    input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False
                )
        with amp():
            student_out = train_student(
                input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False
            )
            loss, parts = combined_loss(student_out, teacher_out, args)
            scaled_loss = loss / args.grad_accum
        if not torch.isfinite(scaled_loss):
            diverged = True
            print("ERROR: non-finite distillation loss")
            break
        scaler.scale(scaled_loss).backward()
        micro += 1
        seen_tokens += ids.numel()
        for key in running:
            running[key] += parts[key]
        running_n += 1
        if micro % args.grad_accum:
            continue

        next_update = update + 1
        lr_mult = cosine_lr(next_update, total_updates, warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_mult
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(grad_norm):
            diverged = True
            print("ERROR: non-finite gradient norm")
            break
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update = next_update

        if update == 1 or update % args.log_every == 0:
            elapsed = max(time.perf_counter() - window_start, 1e-6)
            tok_s = (args.batch_size * args.context_length * args.grad_accum * max(args.log_every, 1)) / elapsed
            avg = {k: v / max(running_n, 1) for k, v in running.items()}
            print(
                f"update={update}/{total_updates} tokens={seen_tokens:,} ce={avg['ce']:.4f} "
                f"kl={avg['kl']:.4f} hidden={avg['hidden']:.4f} total={avg['total']:.4f} "
                f"grad={grad_norm:.3f} lr={optimizer.param_groups[0]['lr']:.2e} tok/s~{tok_s:,.0f}"
            )
            running = {"ce": 0.0, "kl": 0.0, "hidden": 0.0, "total": 0.0}
            running_n = 0
            window_start = time.perf_counter()

        if update == total_updates or (args.eval_every > 0 and update % args.eval_every == 0):
            metrics = evaluate(teacher, student, eval_batches, device, dtype, args)
            history.append({"update": update, "tokens": seen_tokens, **metrics})
            print(
                f"eval={update}: student_ce={metrics['student_ce']:.4f} ppl={metrics['student_ppl']:.2f} "
                f"teacher_ce={metrics['teacher_ce']:.4f} KL={metrics['kl']:.4f} hidden={metrics['hidden']:.4f}"
            )
            if not all(math.isfinite(metrics[k]) for k in ("student_ce", "kl", "hidden")):
                diverged = True
                break
            if metrics["student_ce"] < best["student_ce"]:
                best = metrics
                best_update = update
                save_cenn_student(
                    student,
                    best_dir,
                    config=cenn_config,
                    base_model=args.base_model,
                    extra_metadata={"distilled": True, "tokens": seen_tokens},
                )
                tokenizer.save_pretrained(best_dir)
                print(f"new best CeNN student saved at update {update}")
            student.train()

    final = evaluate(teacher, student, eval_batches, device, dtype, args) if not diverged else None
    if not diverged:
        save_cenn_student(
            student,
            output_dir,
            config=cenn_config,
            base_model=args.base_model,
            extra_metadata={"distilled": True, "tokens": seen_tokens},
        )
        tokenizer.save_pretrained(output_dir)

    initial_gap = initial["student_ce"] - initial["teacher_ce"]
    best_gap = best["student_ce"] - best["teacher_ce"]
    recovery = (initial_gap - best_gap) / max(initial_gap, 1e-12) if initial_gap > 0 else 0.0
    relative_ce_improvement = (initial["student_ce"] - best["student_ce"]) / max(initial["student_ce"], 1e-12)
    status = "diverged" if diverged else (
        "healthy" if relative_ce_improvement >= args.health_min_relative_ce_improvement else "warning_no_improvement"
    )
    report = {
        "status": status,
        "architecture": "cenn-only-replacement",
        "transformer_layers_remaining": 0,
        "base_model_teacher": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "validation_split": "deterministic text hash: buckets 990-999 / 1000",
        "context_length": args.context_length,
        "requested_tokens": args.max_tokens,
        "seen_tokens": seen_tokens,
        "updates_completed": update,
        "cenn_steps": args.steps,
        "cenn_dilations": list(dilations),
        "cenn_receptive_field": replacement.cenn.receptive_field,
        "parameters": summary,
        "distillation": {
            "temperature": args.temperature,
            "ce_weight": args.ce_weight,
            "kl_weight": args.kl_weight,
            "hidden_weight": args.hidden_weight,
        },
        "initial": initial,
        "best": best,
        "best_update": best_update,
        "final": final,
        "relative_student_ce_improvement": relative_ce_improvement,
        "teacher_gap_recovery_fraction": recovery,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "eval_history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "distillation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best_dir.exists():
        (best_dir / "distillation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 80)
    print(
        f"DISTILLATION HEALTH: {status.upper()} | student CE {initial['student_ce']:.4f} -> "
        f"{best['student_ce']:.4f} | teacher={best['teacher_ce']:.4f} | "
        f"teacher-gap recovery={recovery * 100:.2f}%"
    )
    print(f"best checkpoint: {best_dir if best_dir.exists() else output_dir}")
    print("=" * 80)
    if diverged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
