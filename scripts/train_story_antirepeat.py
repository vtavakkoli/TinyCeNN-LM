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
from transformers import AutoTokenizer

from tinycenn_lm.sharded_moe import (
    ShardedMoECeNNReplacementLayer,
    build_sharded_moe_student,
    freeze_sharded_moe_interfaces,
    save_sharded_moe_student,
    sharded_router_stats,
)
from tinycenn_lm.story import repetition_unlikelihood_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast TinyStories specialization with anti-repetition loss")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", default="checkpoints/tinycenn-story-antirepeat")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=20_000_000)
    p.add_argument("--max-runtime-minutes", type=float, default=45.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--repeat-weight", type=float, default=0.20)
    p.add_argument("--repeat-window", type=int, default=32)
    p.add_argument("--router-aux-weight", type=float, default=5e-4)
    p.add_argument("--router-z-weight", type=float, default=1e-4)
    p.add_argument("--shuffle-buffer", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
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
    return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def story_blocks(dataset, tokenizer, text_field: str, context_length: int):
    prefix = tokenizer("Story:\n", add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")

    buffer: list[int] = []
    for row in dataset:
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        ids = prefix + tokenizer(text, add_special_tokens=False)["input_ids"] + [eos]
        if len(ids) < 12:
            continue
        buffer.extend(ids)
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


def get_config(model):
    layer = next((m for m in model.modules() if isinstance(m, ShardedMoECeNNReplacementLayer)), None)
    if layer is None:
        raise RuntimeError("source checkpoint is not a sharded MoE-CeNN model")
    return layer.config


def save_checkpoint(model, tokenizer, output_dir: Path, config, metadata: dict) -> None:
    save_sharded_moe_student(model, output_dir, config=config, extra_metadata=metadata)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device={device} dtype={dtype}")

    source_dir = Path(args.source_dir).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(source_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_sharded_moe_student(source_dir, device=device, dtype=dtype)
    freeze_sharded_moe_interfaces(model)
    config = get_config(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"parameters total={total_count:,} trainable={trainable_count:,}")

    raw = load_dataset(args.dataset, split=args.split, streaming=True)
    raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    blocks = story_blocks(raw, tokenizer, args.text_field, args.context_length)
    batch_iter = iter(batches(blocks, args.batch_size))

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext

    tokens_per_update = args.batch_size * args.context_length * args.grad_accum
    total_updates = math.ceil(args.max_tokens / tokens_per_update)
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    seen_tokens = 0
    micro = 0
    update = 0
    stop_reason = "token_budget"
    last = {"ce": float("nan"), "repeat": float("nan"), "total": float("nan")}

    while update < total_updates:
        if (time.perf_counter() - start) / 60.0 >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            break

        ids = next(batch_iter).to(device, non_blocking=True)
        with amp():
            out = model(input_ids=ids, labels=ids, use_cache=False)
            repeat_loss = repetition_unlikelihood_loss(
                out.logits, ids, window=args.repeat_window
            )
            router = sharded_router_stats(model)
            loss = (
                out.loss
                + args.repeat_weight * repeat_loss
                + args.router_aux_weight * router["load_balance"]
                + args.router_z_weight * router["z_loss"]
            )
            scaled_loss = loss / args.grad_accum

        if not torch.isfinite(scaled_loss):
            raise RuntimeError("non-finite story training loss")
        scaler.scale(scaled_loss).backward()
        micro += 1
        seen_tokens += ids.numel()
        last = {
            "ce": float(out.loss.detach().float()),
            "repeat": float(repeat_loss.detach().float()),
            "total": float(loss.detach().float()),
        }
        if micro % args.grad_accum:
            continue

        update += 1
        lr_mult = cosine_lr(update, total_updates, warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_mult
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if update == 1 or update % args.log_every == 0:
            stats = sharded_router_stats(model)
            route_mix = float(stats["route_mix"].detach().float())
            print(
                f"update={update}/{total_updates} tokens={seen_tokens:,} "
                f"ce={last['ce']:.4f} repeat={last['repeat']:.5f} "
                f"route_mix={route_mix:.4f} grad={grad_norm:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if args.save_every > 0 and update % args.save_every == 0:
            save_checkpoint(
                model,
                tokenizer,
                output_dir,
                config,
                {
                    "task": "TinyStories anti-repetition specialization",
                    "seen_tokens": seen_tokens,
                    "updates": update,
                    "source_dir": str(source_dir),
                },
            )

    elapsed = time.perf_counter() - start
    save_checkpoint(
        model,
        tokenizer,
        output_dir,
        config,
        {
            "task": "TinyStories anti-repetition specialization",
            "seen_tokens": seen_tokens,
            "updates": update,
            "source_dir": str(source_dir),
            "stop_reason": stop_reason,
        },
    )

    model.eval()
    with torch.inference_mode():
        probe = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
        _ = model(input_ids=probe, use_cache=False)
    stats = sharded_router_stats(model)
    report = {
        "status": "trained",
        "task": "short-story-specialization",
        "dataset": args.dataset,
        "evaluation_performed": False,
        "stop_reason": stop_reason,
        "seen_tokens": seen_tokens,
        "updates": update,
        "trainable_parameters": trainable_count,
        "learning_rate": args.learning_rate,
        "repeat_weight": args.repeat_weight,
        "repeat_window": args.repeat_window,
        "last_training_ce": last["ce"],
        "last_repetition_unlikelihood": last["repeat"],
        "route_mix": float(stats["route_mix"].detach().float()),
        "elapsed_minutes": elapsed / 60.0,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        ),
    }
    (output_dir / "story_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
