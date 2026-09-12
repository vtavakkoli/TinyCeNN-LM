#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from tinycenn_lm.sharded_moe import ShardedMoECeNNReplacementLayer
from tinycenn_lm.story import repetition_unlikelihood_loss
from tinycenn_lm.story_v2 import (
    StoryV2Config,
    StoryV2ReplacementLayer,
    build_story_v2_from_story_v1,
    freeze_story_v2_interfaces,
    save_story_v2_student,
    story_v2_parameter_summary,
    story_v2_router_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TinyCeNN Story-v2 continuation specialization")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", default="checkpoints/tinycenn-story-v2")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=15_000_000)
    p.add_argument("--max-runtime-minutes", type=float, default=45.0)
    p.add_argument("--learning-rate", type=float, default=7e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--repeat-weight", type=float, default=0.08)
    p.add_argument("--repeat-window", type=int, default=24)
    p.add_argument("--router-aux-weight", type=float, default=1e-4)
    p.add_argument("--router-z-weight", type=float, default=5e-5)
    p.add_argument("--shuffle-buffer", type=int, default=4096)
    p.add_argument("--memory-rank", type=int, default=32)
    p.add_argument("--head-rank", type=int, default=4)
    p.add_argument("--max-trainable-params", type=int, default=650_000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=43)
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
    return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def split_opening(text: str) -> tuple[str, str] | None:
    text = " ".join(str(text).strip().split())
    if len(text) < 120:
        return None
    match = re.search(r"[.!?](?:[\"']?)(?:\s+|$)", text)
    if match and match.end() >= 20 and len(text) - match.end() >= 60:
        return text[: match.end()].strip(), text[match.end() :].strip()

    words = text.split()
    if len(words) < 30:
        return None
    cut = max(8, min(24, len(words) // 5))
    opening = " ".join(words[:cut]).strip()
    continuation = " ".join(words[cut:]).strip()
    if not continuation:
        return None
    return opening, continuation


def continuation_examples(dataset, tokenizer, text_field: str, context_length: int):
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if eos is None or pad is None:
        raise ValueError("tokenizer must define eos/pad token")

    for row in dataset:
        split = split_opening(row.get(text_field, ""))
        if split is None:
            continue
        opening, continuation = split
        prompt_text = f"Story beginning:\n{opening}\nContinue the story:\n"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"] + [eos]
        if len(prompt_ids) >= context_length - 24:
            continue
        continuation_ids = continuation_ids[: context_length - len(prompt_ids)]
        if len(continuation_ids) < 24:
            continue

        real_ids = prompt_ids + continuation_ids
        pad_len = context_length - len(real_ids)
        input_ids = real_ids + [pad] * pad_len
        attention_mask = [1] * len(real_ids) + [0] * pad_len
        labels = [-100] * len(prompt_ids) + continuation_ids + [-100] * pad_len
        yield (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            len(continuation_ids),
        )


def batches(examples, batch_size: int):
    pending = []
    for example in examples:
        pending.append(example)
        if len(pending) == batch_size:
            ids = torch.stack([x[0] for x in pending])
            mask = torch.stack([x[1] for x in pending])
            labels = torch.stack([x[2] for x in pending])
            target_tokens = sum(x[3] for x in pending)
            yield ids, mask, labels, target_tokens
            pending.clear()


def get_sharded_config(model):
    for module in model.modules():
        if isinstance(module, StoryV2ReplacementLayer):
            return module.cenn.config
        if isinstance(module, ShardedMoECeNNReplacementLayer):
            return module.config
    raise RuntimeError("could not find sharded CeNN configuration")


def save_checkpoint(model, tokenizer, output_dir: Path, story_config, sharded_config, base_model: str, metadata: dict) -> None:
    save_story_v2_student(
        model,
        output_dir,
        story_config=story_config,
        sharded_config=sharded_config,
        base_model=base_model,
        extra_metadata=metadata,
    )
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
    source_meta = json.loads((source_dir / "sharded_moe_student_config.json").read_text())
    base_model = source_meta.get("base_model", "arnir0/Tiny-LLM")
    tokenizer = AutoTokenizer.from_pretrained(source_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    story_config = StoryV2Config(memory_rank=args.memory_rank, head_rank=args.head_rank)
    model = build_story_v2_from_story_v1(
        source_dir,
        story_config=story_config,
        device=device,
        dtype=dtype,
    )
    freeze_story_v2_interfaces(model)
    sharded_config = get_sharded_config(model)
    params = story_v2_parameter_summary(model)
    print("parameter summary:", json.dumps(params, indent=2))
    if int(params["trainable"]) > args.max_trainable_params:
        raise RuntimeError(
            f"Story-v2 trainable parameter budget exceeded: {params['trainable']} > {args.max_trainable_params}"
        )

    raw = load_dataset(args.dataset, split=args.split, streaming=True)
    raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    example_iter = continuation_examples(raw, tokenizer, args.text_field, args.context_length)
    batch_iter = iter(batches(example_iter, args.batch_size))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext

    estimated_target_tokens_per_update = args.batch_size * 128 * args.grad_accum
    total_updates = max(1, math.ceil(args.max_tokens / estimated_target_tokens_per_update))
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    seen_target_tokens = 0
    micro = 0
    update = 0
    stop_reason = "token_budget"
    last = {"ce": float("nan"), "repeat": float("nan"), "total": float("nan")}

    while seen_target_tokens < args.max_tokens:
        if (time.perf_counter() - start) / 60.0 >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            break

        ids, mask, labels, target_tokens = next(batch_iter)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp():
            out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
            repeat_loss = repetition_unlikelihood_loss(out.logits, labels, window=args.repeat_window)
            router = story_v2_router_stats(model)
            loss = (
                out.loss
                + args.repeat_weight * repeat_loss
                + args.router_aux_weight * router["load_balance"]
                + args.router_z_weight * router["z_loss"]
            )
            scaled_loss = loss / args.grad_accum

        if not torch.isfinite(scaled_loss):
            raise RuntimeError("non-finite Story-v2 training loss")
        scaler.scale(scaled_loss).backward()
        micro += 1
        seen_target_tokens += int(target_tokens)
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
            router = story_v2_router_stats(model)
            print(
                f"update={update} target_tokens={seen_target_tokens:,} "
                f"ce={last['ce']:.4f} repeat={last['repeat']:.5f} "
                f"route_mix={float(router['route_mix'].detach().float()):.4f} "
                f"grad={grad_norm:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if args.save_every > 0 and update % args.save_every == 0:
            save_checkpoint(
                model, tokenizer, output_dir, story_config, sharded_config, base_model,
                {
                    "task": "TinyCeNN Story-v2 continuation specialization",
                    "seen_target_tokens": seen_target_tokens,
                    "updates": update,
                    "source_dir": str(source_dir),
                },
            )

    elapsed = time.perf_counter() - start
    save_checkpoint(
        model, tokenizer, output_dir, story_config, sharded_config, base_model,
        {
            "task": "TinyCeNN Story-v2 continuation specialization",
            "seen_target_tokens": seen_target_tokens,
            "updates": update,
            "source_dir": str(source_dir),
            "stop_reason": stop_reason,
        },
    )

    report = {
        "status": "trained",
        "architecture": "tinycenn-story-v2-memory-head",
        "task": "short-story-continuation",
        "dataset": args.dataset,
        "evaluation_performed": False,
        "stop_reason": stop_reason,
        "seen_target_tokens": seen_target_tokens,
        "updates": update,
        "parameters": params,
        "memory_rank": args.memory_rank,
        "head_rank": args.head_rank,
        "learning_rate": args.learning_rate,
        "repeat_weight": args.repeat_weight,
        "repeat_window": args.repeat_window,
        "last_training_ce": last["ce"],
        "last_repetition_unlikelihood": last["repeat"],
        "elapsed_minutes": elapsed / 60.0,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
        ),
    }
    (output_dir / "story_v2_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
