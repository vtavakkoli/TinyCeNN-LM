#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm import (
    DEFAULT_BASE_MODEL,
    CeNNConfig,
    freeze_for_adapter_training,
    inject_cenn,
    save_adapter,
    trainable_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a fast CeNN residual on arnir0/Tiny-LLM")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="checkpoints/tinycenn-base")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--dilations", default="1,2,4,8")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--train-lm-head", action="store_true")
    p.add_argument("--train-embeddings", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def token_blocks(dataset, tokenizer, text_field: str, block_size: int) -> Iterator[torch.Tensor]:
    buffer: list[int] = []
    offset = 0
    eos = tokenizer.eos_token_id
    for example in dataset:
        text = example.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids.append(eos)
        buffer.extend(ids)

        while len(buffer) - offset >= block_size:
            block = buffer[offset : offset + block_size]
            offset += block_size
            yield torch.tensor(block, dtype=torch.long)

        # Periodically compact without doing O(n) work for every block.
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


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    if args.context_length < 8:
        raise ValueError("context length is too small")
    if args.max_tokens < args.context_length * args.batch_size:
        raise ValueError("max-tokens must cover at least one batch")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"attn_implementation": "sdpa"}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    dilations = tuple(int(x) for x in args.dilations.split(",") if x.strip())
    cenn_config = CeNNConfig(
        hidden_size=model.config.hidden_size,
        kernel_size=args.kernel_size,
        expansion=args.expansion,
        steps=args.steps,
        dilations=dilations,
        rms_norm_eps=float(getattr(model.config, "rms_norm_eps", 1e-5)),
        dropout=args.dropout,
    )
    inject_cenn(model, config=cenn_config, layer_indices=(0,))
    freeze_for_adapter_training(
        model,
        train_lm_head=args.train_lm_head,
        train_embeddings=args.train_embeddings,
    )
    model.config.use_cache = False
    model.to(device)
    model.train()
    train_model = model
    if not args.no_compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            train_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print("torch.compile enabled (reduce-overhead)")
        except Exception as exc:
            print(f"torch.compile unavailable; continuing in eager mode: {exc}")

    summary = trainable_parameter_summary(model)
    print(
        "parameters: "
        f"total={summary['total']:,} trainable={summary['trainable']:,} "
        f"({summary['trainable_percent']:.2f}%)"
    )
    print(f"CeNN recurrent steps={args.steps}, dilations={dilations}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            trainable, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    tokens_per_microbatch = args.batch_size * args.context_length
    tokens_per_update = tokens_per_microbatch * args.grad_accum
    total_updates = max(1, math.ceil(args.max_tokens / tokens_per_update))
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)
    batches = batch_blocks(
        token_blocks(dataset, tokenizer, args.text_field, args.context_length),
        args.batch_size,
    )

    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if device.type == "cuda"
        else nullcontext
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and dtype == torch.float16)
    )

    optimizer.zero_grad(set_to_none=True)
    seen_tokens = 0
    update = 0
    micro = 0
    running_loss = 0.0
    window_microbatches = 0
    window_tokens = 0
    window_start = time.perf_counter()

    # Run complete optimizer updates even when max_tokens is not exactly divisible
    # by the effective batch. The final run may exceed max_tokens by <1 update,
    # but never drops partially accumulated gradients.
    while update < total_updates:
        input_ids = next(batches).to(device, non_blocking=True)
        with amp_context():
            outputs = train_model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss / args.grad_accum

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}: {loss.item()}")

        scaler.scale(loss).backward()
        micro += 1
        batch_tokens = input_ids.numel()
        seen_tokens += batch_tokens
        window_tokens += batch_tokens
        running_loss += loss.detach().float().item() * args.grad_accum
        window_microbatches += 1

        if micro % args.grad_accum != 0:
            continue

        next_update = update + 1
        mult = cosine_multiplier(next_update, total_updates, warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * mult

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update = next_update

        if update % args.log_every == 0 or update == 1:
            elapsed = max(time.perf_counter() - window_start, 1e-6)
            tps = window_tokens / elapsed
            avg_loss = running_loss / max(window_microbatches, 1)
            lr = optimizer.param_groups[0]["lr"]
            vram = (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            )
            print(
                f"update={update}/{total_updates} tokens={seen_tokens:,} "
                f"loss={avg_loss:.4f} lr={lr:.2e} tok/s={tps:,.0f} "
                f"peak_vram={vram:.2f}GiB"
            )
            running_loss = 0.0
            window_microbatches = 0
            window_tokens = 0
            window_start = time.perf_counter()

        if args.save_every > 0 and update % args.save_every == 0:
            save_adapter(
                model,
                f"{args.output_dir}-step-{update}",
                base_model=args.base_model,
                layer_indices=(0,),
                config=cenn_config,
            )

    save_adapter(
        model,
        args.output_dir,
        base_model=args.base_model,
        layer_indices=(0,),
        config=cenn_config,
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
