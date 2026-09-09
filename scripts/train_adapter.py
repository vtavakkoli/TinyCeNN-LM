#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Train and health-check a CeNN residual on Tiny-LLM")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="checkpoints/tinycenn-base")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1_000_000)
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
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-shuffle-buffer", type=int, default=2000)
    p.add_argument("--health-min-improvement", type=float, default=0.002)
    p.add_argument("--divergence-factor", type=float, default=1.25)
    p.add_argument("--fail-on-no-improvement", action="store_true")
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


def collect_eval_batches(dataset, tokenizer, text_field: str, block_size: int, batch_size: int, count: int):
    batches = batch_blocks(token_blocks(dataset, tokenizer, text_field, block_size), batch_size)
    out: list[torch.Tensor] = []
    for _ in range(count):
        try:
            out.append(next(batches))
        except StopIteration:
            break
    if not out:
        raise RuntimeError("could not build any evaluation batches")
    return out


def cosine_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))


@torch.inference_mode()
def evaluate(model, batches: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if device.type == "cuda"
        else nullcontext
    )
    for cpu_ids in batches:
        input_ids = cpu_ids.to(device, non_blocking=True)
        with amp_context():
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        n = input_ids.numel()
        loss = float(outputs.loss.detach().float().item())
        if not math.isfinite(loss):
            return float("inf")
        total_loss += loss * n
        total_tokens += n
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"health report: {path}")


def main() -> None:
    args = parse_args()
    if args.context_length < 8:
        raise ValueError("context length is too small")
    if args.max_tokens < args.context_length * args.batch_size:
        raise ValueError("max-tokens must cover at least one batch")
    if args.eval_batches < 1 or args.eval_batch_size < 1:
        raise ValueError("evaluation batch settings must be >= 1")
    if args.divergence_factor <= 1.0:
        raise ValueError("divergence-factor must be > 1.0")

    output_dir = Path(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
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

    summary = trainable_parameter_summary(model)
    print(
        "parameters: "
        f"total={summary['total']:,} trainable={summary['trainable']:,} "
        f"({summary['trainable_percent']:.2f}%)"
    )
    print(f"CeNN recurrent steps={args.steps}, dilations={dilations}")

    # FineWeb sample-10BT has no official validation split. A separately shuffled,
    # fixed monitoring sample is therefore used to detect improvement/divergence.
    # This is a health check, not a publication-grade held-out benchmark.
    print("loading evaluation monitoring sample...")
    eval_dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed + 10_000, buffer_size=args.eval_shuffle_buffer)
    eval_batches = collect_eval_batches(
        eval_dataset,
        tokenizer,
        args.text_field,
        args.context_length,
        args.eval_batch_size,
        args.eval_batches,
    )

    initial_eval = evaluate(model, eval_batches, device, dtype)
    if not math.isfinite(initial_eval):
        raise FloatingPointError("initial evaluation loss is non-finite")
    print(
        f"initial_eval_loss={initial_eval:.4f} "
        f"initial_ppl={safe_perplexity(initial_eval):.2f}"
    )

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

    train_dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)
    batches = batch_blocks(
        token_blocks(train_dataset, tokenizer, args.text_field, args.context_length),
        args.batch_size,
    )

    model.train()
    train_model = model
    if not args.no_compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            train_model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print("torch.compile enabled (reduce-overhead)")
        except Exception as exc:
            print(f"torch.compile unavailable; continuing in eager mode: {exc}")

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
    run_start = window_start
    best_eval = initial_eval
    best_update = 0
    eval_history = [
        {"update": 0, "tokens": 0, "loss": initial_eval, "perplexity": safe_perplexity(initial_eval)}
    ]
    diverged = False

    while update < total_updates:
        input_ids = next(batches).to(device, non_blocking=True)
        with amp_context():
            outputs = train_model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = outputs.loss / args.grad_accum

        if not torch.isfinite(loss):
            diverged = True
            print(f"ERROR: non-finite training loss at update {update}")
            break

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
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(grad_norm):
            diverged = True
            print(f"ERROR: non-finite gradient norm at update {update}")
            break

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
                f"loss={avg_loss:.4f} lr={lr:.2e} grad={grad_norm:.3f} "
                f"tok/s={tps:,.0f} peak_vram={vram:.2f}GiB"
            )
            running_loss = 0.0
            window_microbatches = 0
            window_tokens = 0
            window_start = time.perf_counter()

        should_eval = (
            update == total_updates
            or (args.eval_every > 0 and update % args.eval_every == 0)
        )
        if should_eval:
            eval_loss = evaluate(model, eval_batches, device, dtype)
            eval_history.append(
                {
                    "update": update,
                    "tokens": seen_tokens,
                    "loss": eval_loss,
                    "perplexity": safe_perplexity(eval_loss) if math.isfinite(eval_loss) else None,
                }
            )
            print(
                f"eval update={update}: loss={eval_loss:.4f} "
                f"ppl={safe_perplexity(eval_loss):.2f}"
            )
            if not math.isfinite(eval_loss) or eval_loss > initial_eval * args.divergence_factor:
                diverged = True
                print(
                    "ERROR: evaluation indicates divergence "
                    f"(threshold={initial_eval * args.divergence_factor:.4f})"
                )
                break
            if eval_loss < best_eval:
                best_eval = eval_loss
                best_update = update
                save_adapter(
                    model,
                    f"{args.output_dir}-best",
                    base_model=args.base_model,
                    layer_indices=(0,),
                    config=cenn_config,
                )
                print(f"new best adapter saved at update {update}")

        if args.save_every > 0 and update % args.save_every == 0:
            save_adapter(
                model,
                f"{args.output_dir}-step-{update}",
                base_model=args.base_model,
                layer_indices=(0,),
                config=cenn_config,
            )

    final_eval = evaluate(model, eval_batches, device, dtype) if not diverged else float("inf")
    improvement = (initial_eval - best_eval) / max(initial_eval, 1e-12)
    status = "diverged" if diverged else (
        "healthy" if improvement >= args.health_min_improvement else "warning_no_improvement"
    )

    if not diverged:
        save_adapter(
            model,
            output_dir,
            base_model=args.base_model,
            layer_indices=(0,),
            config=cenn_config,
        )
        tokenizer.save_pretrained(output_dir)

    elapsed_seconds = time.perf_counter() - run_start
    report = {
        "status": status,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "context_length": args.context_length,
        "requested_tokens": args.max_tokens,
        "seen_tokens": seen_tokens,
        "updates_completed": update,
        "total_updates": total_updates,
        "cenn_steps": args.steps,
        "cenn_dilations": list(dilations),
        "parameters": summary,
        "initial_eval_loss": initial_eval,
        "initial_perplexity": safe_perplexity(initial_eval),
        "final_eval_loss": final_eval if math.isfinite(final_eval) else None,
        "final_perplexity": safe_perplexity(final_eval) if math.isfinite(final_eval) else None,
        "best_eval_loss": best_eval,
        "best_perplexity": safe_perplexity(best_eval),
        "best_update": best_update,
        "relative_best_improvement": improvement,
        "health_min_improvement": args.health_min_improvement,
        "divergence_factor": args.divergence_factor,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        ),
        "evaluation_note": (
            "FineWeb sample-10BT has no official validation split; evaluation uses a fixed "
            "separately shuffled monitoring sample and is intended for training-health checks."
        ),
        "eval_history": eval_history,
    }
    write_report(output_dir, report)

    print("=" * 72)
    print(
        f"TRAINING HEALTH: {status.upper()} | "
        f"initial_loss={initial_eval:.4f} best_loss={best_eval:.4f} "
        f"improvement={improvement * 100:.2f}%"
    )
    if status == "healthy":
        print(f"best adapter: {args.output_dir}-best (update {best_update})")
    elif status == "warning_no_improvement":
        print("WARNING: run stayed finite but did not improve enough on the monitoring sample.")
    else:
        print("ERROR: training diverged; inspect learning rate, gradients and report history.")
    print("=" * 72)

    if diverged:
        raise SystemExit(2)
    if status == "warning_no_improvement" and args.fail_on_no_improvement:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
