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

from tinycenn_lm.smollm2_amcenn import (
    DEFAULT_SMOLLM2,
    AMCeNNAttention,
    ShardedTop2LlamaMLP,
    SmolAMCeNNConfig,
    amcenn_parameter_summary,
    amcenn_router_stats,
    freeze_smollm2_for_amcenn_training,
    replace_smollm2_core,
    save_smollm2_amcenn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill SmolLM2 self-attention into recurrent AM-CeNN")
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--output-dir", default="checkpoints/smollm2-amcenn-top2")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--max-runtime-minutes", type=float, default=45.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--router-aux-weight", type=float, default=1e-4)
    p.add_argument("--router-z-weight", type=float, default=1e-5)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--shuffle-buffer", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ffn-shards", action="store_true")
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


def distillation_kl(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    per_token = F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean() * (temperature**2)


def structural_assertions(model) -> None:
    layers = model.model.layers
    if len(layers) != int(model.config.num_hidden_layers):
        raise RuntimeError("decoder layer count changed unexpectedly")
    if not all(isinstance(layer.self_attn, AMCeNNAttention) for layer in layers):
        raise RuntimeError("not every self-attention layer was replaced by AM-CeNN")
    if not all(isinstance(layer.mlp, ShardedTop2LlamaMLP) for layer in layers):
        raise RuntimeError("not every FFN was replaced by 8-shard Top-2 FFN")
    if any("LlamaAttention" in m.__class__.__name__ for m in model.modules()):
        raise RuntimeError("Transformer self-attention module remains in student")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.config.use_cache = False

    student = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    config = SmolAMCeNNConfig(
        feature_dim=args.feature_dim,
        num_shards=args.num_shards,
        top_k=args.top_k,
        feature_seed=1234 + args.seed,
    )
    replace_smollm2_core(student, config)
    freeze_smollm2_for_amcenn_training(student, train_ffn_shards=args.train_ffn_shards)
    structural_assertions(student)
    params = amcenn_parameter_summary(student)
    print("parameter summary:", json.dumps(params, indent=2))

    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    )
    raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    batch_iter = iter(batches(token_blocks(raw, tokenizer, args.text_field, args.context_length), args.batch_size))

    trainable = [p for p in student.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except Exception:
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    tokens_per_update = args.batch_size * args.context_length * args.grad_accum
    total_updates = max(1, math.ceil(args.max_tokens / tokens_per_update))
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    seen_tokens = 0
    micro = 0
    update = 0
    stop_reason = "token_budget"
    last = {"ce": float("nan"), "kl": float("nan"), "total": float("nan")}

    while seen_tokens < args.max_tokens:
        if (time.perf_counter() - start) / 60.0 >= args.max_runtime_minutes:
            stop_reason = "runtime_budget"
            break

        ids = next(batch_iter).to(device, non_blocking=True)
        with torch.inference_mode(), amp():
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits

        with amp():
            out = student(input_ids=ids, labels=ids, use_cache=False)
            kl = distillation_kl(out.logits, teacher_logits, args.temperature)
            router = amcenn_router_stats(student)
            loss = (
                args.ce_weight * out.loss
                + args.kl_weight * kl
                + args.router_aux_weight * router["load_balance"]
                + args.router_z_weight * router["z_loss"]
            )
            scaled_loss = loss / args.grad_accum

        if not torch.isfinite(scaled_loss):
            raise RuntimeError("non-finite AM-CeNN training loss")
        scaler.scale(scaled_loss).backward()
        micro += 1
        seen_tokens += ids.numel()
        last = {
            "ce": float(out.loss.detach().float()),
            "kl": float(kl.detach().float()),
            "total": float(loss.detach().float()),
        }
        del teacher_logits, out
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
            router = amcenn_router_stats(student)
            print(
                f"update={update}/{total_updates} tokens={seen_tokens:,} "
                f"ce={last['ce']:.4f} kl={last['kl']:.4f} "
                f"route_mix={float(router['route_mix']):.5f} "
                f"entropy={float(router['entropy']):.4f} "
                f"grad={grad_norm:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    elapsed = time.perf_counter() - start
    student.eval()
    with torch.inference_mode():
        probe = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
        _ = student(input_ids=probe, use_cache=False)
    router = amcenn_router_stats(student)
    report = {
        "status": "trained",
        "architecture": "smollm2-amcenn-top2",
        "base_model": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "evaluation_performed": False,
        "stop_reason": stop_reason,
        "seen_tokens": seen_tokens,
        "updates": update,
        "context_length": args.context_length,
        "parameters": params,
        "feature_dim": args.feature_dim,
        "num_shards": args.num_shards,
        "top_k": args.top_k,
        "train_ffn_shards": bool(args.train_ffn_shards),
        "last_training_ce": last["ce"],
        "last_distillation_kl": last["kl"],
        "mean_route_mix": float(router["route_mix"]),
        "mean_router_entropy": float(router["entropy"]),
        "elapsed_minutes": elapsed / 60.0,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
    }
    save_smollm2_amcenn(
        student,
        output_dir,
        config=config,
        base_model=args.base_model,
        extra_metadata=report,
    )
    tokenizer.save_pretrained(output_dir)
    (output_dir / "smollm2_amcenn_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
