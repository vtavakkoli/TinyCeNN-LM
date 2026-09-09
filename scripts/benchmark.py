#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm import DEFAULT_BASE_MODEL, CeNNConfig, inject_cenn, load_adapter


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Tiny-LLM vs TinyCeNN-LM")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    return p.parse_args()


def timed_forward(model, input_ids, warmup: int, iterations: int):
    model.eval()
    if input_ids.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids=input_ids, use_cache=False)
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(input_ids=input_ids, use_cache=False)
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    total_tokens = input_ids.numel() * iterations
    peak = torch.cuda.max_memory_allocated() / 1024**3 if input_ids.is_cuda else 0.0
    return total_tokens / elapsed, peak


def load_base(model_id, dtype):
    kwargs = {"attn_implementation": "sdpa"}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    vocab = len(tokenizer)
    input_ids = torch.randint(
        0, vocab, (args.batch_size, args.context_length), device=device
    )

    base = load_base(args.base_model, dtype).to(device)
    base_tps, base_vram = timed_forward(base, input_ids, args.warmup, args.iterations)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hybrid = load_base(args.base_model, dtype)
    config = CeNNConfig(hidden_size=hybrid.config.hidden_size, steps=args.steps)
    inject_cenn(hybrid, config=config)
    if args.adapter:
        load_adapter(hybrid, args.adapter)
    hybrid.to(device)
    cenn_tps, cenn_vram = timed_forward(hybrid, input_ids, args.warmup, args.iterations)

    print("\nModel                 tok/s        peak VRAM    rel. speed")
    print("--------------------------------------------------------")
    print(f"Tiny-LLM          {base_tps:10,.0f}     {base_vram:6.2f} GiB      1.00x")
    print(
        f"TinyCeNN-LM x{args.steps:<2}  {cenn_tps:10,.0f}     {cenn_vram:6.2f} GiB      "
        f"{cenn_tps/base_tps:5.2f}x"
    )


if __name__ == "__main__":
    main()
