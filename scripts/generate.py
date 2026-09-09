#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from tinycenn_lm import build_from_adapter


def main():
    p = argparse.ArgumentParser(description="Generate text with a trained TinyCeNN-LM adapter")
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompt", default="The future of efficient AI is")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    model = build_from_adapter(args.adapter, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            use_cache=True,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
