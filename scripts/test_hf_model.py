#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from tinycenn_lm import build_from_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a TinyCeNN-LM adapter from Hugging Face and run sanity tests"
    )
    p.add_argument(
        "--repo-id",
        default=os.environ.get("HF_MODEL_REPO", ""),
        help="Hugging Face model repo, e.g. user/TinyCeNN-LM-Base",
    )
    p.add_argument("--revision", default=os.environ.get("HF_REVISION", "main"))
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to test; may be specified multiple times",
    )
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit(
            "HF model repo is required. Set HF_MODEL_REPO=user/TinyCeNN-LM-Base "
            "or pass --repo-id."
        )

    token = os.environ.get("HF_TOKEN") or None
    print(f"Downloading {args.repo_id}@{args.revision} from Hugging Face...")
    adapter_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            token=token,
        )
    )

    required = [adapter_dir / "cenn_adapter.pt", adapter_dir / "cenn_config.json"]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Hugging Face repo is missing required adapter files: {missing}")

    report_path = adapter_dir / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("Published training status:", report.get("status", "unknown"))
        print("Published best eval loss:", report.get("best_eval_loss"))
        if report.get("status") == "diverged":
            raise RuntimeError("Published checkpoint is marked as diverged")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")

    model = build_from_adapter(adapter_dir, device=device, dtype=dtype)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    except Exception:
        metadata = json.loads((adapter_dir / "cenn_config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(metadata["base_model"], use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sanity_text = "TinyCeNN-LM is a compact research language model."
    sanity = tokenizer(sanity_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**sanity, labels=sanity["input_ids"], use_cache=False)
    loss = float(out.loss.detach().cpu())
    if not math.isfinite(loss):
        raise RuntimeError(f"non-finite sanity loss: {loss}")
    ppl = math.exp(min(loss, 20.0))
    print(f"sanity_loss={loss:.4f} sanity_ppl={ppl:.2f}")

    prompts = args.prompts or [
        "The capital of Austria is",
        "Artificial intelligence can help",
        "A small language model",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if int(generated.shape[1]) <= input_len:
            raise RuntimeError(f"generation produced no new tokens for prompt: {prompt!r}")
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print("\nPROMPT:", prompt)
        print(text)

    print("\nTinyCeNN-LM Hugging Face reload test: PASS")


if __name__ == "__main__":
    main()
