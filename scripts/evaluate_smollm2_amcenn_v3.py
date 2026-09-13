#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.smollm2_amcenn_v3 import (
    build_smollm2_amcenn_v3,
    v3_attention_stats,
)


PROMPTS = [
    "The capital of Austria is",
    "Artificial intelligence can help people",
    "Once upon a time, a small robot was lost in a park.",
    "A small language model can",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare AM-CeNN hybrid v3 against its SmolLM2 teacher")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", default="v3_evaluation.json")
    p.add_argument("--eval-tokens", type=int, default=16384)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--seed", type=int, default=9001)
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def token_blocks(dataset, tokenizer, context_length: int):
    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos)
        while len(buffer) >= context_length:
            yield torch.tensor(buffer[:context_length], dtype=torch.long).unsqueeze(0)
            del buffer[:context_length]


def distillation_kl(student_logits, teacher_logits, temperature=2.0):
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    return (
        F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none")
        .sum(dim=-1)
        .mean()
        * (temperature ** 2)
    )


def repeated_ngram_fraction(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def main() -> None:
    args = parse_args()
    root = Path(args.model_dir).resolve()
    output = Path(args.output).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)

    meta = json.loads((root / "smollm2_amcenn_v3_config.json").read_text())
    base_model = meta["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = build_smollm2_amcenn_v3(root, device=device, dtype=dtype)
    teacher = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    student.eval()
    teacher.eval()
    student.config.use_cache = False
    teacher.config.use_cache = False

    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=2048)
    blocks = token_blocks(raw, tokenizer, args.context_length)

    teacher_losses, student_losses, kls = [], [], []
    seen = 0
    with torch.no_grad():
        while seen < args.eval_tokens:
            ids = next(blocks).to(device)
            t = teacher(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            s = student(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            teacher_losses.append(t.loss.detach().float())
            student_losses.append(s.loss.detach().float())
            kls.append(distillation_kl(s.logits, t.logits).detach().float())
            seen += ids.numel()

    teacher_ce = float(torch.stack(teacher_losses).mean().cpu())
    student_ce = float(torch.stack(student_losses).mean().cpu())
    kl = float(torch.stack(kls).mean().cpu())

    generations = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            teacher_ids = teacher.generate(**inputs, **kwargs)
            student_ids = student.generate(**inputs, **kwargs)
        teacher_text = tokenizer.decode(teacher_ids[0], skip_special_tokens=True)
        student_text = tokenizer.decode(student_ids[0], skip_special_tokens=True)
        generations.append(
            {
                "prompt": prompt,
                "teacher": teacher_text,
                "student": student_text,
                "teacher_repeat_3gram_fraction": repeated_ngram_fraction(teacher_text, 3),
                "student_repeat_3gram_fraction": repeated_ngram_fraction(student_text, 3),
            }
        )

    report = {
        "status": "PASS",
        "architecture": "smollm2-amcenn-hybrid-v3",
        "model_dir": str(root),
        "base_model": base_model,
        "device": str(device),
        "dtype": str(dtype),
        "eval_tokens": seen,
        "teacher_ce": teacher_ce,
        "student_ce": student_ce,
        "ce_gap": student_ce - teacher_ce,
        "teacher_perplexity": math.exp(min(teacher_ce, 20.0)),
        "student_perplexity": math.exp(min(student_ce, 20.0)),
        "student_teacher_kl": kl,
        "gate_stats": v3_attention_stats(student),
        "generations": generations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
