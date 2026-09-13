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

from tinycenn_lm.smollm2_amcenn_v4 import build_smollm2_amcenn_v4, v4_attention_stats


PROMPTS = [
    "The capital of Austria is",
    "Artificial intelligence can help people",
    "Once upon a time, a small robot was lost in a park.",
    "A small language model can",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SmolLM2 AM-CeNN adaptive v4 against its teacher")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--eval-tokens", type=int, default=8192)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=9107)
    return p.parse_args()


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def token_blocks(dataset, tokenizer, context_length):
    eos = tokenizer.eos_token_id
    buffer = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(eos)
        while len(buffer) >= context_length:
            yield torch.tensor(buffer[:context_length], dtype=torch.long)
            del buffer[:context_length]


def distillation_kl(student_logits, teacher_logits, temperature=2.0):
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    return F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction="none").sum(-1).mean() * temperature**2


def threegram_repeat_fraction(text: str) -> float:
    words = text.split()
    grams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def recent_repeat_fraction(token_ids: torch.Tensor, window: int = 16) -> float:
    values = token_ids.tolist()
    if len(values) < 2:
        return 0.0
    repeats = total = 0
    for i in range(1, len(values)):
        start = max(0, i - window)
        repeats += int(values[i] in values[start:i])
        total += 1
    return repeats / max(total, 1)


def main():
    args = parse_args()
    root = Path(args.model_dir).resolve()
    out_path = Path(args.output).resolve()
    metadata = json.loads((root / "smollm2_amcenn_v4_config.json").read_text())
    base_model = metadata["base_model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device).eval()
    student = build_smollm2_amcenn_v4(root, device=device, dtype=dtype).eval()
    teacher.config.use_cache = False
    student.config.use_cache = False

    raw = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=True).shuffle(
        seed=args.seed, buffer_size=2048
    )
    blocks = token_blocks(raw, tokenizer, args.context_length)
    teacher_losses, student_losses, kls = [], [], []
    seen = 0
    with torch.no_grad():
        while seen < args.eval_tokens:
            ids = next(blocks).unsqueeze(0).to(device)
            t = teacher(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            s = student(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            teacher_losses.append(t.loss.detach().float())
            student_losses.append(s.loss.detach().float())
            kls.append(distillation_kl(s.logits, t.logits).detach().float())
            seen += ids.numel()

    teacher_ce = float(torch.stack(teacher_losses).mean().cpu())
    student_ce = float(torch.stack(student_losses).mean().cpu())

    generations = []
    with torch.no_grad():
        for prompt in PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = encoded.input_ids.shape[1]
            kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            teacher_ids = teacher.generate(**encoded, **kwargs)[0]
            student_ids = student.generate(**encoded, **kwargs)[0]
            teacher_text = tokenizer.decode(teacher_ids, skip_special_tokens=True)
            student_text = tokenizer.decode(student_ids, skip_special_tokens=True)
            generations.append({
                "prompt": prompt,
                "teacher": teacher_text,
                "student": student_text,
                "teacher_repeat_3gram_fraction": threegram_repeat_fraction(teacher_text),
                "student_repeat_3gram_fraction": threegram_repeat_fraction(student_text),
                "teacher_recent_token_repeat_fraction": recent_repeat_fraction(teacher_ids[input_len:]),
                "student_recent_token_repeat_fraction": recent_repeat_fraction(student_ids[input_len:]),
            })

    report = {
        "status": "PASS",
        "architecture": "smollm2-amcenn-adaptive-v4",
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
        "student_teacher_kl": float(torch.stack(kls).mean().cpu()),
        "attention_stats": v4_attention_stats(student),
        "generations": generations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
