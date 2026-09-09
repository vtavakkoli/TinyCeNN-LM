#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm import build_cenn_student
from tinycenn_lm.distill_utils import (
    collect_eval_batches,
    evaluation_fingerprint,
    evaluate_distillation,
    partition_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the rigorous held-out benchmark for a distilled CeNN student")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--student-dir")
    source.add_argument("--hf-repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--ce-tolerance", type=float, default=0.02)
    p.add_argument("--skip-parity-check", action="store_true")
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_student(args) -> Path:
    if args.student_dir:
        return Path(args.student_dir).expanduser().resolve()
    path = snapshot_download(repo_id=args.hf_repo, revision=args.revision, repo_type="model")
    return Path(path)


def main() -> None:
    args = parse_args()
    student_dir = resolve_student(args)
    report_path = student_dir / "distillation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing distillation report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("benchmark_protocol") != "rigorous-v2":
        raise RuntimeError(
            "checkpoint was not produced by rigorous-v2 benchmark protocol; "
            "run the rigorous continuation notebook first"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    base_model = report["base_model_teacher"]
    context_length = int(report["context_length"])
    eval_cfg = report["evaluation"]
    eval_batches_count = int(eval_cfg["batches"])
    eval_batch_size = int(eval_cfg["batch_size"])
    dataset_split = report.get("dataset_split", "train")
    text_field = report.get("text_field", "text")

    tokenizer = AutoTokenizer.from_pretrained(student_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if device.type == "cuda":
        load_kwargs["dtype"] = dtype
    teacher = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs).to(device)
    teacher.config.use_cache = False
    teacher.eval()

    student = build_cenn_student(student_dir, device=device, dtype=dtype)
    student.eval()

    eval_raw = load_dataset(
        report["dataset"],
        report["dataset_config"],
        split=dataset_split,
        streaming=True,
    )
    eval_rows = partition_rows(eval_raw, text_field, validation=True)
    batches = collect_eval_batches(
        eval_rows,
        tokenizer,
        text_field,
        context_length,
        eval_batch_size,
        eval_batches_count,
    )
    fingerprint = evaluation_fingerprint(batches)
    expected_fingerprint = eval_cfg["fingerprint_sha256"]
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            "held-out benchmark fingerprint mismatch: "
            f"expected={expected_fingerprint} actual={fingerprint}"
        )

    distill = report["distillation"]
    metrics = evaluate_distillation(
        teacher,
        student,
        batches,
        device=device,
        dtype=dtype,
        temperature=float(distill["temperature"]),
        kl_chunk_rows=int(distill.get("kl_chunk_rows", 256)),
        ce_weight=float(distill["ce_weight"]),
        kl_weight=float(distill["kl_weight"]),
        hidden_weight=float(distill["hidden_weight"]),
    )

    expected_ce = float(report["best"]["student_ce"])
    actual_ce = float(metrics["student_ce"])
    ce_delta = abs(actual_ce - expected_ce)
    result = {
        "source": str(student_dir),
        "benchmark_protocol": report["benchmark_protocol"],
        "fingerprint_sha256": fingerprint,
        "eval_tokens": metrics["eval_tokens"],
        "student_ce": actual_ce,
        "student_ppl": metrics["student_ppl"],
        "teacher_ce": metrics["teacher_ce"],
        "teacher_ppl": metrics["teacher_ppl"],
        "kl": metrics["kl"],
        "hidden": metrics["hidden"],
        "expected_best_student_ce": expected_ce,
        "ce_absolute_delta": ce_delta,
        "ce_tolerance": args.ce_tolerance,
    }
    print(json.dumps(result, indent=2))

    if not all(
        math.isfinite(float(metrics[key]))
        for key in ("student_ce", "teacher_ce", "kl", "hidden")
    ):
        raise RuntimeError("non-finite benchmark metric")
    if not args.skip_parity_check and ce_delta > args.ce_tolerance:
        raise RuntimeError(
            f"remote checkpoint CE parity failed: delta={ce_delta:.6f} > tolerance={args.ce_tolerance:.6f}"
        )
    print("RIGOROUS REMOTE BENCHMARK: PASS")


if __name__ == "__main__":
    main()
