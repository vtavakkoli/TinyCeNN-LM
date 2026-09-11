#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.distill_utils import (
    collect_eval_batches,
    combined_loss,
    evaluation_fingerprint,
    partition_rows,
)
from tinycenn_lm.sharded_moe import build_sharded_moe_student, sharded_router_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce sharded MoE-CeNN held-out benchmark")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--student-dir")
    source.add_argument("--hf-repo")
    p.add_argument("--revision", default="main")
    p.add_argument("--ce-tolerance", type=float, default=0.02)
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main() -> None:
    args = parse_args()
    root = (
        Path(args.student_dir).expanduser().resolve()
        if args.student_dir
        else Path(snapshot_download(repo_id=args.hf_repo, revision=args.revision, repo_type="model"))
    )
    report_path = root / "sharded_moe_distillation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("architecture") != "sharded-moe-cenn-top2-replacement":
        raise RuntimeError("checkpoint is not the parameter-neutral sharded MoE model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if device.type == "cuda":
        load_kwargs["dtype"] = dtype
    teacher = AutoModelForCausalLM.from_pretrained(report["base_model_teacher"], **load_kwargs).to(device).eval()
    teacher.config.use_cache = False
    student = build_sharded_moe_student(root, device=device, dtype=dtype).eval()

    eval_cfg = report["evaluation"]
    raw = load_dataset(
        report["dataset"],
        report["dataset_config"],
        split=report.get("dataset_split", "train"),
        streaming=True,
    )
    rows = partition_rows(raw, report.get("text_field", "text"), validation=True)
    batches = collect_eval_batches(
        rows,
        tokenizer,
        report.get("text_field", "text"),
        int(report["context_length"]),
        int(eval_cfg["batch_size"]),
        int(eval_cfg["batches"]),
    )
    fingerprint = evaluation_fingerprint(batches)
    if fingerprint != eval_cfg["fingerprint_sha256"]:
        raise RuntimeError("held-out benchmark fingerprint mismatch")

    distill_cfg = report["distillation"]
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    student_ce = teacher_ce = kl = hidden = 0.0
    shard_fraction = torch.zeros(int(report["num_shards"]), dtype=torch.float64)
    route_mix = 0.0
    with torch.inference_mode():
        for cpu_ids in batches:
            ids = cpu_ids.to(device)
            with amp():
                t = teacher(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
                s = student(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
                _, parts = combined_loss(
                    s,
                    t,
                    temperature=float(distill_cfg["temperature"]),
                    kl_chunk_rows=256,
                    ce_weight=float(distill_cfg["ce_weight"]),
                    kl_weight=float(distill_cfg["kl_weight"]),
                    hidden_weight=float(distill_cfg["hidden_weight"]),
                )
                router = sharded_router_stats(student)
            student_ce += float(s.loss.detach().float())
            teacher_ce += float(t.loss.detach().float())
            kl += parts["kl"]
            hidden += parts["hidden"]
            shard_fraction += router["shard_fraction"].detach().float().cpu().double()
            route_mix += float(router["route_mix"].detach().float())

    n = max(len(batches), 1)
    student_ce /= n
    teacher_ce /= n
    kl /= n
    hidden /= n
    shard_fraction /= n
    route_mix /= n
    expected_ce = float(report["best"]["student_ce"])
    delta = abs(student_ce - expected_ce)
    result = {
        "benchmark_protocol": report["benchmark_protocol"],
        "fingerprint_sha256": fingerprint,
        "student_ce": student_ce,
        "student_ppl": math.exp(min(student_ce, 30.0)),
        "teacher_ce": teacher_ce,
        "teacher_ppl": math.exp(min(teacher_ce, 30.0)),
        "kl": kl,
        "hidden": hidden,
        "route_mix": route_mix,
        "shard_fraction": shard_fraction.tolist(),
        "expected_best_ce": expected_ce,
        "ce_absolute_delta": delta,
        "ce_tolerance": args.ce_tolerance,
    }
    print(json.dumps(result, indent=2))
    if delta > args.ce_tolerance:
        raise RuntimeError(f"remote CE parity failed: {delta:.6f} > {args.ce_tolerance:.6f}")
    print("SHARDED MOE REMOTE BENCHMARK: PASS")


if __name__ == "__main__":
    main()
