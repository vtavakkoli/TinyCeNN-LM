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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm import (
    DEFAULT_BASE_MODEL,
    CeNNConfig,
    freeze_student_interfaces,
    load_cenn_student_weights,
    replace_transformer_with_cenn,
    save_cenn_student,
    student_parameter_summary,
)
from tinycenn_lm.distill_utils import (
    batch_blocks,
    buffered_shuffle,
    collect_eval_batches,
    combined_loss,
    evaluation_fingerprint,
    evaluate_distillation,
    partition_rows,
    token_blocks,
)
from tinycenn_lm.training import (
    annealed_weight,
    checkpoint_training_tokens,
    lr_multiplier,
    optimizer_groups,
    plateau_summary,
    stream_resume_offset,
    training_stream_signature,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continue Tiny-LLM → CeNN distillation under the rigorous-v2 protocol")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--dataset-revision", default=None, help="Pin the dataset for reproducible streaming")
    p.add_argument("--text-field", default="text")
    p.add_argument("--output-dir", default="checkpoints/cenn-student-rigorous-v2")
    p.add_argument("--resume-student-dir", default="")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=30_000_000, help="Additional tokens for this run")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-schedule", choices=("cosine", "wsd"), default="cosine")
    p.add_argument("--decay-ratio", type=float, default=0.2)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--train-interfaces", choices=("none", "norm", "all"), default="none")
    p.add_argument("--interface-lr-scale", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--dilations", default="1,2,4,8,16,32,64")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.25)
    p.add_argument("--kl-final-weight", type=float, default=None)
    p.add_argument("--hidden-final-weight", type=float, default=None)
    p.add_argument("--kl-chunk-rows", type=int, default=256)
    p.add_argument("--shuffle-buffer", type=int, default=4096)
    p.add_argument("--train-skip-tokens", type=int, default=None,
                   help="Override the saved training stream offset (0 explicitly restarts it)")
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--health-min-relative-ce-improvement", type=float, default=0.002)
    p.add_argument("--target-gap-recovery", type=float, default=0.90)
    p.add_argument("--plateau-patience", type=int, default=4, help="Held-out evaluation intervals")
    p.add_argument("--plateau-min-delta", type=float, default=0.002, help="Minimum absolute CE improvement")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_previous_report(resume_dir: Path | None) -> dict | None:
    if resume_dir is None:
        return None
    path = resume_dir / "distillation_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_resume_config(resume_dir: Path, config: CeNNConfig, base_model: str) -> None:
    metadata_path = resume_dir / "student_config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing resume metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("base_model") != base_model:
        raise ValueError(
            f"resume base model {metadata.get('base_model')!r} does not match {base_model!r}"
        )
    previous_config = CeNNConfig.from_dict(metadata["cenn"])
    if previous_config != config:
        raise ValueError(
            "resume CeNN configuration does not match requested configuration: "
            f"resume={previous_config.to_dict()} requested={config.to_dict()}"
        )


def eval_model(teacher, student, batches, device, dtype, args):
    return evaluate_distillation(
        teacher,
        student,
        batches,
        device=device,
        dtype=dtype,
        temperature=args.temperature,
        kl_chunk_rows=args.kl_chunk_rows,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        hidden_weight=args.hidden_weight,
    )


def save_best_snapshot(
    student,
    tokenizer,
    best_dir: Path,
    cenn_config: CeNNConfig,
    args,
    *,
    tokens_this_run: int,
    cumulative_tokens: int,
    resume_dir: Path | None,
    data_stream: dict,
) -> None:
    save_cenn_student(
        student,
        best_dir,
        config=cenn_config,
        base_model=args.base_model,
        extra_metadata={
            "distilled": True,
            "benchmark_protocol": "rigorous-v2",
            "tokens_this_run": tokens_this_run,
            "cumulative_tokens": cumulative_tokens,
            "resume_source": str(resume_dir) if resume_dir else None,
            "data_stream": data_stream,
            "train_interfaces": args.train_interfaces,
            "parameter_dtype": "float32",
        },
    )
    tokenizer.save_pretrained(best_dir)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.eval_batches < 1 or args.eval_batch_size < 1:
        raise ValueError("evaluation settings must be >= 1")
    if args.shuffle_buffer < 1:
        raise ValueError("shuffle-buffer must be >= 1")
    if min(args.batch_size, args.grad_accum, args.log_every, args.kl_chunk_rows) < 1:
        raise ValueError("batch-size, grad-accum, log-every and kl-chunk-rows must be positive")
    if args.context_length < 2 or args.grad_clip <= 0 or args.eval_every < 0:
        raise ValueError("invalid context length, grad clip, or evaluation interval")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    lr_multiplier(1, 10, 0, schedule=args.lr_schedule, decay_ratio=args.decay_ratio,
                  min_lr_ratio=args.min_lr_ratio)
    for initial, final in ((args.ce_weight, None), (args.kl_weight, args.kl_final_weight),
                           (args.hidden_weight, args.hidden_final_weight)):
        annealed_weight(initial, final, 0)
    if args.ce_weight <= 0:
        raise ValueError("ce-weight must be positive")
    plateau_summary([], args.plateau_patience, args.plateau_min_delta)
    if not 0.0 < args.target_gap_recovery <= 1.0:
        raise ValueError("target-gap-recovery must be in (0, 1]")

    # A fresh directory keeps reports and weights from different runs together.
    for directory in (Path(args.output_dir), Path(args.output_dir + "-best")):
        if any((directory / name).exists() for name in ("student_config.json", "distillation_report.json")):
            raise FileExistsError(f"output already contains a training run: {directory}; choose a new --output-dir")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {"attn_implementation": "sdpa"}
    if device.type == "cuda":
        load_kwargs["dtype"] = dtype

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device)
    teacher.config.use_cache = False
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="sdpa", dtype=torch.float32
    )
    dilations = tuple(int(x) for x in args.dilations.split(",") if x.strip())
    cenn_config = CeNNConfig(
        hidden_size=student.config.hidden_size,
        kernel_size=args.kernel_size,
        expansion=args.expansion,
        steps=args.steps,
        dilations=dilations,
        rms_norm_eps=float(getattr(student.config, "rms_norm_eps", 1e-5)),
    )
    if len(student.model.layers) != 1:
        raise ValueError("this trainer requires a one-layer teacher, such as arnir0/Tiny-LLM")
    replace_transformer_with_cenn(student, cenn_config, (0,))
    freeze_student_interfaces(student, args.train_interfaces)
    student.to(device)

    summary = student_parameter_summary(student)
    replacement = next(
        module for module in student.modules() if module.__class__.__name__ == "CeNNReplacementLayer"
    )
    print(
        f"student parameters: total={summary['total']:,} trainable={summary['trainable']:,} "
        f"({summary['trainable_percent']:.2f}%)"
    )
    print(
        f"Transformer removed: CeNN steps={args.steps}, "
        f"receptive_field={replacement.cenn.receptive_field}"
    )

    # Build the exact deterministic held-out benchmark once and reuse it for every
    # evaluation in this run. Its token fingerprint is stored in the report and
    # checked again after remote Hugging Face reload.
    eval_raw = load_dataset(args.dataset, args.dataset_config, split=args.split,
                            revision=args.dataset_revision, streaming=True)
    eval_rows = partition_rows(eval_raw, args.text_field, validation=True)
    eval_batches = collect_eval_batches(
        eval_rows,
        tokenizer,
        args.text_field,
        args.context_length,
        args.eval_batch_size,
        args.eval_batches,
    )
    eval_fingerprint = evaluation_fingerprint(eval_batches)
    eval_token_count = sum(batch.numel() for batch in eval_batches)
    print(
        f"held-out benchmark: batches={len(eval_batches)} tokens={eval_token_count:,} "
        f"fingerprint={eval_fingerprint[:16]}..."
    )

    # Cold CeNN evaluation establishes the teacher-gap denominator on the exact
    # same rigorous held-out set, even when this run resumes a previous checkpoint.
    cold_initial = eval_model(teacher, student, eval_batches, device, dtype, args)
    print(
        f"cold CeNN: student_ce={cold_initial['student_ce']:.4f} "
        f"ppl={cold_initial['student_ppl']:.2f} | "
        f"teacher_ce={cold_initial['teacher_ce']:.4f} "
        f"ppl={cold_initial['teacher_ppl']:.2f}"
    )

    resume_dir = Path(args.resume_student_dir).expanduser().resolve() if args.resume_student_dir else None
    previous_report = load_previous_report(resume_dir)
    resume_metadata = {}
    if resume_dir is not None:
        validate_resume_config(resume_dir, cenn_config, args.base_model)
        resume_metadata = json.loads((resume_dir / "student_config.json").read_text())
        load_cenn_student_weights(student, resume_dir, map_location="cpu", strict=True)
        student.to(device=device)
        print(f"resumed CeNN student from: {resume_dir}")
    previous_seen_tokens = checkpoint_training_tokens(resume_metadata, previous_report)
    stream_signature = training_stream_signature(args, tokenizer)
    stream_offset, stream_mode = stream_resume_offset(
        resume_metadata, previous_report, stream_signature, explicit_offset=args.train_skip_tokens,
    )
    def stream_state(tokens):
        return {"signature": stream_signature, "next_token_offset": stream_offset + tokens}

    print(f"training stream: {stream_mode}, skip {stream_offset:,} tokens before optimization")
    if stream_mode == "legacy_estimate":
        print("Legacy checkpoint has no exact cursor: prefix coverage is estimated, not guaranteed disjoint.")
    expected_fp = (previous_report or {}).get("evaluation", {}).get("fingerprint_sha256")
    if expected_fp and expected_fp != eval_fingerprint:
        raise ValueError("held-out fingerprint differs from resumed benchmark; use matching evaluation settings")

    run_start = eval_model(teacher, student, eval_batches, device, dtype, args)
    print(
        f"run start: student_ce={run_start['student_ce']:.4f} "
        f"ppl={run_start['student_ppl']:.2f} | teacher_ce={run_start['teacher_ce']:.4f} "
        f"KL={run_start['kl']:.4f} hidden={run_start['hidden']:.4f}"
    )

    output_dir = Path(args.output_dir)
    best_dir = Path(str(output_dir) + "-best")
    # The resumed checkpoint is a valid candidate and must be materialized as the
    # best-at-update-0 snapshot. This guarantees that report["best"] and the
    # published best weights remain identical even if continuation never improves.
    save_best_snapshot(
        student,
        tokenizer,
        best_dir,
        cenn_config,
        args,
        tokens_this_run=0,
        cumulative_tokens=previous_seen_tokens,
        resume_dir=resume_dir,
        data_stream=stream_state(0),
    )
    print("saved rigorous best-at-update-0 snapshot")

    train_raw = load_dataset(args.dataset, args.dataset_config, split=args.split,
                             revision=args.dataset_revision, streaming=True)
    train_rows = partition_rows(train_raw, args.text_field, validation=False)
    train_rows = buffered_shuffle(
        train_rows,
        buffer_size=args.shuffle_buffer,
        seed=args.seed,
    )
    batches = batch_blocks(
        token_blocks(train_rows, tokenizer, args.text_field, args.context_length, skip_tokens=stream_offset),
        args.batch_size,
    )

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    groups = optimizer_groups(student, args.learning_rate, args.interface_lr_scale, args.weight_decay)
    try:
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    tokens_per_update = args.batch_size * args.context_length * args.grad_accum
    total_updates = max(1, math.ceil(args.max_tokens / tokens_per_update))
    warmup_updates = min(total_updates - 1, int(total_updates * args.warmup_ratio))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and dtype == torch.float16),
    )
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext

    train_student = student
    if not args.no_compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            train_student = torch.compile(student, mode="reduce-overhead", dynamic=False)
            print("torch.compile enabled for student")
        except Exception as exc:
            print(f"torch.compile unavailable, eager mode: {exc}")

    history = [{"update": 0, "tokens_this_run": 0, **run_start}]
    best = run_start
    best_update = 0
    best_tokens = 0
    seen_tokens = 0
    update = 0
    micro = 0
    running = {"ce": 0.0, "kl": 0.0, "hidden": 0.0, "total": 0.0}
    running_n = 0
    window_tokens = 0
    start = time.perf_counter()
    window_start = start
    optimizer.zero_grad(set_to_none=True)
    diverged = False
    exhausted = False
    train_history = []
    committed_tokens = 0

    student.train()
    while update < total_updates:
        try:
            ids = next(batches).to(device, non_blocking=True)
        except StopIteration:
            exhausted = True
            # An incomplete gradient accumulation is not an optimizer update.
            # Leave its tokens available for continuation from this checkpoint.
            seen_tokens = committed_tokens
            optimizer.zero_grad(set_to_none=True)
            print("Training stream exhausted before the requested token budget; saving completed updates.")
            break
        progress = update / max(total_updates - 1, 1)
        train_kl_weight = annealed_weight(args.kl_weight, args.kl_final_weight, progress)
        train_hidden_weight = annealed_weight(args.hidden_weight, args.hidden_final_weight, progress)
        # Teacher is frozen. no_grad keeps tensors usable as fixed KL/hidden targets
        # while avoiding the stricter inference-tensor semantics of inference_mode.
        with torch.no_grad():
            with amp():
                teacher_out = teacher(
                    input_ids=ids,
                    labels=ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
        with amp():
            student_out = train_student(
                input_ids=ids,
                labels=ids,
                output_hidden_states=True,
                use_cache=False,
            )
            loss, parts = combined_loss(
                student_out,
                teacher_out,
                temperature=args.temperature,
                kl_chunk_rows=args.kl_chunk_rows,
                ce_weight=args.ce_weight,
                kl_weight=train_kl_weight,
                hidden_weight=train_hidden_weight,
            )
            scaled_loss = loss / args.grad_accum

        if not torch.isfinite(scaled_loss):
            diverged = True
            print("ERROR: non-finite distillation loss")
            break

        scaler.scale(scaled_loss).backward()
        micro += 1
        batch_tokens = ids.numel()
        seen_tokens += batch_tokens
        window_tokens += batch_tokens
        for key in running:
            running[key] += parts[key]
        running_n += 1
        if micro % args.grad_accum:
            continue

        next_update = update + 1
        lr_mult = lr_multiplier(next_update, total_updates, warmup_updates,
                               schedule=args.lr_schedule, decay_ratio=args.decay_ratio,
                               min_lr_ratio=args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_mult
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
        if not math.isfinite(grad_norm):
            diverged = True
            print("ERROR: non-finite gradient norm")
            break
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        update = next_update
        committed_tokens = seen_tokens

        if update == 1 or update == total_updates or update % args.log_every == 0:
            elapsed = max(time.perf_counter() - window_start, 1e-6)
            tok_s = window_tokens / elapsed
            avg = {key: value / max(running_n, 1) for key, value in running.items()}
            train_history.append({"update": update, "tokens_this_run": seen_tokens, **avg,
                                  "grad_norm": grad_norm, "lr_multiplier": lr_mult,
                                  "kl_weight": train_kl_weight, "hidden_weight": train_hidden_weight})
            print(
                f"update={update}/{total_updates} run_tokens={seen_tokens:,} "
                f"ce={avg['ce']:.4f} kl={avg['kl']:.4f} hidden={avg['hidden']:.4f} "
                f"total={avg['total']:.4f} grad={grad_norm:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} tok/s={tok_s:,.0f}"
            )
            running = {"ce": 0.0, "kl": 0.0, "hidden": 0.0, "total": 0.0}
            running_n = 0
            window_tokens = 0
            window_start = time.perf_counter()

        if update == total_updates or (args.eval_every > 0 and update % args.eval_every == 0):
            metrics = eval_model(teacher, student, eval_batches, device, dtype, args)
            history.append({"update": update, "tokens_this_run": seen_tokens,
                            "lr_multiplier": lr_mult, **metrics})
            plateau = plateau_summary(history, args.plateau_patience, args.plateau_min_delta)
            if plateau["detected"]:
                print("PLATEAU: held-out CE has made little recent progress; see the training/validation traces.")
            print(
                f"eval={update}: student_ce={metrics['student_ce']:.4f} "
                f"ppl={metrics['student_ppl']:.2f} teacher_ce={metrics['teacher_ce']:.4f} "
                f"KL={metrics['kl']:.4f} hidden={metrics['hidden']:.4f}"
            )
            if not all(
                math.isfinite(float(metrics[key])) for key in ("student_ce", "kl", "hidden")
            ):
                diverged = True
                break
            if float(metrics["student_ce"]) < float(best["student_ce"]):
                best = metrics
                best_update = update
                best_tokens = seen_tokens
                save_best_snapshot(
                    student,
                    tokenizer,
                    best_dir,
                    cenn_config,
                    args,
                    tokens_this_run=seen_tokens,
                    cumulative_tokens=previous_seen_tokens + seen_tokens,
                    resume_dir=resume_dir,
                    data_stream=stream_state(seen_tokens),
                )
                print(f"new best CeNN student saved at update {update}")
            student.train()

    final = eval_model(teacher, student, eval_batches, device, dtype, args) if not diverged else None
    if final is not None and (not history or history[-1]["update"] != update):
        history.append({"update": update, "tokens_this_run": seen_tokens, **final})
        if float(final["student_ce"]) < float(best["student_ce"]):
            best, best_update, best_tokens = final, update, seen_tokens
            save_best_snapshot(student, tokenizer, best_dir, cenn_config, args,
                               tokens_this_run=seen_tokens,
                               cumulative_tokens=previous_seen_tokens + seen_tokens,
                               resume_dir=resume_dir, data_stream=stream_state(seen_tokens))
    cumulative_tokens = previous_seen_tokens + seen_tokens
    if not diverged:
        save_best_snapshot(student, tokenizer, output_dir, cenn_config, args,
                           tokens_this_run=seen_tokens, cumulative_tokens=cumulative_tokens,
                           resume_dir=resume_dir, data_stream=stream_state(seen_tokens))

    teacher_ce = float(cold_initial["teacher_ce"])
    cold_gap = float(cold_initial["student_ce"]) - teacher_ce
    best_gap = float(best["student_ce"]) - float(best["teacher_ce"])
    recovery = (cold_gap - best_gap) / max(cold_gap, 1e-12) if cold_gap > 0 else 0.0
    run_relative_ce_improvement = (
        (float(run_start["student_ce"]) - float(best["student_ce"]))
        / max(float(run_start["student_ce"]), 1e-12)
    )

    if diverged:
        status = "diverged"
    elif exhausted:
        status = "data_exhausted"
    elif recovery >= args.target_gap_recovery:
        status = "target_reached"
    elif run_relative_ce_improvement >= args.health_min_relative_ce_improvement:
        status = "healthy_progress"
    else:
        status = "warning_no_improvement"

    report = {
        "status": status,
        "benchmark_protocol": "rigorous-v2",
        "architecture": "cenn-only-replacement",
        "transformer_layers_remaining": 0,
        "base_model_teacher": args.base_model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.split,
        "text_field": args.text_field,
        "validation_split": "deterministic text hash: buckets 990-999 / 1000",
        "training_shuffle": {
            "method": "deterministic bounded-memory buffered shuffle",
            "buffer_size": args.shuffle_buffer,
            "seed": args.seed,
        },
        "evaluation": {
            "batches": len(eval_batches),
            "batch_size": args.eval_batch_size,
            "tokens": eval_token_count,
            "fingerprint_sha256": eval_fingerprint,
            "eval_every_updates": args.eval_every,
        },
        "context_length": args.context_length,
        "requested_additional_tokens": args.max_tokens,
        "seen_tokens_this_run": seen_tokens,
        "previous_training_tokens": previous_seen_tokens,
        "cumulative_training_tokens": cumulative_tokens,
        "updates_completed_this_run": update,
        "resume_student_dir": str(resume_dir) if resume_dir else None,
        "cenn_steps": args.steps,
        "cenn_dilations": list(dilations),
        "cenn_receptive_field": replacement.cenn.receptive_field,
        "parameters": summary,
        "training_precision": {"parameters": "float32", "autocast": str(dtype)},
        "train_interfaces": args.train_interfaces,
        "optimizer_continuation": "new AdamW state and schedule; checkpoint weights and data cursor retained",
        "training_stream": {"resume_mode": stream_mode, "start_token_offset": stream_offset,
                            **stream_state(seen_tokens)},
        "plateau": {**plateau_summary(history, args.plateau_patience, args.plateau_min_delta),
                    "patience": args.plateau_patience, "min_delta": args.plateau_min_delta},
        "distillation": {
            "temperature": args.temperature,
            "ce_weight": args.ce_weight,
            "kl_weight": args.kl_weight,
            "hidden_weight": args.hidden_weight,
            "kl_chunk_rows": args.kl_chunk_rows,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lr_schedule": args.lr_schedule,
            "decay_ratio": args.decay_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "interface_lr_scale": args.interface_lr_scale,
            "kl_final_weight": args.kl_final_weight,
            "hidden_final_weight": args.hidden_final_weight,
            "evaluation_weights": "fixed initial weights; training weights may anneal",
        },
        "cold_initial": cold_initial,
        "run_start": run_start,
        "best": best,
        "best_update_this_run": best_update,
        "final": final,
        "run_relative_student_ce_improvement": run_relative_ce_improvement,
        "teacher_gap_recovery_fraction": recovery,
        "target_gap_recovery_fraction": args.target_gap_recovery,
        "target_gap_recovery_reached": recovery >= args.target_gap_recovery,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        ),
        "eval_history": history,
        "train_history": train_history,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for directory, kind, checkpoint_tokens, metrics in (
        (output_dir, "final", seen_tokens, final), (best_dir, "best", best_tokens, best),
    ):
        report["checkpoint"] = {
            "kind": kind, "available": metrics is not None,
            "tokens_this_run": checkpoint_tokens,
            "cumulative_training_tokens": previous_seen_tokens + checkpoint_tokens,
            "metrics": metrics, "data_stream": stream_state(checkpoint_tokens),
        }
        (directory / "distillation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print(
        f"DISTILLATION: {status.upper()} | run CE {float(run_start['student_ce']):.4f} -> "
        f"{float(best['student_ce']):.4f} | teacher={float(best['teacher_ce']):.4f} | "
        f"global teacher-gap recovery={recovery * 100:.2f}% | "
        f"cumulative tokens={cumulative_tokens:,}"
    )
    print(f"best checkpoint: {best_dir}")
    print("=" * 88)
    if diverged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
