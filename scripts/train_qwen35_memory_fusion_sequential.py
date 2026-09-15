#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from scripts import train_smollm2_memory_fusion_sequential as seq
from tinycenn_lm.qwen3_5_memory_fusion import (
    DEFAULT_QWEN35,
    Qwen35MemoryFusionConfig,
    freeze_current_layer_only,
    full_attention_layers,
    load_selected_attention_state,
    replace_attention_layers,
    selected_attention_state,
    structural_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sequential Qwen3.5 Memory Fusion conversion. V1 replaces only the "
            "model's original full-attention anchors while preserving Qwen3.5's "
            "native Gated DeltaNet linear-attention layers."
        )
    )
    p.add_argument("--base-model", default=DEFAULT_QWEN35)
    p.add_argument("--model-revision", default="main")
    p.add_argument("--output-dir", default="checkpoints/qwen35-0.8b-memory-fusion-sequential-r64")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--memory-rank", type=int, choices=(32, 48, 64), default=64)
    p.add_argument("--target-layers", default="auto", help="auto or comma-separated Qwen3.5 full-attention layer indices")
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--shuffle-buffer", type=int, default=2048)

    p.add_argument("--min-layer-steps", type=int, default=50)
    p.add_argument("--max-layer-steps", type=int, default=300)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--layer-lr", type=float, default=2e-4)
    p.add_argument("--teacher-alpha-start", type=float, default=0.90)
    p.add_argument("--teacher-alpha-end", type=float, default=0.00)
    p.add_argument("--layer-kl-weight", type=float, default=0.08)
    p.add_argument("--layer-ce-weight", type=float, default=0.08)
    p.add_argument("--cosine-weight", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--logit-chunk", type=int, default=4)
    p.add_argument("--accept-nmse", type=float, default=0.20)
    p.add_argument("--accept-cosine", type=float, default=0.90)
    p.add_argument("--accept-incremental-delta-nll", type=float, default=0.015)
    p.add_argument("--accept-cumulative-delta-nll", type=float, default=0.05)
    p.add_argument("--probe-blocks", type=int, default=4)
    p.add_argument("--probe-context", type=int, default=128)
    p.add_argument("--strict-acceptance", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-runtime-minutes", type=float, default=240.0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    if args.context_length < 16 or args.probe_context < 16:
        p.error("context lengths must be >= 16")
    if args.logit_chunk < 1:
        p.error("logit-chunk must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_factory(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=dtype)
    return nullcontext


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def target_layers_for(model, requested: str) -> list[int]:
    available = full_attention_layers(model)
    if requested.strip().lower() == "auto":
        return available
    chosen = list(dict.fromkeys(int(x.strip()) for x in requested.split(",") if x.strip()))
    if not chosen or not set(chosen).issubset(set(available)):
        raise ValueError(f"target layers {chosen} must be a nonempty subset of full-attention layers {available}")
    return chosen


def chunked_distillation_kl(student_logits, teacher_logits, temperature: float, chunk: int) -> torch.Tensor:
    """Chunk token positions to bound Qwen3.5's 248k-vocabulary KL temporaries."""
    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    total = s.new_zeros((), dtype=torch.float32)
    n = s.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ss = s[start:end].float() / temperature
        with torch.no_grad():
            tt = t[start:end].float() / temperature
            teacher_prob = F.softmax(tt, dim=-1)
        total = total + F.kl_div(
            F.log_softmax(ss, dim=-1), teacher_prob, reduction="sum"
        ) * (temperature ** 2 / n)
    return total


def acceptance_passes(*, nmse, cosine, incremental, cumulative, args) -> bool:
    return (
        nmse <= args.accept_nmse
        and cosine >= args.accept_cosine
        and incremental <= args.accept_incremental_delta_nll
        and cumulative <= args.accept_cumulative_delta_nll
    )


def train_one_replacement(*, teacher, student, layer_idx, batch_iter, probe_blocks,
                          teacher_probe_nll, pre_probe_nll, args, device, amp) -> dict:
    trainable = freeze_current_layer_only(student, layer_idx, train_output_projection=True)
    optimizer = seq.make_optimizer(trainable, args.layer_lr, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and choose_dtype(device) == torch.float16)
    )
    student.train()
    best = None

    for step in range(1, args.max_layer_steps + 1):
        ids = next(batch_iter).to(device, non_blocking=True)
        t_capture, t_out = seq.capture_attention_input(teacher, ids, layer_idx, amp, with_output=True)
        s_capture, s_out = seq.capture_attention_input(student, ids, layer_idx, amp, with_output=True)
        alpha = seq.alpha_for_step(
            step, args.max_layer_steps, args.teacher_alpha_start, args.teacher_alpha_end
        )
        mixed_hidden = (
            alpha * t_capture["hidden"].detach()
            + (1.0 - alpha) * s_capture["hidden"].detach()
        )
        kwargs = seq.attention_kwargs_from_capture(t_capture)
        with torch.no_grad(), amp():
            target = seq.call_attention(teacher.model.layers[layer_idx].self_attn, mixed_hidden, kwargs)
        with amp():
            pred = seq.call_attention(student.model.layers[layer_idx].self_attn, mixed_hidden, kwargs)
            functional = seq.alignment_loss(pred, target, args.cosine_weight)
            kl = chunked_distillation_kl(s_out.logits, t_out.logits, args.temperature, args.logit_chunk)
            total = functional + args.layer_kl_weight * kl + args.layer_ce_weight * s_out.loss

        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at layer {layer_idx}, step {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            nmse_batch, cosine_batch = seq.alignment_metrics(pred.detach(), target.detach())
            print(
                f"layer={layer_idx:02d} step={step:03d}/{args.max_layer_steps} alpha={alpha:.3f} "
                f"functional={float(functional.detach()):.4f} nmse={float(nmse_batch):.4f} "
                f"cos={float(cosine_batch):.4f} kl={float(kl.detach()):.4f} "
                f"ce={float(s_out.loss.detach()):.4f} grad={grad_norm:.3f}", flush=True
            )

        should_check = step >= args.min_layer_steps and (
            step % args.check_every == 0 or step == args.max_layer_steps
        )
        if should_check:
            nmse, cosine = seq.real_hidden_function_metrics(teacher, student, ids, layer_idx, amp)
            current_probe_nll = seq.probe_nll(student, probe_blocks, device, amp)
            student.train()
            incremental = current_probe_nll - pre_probe_nll
            cumulative = current_probe_nll - teacher_probe_nll
            passed = acceptance_passes(
                nmse=nmse, cosine=cosine, incremental=incremental,
                cumulative=cumulative, args=args,
            )
            candidate = {
                "step": step,
                "nmse": nmse,
                "cosine": cosine,
                "probe_nll": current_probe_nll,
                "incremental_delta_nll": incremental,
                "cumulative_delta_nll": cumulative,
                "passed": passed,
            }
            if best is None or (
                candidate["incremental_delta_nll"] < best["incremental_delta_nll"]
                and candidate["nmse"] <= args.accept_nmse * 1.25
            ):
                best = candidate
            print(
                f"  ACCEPTANCE CHECK layer={layer_idx:02d}: NMSE={nmse:.4f} (≤{args.accept_nmse:.4f}) "
                f"cos={cosine:.4f} (≥{args.accept_cosine:.4f}) "
                f"ΔNLL_inc={incremental:+.5f} (≤{args.accept_incremental_delta_nll:+.5f}) "
                f"ΔNLL_total={cumulative:+.5f} (≤{args.accept_cumulative_delta_nll:+.5f}) "
                f"=> {'PASS' if passed else 'continue'}",
                flush=True,
            )
            if passed:
                best = candidate
                break
        del t_out, s_out, pred, target, total

    if best is None:
        raise RuntimeError("acceptance was never evaluated")
    return {
        "layer": layer_idx,
        "accepted": bool(best["passed"]),
        "steps": int(best["step"]),
        "nmse": float(best["nmse"]),
        "cosine": float(best["cosine"]),
        "probe_nll": float(best["probe_nll"]),
        "incremental_delta_nll": float(best["incremental_delta_nll"]),
        "cumulative_delta_nll": float(best["cumulative_delta_nll"]),
    }


def save_progress(output_dir: Path, student, config, target_layers, accepted_layers, layer_reports, *, stage: str) -> None:
    payload = {
        "format_version": 1,
        "stage": stage,
        "target_layers": list(target_layers),
        "accepted_layers": list(accepted_layers),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "attention_state": selected_attention_state(student, accepted_layers),
    }
    atomic_torch(output_dir / "sequential_progress.pt", payload)
    atomic_json(output_dir / "sequential_progress.json", {k: v for k, v in payload.items() if k != "attention_state"})


def load_progress(output_dir: Path, student, config, target_layers, *, resume: bool):
    path = output_dir / "sequential_progress.pt"
    if not resume or not path.exists():
        return [], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    accepted = [int(x) for x in payload.get("accepted_layers", [])]
    if payload.get("target_layers") != list(target_layers):
        raise RuntimeError("resume target layers differ from this run")
    if int(payload.get("config", {}).get("memory_rank", -1)) != config.memory_rank:
        raise RuntimeError("resume memory rank differs from this run")
    if accepted != list(target_layers[:len(accepted)]):
        raise RuntimeError(f"accepted layers are not a target-layer prefix: {accepted}")
    if accepted:
        replace_attention_layers(student, config, accepted)
        load_selected_attention_state(student, payload["attention_state"], accepted)
    print(f"RESUME: accepted full-attention layers={accepted}", flush=True)
    return accepted, list(payload.get("layer_reports", []))


def save_in_progress(output_dir: Path, student, config, target_layers, accepted_layers, current_layer,
                     pre_probe_nll, rounds_completed, layer_reports, status):
    layers = list(accepted_layers) + [int(current_layer)]
    payload = {
        "format_version": 1,
        "status": status,
        "target_layers": list(target_layers),
        "accepted_layers": list(accepted_layers),
        "current_layer": int(current_layer),
        "pre_probe_nll": float(pre_probe_nll),
        "rounds_completed": int(rounds_completed),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "attention_state": selected_attention_state(student, layers),
    }
    atomic_torch(output_dir / "sequential_in_progress.pt", payload)
    atomic_json(output_dir / "sequential_in_progress.json", {k: v for k, v in payload.items() if k != "attention_state"})


def load_in_progress(output_dir: Path, student, config, target_layers, accepted_layers, *, resume: bool):
    path = output_dir / "sequential_in_progress.pt"
    if not resume or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("target_layers") != list(target_layers):
        return None
    if [int(x) for x in payload.get("accepted_layers", [])] != list(accepted_layers):
        return None
    expected_current = target_layers[len(accepted_layers)] if len(accepted_layers) < len(target_layers) else None
    if expected_current is None or int(payload.get("current_layer", -1)) != int(expected_current):
        return None
    replace_attention_layers(student, config, [expected_current])
    load_selected_attention_state(
        student, payload["attention_state"], list(accepted_layers) + [expected_current]
    )
    print(
        f"RESUME CURRENT LAYER: layer={expected_current}, "
        f"rounds_completed={payload.get('rounds_completed', 0)}",
        flush=True,
    )
    return payload


def remove_in_progress(output_dir: Path) -> None:
    for name in ("sequential_in_progress.pt", "sequential_in_progress.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rounds = max(1, int(os.environ.get("SEQUENTIAL_MAX_ROUNDS_PER_RUN", "4")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    amp = amp_factory(device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} dtype={dtype} memory_rank={args.memory_rank}", flush=True)
    print(f"output_dir={output_dir} max_rounds_this_run={max_rounds}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Text-only loading intentionally skips the checkpoint's vision tower.
    teacher = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval().requires_grad_(False)
    student = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    teacher.config.use_cache = False
    student.config.use_cache = False

    text_config = student.config.get_text_config(decoder=True) if hasattr(student.config, "get_text_config") else student.config
    if text_config.model_type != "qwen3_5_text":
        raise ValueError(f"expected qwen3_5_text, got {text_config.model_type}")
    target_layers = target_layers_for(student, args.target_layers)
    print(
        f"Qwen3.5 layers={text_config.num_hidden_layers}; "
        f"native linear={sum(x == 'linear_attention' for x in text_config.layer_types)}; "
        f"full-attention targets={target_layers}",
        flush=True,
    )

    config = Qwen35MemoryFusionConfig(feature_dim=args.feature_dim, memory_rank=args.memory_rank)
    accepted_layers, layer_reports = load_progress(
        output_dir, student, config, target_layers, resume=args.resume
    )
    in_progress = load_in_progress(
        output_dir, student, config, target_layers, accepted_layers, resume=args.resume
    )

    raw = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=True).shuffle(
        seed=args.seed, buffer_size=args.shuffle_buffer
    )
    batch_iter = iter(seq.batches(
        seq.token_blocks(raw, tokenizer, args.text_field, args.context_length), args.batch_size
    ))
    probe_blocks = seq.build_probe_blocks(tokenizer, context=args.probe_context, count=args.probe_blocks)
    teacher_probe_nll = seq.probe_nll(teacher, probe_blocks, device, amp)
    print(f"teacher probe NLL={teacher_probe_nll:.6f}", flush=True)

    start = time.perf_counter()
    for layer_idx in target_layers:
        if layer_idx in accepted_layers:
            continue
        if accepted_layers != target_layers[:target_layers.index(layer_idx)]:
            raise RuntimeError(f"accepted layers must be a target prefix before {layer_idx}: {accepted_layers}")
        if (time.perf_counter() - start) / 60 >= args.max_runtime_minutes * 0.70:
            atomic_json(output_dir / "sequential_run_status.json", {
                "status": "paused_runtime_budget",
                "accepted_layers": accepted_layers,
                "next_layer": layer_idx,
                "message": "Rerun with --resume to continue.",
            })
            return 0

        print("\n" + "=" * 100, flush=True)
        print(f"QWEN3.5 FULL-ATTENTION REPLACEMENT: layer {layer_idx}", flush=True)
        print("=" * 100, flush=True)
        if in_progress is not None and int(in_progress["current_layer"]) == layer_idx:
            pre_probe_nll = float(in_progress["pre_probe_nll"])
            rounds_completed = int(in_progress.get("rounds_completed", 0))
            print(f"continuing saved layer {layer_idx}; pre-replacement NLL={pre_probe_nll:.6f}", flush=True)
        else:
            pre_probe_nll = seq.probe_nll(student, probe_blocks, device, amp)
            print(
                f"before replacement NLL={pre_probe_nll:.6f}; "
                f"Δteacher={pre_probe_nll-teacher_probe_nll:+.6f}",
                flush=True,
            )
            replace_attention_layers(student, config, [layer_idx])
            rounds_completed = 0

        accepted = False
        last_report = None
        for local_round in range(1, max_rounds + 1):
            global_round = rounds_completed + local_round
            round_args = copy.copy(args)
            if global_round > 1:
                round_args.teacher_alpha_start = min(0.25, args.teacher_alpha_start)
                round_args.teacher_alpha_end = 0.0
            print(f"\n--- layer {layer_idx} round {global_round} ---", flush=True)
            report = train_one_replacement(
                teacher=teacher,
                student=student,
                layer_idx=layer_idx,
                batch_iter=batch_iter,
                probe_blocks=probe_blocks,
                teacher_probe_nll=teacher_probe_nll,
                pre_probe_nll=pre_probe_nll,
                args=round_args,
                device=device,
                amp=amp,
            )
            report["round"] = global_round
            layer_reports.append(report)
            last_report = report
            if report["accepted"]:
                accepted = True
                break
            save_in_progress(
                output_dir, student, config, target_layers, accepted_layers, layer_idx,
                pre_probe_nll, global_round, layer_reports, "current_layer_needs_more_training"
            )
            print(f"Layer {layer_idx} not accepted yet; weights saved for resume.", flush=True)

        if not accepted:
            rounds_completed += max_rounds
            save_in_progress(
                output_dir, student, config, target_layers, accepted_layers, layer_idx,
                pre_probe_nll, rounds_completed, layer_reports, "current_layer_needs_more_training"
            )
            status = {
                "status": "current_layer_needs_more_training",
                "target_layers": target_layers,
                "accepted_layers": accepted_layers,
                "current_layer": layer_idx,
                "rounds_completed": rounds_completed,
                "last_report": last_report,
                "message": "No next full-attention layer was replaced. Rerun to continue this same layer.",
            }
            atomic_json(output_dir / "sequential_run_status.json", status)
            print("\nNOT A CRASH:", json.dumps(status, indent=2), flush=True)
            return 0

        accepted_layers.append(layer_idx)
        save_progress(
            output_dir,
            student,
            config,
            target_layers,
            accepted_layers,
            layer_reports,
            stage=f"accepted_layer_{layer_idx}",
        )
        remove_in_progress(output_dir)
        in_progress = None
        print(f"✅ accepted Qwen3.5 full-attention layer {layer_idx}", flush=True)

    final_nll = seq.probe_nll(student, probe_blocks, device, amp)
    report = {
        "status": "all_target_full_attention_layers_accepted",
        "architecture": "qwen3.5-memory-fusion-sequential-v1",
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "target_layers": target_layers,
        "accepted_layers": accepted_layers,
        "feature_dim": args.feature_dim,
        "memory_rank": args.memory_rank,
        "teacher_probe_nll": teacher_probe_nll,
        "final_probe_nll": final_nll,
        "final_probe_delta_nll": final_nll - teacher_probe_nll,
        "structure": structural_summary(student),
        "layer_reports": layer_reports,
        "elapsed_minutes": (time.perf_counter() - start) / 60,
    }
    atomic_torch(output_dir / "qwen35_memory_fusion_full_state.pt", {
        "format": "qwen3.5-memory-fusion-sequential-v1",
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "target_layers": target_layers,
        "config": config.to_dict(),
        "attention_state": selected_attention_state(student, accepted_layers),
        "report": report,
    })
    atomic_json(output_dir / "sequential_training_report.json", report)
    atomic_json(output_dir / "sequential_run_status.json", {
        "status": "complete",
        "accepted_layers": accepted_layers,
        "target_layers": target_layers,
    })
    tokenizer.save_pretrained(output_dir)
    print("\nFINAL REPORT\n" + json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted. Saved accepted/current-layer checkpoints remain resumable.",
            file=sys.stderr,
            flush=True,
        )
        raise
