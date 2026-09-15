#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the tested training/math helpers from the original sequential trainer.
import train_smollm2_memory_fusion_sequential as seq
from tinycenn_lm.smollm2_memory_fusion import (
    SmolMemoryFusionConfig,
    replace_attention_layers,
    structural_summary,
)


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _atomic_json_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _selected_attention_state(student, layers: list[int]) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"model.layers.{i}.self_attn." for i in layers)
    return {
        k: v.detach().cpu()
        for k, v in student.state_dict().items()
        if prefixes and k.startswith(prefixes)
    }


def save_in_progress(
    output_dir: Path,
    student,
    config: SmolMemoryFusionConfig,
    *,
    accepted_layers: list[int],
    current_layer: int,
    pre_probe_nll: float,
    rounds_completed: int,
    layer_reports: list[dict],
    status: str,
) -> None:
    layers = accepted_layers + [current_layer]
    payload = {
        "format_version": 2,
        "status": status,
        "accepted_layers": list(accepted_layers),
        "current_layer": int(current_layer),
        "pre_probe_nll": float(pre_probe_nll),
        "rounds_completed": int(rounds_completed),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "attention_state": _selected_attention_state(student, layers),
    }
    _atomic_torch_save(payload, output_dir / "sequential_in_progress.pt")
    _atomic_json_save(
        {k: v for k, v in payload.items() if k != "attention_state"},
        output_dir / "sequential_in_progress.json",
    )


def load_in_progress(
    output_dir: Path,
    student,
    config: SmolMemoryFusionConfig,
    accepted_layers: list[int],
    *,
    resume: bool,
):
    path = output_dir / "sequential_in_progress.pt"
    if not resume or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("config", {}).get("memory_rank", -1)) != config.memory_rank:
        print("Ignoring in-progress checkpoint: memory rank changed.")
        return None
    saved_accepted = [int(x) for x in payload.get("accepted_layers", [])]
    if saved_accepted != accepted_layers:
        print("Ignoring stale in-progress checkpoint: accepted prefix changed.")
        return None
    current = int(payload["current_layer"])
    if current != len(accepted_layers):
        print("Ignoring stale in-progress checkpoint: current layer is not next prefix layer.")
        return None
    replace_attention_layers(student, config, [current])
    incompatible = student.load_state_dict(payload["attention_state"], strict=False)
    expected_prefixes = tuple(
        f"model.layers.{i}.self_attn." for i in accepted_layers + [current]
    )
    missing = [
        k for k in incompatible.missing_keys
        if expected_prefixes and k.startswith(expected_prefixes)
    ]
    if missing:
        raise RuntimeError(f"in-progress checkpoint missing attention keys: {missing[:8]}")
    print(
        f"RESUME CURRENT LAYER: layer={current}, "
        f"rounds_completed={payload.get('rounds_completed', 0)}"
    )
    return payload


def remove_in_progress(output_dir: Path) -> None:
    for name in ("sequential_in_progress.pt", "sequential_in_progress.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_status(output_dir: Path, payload: dict) -> None:
    _atomic_json_save(payload, output_dir / "sequential_run_status.json")


def main() -> int:
    args = seq.parse_args()
    seq.set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Number of 300-step rounds to attempt in one Colab session before returning
    # gracefully. Reopening the notebook continues from exactly those weights.
    max_rounds_this_run = max(1, int(os.environ.get("SEQUENTIAL_MAX_ROUNDS_PER_RUN", "4")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = seq.choose_dtype(device)
    amp = seq.amp_factory(device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"device={device} dtype={dtype} memory_rank={args.memory_rank}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"max_rounds_this_run={max_rounds_this_run}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    teacher.eval()
    teacher.config.use_cache = False
    teacher.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    student.config.use_cache = False

    config = SmolMemoryFusionConfig(
        feature_dim=args.feature_dim,
        memory_rank=args.memory_rank,
        train_output_projection=True,
    )

    accepted_layers, layer_reports = seq.load_progress_if_available(
        output_dir, student, config, resume=args.resume
    )
    in_progress = load_in_progress(
        output_dir, student, config, accepted_layers, resume=args.resume
    )

    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    batch_iter = iter(
        seq.batches(
            seq.token_blocks(raw, tokenizer, args.text_field, args.context_length),
            args.batch_size,
        )
    )

    probe_blocks = seq.build_probe_blocks(
        tokenizer, context=args.probe_context, count=args.probe_blocks
    )
    teacher_probe_nll = seq.probe_nll(teacher, probe_blocks, device, amp)
    print(f"teacher probe NLL={teacher_probe_nll:.6f}", flush=True)

    start_time = time.perf_counter()
    num_layers = int(student.config.num_hidden_layers)

    for layer_idx in range(num_layers):
        if layer_idx in accepted_layers:
            continue
        if accepted_layers != list(range(layer_idx)):
            raise RuntimeError(
                f"accepted layers must be a contiguous prefix before {layer_idx}: {accepted_layers}"
            )

        # Graceful pause, not a subprocess failure.
        elapsed_min = (time.perf_counter() - start_time) / 60.0
        if elapsed_min >= args.max_runtime_minutes * 0.70:
            write_status(output_dir, {
                "status": "paused_runtime_budget",
                "accepted_layers": accepted_layers,
                "next_layer": layer_idx,
                "message": "Rerun the notebook with RESUME=True.",
            })
            print("PAUSED: runtime budget reached; persistent progress is safe.", flush=True)
            return 0

        print("\n" + "=" * 110, flush=True)
        print(f"SEQUENTIAL REPLACEMENT: layer {layer_idx}/{num_layers - 1}", flush=True)
        print("=" * 110, flush=True)

        if in_progress is not None and int(in_progress["current_layer"]) == layer_idx:
            pre_probe_nll = float(in_progress["pre_probe_nll"])
            rounds_completed = int(in_progress.get("rounds_completed", 0))
            print(
                f"continuing layer {layer_idx} from saved weights; "
                f"original pre-replacement NLL={pre_probe_nll:.6f}",
                flush=True,
            )
        else:
            pre_probe_nll = seq.probe_nll(student, probe_blocks, device, amp)
            print(
                f"before replacement: probe NLL={pre_probe_nll:.6f}, "
                f"Δ vs teacher={pre_probe_nll - teacher_probe_nll:+.6f}",
                flush=True,
            )
            replace_attention_layers(student, config, [layer_idx])
            rounds_completed = 0

        accepted = False
        best_report = None
        for local_round in range(1, max_rounds_this_run + 1):
            global_round = rounds_completed + local_round
            round_args = copy.copy(args)
            # First round performs full teacher→student curriculum. Later rounds
            # focus increasingly on the real student distribution instead of
            # repeatedly jumping back to 90% teacher hidden states.
            if global_round > 1:
                round_args.teacher_alpha_start = min(0.25, args.teacher_alpha_start)
                round_args.teacher_alpha_end = 0.0

            print(
                f"\n--- layer {layer_idx} training round {global_round} "
                f"({round_args.max_layer_steps} max steps) ---",
                flush=True,
            )
            report = seq.train_one_replacement(
                teacher=teacher,
                student=student,
                layer_idx=layer_idx,
                batch_iter=batch_iter,
                probe_blocks=probe_blocks,
                teacher_probe_nll=teacher_probe_nll,
                pre_replacement_probe_nll=pre_probe_nll,
                args=round_args,
                device=device,
                amp=amp,
            )
            best_report = report
            layer_reports.append({**report, "round": global_round})

            if report["accepted"]:
                accepted = True
                break

            save_in_progress(
                output_dir,
                student,
                config,
                accepted_layers=accepted_layers,
                current_layer=layer_idx,
                pre_probe_nll=pre_probe_nll,
                rounds_completed=global_round,
                layer_reports=layer_reports,
                status="current_layer_needs_more_training",
            )
            print(
                f"Layer {layer_idx} not accepted yet; current trained weights saved. "
                "Continuing instead of discarding them.",
                flush=True,
            )

            elapsed_min = (time.perf_counter() - start_time) / 60.0
            if elapsed_min >= args.max_runtime_minutes * 0.70:
                write_status(output_dir, {
                    "status": "paused_runtime_budget",
                    "accepted_layers": accepted_layers,
                    "current_layer": layer_idx,
                    "rounds_completed": global_round,
                    "last_report": report,
                    "message": "Rerun the notebook; current layer weights will resume.",
                })
                print("PAUSED: runtime budget reached during current layer.", flush=True)
                return 0

        if not accepted:
            # This is not a software error. The scientific acceptance gate simply
            # has not passed yet. Persist and exit successfully so Colab shows the
            # diagnostic instead of an opaque CalledProcessError.
            rounds_completed += max_rounds_this_run
            save_in_progress(
                output_dir,
                student,
                config,
                accepted_layers=accepted_layers,
                current_layer=layer_idx,
                pre_probe_nll=pre_probe_nll,
                rounds_completed=rounds_completed,
                layer_reports=layer_reports,
                status="current_layer_needs_more_training",
            )
            status = {
                "status": "current_layer_needs_more_training",
                "accepted_layers": accepted_layers,
                "current_layer": layer_idx,
                "rounds_completed": rounds_completed,
                "last_report": best_report,
                "message": "No next Transformer layer was replaced. Rerun to continue this same layer.",
            }
            write_status(output_dir, status)
            print("\nNOT A CRASH:", json.dumps(status, indent=2), flush=True)
            return 0

        accepted_layers.append(layer_idx)
        seq.save_progress(
            output_dir,
            student,
            config,
            accepted_layers,
            layer_reports,
            stage=f"accepted_layer_{layer_idx}",
        )
        remove_in_progress(output_dir)
        in_progress = None
        print(f"✅ accepted layer {layer_idx}; persistent progress saved", flush=True)

    if accepted_layers != list(range(num_layers)):
        raise RuntimeError("not all layers were accepted")

    summary = structural_summary(student)
    if summary["memory_fusion_layers"] != num_layers or summary["transformer_attention_layers"] != 0:
        raise RuntimeError(f"unexpected final structure: {summary}")

    report = {
        "status": "all_layers_accepted",
        "architecture": "smollm2-memory-fusion-sequential-full-v2",
        "base_model": args.base_model,
        "memory_rank": args.memory_rank,
        "feature_dim": args.feature_dim,
        "context_length": args.context_length,
        "teacher_probe_nll": teacher_probe_nll,
        "accepted_layers": accepted_layers,
        "layer_reports": layer_reports,
        "integrated_stages": [],
    }

    print("\nALL REPLACEMENTS ACCEPTED — STARTING INTEGRATED TRAINING", flush=True)
    integrated_specs = [
        ("core_o", args.core_o_tokens, args.core_o_lr),
        ("core_o_norm", args.norm_tokens, args.norm_lr),
        ("full", args.full_tokens, args.full_lr),
    ]
    for stage_name, token_budget, lr in integrated_specs:
        stage_report = seq.run_integrated_stage(
            teacher=teacher,
            student=student,
            batch_iter=batch_iter,
            stage_name=stage_name,
            token_budget=token_budget,
            lr=lr,
            args=args,
            device=device,
            amp=amp,
            output_dir=output_dir,
            config=config,
            report=report,
        )
        report["integrated_stages"].append(stage_report)
        seq.save_full_state(
            output_dir,
            student,
            config,
            report,
            filename=f"stage_{stage_name}_full_state.pt",
        )
        print(f"✅ completed {stage_name}; full checkpoint saved", flush=True)

    student.eval()
    final_probe_nll = seq.probe_nll(student, probe_blocks, device, amp)
    report["status"] = "complete"
    report["final_probe_nll"] = final_probe_nll
    report["final_probe_delta_nll"] = final_probe_nll - teacher_probe_nll
    report["parameters"] = seq.parameter_summary(student)
    report["structure"] = structural_summary(student)
    report["elapsed_minutes"] = (time.perf_counter() - start_time) / 60.0
    report["peak_vram_gib"] = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0
    )

    seq.save_full_state(output_dir, student, config, report)
    _atomic_json_save(report, output_dir / "sequential_training_report.json")
    write_status(output_dir, {"status": "complete", "accepted_layers": accepted_layers})
    tokenizer.save_pretrained(output_dir)
    print("\nFINAL REPORT", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Any accepted/current-layer checkpoint already written is preserved.", file=sys.stderr, flush=True)
        raise
    except Exception as exc:
        # Always expose the actual child error in Colab instead of only the outer
        # CalledProcessError.
        import traceback
        traceback.print_exc()
        try:
            out = Path(seq.parse_args().output_dir)
            _atomic_json_save(
                {
                    "status": "software_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                out / "sequential_last_error.json",
            )
        except Exception:
            pass
        raise
