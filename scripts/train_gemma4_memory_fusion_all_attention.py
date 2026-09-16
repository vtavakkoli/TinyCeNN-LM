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
from transformers import AutoTokenizer, Gemma4ForCausalLM

from scripts import train_smollm2_memory_fusion_sequential as seq
from tinycenn_lm.gemma4_memory_fusion import (
    DEFAULT_GEMMA4,
    DualGemma4Attention,
    Gemma4MemoryFusionConfig,
    all_attention_layers,
    ensure_dual_layer,
    ensure_dual_layers,
    freeze_current_layer_only,
    load_selected_memory_state,
    selected_memory_state,
    set_attention_mode,
    structural_summary,
    text_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sequentially replace every Gemma 4 E2B text-attention layer with TinyCeNN Memory Fusion. "
            "Strictly accepted layers are preferred; when --force-all is enabled, the closest saved "
            "candidate is selected after --force-after-rounds so the experiment can reach 35/35."
        )
    )
    p.add_argument("--base-model", default=DEFAULT_GEMMA4)
    p.add_argument("--model-revision", default="main")
    p.add_argument("--output-dir", default="checkpoints/gemma4-e2b-memory-fusion-all-r64")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--memory-rank", type=int, choices=(32, 48, 64), default=64)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--shuffle-buffer", type=int, default=2048)

    p.add_argument("--min-layer-steps", type=int, default=50)
    p.add_argument("--max-layer-steps", type=int, default=150)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--layer-lr", type=float, default=2e-4)
    p.add_argument("--teacher-alpha-start", type=float, default=0.90)
    p.add_argument("--teacher-alpha-end", type=float, default=0.00)
    p.add_argument("--layer-kl-weight", type=float, default=0.06)
    p.add_argument("--layer-ce-weight", type=float, default=0.08)
    p.add_argument("--cosine-weight", type=float, default=0.25)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--logit-chunk", type=int, default=2)
    p.add_argument("--accept-nmse", type=float, default=0.20)
    p.add_argument("--accept-cosine", type=float, default=0.90)
    p.add_argument("--accept-incremental-delta-nll", type=float, default=0.015)
    p.add_argument("--accept-cumulative-delta-nll", type=float, default=0.05)
    p.add_argument("--probe-blocks", type=int, default=3)
    p.add_argument("--probe-context", type=int, default=64)
    p.add_argument("--force-all", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-after-rounds", type=int, default=2)

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-runtime-minutes", type=float, default=420.0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    if args.context_length < 16 or args.probe_context < 16:
        p.error("context lengths must be >= 16")
    if args.logit_chunk < 1 or args.force_after_rounds < 1:
        p.error("logit-chunk and force-after-rounds must be positive")
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


def chunked_distillation_kl(student_logits, teacher_logits, temperature: float, chunk: int) -> torch.Tensor:
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
        total = total + F.kl_div(F.log_softmax(ss, dim=-1), teacher_prob, reduction="sum") * (
            temperature**2 / n
        )
    return total


def acceptance_passes(*, nmse, cosine, incremental, cumulative, args) -> bool:
    return (
        nmse <= args.accept_nmse
        and cosine >= args.accept_cosine
        and incremental <= args.accept_incremental_delta_nll
        and cumulative <= args.accept_cumulative_delta_nll
    )


def closeness_score(*, nmse, cosine, incremental, cumulative, args) -> float:
    """0 means all gates pass; larger values quantify normalized gate miss distance."""
    nmse_miss = max(0.0, nmse / args.accept_nmse - 1.0)
    cosine_miss = max(0.0, (args.accept_cosine - cosine) / max(1e-6, 1.0 - args.accept_cosine))
    inc_miss = max(0.0, (incremental - args.accept_incremental_delta_nll) / max(0.005, abs(args.accept_incremental_delta_nll)))
    cum_miss = max(0.0, (cumulative - args.accept_cumulative_delta_nll) / max(0.01, abs(args.accept_cumulative_delta_nll)))
    return 0.20 * nmse_miss + 0.20 * cosine_miss + 0.40 * inc_miss + 0.20 * cum_miss


def _alignment_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    p = pred.float()
    t = target.float()
    nmse = float(((p - t).pow(2).mean() / t.pow(2).mean().clamp_min(1e-8)).detach())
    cosine = float(F.cosine_similarity(p.flatten(1), t.flatten(1), dim=-1).mean().detach())
    return nmse, cosine


def _alignment_loss(pred: torch.Tensor, target: torch.Tensor, cosine_weight: float) -> torch.Tensor:
    p = pred.float()
    t = target.float()
    nmse = (p - t).pow(2).mean() / t.pow(2).mean().clamp_min(1e-8)
    cosine = F.cosine_similarity(p.flatten(1), t.flatten(1), dim=-1).mean()
    return nmse + cosine_weight * (1.0 - cosine)


def _call(module: DualGemma4Attention, which: str, hidden: torch.Tensor, capture: dict) -> torch.Tensor:
    target = module.original if which == "original" else module.memory
    out = target(
        hidden,
        capture["position_embeddings"],
        capture.get("attention_mask"),
        copy.deepcopy(capture.get("shared_kv_states")),
        past_key_values=None,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def probe_nll(model, blocks: list[torch.Tensor], device, amp, mode: str) -> float:
    set_attention_mode(model, mode)
    model.eval()
    losses = []
    for block in blocks:
        x = block.unsqueeze(0).to(device)
        with amp():
            out = model(input_ids=x, labels=x, use_cache=False, return_dict=True)
        losses.append(float(out.loss.detach().float()))
    return sum(losses) / len(losses)


@torch.no_grad()
def real_hidden_metrics(model, ids, layer_idx, device, amp) -> tuple[float, float]:
    module = text_model(model).layers[layer_idx].self_attn
    if not isinstance(module, DualGemma4Attention):
        raise TypeError("current layer is not dual-wrapped")
    set_attention_mode(model, "original")
    with amp():
        model(input_ids=ids, use_cache=False, return_dict=True)
    teacher_capture = copy.deepcopy(module.last_call)
    set_attention_mode(model, "memory")
    with amp():
        model(input_ids=ids, use_cache=False, return_dict=True)
    student_hidden = module.last_call["hidden_states"].detach()
    target = _call(module, "original", student_hidden, teacher_capture)
    pred = _call(module, "memory", student_hidden, teacher_capture)
    return _alignment_metrics(pred, target)


def _snapshot_current(model, layer_idx: int) -> dict[str, torch.Tensor]:
    module = text_model(model).layers[layer_idx].self_attn
    assert isinstance(module, DualGemma4Attention)
    return {k: v.detach().cpu().clone() for k, v in module.memory.state_dict().items()}


def _restore_current(model, layer_idx: int, state: dict[str, torch.Tensor]) -> None:
    module = text_model(model).layers[layer_idx].self_attn
    assert isinstance(module, DualGemma4Attention)
    module.memory.load_state_dict(state, strict=True)


def train_one_round(*, model, layer_idx, batch_iter, probe_blocks, teacher_probe_nll,
                    pre_probe_nll, args, device, amp) -> dict:
    module = text_model(model).layers[layer_idx].self_attn
    if not isinstance(module, DualGemma4Attention):
        raise TypeError(f"layer {layer_idx} is not DualGemma4Attention")
    trainable = freeze_current_layer_only(model, layer_idx, train_output_projection=True)
    optimizer = seq.make_optimizer(trainable, args.layer_lr, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and choose_dtype(device) == torch.float16))
    model.eval()
    best = None
    best_state = None

    for step in range(1, args.max_layer_steps + 1):
        ids = next(batch_iter).to(device, non_blocking=True)

        set_attention_mode(model, "original")
        with torch.no_grad(), amp():
            t_out = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
        teacher_logits = t_out.logits.detach()
        teacher_hidden = module.last_call["hidden_states"].detach()
        teacher_capture = copy.deepcopy(module.last_call)
        del t_out

        set_attention_mode(model, "memory")
        with amp():
            s_out = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
        student_hidden = module.last_call["hidden_states"].detach()
        alpha = seq.alpha_for_step(step, args.max_layer_steps, args.teacher_alpha_start, args.teacher_alpha_end)
        mixed_hidden = alpha * teacher_hidden + (1.0 - alpha) * student_hidden

        with torch.no_grad(), amp():
            target = _call(module, "original", mixed_hidden, teacher_capture)
        with amp():
            pred = _call(module, "memory", mixed_hidden, teacher_capture)
            functional = _alignment_loss(pred, target, args.cosine_weight)
            kl = chunked_distillation_kl(s_out.logits, teacher_logits, args.temperature, args.logit_chunk)
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
            nmse_batch, cosine_batch = _alignment_metrics(pred.detach(), target.detach())
            print(
                f"layer={layer_idx:02d} step={step:03d}/{args.max_layer_steps} alpha={alpha:.3f} "
                f"functional={float(functional.detach()):.4f} nmse={nmse_batch:.4f} cos={cosine_batch:.4f} "
                f"kl={float(kl.detach()):.4f} ce={float(s_out.loss.detach()):.4f} grad={grad_norm:.3f}",
                flush=True,
            )

        should_check = step >= args.min_layer_steps and (step % args.check_every == 0 or step == args.max_layer_steps)
        if should_check:
            nmse, cosine = real_hidden_metrics(model, ids, layer_idx, device, amp)
            current_probe_nll = probe_nll(model, probe_blocks, device, amp, "memory")
            incremental = current_probe_nll - pre_probe_nll
            cumulative = current_probe_nll - teacher_probe_nll
            passed = acceptance_passes(nmse=nmse, cosine=cosine, incremental=incremental, cumulative=cumulative, args=args)
            distance = closeness_score(nmse=nmse, cosine=cosine, incremental=incremental, cumulative=cumulative, args=args)
            candidate = {
                "step": step,
                "nmse": nmse,
                "cosine": cosine,
                "probe_nll": current_probe_nll,
                "incremental_delta_nll": incremental,
                "cumulative_delta_nll": cumulative,
                "strict_accepted": passed,
                "closeness_score": distance,
            }
            if best is None or (distance, cumulative, incremental) < (
                best["closeness_score"], best["cumulative_delta_nll"], best["incremental_delta_nll"]
            ):
                best = candidate
                best_state = _snapshot_current(model, layer_idx)
            print(
                f"  ACCEPTANCE CHECK layer={layer_idx:02d}: NMSE={nmse:.4f} (≤{args.accept_nmse:.4f}) "
                f"cos={cosine:.4f} (≥{args.accept_cosine:.4f}) "
                f"ΔNLL_inc={incremental:+.5f} (≤{args.accept_incremental_delta_nll:+.5f}) "
                f"ΔNLL_total={cumulative:+.5f} (≤{args.accept_cumulative_delta_nll:+.5f}) "
                f"distance={distance:.4f} => {'STRICT PASS' if passed else 'continue'}",
                flush=True,
            )
            if passed:
                break

        del s_out, pred, target, total, teacher_logits

    if best is None or best_state is None:
        raise RuntimeError("acceptance was never evaluated")
    _restore_current(model, layer_idx, best_state)
    return best


def save_progress(output_dir: Path, model, config, target_layers, replaced_layers, layer_reports, *, stage: str) -> None:
    payload = {
        "format_version": 1,
        "stage": stage,
        "target_layers": list(target_layers),
        "replaced_layers": list(replaced_layers),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "memory_state": selected_memory_state(model, replaced_layers),
    }
    atomic_torch(output_dir / "sequential_progress.pt", payload)
    atomic_json(output_dir / "sequential_progress.json", {k: v for k, v in payload.items() if k != "memory_state"})


def load_progress(output_dir: Path, model, config, target_layers, *, resume: bool):
    path = output_dir / "sequential_progress.pt"
    if not resume or not path.exists():
        return [], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    replaced = [int(x) for x in payload.get("replaced_layers", [])]
    if payload.get("target_layers") != list(target_layers):
        raise RuntimeError("resume target layers differ from this run")
    if int(payload.get("config", {}).get("memory_rank", -1)) != config.memory_rank:
        raise RuntimeError("resume memory rank differs from this run")
    if replaced != list(target_layers[: len(replaced)]):
        raise RuntimeError(f"replaced layers are not a prefix: {replaced}")
    if replaced:
        ensure_dual_layers(model, config, replaced)
        load_selected_memory_state(model, payload["memory_state"], replaced)
        set_attention_mode(model, "memory")
    print(f"RESUME: replaced attention layers={len(replaced)}/{len(target_layers)}", flush=True)
    return replaced, list(payload.get("layer_reports", []))


def save_in_progress(output_dir: Path, model, config, target_layers, replaced_layers, current_layer,
                     pre_probe_nll, rounds_completed, layer_reports, status):
    layers = list(replaced_layers) + [int(current_layer)]
    payload = {
        "format_version": 1,
        "status": status,
        "target_layers": list(target_layers),
        "replaced_layers": list(replaced_layers),
        "current_layer": int(current_layer),
        "pre_probe_nll": float(pre_probe_nll),
        "rounds_completed": int(rounds_completed),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "memory_state": selected_memory_state(model, layers),
    }
    atomic_torch(output_dir / "sequential_in_progress.pt", payload)
    atomic_json(output_dir / "sequential_in_progress.json", {k: v for k, v in payload.items() if k != "memory_state"})


def load_in_progress(output_dir: Path, model, config, target_layers, replaced_layers, *, resume: bool):
    path = output_dir / "sequential_in_progress.pt"
    if not resume or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("target_layers") != list(target_layers):
        return None
    if [int(x) for x in payload.get("replaced_layers", [])] != list(replaced_layers):
        return None
    expected = target_layers[len(replaced_layers)] if len(replaced_layers) < len(target_layers) else None
    if expected is None or int(payload.get("current_layer", -1)) != int(expected):
        return None
    ensure_dual_layer(model, config, expected)
    load_selected_memory_state(model, payload["memory_state"], list(replaced_layers) + [expected])
    set_attention_mode(model, "memory")
    print(f"RESUME CURRENT LAYER: layer={expected}, rounds_completed={payload.get('rounds_completed', 0)}", flush=True)
    return payload


def remove_in_progress(output_dir: Path) -> None:
    for name in ("sequential_in_progress.pt", "sequential_in_progress.json"):
        p = output_dir / name
        if p.exists():
            p.unlink()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    amp = amp_factory(device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} dtype={dtype} memory_rank={args.memory_rank}", flush=True)
    print(f"output_dir={output_dir} force_all={args.force_all} force_after_rounds={args.force_after_rounds}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # One base model only. Dual attention wrappers switch between the frozen native
    # path and the trainable Memory Fusion path, avoiding two ~10 GB BF16 copies.
    model = Gemma4ForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    text_config = model.config.get_text_config(decoder=True) if hasattr(model.config, "get_text_config") else model.config
    if text_config.model_type != "gemma4_text":
        raise ValueError(f"expected gemma4_text, got {text_config.model_type}")
    target_layers = all_attention_layers(model)
    kinds = list(text_config.layer_types)
    print(
        f"Gemma 4 text layers={len(target_layers)}; sliding={sum(x == 'sliding_attention' for x in kinds)}; "
        f"full={sum(x == 'full_attention' for x in kinds)}; targets=ALL 0..{target_layers[-1]}",
        flush=True,
    )

    config = Gemma4MemoryFusionConfig(feature_dim=args.feature_dim, memory_rank=args.memory_rank)
    replaced_layers, layer_reports = load_progress(output_dir, model, config, target_layers, resume=args.resume)
    in_progress = load_in_progress(output_dir, model, config, target_layers, replaced_layers, resume=args.resume)

    raw = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=True).shuffle(
        seed=args.seed, buffer_size=args.shuffle_buffer
    )
    batch_iter = iter(seq.batches(seq.token_blocks(raw, tokenizer, args.text_field, args.context_length), args.batch_size))
    probe_blocks = seq.build_probe_blocks(tokenizer, context=args.probe_context, count=args.probe_blocks)
    teacher_probe_nll = probe_nll(model, probe_blocks, device, amp, "original")
    print(f"original Gemma 4 probe NLL={teacher_probe_nll:.6f}", flush=True)

    start = time.perf_counter()
    for layer_idx in target_layers:
        if layer_idx in replaced_layers:
            continue
        if replaced_layers != target_layers[: target_layers.index(layer_idx)]:
            raise RuntimeError(f"replaced layers must be a prefix before {layer_idx}: {replaced_layers}")
        if (time.perf_counter() - start) / 60 >= args.max_runtime_minutes * 0.92:
            atomic_json(output_dir / "sequential_run_status.json", {
                "status": "paused_runtime_budget",
                "target_layers": target_layers,
                "replaced_layers": replaced_layers,
                "next_layer": layer_idx,
                "message": "Rerun with --resume to continue.",
            })
            return 0

        print("\n" + "=" * 100, flush=True)
        print(f"GEMMA4 ALL-ATTENTION REPLACEMENT: layer {layer_idx} ({kinds[layer_idx]})", flush=True)
        print("=" * 100, flush=True)

        if in_progress is not None and int(in_progress["current_layer"]) == layer_idx:
            pre_probe_nll = float(in_progress["pre_probe_nll"])
            rounds_completed = int(in_progress.get("rounds_completed", 0))
            print(f"continuing saved layer {layer_idx}; pre-replacement NLL={pre_probe_nll:.6f}", flush=True)
        else:
            pre_probe_nll = probe_nll(model, probe_blocks, device, amp, "memory") if replaced_layers else teacher_probe_nll
            print(f"before replacement NLL={pre_probe_nll:.6f}; Δoriginal={pre_probe_nll-teacher_probe_nll:+.6f}", flush=True)
            ensure_dual_layer(model, config, layer_idx)
            set_attention_mode(model, "memory")
            rounds_completed = 0

        strict = False
        selected_report = None
        while rounds_completed < args.force_after_rounds:
            global_round = rounds_completed + 1
            round_args = copy.copy(args)
            if global_round > 1:
                round_args.teacher_alpha_start = min(0.25, args.teacher_alpha_start)
                round_args.teacher_alpha_end = 0.0
                round_args.layer_lr = args.layer_lr * 0.5
            print(f"\n--- layer {layer_idx} round {global_round}/{args.force_after_rounds} ---", flush=True)
            report = train_one_round(
                model=model,
                layer_idx=layer_idx,
                batch_iter=batch_iter,
                probe_blocks=probe_blocks,
                teacher_probe_nll=teacher_probe_nll,
                pre_probe_nll=pre_probe_nll,
                args=round_args,
                device=device,
                amp=amp,
            )
            report.update({"layer": layer_idx, "layer_type": kinds[layer_idx], "round": global_round})
            layer_reports.append(report)
            selected_report = report
            rounds_completed = global_round
            save_in_progress(
                output_dir, model, config, target_layers, replaced_layers, layer_idx,
                pre_probe_nll, rounds_completed, layer_reports, "current_layer_training"
            )
            if report["strict_accepted"]:
                strict = True
                break

        if not strict and not args.force_all:
            atomic_json(output_dir / "sequential_run_status.json", {
                "status": "current_layer_needs_more_training",
                "target_layers": target_layers,
                "replaced_layers": replaced_layers,
                "current_layer": layer_idx,
                "rounds_completed": rounds_completed,
                "last_report": selected_report,
            })
            return 0

        selection = "strict" if strict else "closest_fallback"
        assert selected_report is not None
        selected_report["selected"] = True
        selected_report["selection"] = selection
        selected_report["accepted"] = True
        replaced_layers.append(layer_idx)
        save_progress(
            output_dir, model, config, target_layers, replaced_layers, layer_reports,
            stage=f"replaced_layer_{layer_idx}_{selection}",
        )
        remove_in_progress(output_dir)
        in_progress = None
        icon = "✅" if strict else "≈"
        print(
            f"{icon} selected Gemma 4 layer {layer_idx}: {selection}; "
            f"distance={selected_report['closeness_score']:.4f}; replaced={len(replaced_layers)}/{len(target_layers)}",
            flush=True,
        )

    final_nll = probe_nll(model, probe_blocks, device, amp, "memory")
    strict_layers = sorted({int(r["layer"]) for r in layer_reports if r.get("selected") and r.get("selection") == "strict"})
    fallback_layers = sorted({int(r["layer"]) for r in layer_reports if r.get("selected") and r.get("selection") == "closest_fallback"})
    report = {
        "status": "all_attention_replaced",
        "architecture": "gemma4-e2b-memory-fusion-all-attention-v1",
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "target_layers": target_layers,
        "replaced_layers": replaced_layers,
        "strict_layers": strict_layers,
        "closest_fallback_layers": fallback_layers,
        "feature_dim": args.feature_dim,
        "memory_rank": args.memory_rank,
        "original_probe_nll": teacher_probe_nll,
        "final_probe_nll": final_nll,
        "final_probe_delta_nll": final_nll - teacher_probe_nll,
        "structure": structural_summary(model),
        "layer_reports": layer_reports,
        "elapsed_minutes": (time.perf_counter() - start) / 60,
    }
    atomic_torch(output_dir / "gemma4_memory_fusion_full_state.pt", {
        "format": "gemma4-e2b-memory-fusion-all-attention-v1",
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "target_layers": target_layers,
        "config": config.to_dict(),
        "memory_state": selected_memory_state(model, replaced_layers),
        "report": report,
    })
    atomic_json(output_dir / "sequential_training_report.json", report)
    atomic_json(output_dir / "sequential_run_status.json", {
        "status": "complete",
        "target_layers": target_layers,
        "replaced_layers": replaced_layers,
        "strict_layers": strict_layers,
        "closest_fallback_layers": fallback_layers,
    })
    tokenizer.save_pretrained(output_dir)
    print("\nFINAL REPORT\n" + json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Saved progress remains resumable.", file=sys.stderr, flush=True)
        raise
