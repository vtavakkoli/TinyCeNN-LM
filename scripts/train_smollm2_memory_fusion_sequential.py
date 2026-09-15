#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.smollm2_memory_fusion import (
    DEFAULT_SMOLLM2,
    MemoryFusionLlamaAttention,
    SmolMemoryFusionConfig,
    parameter_summary,
    replace_attention_layers,
    structural_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sequential teacher-guided SmolLM2 Memory Fusion conversion with "
            "per-layer acceptance gates and staged whole-model distillation."
        )
    )
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--output-dir", default="checkpoints/smollm2-memory-fusion-sequential-r64")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--memory-rank", type=int, choices=(32, 48, 64), default=64)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--shuffle-buffer", type=int, default=2048)

    p.add_argument("--min-layer-steps", type=int, default=50)
    p.add_argument("--max-layer-steps", type=int, default=300)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--layer-lr", type=float, default=2e-4)
    p.add_argument("--teacher-alpha-start", type=float, default=0.90)
    p.add_argument("--teacher-alpha-end", type=float, default=0.00)
    p.add_argument("--layer-kl-weight", type=float, default=0.10)
    p.add_argument("--layer-ce-weight", type=float, default=0.05)
    p.add_argument("--cosine-weight", type=float, default=0.25)
    p.add_argument("--accept-nmse", type=float, default=0.20)
    p.add_argument("--accept-cosine", type=float, default=0.90)
    p.add_argument("--accept-incremental-delta-nll", type=float, default=0.015)
    p.add_argument("--accept-cumulative-delta-nll", type=float, default=0.05)
    p.add_argument("--probe-blocks", type=int, default=4)
    p.add_argument("--probe-context", type=int, default=128)
    p.add_argument("--strict-acceptance", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--core-o-tokens", type=int, default=50_000)
    p.add_argument("--core-o-lr", type=float, default=3e-5)
    p.add_argument("--norm-tokens", type=int, default=50_000)
    p.add_argument("--norm-lr", type=float, default=8e-6)
    p.add_argument("--full-tokens", type=int, default=100_000)
    p.add_argument("--full-lr", type=float, default=3e-6)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=0.25)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--hidden-weight", type=float, default=0.5)

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--backup-every-updates", type=int, default=50)
    p.add_argument("--max-runtime-minutes", type=float, default=240.0)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


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


def token_blocks(dataset, tokenizer, text_field: str, context_length: int):
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    buffer: list[int] = []
    for row in dataset:
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos)
        while len(buffer) >= context_length:
            yield torch.tensor(buffer[:context_length], dtype=torch.long)
            del buffer[:context_length]


def batches(blocks, batch_size: int):
    pending = []
    for block in blocks:
        pending.append(block)
        if len(pending) == batch_size:
            yield torch.stack(pending)
            pending.clear()


def make_optimizer(params, lr: float, device: torch.device):
    params = list(params)
    if not params:
        raise ValueError("optimizer received no trainable parameters")
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
    except Exception:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def distillation_kl(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    per_token = F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean() * (temperature ** 2)


def representation_loss(student_hidden, teacher_hidden) -> torch.Tensor:
    max_idx = min(len(student_hidden), len(teacher_hidden)) - 1
    candidates = (6, 12, 18, 24, 30)
    indices = [i for i in candidates if i <= max_idx]
    if not indices:
        indices = [max_idx]
    terms = []
    for idx in indices:
        s = student_hidden[idx].float()
        t = teacher_hidden[idx].float()
        cosine = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        nmse = (s - t).square().mean() / t.square().mean().clamp_min(1e-5)
        terms.append(cosine + 0.25 * nmse)
    return torch.stack(terms).mean()


def alignment_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred32 = pred.float()
    target32 = target.float()
    mse = (pred32 - target32).square().mean()
    nmse = mse / target32.square().mean().clamp_min(1e-5)
    cosine = F.cosine_similarity(pred32, target32, dim=-1).mean()
    return nmse, cosine


def alignment_loss(pred: torch.Tensor, target: torch.Tensor, cosine_weight: float) -> torch.Tensor:
    nmse, cosine = alignment_metrics(pred, target)
    return nmse + cosine_weight * (1.0 - cosine)


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(v) for v in value)
    if isinstance(value, list):
        return [_detach_tree(v) for v in value]
    return value


def capture_attention_input(model, ids: torch.Tensor, layer_idx: int, amp, *, with_output: bool):
    capture: dict[str, Any] = {}
    module = model.model.layers[layer_idx].self_attn

    def pre_hook(mod, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("attention hidden_states not found")
        capture["hidden"] = hidden
        for key in (
            "position_embeddings",
            "position_ids",
            "attention_mask",
            "cache_position",
        ):
            if key in kwargs and kwargs[key] is not None:
                capture[key] = _detach_tree(kwargs[key])

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        if with_output:
            with amp():
                out = model(
                    input_ids=ids,
                    labels=ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            with torch.no_grad(), amp():
                out = model(
                    input_ids=ids,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
    finally:
        handle.remove()
    if "hidden" not in capture:
        raise RuntimeError(f"failed to capture layer {layer_idx} attention input")
    return capture, out


def attention_kwargs_from_capture(capture: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"use_cache": False}
    for key in (
        "position_embeddings",
        "position_ids",
        "attention_mask",
        "cache_position",
    ):
        if key in capture:
            kwargs[key] = capture[key]
    return kwargs


def call_attention(module, hidden: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
    out = module(hidden, **kwargs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def alpha_for_step(step: int, max_steps: int, start: float, end: float) -> float:
    if max_steps <= 1:
        return float(end)
    progress = min(max((step - 1) / (max_steps - 1), 0.0), 1.0)
    return float(start + progress * (end - start))


def freeze_current_layer_only(student, layer_idx: int) -> list[torch.nn.Parameter]:
    for p in student.parameters():
        p.requires_grad = False
    module = student.model.layers[layer_idx].self_attn
    if not isinstance(module, MemoryFusionLlamaAttention):
        raise TypeError(f"layer {layer_idx} is not MemoryFusionLlamaAttention")
    trainable = []
    for p in module.core.parameters():
        p.requires_grad = True
        trainable.append(p)
    for p in module.o_proj.parameters():
        p.requires_grad = True
        trainable.append(p)
    return trainable


def select_integrated_stage_params(student, stage: str) -> list[torch.nn.Parameter]:
    for p in student.parameters():
        p.requires_grad = False

    if stage in {"core_o", "core_o_norm"}:
        for module in student.modules():
            if isinstance(module, MemoryFusionLlamaAttention):
                for p in module.core.parameters():
                    p.requires_grad = True
                for p in module.o_proj.parameters():
                    p.requires_grad = True

        if stage == "core_o_norm":
            for name, p in student.named_parameters():
                if "norm" in name.lower():
                    p.requires_grad = True
    elif stage == "full":
        for p in student.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown integrated stage {stage!r}")

    return [p for p in student.parameters() if p.requires_grad]


def _layer_prefix(idx: int) -> str:
    return f"model.layers.{idx}.self_attn."


def progressive_attention_state(student, accepted_layers: list[int]) -> dict[str, torch.Tensor]:
    prefixes = tuple(_layer_prefix(i) for i in accepted_layers)
    return {
        k: v.detach().cpu()
        for k, v in student.state_dict().items()
        if prefixes and k.startswith(prefixes)
    }


def save_progress(
    output_dir: Path,
    student,
    config: SmolMemoryFusionConfig,
    accepted_layers: list[int],
    layer_reports: list[dict],
    *,
    stage: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "stage": stage,
        "accepted_layers": accepted_layers,
        "config": config.to_dict(),
        "layer_reports": layer_reports,
        "attention_state": progressive_attention_state(student, accepted_layers),
    }
    tmp = output_dir / "sequential_progress.tmp"
    final = output_dir / "sequential_progress.pt"
    torch.save(payload, tmp)
    tmp.replace(final)
    (output_dir / "sequential_progress.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "stage": stage,
                "accepted_layers": accepted_layers,
                "config": config.to_dict(),
                "layer_reports": layer_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_progress_if_available(
    output_dir: Path,
    student,
    config: SmolMemoryFusionConfig,
    *,
    resume: bool,
) -> tuple[list[int], list[dict]]:
    path = output_dir / "sequential_progress.pt"
    if not resume or not path.exists():
        return [], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    accepted = [int(x) for x in payload.get("accepted_layers", [])]
    if payload.get("config", {}).get("memory_rank") != config.memory_rank:
        raise RuntimeError("resume checkpoint memory rank differs from requested rank")
    if accepted:
        replace_attention_layers(student, config, accepted)
        incompatible = student.load_state_dict(payload["attention_state"], strict=False)
        expected = {
            k
            for k in student.state_dict()
            if any(k.startswith(_layer_prefix(i)) for i in accepted)
        }
        missing = [k for k in incompatible.missing_keys if k in expected]
        if missing:
            raise RuntimeError(f"resume checkpoint missing accepted-layer keys: {missing[:8]}")
    print(f"RESUME: accepted layers={accepted}")
    return accepted, list(payload.get("layer_reports", []))


def save_full_state(
    output_dir: Path,
    student,
    config: SmolMemoryFusionConfig,
    metadata: dict,
    *,
    filename: str = "smollm2_memory_fusion_sequential_full.pt",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / filename
    tmp = output_dir / (filename + ".tmp")
    torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()}, tmp)
    tmp.replace(state_path)
    meta = {
        "format_version": 1,
        "architecture": "smollm2-memory-fusion-sequential-full",
        "base_model": metadata["base_model"],
        "memory_fusion": config.to_dict(),
        "training": metadata,
        "state_file": filename,
    }
    (output_dir / "smollm2_memory_fusion_sequential_config.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def build_probe_blocks(tokenizer, context: int, count: int) -> list[torch.Tensor]:
    try:
        wiki = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="validation",
        )
    except Exception:
        texts = [
            "The history of science is shaped by observation, measurement, and careful comparison.",
            "A language model predicts the next token from the sequence that came before it.",
            "Vienna is the capital of Austria and has a long history of music, science, and public administration.",
            "Neural networks can approximate complex functions when their parameters are trained on representative data.",
            "The experiment should separate training data from the probe used to decide whether a replacement is acceptable.",
            "Reliable evaluation requires the same inputs, tokenizer, context length, and scoring rule for every model.",
            "A recurrent memory can carry information through a sequence without constructing a full dense attention matrix.",
            "When a model component is replaced, small approximation errors can accumulate across many layers.",
        ]
        stream = "\n".join(texts)
    else:
        stream = "\n".join(str(x) for x in wiki["text"] if str(x).strip())

    ids = tokenizer(stream, add_special_tokens=False)["input_ids"]
    need = context
    blocks = []
    for start in range(0, len(ids) - need + 1, need):
        blocks.append(torch.tensor(ids[start : start + need], dtype=torch.long))
        if len(blocks) >= count:
            break
    if not blocks:
        raise RuntimeError("could not construct probe blocks")
    return blocks


@torch.no_grad()
def probe_nll(model, probe_blocks: list[torch.Tensor], device: torch.device, amp) -> float:
    model.eval()
    losses = []
    for block in probe_blocks:
        x = block.unsqueeze(0).to(device)
        with amp():
            out = model(input_ids=x, labels=x, use_cache=False, return_dict=True)
        losses.append(float(out.loss.detach().float()))
    return sum(losses) / len(losses)


@torch.no_grad()
def real_hidden_function_metrics(
    teacher,
    student,
    ids: torch.Tensor,
    layer_idx: int,
    amp,
) -> tuple[float, float]:
    t_capture, _ = capture_attention_input(teacher, ids, layer_idx, amp, with_output=False)
    s_capture, _ = capture_attention_input(student, ids, layer_idx, amp, with_output=False)
    real_hidden = s_capture["hidden"].detach()
    kwargs = attention_kwargs_from_capture(t_capture)
    target = call_attention(teacher.model.layers[layer_idx].self_attn, real_hidden, kwargs)
    pred = call_attention(student.model.layers[layer_idx].self_attn, real_hidden, kwargs)
    nmse, cosine = alignment_metrics(pred, target)
    return float(nmse), float(cosine)


def acceptance_passes(
    *,
    nmse: float,
    cosine: float,
    incremental_delta_nll: float,
    cumulative_delta_nll: float,
    args: argparse.Namespace,
) -> bool:
    return (
        nmse <= args.accept_nmse
        and cosine >= args.accept_cosine
        and incremental_delta_nll <= args.accept_incremental_delta_nll
        and cumulative_delta_nll <= args.accept_cumulative_delta_nll
    )


def train_one_replacement(
    *,
    teacher,
    student,
    layer_idx: int,
    batch_iter,
    probe_blocks,
    teacher_probe_nll: float,
    pre_replacement_probe_nll: float,
    args,
    device,
    amp,
) -> dict:
    trainable = freeze_current_layer_only(student, layer_idx)
    student.train()
    optimizer = make_optimizer(trainable, args.layer_lr, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and choose_dtype(device) == torch.float16)
    )
    best: dict[str, float | int | bool] | None = None
    accepted = False

    for step in range(1, args.max_layer_steps + 1):
        ids = next(batch_iter).to(device, non_blocking=True)

        t_capture, t_out = capture_attention_input(
            teacher, ids, layer_idx, amp, with_output=True
        )
        s_capture, s_out = capture_attention_input(
            student, ids, layer_idx, amp, with_output=True
        )

        alpha = alpha_for_step(
            step,
            args.max_layer_steps,
            args.teacher_alpha_start,
            args.teacher_alpha_end,
        )
        mixed_hidden = (
            alpha * t_capture["hidden"].detach()
            + (1.0 - alpha) * s_capture["hidden"].detach()
        )
        kwargs = attention_kwargs_from_capture(t_capture)

        with torch.no_grad(), amp():
            target = call_attention(
                teacher.model.layers[layer_idx].self_attn,
                mixed_hidden,
                kwargs,
            )

        with amp():
            pred = call_attention(
                student.model.layers[layer_idx].self_attn,
                mixed_hidden,
                kwargs,
            )
            functional = alignment_loss(pred, target, args.cosine_weight)
            kl = distillation_kl(s_out.logits, t_out.logits, args.temperature)
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
            nmse_batch, cosine_batch = alignment_metrics(pred.detach(), target.detach())
            print(
                f"layer={layer_idx:02d} step={step:03d}/{args.max_layer_steps} "
                f"alpha={alpha:.3f} functional={float(functional.detach()):.4f} "
                f"nmse={float(nmse_batch):.4f} cos={float(cosine_batch):.4f} "
                f"kl={float(kl.detach()):.4f} ce={float(s_out.loss.detach()):.4f} "
                f"grad={grad_norm:.3f}"
            )

        should_check = (
            step >= args.min_layer_steps
            and (step % args.check_every == 0 or step == args.max_layer_steps)
        )
        if should_check:
            nmse, cosine = real_hidden_function_metrics(
                teacher, student, ids, layer_idx, amp
            )
            current_probe_nll = probe_nll(student, probe_blocks, device, amp)
            student.train()
            incremental = current_probe_nll - pre_replacement_probe_nll
            cumulative = current_probe_nll - teacher_probe_nll
            passed = acceptance_passes(
                nmse=nmse,
                cosine=cosine,
                incremental_delta_nll=incremental,
                cumulative_delta_nll=cumulative,
                args=args,
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
                candidate["nmse"] < best["nmse"]
                and candidate["cumulative_delta_nll"] <= best["cumulative_delta_nll"] + 0.01
            ):
                best = candidate
            print(
                f"  ACCEPTANCE CHECK layer={layer_idx:02d}: "
                f"NMSE={nmse:.4f} (≤{args.accept_nmse:.4f}) "
                f"cos={cosine:.4f} (≥{args.accept_cosine:.4f}) "
                f"ΔNLL_inc={incremental:+.5f} (≤{args.accept_incremental_delta_nll:+.5f}) "
                f"ΔNLL_total={cumulative:+.5f} (≤{args.accept_cumulative_delta_nll:+.5f}) "
                f"=> {'PASS' if passed else 'continue'}"
            )
            if passed:
                accepted = True
                best = candidate
                break

        del t_out, s_out, pred, target, total

    if best is None:
        raise RuntimeError("acceptance was never evaluated")
    return {
        "layer": layer_idx,
        "accepted": accepted,
        "steps": int(best["step"]),
        "nmse": float(best["nmse"]),
        "cosine": float(best["cosine"]),
        "probe_nll": float(best["probe_nll"]),
        "incremental_delta_nll": float(best["incremental_delta_nll"]),
        "cumulative_delta_nll": float(best["cumulative_delta_nll"]),
    }


def run_integrated_stage(
    *,
    teacher,
    student,
    batch_iter,
    stage_name: str,
    token_budget: int,
    lr: float,
    args,
    device,
    amp,
    output_dir: Path,
    config: SmolMemoryFusionConfig,
    report: dict,
) -> dict:
    if token_budget <= 0:
        return {"stage": stage_name, "tokens": 0, "updates": 0, "skipped": True}

    trainable = select_integrated_stage_params(student, stage_name)
    optimizer = make_optimizer(trainable, lr, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and choose_dtype(device) == torch.float16)
    )
    student.train()
    optimizer.zero_grad(set_to_none=True)

    seen_tokens = 0
    micro = 0
    updates = 0
    last = {}
    start = time.perf_counter()

    while seen_tokens < token_budget:
        ids = next(batch_iter).to(device, non_blocking=True)
        with torch.no_grad(), amp():
            t_out = teacher(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        with amp():
            s_out = student(
                input_ids=ids,
                labels=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            kl = distillation_kl(s_out.logits, t_out.logits, args.temperature)
            hidden = representation_loss(s_out.hidden_states, t_out.hidden_states)
            loss = (
                args.ce_weight * s_out.loss
                + args.kl_weight * kl
                + args.hidden_weight * hidden
            )
            scaled = loss / args.grad_accum

        if not torch.isfinite(scaled):
            raise RuntimeError(f"non-finite loss during integrated stage {stage_name}")

        scaler.scale(scaled).backward()
        micro += 1
        seen_tokens += ids.numel()
        last = {
            "ce": float(s_out.loss.detach().float()),
            "kl": float(kl.detach().float()),
            "hidden": float(hidden.detach().float()),
            "total": float(loss.detach().float()),
        }
        del t_out, s_out

        if micro % args.grad_accum:
            continue

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

        if updates == 1 or updates % args.log_every == 0:
            print(
                f"stage={stage_name} update={updates} tokens={seen_tokens:,}/{token_budget:,} "
                f"ce={last['ce']:.4f} kl={last['kl']:.4f} hidden={last['hidden']:.4f} "
                f"grad={grad_norm:.3f}"
            )

        if args.backup_every_updates > 0 and updates % args.backup_every_updates == 0:
            backup_meta = dict(report)
            backup_meta["integrated_stage"] = stage_name
            backup_meta["integrated_stage_tokens"] = seen_tokens
            save_full_state(
                output_dir,
                student,
                config,
                backup_meta,
                filename="live_sequential_full_state.pt",
            )
            print("  persistent full-state backup saved")

    elapsed = time.perf_counter() - start
    return {
        "stage": stage_name,
        "tokens": seen_tokens,
        "updates": updates,
        "lr": lr,
        "elapsed_minutes": elapsed / 60.0,
        "last": last,
        "trainable_parameters": sum(p.numel() for p in trainable),
    }


def main() -> None:
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

    print(f"device={device} dtype={dtype} memory_rank={args.memory_rank}")
    print(f"output_dir={output_dir}")

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

    accepted_layers, layer_reports = load_progress_if_available(
        output_dir, student, config, resume=args.resume
    )

    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    batch_iter = iter(
        batches(
            token_blocks(raw, tokenizer, args.text_field, args.context_length),
            args.batch_size,
        )
    )

    probe_blocks = build_probe_blocks(
        tokenizer,
        context=args.probe_context,
        count=args.probe_blocks,
    )
    teacher_probe_nll = probe_nll(teacher, probe_blocks, device, amp)
    print(f"teacher probe NLL={teacher_probe_nll:.6f}")

    start_time = time.perf_counter()
    num_layers = int(student.config.num_hidden_layers)

    for layer_idx in range(num_layers):
        if layer_idx in accepted_layers:
            continue
        if accepted_layers != list(range(layer_idx)):
            raise RuntimeError(
                f"accepted layers must be a contiguous prefix before layer {layer_idx}: "
                f"{accepted_layers}"
            )
        if (time.perf_counter() - start_time) / 60.0 >= args.max_runtime_minutes * 0.70:
            raise RuntimeError(
                "runtime budget reached during sequential replacement; progress was "
                "saved and the same command can resume from the last accepted layer"
            )

        print("\n" + "=" * 110)
        print(f"SEQUENTIAL REPLACEMENT: layer {layer_idx}/{num_layers - 1}")
        print("=" * 110)

        pre_probe_nll = probe_nll(student, probe_blocks, device, amp)
        print(
            f"before replacement: probe NLL={pre_probe_nll:.6f}, "
            f"Δ vs teacher={pre_probe_nll - teacher_probe_nll:+.6f}"
        )

        replace_attention_layers(student, config, [layer_idx])

        layer_report = train_one_replacement(
            teacher=teacher,
            student=student,
            layer_idx=layer_idx,
            batch_iter=batch_iter,
            probe_blocks=probe_blocks,
            teacher_probe_nll=teacher_probe_nll,
            pre_replacement_probe_nll=pre_probe_nll,
            args=args,
            device=device,
            amp=amp,
        )
        layer_reports.append(layer_report)

        if not layer_report["accepted"] and args.strict_acceptance:
            save_progress(
                output_dir,
                student,
                config,
                accepted_layers,
                layer_reports,
                stage=f"layer_{layer_idx}_rejected",
            )
            (output_dir / "sequential_training_report.json").write_text(
                json.dumps(
                    {
                        "status": "stopped_on_rejected_layer",
                        "base_model": args.base_model,
                        "memory_rank": args.memory_rank,
                        "accepted_layers": accepted_layers,
                        "layer_reports": layer_reports,
                        "thresholds": {
                            "nmse": args.accept_nmse,
                            "cosine": args.accept_cosine,
                            "incremental_delta_nll": args.accept_incremental_delta_nll,
                            "cumulative_delta_nll": args.accept_cumulative_delta_nll,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"layer {layer_idx} did not satisfy acceptance criteria; "
                "the next Transformer layer was NOT replaced"
            )

        if not layer_report["accepted"]:
            print("WARNING: relaxed mode accepts a layer that did not pass all thresholds")

        accepted_layers.append(layer_idx)
        save_progress(
            output_dir,
            student,
            config,
            accepted_layers,
            layer_reports,
            stage=f"accepted_layer_{layer_idx}",
        )
        print(f"✅ accepted layer {layer_idx}; persistent progress saved")

    if accepted_layers != list(range(num_layers)):
        raise RuntimeError("not all layers were accepted")

    summary = structural_summary(student)
    if summary["memory_fusion_layers"] != num_layers or summary["transformer_attention_layers"] != 0:
        raise RuntimeError(f"unexpected final structure: {summary}")

    report = {
        "status": "all_layers_accepted",
        "architecture": "smollm2-memory-fusion-sequential-full",
        "base_model": args.base_model,
        "memory_rank": args.memory_rank,
        "feature_dim": args.feature_dim,
        "context_length": args.context_length,
        "teacher_probe_nll": teacher_probe_nll,
        "accepted_layers": accepted_layers,
        "layer_reports": layer_reports,
        "thresholds": {
            "nmse": args.accept_nmse,
            "cosine": args.accept_cosine,
            "incremental_delta_nll": args.accept_incremental_delta_nll,
            "cumulative_delta_nll": args.accept_cumulative_delta_nll,
        },
        "integrated_stages": [],
    }

    print("\n" + "=" * 110)
    print("ALL 30 REPLACEMENTS ACCEPTED — STARTING INTEGRATED TRAINING")
    print("=" * 110)

    integrated_specs = [
        ("core_o", args.core_o_tokens, args.core_o_lr),
        ("core_o_norm", args.norm_tokens, args.norm_lr),
        ("full", args.full_tokens, args.full_lr),
    ]
    for stage_name, token_budget, lr in integrated_specs:
        if (time.perf_counter() - start_time) / 60.0 >= args.max_runtime_minutes:
            print("runtime budget reached before remaining integrated stages")
            report["status"] = "runtime_budget_after_acceptance"
            break
        print(f"\n--- integrated stage: {stage_name} ---")
        stage_report = run_integrated_stage(
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
        save_full_state(
            output_dir,
            student,
            config,
            report,
            filename=f"stage_{stage_name}_full_state.pt",
        )
        print(f"✅ completed {stage_name}; full persistent checkpoint saved")

    student.eval()
    final_probe_nll = probe_nll(student, probe_blocks, device, amp)
    report["final_probe_nll"] = final_probe_nll
    report["final_probe_delta_nll"] = final_probe_nll - teacher_probe_nll
    report["parameters"] = parameter_summary(student)
    report["structure"] = structural_summary(student)
    report["elapsed_minutes"] = (time.perf_counter() - start_time) / 60.0
    report["peak_vram_gib"] = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    save_full_state(output_dir, student, config, report)
    (output_dir / "sequential_training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    tokenizer.save_pretrained(output_dir)

    print("\nFINAL REPORT")
    print(json.dumps(report, indent=2))
    print("saved:", output_dir)


if __name__ == "__main__":
    main()
