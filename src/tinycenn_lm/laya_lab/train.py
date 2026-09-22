from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .core import LayaLabConfig
from .factory import make_replacement
from .data import _batch_from_items


def _local_metrics(pred: Tensor, target: Tensor, valid2d: Tensor):
    mask = valid2d[:, :, None].float()
    diff2 = ((pred.float() - target.float()) ** 2) * mask
    mse = diff2.sum() / (
        mask.sum() * pred.shape[-1]
    ).clamp_min(1.0)
    denom = ((target.float() ** 2) * mask).sum() / (
        mask.sum() * target.shape[-1]
    ).clamp_min(1.0)
    nmse = mse / denom.clamp_min(1e-8)

    p = pred.float()[valid2d]
    t = target.float()[valid2d]
    cosine = (
        F.cosine_similarity(p, t, dim=-1).mean()
        if p.numel()
        else pred.new_tensor(0.0)
    )
    return nmse, cosine


def _score_metrics(
    output_nmse: Tensor,
    output_cosine: Tensor,
    core_nmse: Tensor,
    core_cosine: Tensor,
    architecture: str | None = None,
):
    if architecture == "integrated_memory_v22":
        # V2.2 transfer already meets NMSE / decision-level gates on the best
        # full-attention layers, while output direction is the remaining
        # bottleneck. Keep the acceptance gates unchanged and optimize more
        # directly for cosine fidelity instead of relaxing the gate.
        return (
            0.75 * output_nmse
            + 1.00 * (1.0 - output_cosine)
            + 0.35 * core_nmse
            + 0.30 * (1.0 - core_cosine)
        )

    # Preserve the established objective for the other Laya replacements.
    return (
        output_nmse
        + 0.35 * (1.0 - output_cosine)
        + 0.50 * core_nmse
        + 0.15 * (1.0 - core_cosine)
    )


def train_one_replacement(
    teacher_agent,
    layer_idx: int,
    cfg: LayaLabConfig,
    train_items: list[dict],
    steps: int,
    batch_size: int,
    verbose: bool = True,
):
    device = teacher_agent.device
    teacher_layer = teacher_agent.model.encoder.layers[layer_idx]
    replacement = make_replacement(teacher_layer.attn, cfg).to(device)

    proj_dtype = teacher_layer.attn.Wqkv.weight.dtype
    replacement.Wqkv.to(device=device, dtype=proj_dtype)
    replacement.Wo.to(device=device, dtype=proj_dtype)

    for name, p in replacement.named_parameters():
        if name.startswith("Wqkv.") or name.startswith("Wo."):
            p.requires_grad = False
        elif p.is_floating_point():
            p.data = p.data.float()

    params = replacement.trainable_core_parameters()
    if not params:
        raise RuntimeError("replacement has no trainable core parameters")

    opt = torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    warmup = max(1, min(20, steps // 10))

    def lr_scale(step_idx: int):
        if step_idx < warmup:
            return 0.20 + 0.80 * float(step_idx + 1) / float(warmup)
        progress = float(step_idx - warmup) / float(max(1, steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 0.10 + 0.90 * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)
    capture: dict[str, Any] = {}

    def pre_hook(module, args, kwargs):
        capture["x"] = args[0].detach()
        pos = kwargs.get("position_embeddings")
        capture["pos"] = (
            None if pos is None else tuple(x.detach() for x in pos)
        )
        mask = kwargs.get("attention_mask")
        capture["mask"] = None if mask is None else mask.detach()

    def wo_pre_hook(module, args):
        if not args:
            return
        flat = args[0].detach()
        if flat.ndim != 3 or flat.shape[-1] != replacement.hidden_size:
            return
        b, t, _ = flat.shape
        capture["core"] = (
            flat.reshape(
                b,
                t,
                replacement.num_heads,
                replacement.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def post_hook(module, args, kwargs, output):
        capture["y"] = output[0].detach()

    h1 = teacher_layer.attn.register_forward_pre_hook(
        pre_hook, with_kwargs=True
    )
    h2 = teacher_layer.attn.Wo.register_forward_pre_hook(wo_pre_hook)
    h3 = teacher_layer.attn.register_forward_hook(
        post_hook, with_kwargs=True
    )

    teacher_agent.model.eval()
    pad_id = teacher_agent.tok.pad_token_id
    use_amp = device.type == "cuda"

    # Fixed probe selection fixes the old bug where best checkpoints were chosen
    # by comparing losses from different training minibatches.
    min_probe = max(8, batch_size * 2)
    probe_count = min(
        max(min_probe, len(train_items) // 10),
        max(min_probe, len(train_items) // 4),
    )
    probe_count = min(
        probe_count,
        max(batch_size, len(train_items) - batch_size),
    )

    if len(train_items) > probe_count + batch_size:
        fit_items = train_items[:-probe_count]
        probe_items = train_items[-probe_count:]
    else:
        fit_items = train_items
        probe_items = train_items

    best_score = float("inf")
    best_state = None
    best_metrics = {
        "nmse": float("inf"),
        "cosine": -1.0,
        "core_nmse": float("inf"),
        "core_cosine": -1.0,
    }

    def run_teacher(batch):
        args = [
            batch[k].to(device)
            for k in (
                "input_ids",
                "attention_mask",
                "marker_pos",
                "marker_mask",
                "qtype",
            )
        ]
        capture.clear()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=teacher_agent.dtype,
            enabled=use_amp,
        ):
            teacher_agent.model(*args)
        if "y" not in capture or "core" not in capture:
            raise RuntimeError(
                "ModernBERT hooks did not capture projected output "
                "and pre-Wo attention core"
            )
        return {
            "x": capture["x"],
            "pos": capture["pos"],
            "mask": capture["mask"],
            "y": capture["y"],
            "core": capture["core"],
        }

    probe_batch = _batch_from_items(
        probe_items,
        0,
        min(len(probe_items), max(8, batch_size * 2)),
        pad_id,
    )
    probe_valid = probe_batch["attention_mask"].to(device).bool()
    probe = run_teacher(probe_batch)

    def core_metrics(core_pred: Tensor, core_target: Tensor, valid: Tensor):
        pred_flat = core_pred.transpose(1, 2).reshape(
            core_pred.shape[0], core_pred.shape[2], -1
        )
        target_flat = core_target.transpose(1, 2).reshape(
            core_target.shape[0], core_target.shape[2], -1
        )
        return _local_metrics(pred_flat, target_flat, valid)

    def probe_metrics():
        replacement.eval()
        with torch.no_grad():
            pred, _ = replacement(
                probe["x"],
                position_embeddings=probe["pos"],
                attention_mask=probe["mask"],
            )
            output_nmse, output_cosine = _local_metrics(
                pred, probe["y"], probe_valid
            )
            core_pred = replacement.last_core_output
            if core_pred is None:
                raise RuntimeError("replacement did not expose core output")
            core_nmse, core_cosine = core_metrics(
                core_pred, probe["core"], probe_valid
            )
            score = _score_metrics(
                output_nmse,
                output_cosine,
                core_nmse,
                core_cosine,
                cfg.architecture,
            )
        return (
            output_nmse,
            output_cosine,
            core_nmse,
            core_cosine,
            score,
        )

    # With the validated 1e-2 LR, V2.2 can cross the strict local gates much
    # earlier than the old low-LR schedule. Probe more frequently and stop only
    # after two consecutive strong-margin probe passes. The best checkpoint is
    # still selected from the fixed held-out probe.
    eval_every = (
        max(50, steps // 12)
        if cfg.architecture == "integrated_memory_v22"
        else max(10, steps // 10)
    )
    strong_probe_streak = 0
    stopped_early = False

    try:
        for step in range(steps):
            batch = _batch_from_items(
                fit_items,
                step * batch_size,
                batch_size,
                pad_id,
            )
            target = run_teacher(batch)

            replacement.train()
            replacement.out_drop.eval()
            opt.zero_grad(set_to_none=True)

            pred, _ = replacement(
                target["x"],
                position_embeddings=target["pos"],
                attention_mask=target["mask"],
            )

            valid = batch["attention_mask"].to(device).bool()
            output_nmse, output_cosine = _local_metrics(
                pred, target["y"], valid
            )

            core_pred = replacement.last_core_output
            if core_pred is None:
                raise RuntimeError("replacement did not expose core output")
            core_nmse, core_cosine = core_metrics(
                core_pred, target["core"], valid
            )

            loss = _score_metrics(
                output_nmse,
                output_cosine,
                core_nmse,
                core_cosine,
                cfg.architecture,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite replacement loss at step {step + 1}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

            should_probe = (
                step == 0
                or (step + 1) % eval_every == 0
                or (step + 1) == steps
            )
            if should_probe:
                (
                    val_nmse,
                    val_cosine,
                    val_core_nmse,
                    val_core_cosine,
                    val_score,
                ) = probe_metrics()
                score_value = float(val_score.detach())

                if score_value < best_score:
                    best_score = score_value
                    best_metrics = {
                        "nmse": float(val_nmse.detach()),
                        "cosine": float(val_cosine.detach()),
                        "core_nmse": float(val_core_nmse.detach()),
                        "core_cosine": float(val_core_cosine.detach()),
                        "step": step + 1,
                        "selection": "fixed_local_probe",
                    }
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in replacement.state_dict().items()
                    }

                if verbose:
                    print(
                        f"layer={layer_idx:02d} "
                        f"step={step + 1:04d}/{steps} "
                        f"train_nmse={float(output_nmse.detach()):.4f} "
                        f"train_cos={float(output_cosine.detach()):.4f} "
                        f"val_nmse={float(val_nmse.detach()):.4f} "
                        f"val_cos={float(val_cosine.detach()):.4f} "
                        f"val_core_nmse={float(val_core_nmse.detach()):.4f}"
                    )

                if (
                    cfg.architecture == "integrated_memory_v22"
                    and cfg.early_stop_local
                    and (step + 1) >= min(400, steps)
                ):
                    strong_local = (
                        float(val_nmse.detach()) <= 0.20
                        and float(val_cosine.detach()) >= 0.90
                        and float(val_core_nmse.detach()) <= 0.20
                        and float(val_core_cosine.detach()) >= 0.90
                    )
                    strong_probe_streak = strong_probe_streak + 1 if strong_local else 0
                    if strong_probe_streak >= 2:
                        stopped_early = True
                        if verbose:
                            print(
                                f"layer={layer_idx:02d} early-stop at step "
                                f"{step + 1}: strong local fidelity held for "
                                "two consecutive probes"
                            )
                        break
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    if best_state is not None:
        replacement.load_state_dict(best_state)
    if "step" in best_metrics:
        best_metrics["stopped_early"] = stopped_early
        best_metrics["max_steps"] = steps
    replacement.eval()
    return replacement, best_metrics
