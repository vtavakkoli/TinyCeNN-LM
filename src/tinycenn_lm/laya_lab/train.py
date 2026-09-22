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
    mse = diff2.sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    denom = ((target.float() ** 2) * mask).sum() / (
        mask.sum() * target.shape[-1]
    ).clamp_min(1.0)
    nmse = mse / denom.clamp_min(1e-8)
    p = pred.float()[valid2d]
    t = target.float()[valid2d]
    cos = (
        F.cosine_similarity(p, t, dim=-1).mean()
        if p.numel()
        else pred.new_tensor(0.0)
    )
    return nmse, cos


def _core_as_tokens(core: Tensor) -> Tensor:
    """[B,H,T,D] -> [B,T,H*D], matching ModernBERT's Wo input layout."""
    b, h, t, d = core.shape
    return core.transpose(1, 2).reshape(b, t, h * d)


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
        raise RuntimeError("replacement exposes no trainable core parameters")

    opt = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    warmup = max(5, min(40, steps // 20))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(0.1, float(step + 1) / float(warmup))
        if steps <= warmup + 1:
            return 1.0
        progress = (step - warmup) / float(steps - warmup - 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    capture: dict[str, Any] = {}

    def pre_hook(module, args, kwargs):
        capture["x"] = args[0].detach()
        pos = kwargs.get("position_embeddings")
        capture["pos"] = None if pos is None else tuple(x.detach() for x in pos)
        m = kwargs.get("attention_mask")
        capture["mask"] = None if m is None else m.detach()

    def wo_pre_hook(module, args):
        # This is the exact teacher attention core BEFORE the frozen Wo
        # projection, already laid out as [B,T,H*D].
        capture["core"] = args[0].detach()

    def post_hook(module, args, kwargs, output):
        capture["y"] = output[0].detach()

    h1 = teacher_layer.attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = teacher_layer.attn.register_forward_hook(post_hook, with_kwargs=True)
    h3 = teacher_layer.attn.Wo.register_forward_pre_hook(wo_pre_hook)

    teacher_agent.model.eval()
    best_score = float("inf")
    best_state = None
    best_metrics = {
        "nmse": float("inf"),
        "cosine": -1.0,
        "core_nmse": float("inf"),
        "core_cosine": -1.0,
    }
    pad_id = teacher_agent.tok.pad_token_id
    use_amp = device.type == "cuda"

    try:
        for step in range(steps):
            batch = _batch_from_items(
                train_items, step * batch_size, batch_size, pad_id
            )
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
                    "ModernBERT hooks did not capture teacher attention targets"
                )

            replacement.train()
            replacement.out_drop.eval()  # teacher is in eval mode
            opt.zero_grad(set_to_none=True)

            pred, _ = replacement(
                capture["x"],
                position_embeddings=capture["pos"],
                attention_mask=capture["mask"],
            )
            pred_core = replacement.last_core_output
            if pred_core is None:
                raise RuntimeError("replacement did not expose its attention core")

            valid = batch["attention_mask"].to(device).bool()
            out_nmse, out_cos = _local_metrics(pred, capture["y"], valid)

            pred_core_tok = _core_as_tokens(pred_core)
            target_core_tok = capture["core"].float()
            core_nmse, core_cos = _local_metrics(
                pred_core_tok, target_core_tok, valid
            )

            # Attention-transfer objective: supervise the pre-Wo attention core
            # directly, while retaining a projected-output term so the quantity
            # used by the real encoder remains the primary acceptance target.
            loss = (
                0.55 * core_nmse
                + 0.45 * out_nmse
                + 0.15 * (1.0 - core_cos)
                + 0.10 * (1.0 - out_cos)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite replacement loss at step {step + 1}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            score = (
                float(out_nmse.detach())
                + 0.35 * (1.0 - float(out_cos.detach()))
                + 0.25 * float(core_nmse.detach())
            )
            if score < best_score:
                best_score = score
                best_metrics = {
                    "nmse": float(out_nmse.detach()),
                    "cosine": float(out_cos.detach()),
                    "core_nmse": float(core_nmse.detach()),
                    "core_cosine": float(core_cos.detach()),
                    "step": step + 1,
                }
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in replacement.state_dict().items()
                }

            if verbose and (
                (step + 1) == 1
                or (step + 1) % max(10, steps // 5) == 0
            ):
                print(
                    f"layer={layer_idx:02d} step={step + 1:04d}/{steps} "
                    f"out_nmse={float(out_nmse.detach()):.4f} "
                    f"out_cos={float(out_cos.detach()):.4f} "
                    f"core_nmse={float(core_nmse.detach()):.4f} "
                    f"core_cos={float(core_cos.detach()):.4f}"
                )
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    if best_state is not None:
        replacement.load_state_dict(best_state)
    replacement.eval()
    return replacement, best_metrics
