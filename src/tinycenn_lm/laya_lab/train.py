from __future__ import annotations
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
    denom = ((target.float() ** 2) * mask).sum() / (mask.sum() * target.shape[-1]).clamp_min(1.0)
    nmse = mse / denom.clamp_min(1e-8)
    p = pred.float()[valid2d]
    t = target.float()[valid2d]
    cos = F.cosine_similarity(p, t, dim=-1).mean() if p.numel() else pred.new_tensor(0.0)
    return nmse, cos

def train_one_replacement(teacher_agent, layer_idx: int, cfg: LayaLabConfig,
                          train_items: list[dict], steps: int, batch_size: int,
                          verbose: bool = True):
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
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    capture: dict[str, Any] = {}

    def pre_hook(module, args, kwargs):
        capture["x"] = args[0].detach()
        pos = kwargs.get("position_embeddings")
        capture["pos"] = None if pos is None else tuple(x.detach() for x in pos)
        m = kwargs.get("attention_mask")
        capture["mask"] = None if m is None else m.detach()

    def post_hook(module, args, kwargs, output):
        capture["y"] = output[0].detach()

    h1 = teacher_layer.attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = teacher_layer.attn.register_forward_hook(post_hook, with_kwargs=True)
    teacher_agent.model.eval()
    best_score = float("inf")
    best_state = None
    best_metrics = {"nmse": float("inf"), "cosine": -1.0}
    pad_id = teacher_agent.tok.pad_token_id
    use_amp = device.type == "cuda"
    try:
        for step in range(steps):
            batch = _batch_from_items(train_items, step * batch_size, batch_size, pad_id)
            args = [batch[k].to(device) for k in (
                "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"
            )]
            capture.clear()
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=teacher_agent.dtype, enabled=use_amp
            ):
                teacher_agent.model(*args)
            if "y" not in capture:
                raise RuntimeError("ModernBERT attention hook did not capture the teacher target")
            replacement.train()
            opt.zero_grad(set_to_none=True)
            pred, _ = replacement(
                capture["x"], position_embeddings=capture["pos"], attention_mask=capture["mask"]
            )
            valid = batch["attention_mask"].to(device).bool()
            nmse, cosine = _local_metrics(pred, capture["y"], valid)
            loss = nmse + 0.35 * (1.0 - cosine)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite replacement loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            score = float(nmse.detach()) + 0.35 * (1.0 - float(cosine.detach()))
            if score < best_score:
                best_score = score
                best_metrics = {
                    "nmse": float(nmse.detach()),
                    "cosine": float(cosine.detach()),
                    "step": step + 1,
                }
                best_state = {k: v.detach().cpu().clone() for k, v in replacement.state_dict().items()}
            if verbose and ((step + 1) == 1 or (step + 1) % max(10, steps // 5) == 0):
                print(
                    f"layer={layer_idx:02d} step={step + 1:04d}/{steps} "
                    f"nmse={float(nmse):.4f} cos={float(cosine):.4f}"
                )
    finally:
        h1.remove()
        h2.remove()
    if best_state is not None:
        replacement.load_state_dict(best_state)
    replacement.eval()
    return replacement, best_metrics
