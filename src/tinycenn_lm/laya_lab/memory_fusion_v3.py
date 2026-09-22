from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .core import BaseLayaReplacementAttention, _orthogonal_maps, _valid_tokens
from .data import _batch_from_items, _load_typed_split, build_training_items, dataset_cases
from .evaluate import _accept, adapter_payload, benchmark_latency, evaluate_agent


class _BiGDN2(nn.Module):
    def __init__(self, heads: int, head_dim: int, rank: int):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.mem_q = nn.Parameter(_orthogonal_maps(heads, rank, head_dim))
        self.mem_k = nn.Parameter(_orthogonal_maps(heads, rank, head_dim))
        self.decay_w = nn.Parameter(torch.zeros(heads, rank, head_dim))
        self.decay_bias = nn.Parameter(torch.full((heads, rank), 4.0))
        self.beta_w = nn.Parameter(torch.zeros(heads, head_dim))
        self.beta_bias = nn.Parameter(torch.full((heads,), -1.5))
        self.erase_w = nn.Parameter(torch.zeros(heads, rank, head_dim))
        self.erase_bias = nn.Parameter(torch.full((heads, rank), -0.5))
        self.write_scale = nn.Parameter(torch.zeros(heads, head_dim))
        self.write_bias = nn.Parameter(torch.full((heads, head_dim), -0.5))

    @staticmethod
    def _project(x: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bhtd,hrd->bhtr", x, weight)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, valid: Tensor, *, reverse: bool) -> Tensor:
        if reverse:
            q, k, v, valid = q.flip(2), k.flip(2), v.flip(2), valid.flip(1)
        q, k, v = q.float(), k.float(), v.float()
        qm = F.normalize(self._project(q, self.mem_q.float()), dim=-1)
        km = F.normalize(self._project(k, self.mem_k.float()), dim=-1)
        decay = torch.sigmoid(
            torch.einsum("bhtd,hrd->bhtr", k, self.decay_w.float())
            + self.decay_bias.float()[None, :, None, :]
        )
        beta = torch.sigmoid(
            torch.einsum("bhtd,hd->bht", k, self.beta_w.float())
            + self.beta_bias.float()[None, :, None]
        )
        erase = torch.sigmoid(
            torch.einsum("bhtd,hrd->bhtr", k, self.erase_w.float())
            + self.erase_bias.float()[None, :, None, :]
        )
        write = torch.sigmoid(
            v * self.write_scale.float()[None, :, None, :]
            + self.write_bias.float()[None, :, None, :]
        )
        b, h, t, _ = qm.shape
        state = torch.zeros(
            b, h, self.rank, self.head_dim, device=q.device, dtype=torch.float32
        )
        outputs = []
        for i in range(t):
            pred = torch.einsum("bhr,bhrd->bhd", km[:, :, i], state)
            error = (v[:, :, i] - pred) * write[:, :, i]
            update = torch.einsum(
                "bhr,bhd->bhrd", km[:, :, i] * erase[:, :, i], error
            )
            proposed = (
                state * decay[:, :, i, :, None]
                + beta[:, :, i, None, None] * update
            )
            token_valid = valid[:, i][:, None, None, None]
            state = torch.where(token_valid, proposed, state)
            out_i = torch.einsum("bhr,bhrd->bhd", qm[:, :, i], state)
            out_i = out_i * valid[:, i][:, None, None].float()
            outputs.append(out_i)
        out = torch.stack(outputs, dim=2)
        return out.flip(2) if reverse else out


class MemoryFusionV3Attention(BaseLayaReplacementAttention):
    """Native bidirectional full/sliding-attention MemoryFusion for Laya/ModernBERT."""

    architecture = "memory_fusion_v3"

    def __init__(
        self,
        original: nn.Module,
        feature_dim: int = 64,
        memory_rank: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    ):
        super().__init__(original)
        self.sliding_window = getattr(original, "sliding_window", None)
        if self.sliding_window is not None:
            # HF attention's value can include a FlashAttention +1 offset.
            # The config/mask radius is the actual inclusive token distance.
            radius = getattr(original.config, "sliding_window", None)
            if radius is None and hasattr(original.config, "local_attention"):
                radius = original.config.local_attention // 2
            if radius is not None:
                self.sliding_window = int(radius)
        if self.sliding_window is not None:
            if not isinstance(self.sliding_window, int) or self.sliding_window < 0:
                raise ValueError("Expected ModernBERT sliding_window radius as a nonnegative int")
        self.feature_dim = int(feature_dim)
        self.memory_rank = int(memory_rank)
        self.dilations = tuple(int(d) for d in dilations)
        self.local_bias = nn.Parameter(torch.zeros(len(self.dilations), self.num_heads, 5))
        self.local_mix = nn.Parameter(torch.zeros(self.num_heads, len(self.dilations)))

        self.hedge_q = nn.Parameter(
            _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        )
        self.hedge_k = nn.Parameter(
            _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        )
        self.hedge_q_bias = nn.Parameter(torch.zeros(self.num_heads, self.feature_dim))
        self.hedge_k_bias = nn.Parameter(torch.zeros(self.num_heads, self.feature_dim))
        self.hedge_log_sharpness = nn.Parameter(torch.zeros(self.num_heads))

        self.gdn_fwd = _BiGDN2(self.num_heads, self.head_dim, self.memory_rank)
        self.gdn_bwd = _BiGDN2(self.num_heads, self.head_dim, self.memory_rank)

        # [symmetric-local, full-Hedgehog, direct-V, GDN2-forward, GDN2-backward]
        # The direct-V path is essentially free and gives optimization a strong
        # identity/self-token anchor while the non-local branches learn.
        self.fusion_w = nn.Parameter(torch.zeros(self.num_heads, 5, self.head_dim))
        # Fast phase starts with only the vectorized local + full-sequence
        # Hedgehog branches. Recurrent GDN2 is activated only when the fixed
        # probe is already close enough to benefit from memory refinement.
        prior = torch.tensor([1.20, 0.60, 0.20, -4.0, -4.0])
        self.fusion_bias = nn.Parameter(
            prior[None, :].expand(self.num_heads, -1).clone()
        )
        self.branch_log_gain = nn.Parameter(torch.zeros(self.num_heads, 5))
        self.output_log_gain = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer("_memory_enabled", torch.tensor(False))
        self._memory_active = False
        for p in self.Wo.parameters():
            p.requires_grad = True

    @property
    def memory_enabled(self):
        return self._memory_active

    @memory_enabled.setter
    def memory_enabled(self, enabled):
        if enabled and self.sliding_window is not None:
            raise ValueError("Unbounded recurrent memory cannot preserve a sliding window")
        self._memory_enabled.fill_(enabled)
        self._memory_active = bool(enabled)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older V3 weights lack the flag; their config supplies memory_enabled.
        state_dict.setdefault(prefix + "_memory_enabled", self._memory_enabled.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._memory_active = bool(self._memory_enabled.item())
        if self._memory_active and self.sliding_window is not None:
            raise ValueError("Sliding attention checkpoint cannot enable unbounded memory")

    def _symmetric_local(self, q: Tensor, k: Tensor, v: Tensor, valid: Tensor) -> Tensor:
        q, k, v = q.float(), k.float(), v.float()
        _, _, t, d = q.shape
        pos = torch.arange(t, device=q.device)
        outputs = []
        scale = 1.0 / math.sqrt(d)
        for s, dilation in enumerate(self.dilations):
            offsets = torch.tensor(
                [-2 * dilation, -dilation, 0, dilation, 2 * dilation],
                device=q.device, dtype=torch.long,
            )
            raw_index = pos[:, None] + offsets[None, :]
            inside = (raw_index >= 0) & (raw_index < t)
            if self.sliding_window is not None:
                inside = inside & (offsets.abs()[None, :] <= self.sliding_window)
            index = raw_index.clamp(0, t - 1)
            kn = k[:, :, index, :]
            vn = v[:, :, index, :]
            key_valid = valid[:, index]
            mask = (
                inside[None, None, :, :]
                & key_valid[:, None, :, :]
                & valid[:, None, :, None]
            )
            scores = torch.einsum("bhtd,bhtwd->bhtw", q, kn) * scale
            scores = scores + self.local_bias[s].float()[None, :, None, :]
            scores = scores.masked_fill(~mask, -1e4)
            weights = scores.softmax(dim=-1) * mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            outputs.append(torch.einsum("bhtw,bhtwd->bhtd", weights, vn))
        mix = self.local_mix.float().softmax(dim=-1)
        out = torch.zeros_like(outputs[0])
        for s, value in enumerate(outputs):
            out = out + mix[:, s][None, :, None, None] * value
        return out * valid[:, None, :, None].float()

    def _hedgehog_full(self, q: Tensor, k: Tensor, v: Tensor, valid: Tensor) -> Tensor:
        q, k, v = q.float(), k.float(), v.float()
        sharp = self.hedge_log_sharpness.float().clamp(-1.4, 2.1).exp()
        q_logits = (
            torch.einsum("bhtd,hrd->bhtr", q, self.hedge_q.float())
            + self.hedge_q_bias.float()[None, :, None, :]
        )
        k_logits = (
            torch.einsum("bhtd,hrd->bhtr", k, self.hedge_k.float())
            + self.hedge_k_bias.float()[None, :, None, :]
        )
        qf = (q_logits * sharp[None, :, None, None]).softmax(-1)
        kf = (k_logits * sharp[None, :, None, None]).softmax(-1)
        norm = math.sqrt(self.feature_dim)
        qf, kf = qf * norm, kf * norm
        mask = valid[:, None, :, None].float()
        kf = kf * mask
        vf = v * mask
        if self.sliding_window is not None:
            # Bounded query chunks avoid a T x T matrix and a T x F x D
            # prefix-state tensor. Work is O(T * (window + chunk) * (F + D)).
            outputs = []
            t = q.shape[2]
            radius = self.sliding_window
            for start in range(0, t, 64):
                end = min(start + 64, t)
                lo, hi = max(0, start - radius), min(t, end + radius)
                weights = qf[:, :, start:end] @ kf[:, :, lo:hi].transpose(-1, -2)
                qp = torch.arange(start, end, device=q.device)
                kp = torch.arange(lo, hi, device=q.device)
                allowed = (qp[:, None] - kp[None, :]).abs() <= radius
                weights = weights * allowed[None, None]
                den = weights.sum(-1, keepdim=True).clamp_min(1e-6)
                outputs.append((weights / den) @ vf[:, :, lo:hi])
            return torch.cat(outputs, dim=2) * mask
        memory = torch.einsum("bhtr,bhtd->bhrd", kf, vf)
        z = kf.sum(dim=2)
        num = torch.einsum("bhtr,bhrd->bhtd", qf, memory)
        den = torch.einsum("bhtr,bhr->bht", qf, z).clamp_min(1e-6)[..., None]
        return (num / den) * mask

    def forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None, **kwargs):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        local = self._symmetric_local(q, k, v, valid)
        hedge = self._hedgehog_full(q, k, v, valid)
        direct = v.float() * valid[:, None, :, None].float()
        all_logits = (
            torch.einsum("bhtd,hcd->bhtc", q.float(), self.fusion_w.float())
            + self.fusion_bias.float()[None, :, None, :]
        )
        gains = self.branch_log_gain.float().clamp(-2, 2).exp()
        if self.memory_enabled:
            gdn_fwd = self.gdn_fwd(q, k, v, valid, reverse=False)
            gdn_bwd = self.gdn_bwd(q, k, v, valid, reverse=True)
            branches = torch.stack((local, hedge, direct, gdn_fwd, gdn_bwd), dim=-2)
            branches = branches * gains[None, :, None, :, None]
            weights = all_logits.softmax(dim=-1)
        else:
            branches = torch.stack((local, hedge, direct), dim=-2)
            branches = branches * gains[:, :3][None, :, None, :, None]
            weights = all_logits[..., :3].softmax(dim=-1)
        out = (weights[..., None] * branches).sum(dim=-2)
        out = (
            out
            * self.output_log_gain.float().clamp(-2, 2).exp()[None, :, None, None]
            * valid[:, None, :, None].float()
        )
        return self.finish(out, hidden_states)

    def enable_memory_refinement(self):
        if not self.memory_enabled and self.sliding_window is None:
            self.memory_enabled = True
            with torch.no_grad():
                self.fusion_bias[:, 3:].fill_(-1.5)
            print("  enabling forward/backward GDN2 memory refinement")

    def core_parameters(self) -> list[nn.Parameter]:
        blocked = {id(p) for p in self.Wqkv.parameters()} | {id(p) for p in self.Wo.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in blocked]

    def fast_parameters(self) -> list[nn.Parameter]:
        # Small router/gain tensors can safely move much faster than the
        # orthogonal feature maps and recurrent memories.
        names = {
            "local_bias", "local_mix", "hedge_q_bias", "hedge_k_bias",
            "hedge_log_sharpness", "fusion_w", "fusion_bias",
            "branch_log_gain", "output_log_gain",
        }
        return [p for n, p in self.named_parameters() if n in names and p.requires_grad]

    def slow_core_parameters(self) -> list[nn.Parameter]:
        fast = {id(p) for p in self.fast_parameters()}
        out = {id(p) for p in self.output_parameters()}
        blocked = {id(p) for p in self.Wqkv.parameters()}
        return [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in fast and id(p) not in out and id(p) not in blocked
        ]

    def output_parameters(self) -> list[nn.Parameter]:
        return list(self.Wo.parameters())

    def config_dict(self):
        d = super().config_dict()
        d.update(
            architecture=self.architecture,
            feature_dim=self.feature_dim,
            memory_rank=self.memory_rank,
            dilations=list(self.dilations),
            local_branch="symmetric_qkv_sparse_softmax",
            global_branch=("bidirectional_window_hedgehog" if self.sliding_window is not None else "bidirectional_full_sequence_hedgehog"),
            memory_branches=["gdn2_forward", "gdn2_backward"],
            branches=["symmetric_local", "full_hedgehog", "direct_v", "gdn2_forward", "gdn2_backward"],
            fusion_prior_fast=[1.20, 0.60, 0.20, -4.0, -4.0],
            fusion_prior_memory=[1.20, 0.60, 0.20, -1.5, -1.5],
            memory_enabled=self.memory_enabled,
            train_output_projection=True,
            supported_attention_type=("sliding_attention" if self.sliding_window is not None else "full_attention"),
            sliding_window=self.sliding_window,
        )
        return d


@dataclass
class LayaMemoryFusionV3Config:
    model_id: str = "convaiinnovations/laya"
    seed: int = 2026
    output_dir: str = "/content/laya_tinycenn"
    candidate_layer: int = 12
    target_all_attention: bool = False
    target_layers: tuple[int, ...] | None = None
    enable_recurrent_memory: bool = False
    feature_dim: int = 64
    memory_rank: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    train_cases: int = 480
    train_max_len: int = 512
    batch_size: int = 4
    train_cache_batches: int = 48
    probe_cache_batches: int = 8
    functional_steps: int = 160
    max_rounds: int = 2
    check_every: int = 20
    min_steps_before_check: int = 40
    fast_learning_rate: float = 1.0e-3
    core_learning_rate: float = 5e-4
    output_learning_rate: float = 5e-5
    warmup_steps: int = 10
    round_lr_decay: float = 0.70
    weight_decay: float = 2e-4
    cosine_weight: float = 0.30
    near_gate_cosine_weight: float = 0.85
    near_gate_nmse: float = 0.38
    memory_enable_nmse: float = 0.36
    memory_enable_cosine: float = 0.80
    max_local_nmse: float = 0.20
    min_local_cosine: float = 0.90
    min_teacher_agreement: float = 0.95
    max_mean_kl: float = 0.05
    max_accuracy_drop: float = 0.02
    gate_cases: int = 80
    final_cases: int = 160
    decision_refine_steps: int = 30
    decision_refine_lr_scale: float = 0.25
    decision_kl_weight: float = 0.12
    action_kl_weight: float = 0.01
    distill_temperature: float = 1.0
    cache_on_device: bool = True


def _tree_bytes(x):
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(_tree_bytes(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return sum(_tree_bytes(v) for v in x)
    return 0


def _tree_detach_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, tuple):
        return tuple(_tree_detach_cpu(v) for v in x)
    if isinstance(x, list):
        return [_tree_detach_cpu(v) for v in x]
    return x


def _tree_to(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, tuple):
        return tuple(_tree_to(v, device) for v in x)
    if isinstance(x, list):
        return [_tree_to(v, device) for v in x]
    return x


def _model_args(batch, device):
    return [batch[k].to(device) for k in (
        "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"
    )]


def _capture_teacher_entry(teacher, layer_idx: int, batch: dict) -> dict:
    capture: dict[str, Any] = {}
    module = teacher.model.encoder.layers[layer_idx].attn

    def pre_hook(mod, args, kwargs):
        capture["x"] = (args[0] if args else kwargs["hidden_states"]).detach()
        capture["pos"] = kwargs.get("position_embeddings")
        capture["mask"] = kwargs.get("attention_mask")

    def post_hook(mod, args, kwargs, output):
        capture["y"] = output[0].detach()

    h1 = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = module.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.no_grad(), torch.autocast(
            device_type=teacher.device.type,
            dtype=teacher.dtype,
            enabled=teacher.device.type == "cuda",
        ):
            teacher.model(*_model_args(batch, teacher.device))
    finally:
        h1.remove()
        h2.remove()
    if "x" not in capture or "y" not in capture:
        raise RuntimeError(f"failed to capture teacher layer {layer_idx}")
    return {
        "x": capture["x"].cpu(),
        "y": capture["y"].cpu(),
        "pos": _tree_detach_cpu(capture["pos"]),
        "mask": _tree_detach_cpu(capture["mask"]),
        "valid": batch["attention_mask"].bool().cpu(),
    }


def _build_teacher_cache(
    teacher, layer_idx: int, items: list[dict], *,
    batches: int, batch_size: int, offset: int, label: str,
) -> list[dict]:
    cache = []
    pad_id = teacher.tok.pad_token_id
    started = time.perf_counter()
    print(f"Building {label} teacher cache: {batches} batches × {batch_size}")
    for i in range(batches):
        batch = _batch_from_items(items, offset + i * batch_size, batch_size, pad_id)
        cache.append(_capture_teacher_entry(teacher, layer_idx, batch))
        if (i + 1) == 1 or (i + 1) % 8 == 0 or (i + 1) == batches:
            print(f"  cached {i + 1}/{batches}")
    print(f"{label} cache ready in {time.perf_counter() - started:.1f}s")
    return cache


def _functional_terms(pred: Tensor, target: Tensor, valid2d: Tensor):
    mask = valid2d[:, :, None].float()
    diff2 = ((pred.float() - target.float()) ** 2) * mask
    target2 = (target.float() ** 2) * mask
    nmse = diff2.sum() / target2.sum().clamp_min(1e-8)
    p, t = pred.float()[valid2d], target.float()[valid2d]
    cosine = F.cosine_similarity(p, t, dim=-1).mean() if p.numel() else pred.new_tensor(0.0)
    return nmse, cosine


@torch.no_grad()
def _probe_metrics(replacement, probe_cache: list[dict], device, dtype):
    diff_sum = target_sum = cos_sum = 0.0
    token_count = 0
    replacement.eval()
    for entry in probe_cache:
        x = entry["x"].to(device, non_blocking=True)
        target = entry["y"].to(device, non_blocking=True)
        pos = _tree_to(entry["pos"], device)
        mask = _tree_to(entry["mask"], device)
        valid = entry["valid"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            pred, _ = replacement(x, position_embeddings=pos, attention_mask=mask)
        m = valid[:, :, None].float()
        diff_sum += float((((pred.float() - target.float()) ** 2) * m).sum())
        target_sum += float(((target.float() ** 2) * m).sum())
        p, t = pred.float()[valid], target.float()[valid]
        if p.numel():
            cos_sum += float(F.cosine_similarity(p, t, dim=-1).sum())
            token_count += int(p.shape[0])
    return {
        "nmse": diff_sum / max(target_sum, 1e-8),
        "cosine": cos_sum / max(token_count, 1),
    }


def _state_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _kl(student, teacher, temperature: float, mask=None):
    t = max(float(temperature), 1e-3)
    s, q = student.float() / t, teacher.detach().float() / t
    if mask is not None:
        mask = mask.bool()
        s = s.masked_fill(~mask, -1e4)
        q = q.masked_fill(~mask, -1e4)
    return F.kl_div(
        F.log_softmax(s, -1), F.softmax(q, -1), reduction="none"
    ).sum(-1).mean() * (t * t)


def _train_functional_round(
    replacement,
    train_cache: list[dict],
    probe_cache: list[dict],
    cfg: LayaMemoryFusionV3Config,
    device,
    dtype,
    round_idx: int,
):
    replacement.train()
    replacement.out_drop.eval()
    fast = replacement.fast_parameters()
    slow = replacement.slow_core_parameters()
    core = fast + slow
    outp = replacement.output_parameters()
    for p in core:
        p.requires_grad = True
        if p.is_floating_point():
            p.data = p.data.float()
    for p in outp:
        p.requires_grad = True

    lr_scale = cfg.round_lr_decay ** (round_idx - 1)
    groups = [
        {"params": fast, "lr": cfg.fast_learning_rate * lr_scale, "base_lr": cfg.fast_learning_rate * lr_scale},
        {"params": slow, "lr": cfg.core_learning_rate * lr_scale, "base_lr": cfg.core_learning_rate * lr_scale},
        {"params": outp, "lr": cfg.output_learning_rate * lr_scale, "base_lr": cfg.output_learning_rate * lr_scale},
    ]
    try:
        opt = torch.optim.AdamW(
            groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
            fused=(device.type == "cuda")
        )
    except Exception:
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    best = _probe_metrics(replacement, probe_cache, device, dtype)
    best.update(step=0, round=round_idx)
    best_state = _state_cpu(replacement)
    best_score = best["nmse"] + cfg.cosine_weight * (1.0 - best["cosine"])
    stale_checks = 0
    lr_cuts = 0
    print(
        f"round {round_idx} start probe: "
        f"NMSE={best['nmse']:.4f} cos={best['cosine']:.4f}"
    )

    rng = random.Random(cfg.seed + 1009 * round_idx)
    order = list(range(len(train_cache)))
    rng.shuffle(order)

    for step in range(1, cfg.functional_steps + 1):
        if (step - 1) > 0 and (step - 1) % len(order) == 0:
            rng.shuffle(order)
        entry = train_cache[order[(step - 1) % len(order)]]

        # Warm up the more aggressive V3-Turbo learning rates to avoid the
        # large first-step gradient spike seen in the previous run.
        warm = min(1.0, step / max(cfg.warmup_steps, 1))
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * warm
        x = entry["x"].to(device, non_blocking=True)
        target = entry["y"].to(device, non_blocking=True)
        pos = _tree_to(entry["pos"], device)
        mask = _tree_to(entry["mask"], device)
        valid = entry["valid"].to(device, non_blocking=True)

        replacement.train()
        replacement.out_drop.eval()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            pred, _ = replacement(x, position_embeddings=pos, attention_mask=mask)
            nmse, cosine = _functional_terms(pred, target, valid)
            # Once reconstruction is in the near-gate region, cosine becomes
            # the limiting metric. Increase its gradient instead of spending
            # hundreds of extra steps optimizing NMSE alone.
            cos_w = (
                cfg.near_gate_cosine_weight
                if nmse.detach().item() <= cfg.near_gate_nmse
                else cfg.cosine_weight
            )
            loss = nmse + cos_w * (1.0 - cosine)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite V3 loss round={round_idx} step={step}"
            )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(core + outp, 1.0)
        opt.step()

        if step == 1 or step % 10 == 0:
            print(
                f"round={round_idx} step={step:03d}/{cfg.functional_steps} "
                f"nmse={nmse.detach().item():.4f} "
                f"cos={cosine.detach().item():.4f} cw={cos_w:.2f} "
                f"lr={opt.param_groups[0]['lr']:.1e} grad={float(grad):.3f}"
            )

        if step < cfg.min_steps_before_check or (
            step % cfg.check_every and step != cfg.functional_steps
        ):
            continue

        probe = _probe_metrics(replacement, probe_cache, device, dtype)
        probe.update(step=step, round=round_idx)
        score = probe["nmse"] + cfg.cosine_weight * (1.0 - probe["cosine"])
        significant_improvement = score < best_score - 0.001
        if score < best_score:
            best, best_score = probe, score
            best_state = _state_cpu(replacement)
        stale_checks = 0 if significant_improvement else stale_checks + 1
        print(
            f"  PROBE: NMSE={probe['nmse']:.4f} cos={probe['cosine']:.4f} "
            f"best={best['nmse']:.4f}/{best['cosine']:.4f}"
        )

        if (
            probe["nmse"] <= cfg.max_local_nmse
            and probe["cosine"] >= cfg.min_local_cosine
        ):
            best, best_state = probe, _state_cpu(replacement)
            replacement.load_state_dict(best_state)
            replacement.eval()
            return best, True

        if (
            cfg.enable_recurrent_memory
            and replacement.sliding_window is None
            and not replacement.memory_enabled
            and probe["nmse"] <= cfg.memory_enable_nmse
            and probe["cosine"] >= cfg.memory_enable_cosine
        ):
            replacement.enable_memory_refinement()
            stale_checks = 0

        if stale_checks >= 3 and lr_cuts < 2:
            for group in opt.param_groups:
                group["base_lr"] *= 0.55
                group["lr"] = group["base_lr"]
            lr_cuts += 1
            stale_checks = 0
            print(
                "  plateau: LR reduced to "
                + ", ".join(f"{g['lr']:.2e}" for g in opt.param_groups)
            )

    replacement.load_state_dict(best_state)
    replacement.eval()
    return best, (
        best["nmse"] <= cfg.max_local_nmse
        and best["cosine"] >= cfg.min_local_cosine
    )


def _split_train_gate_rows(ds, gate_cases: int, seed: int):
    rows = list(ds)
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(str(row.get("workflow", "unknown")), []).append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    gate_rows = []
    keys = [k for k in sorted(grouped) if grouped[k]]
    cursor = 0
    while len(gate_rows) < gate_cases and keys:
        key = keys[cursor % len(keys)]
        gate_rows.append(grouped[key].pop())
        if not grouped[key]:
            keys = [k for k in keys if grouped[k]]
            cursor = 0
        else:
            cursor += 1
    train_rows = []
    for key in sorted(grouped):
        train_rows.extend(grouped[key])
    rng.shuffle(train_rows)
    return train_rows, gate_rows


def _stratified_test_cases(ds, count: int, seed: int):
    rows = list(ds)
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(str(row.get("workflow", "unknown")), []).append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected = []
    keys = [k for k in sorted(grouped) if grouped[k]]
    cursor = 0
    while len(selected) < count and keys:
        key = keys[cursor % len(keys)]
        selected.append(grouped[key].pop())
        if not grouped[key]:
            keys = [k for k in keys if grouped[k]]
            cursor = 0
        else:
            cursor += 1
    return dataset_cases(selected, None)


def _decision_refine(
    teacher, student, replacement, cfg, train_items, train_cache,
    probe_cache, gate_cases, teacher_gate,
):
    device = teacher.device
    params = replacement.core_parameters() + replacement.output_parameters()
    opt = torch.optim.AdamW(
        params,
        lr=cfg.core_learning_rate * cfg.decision_refine_lr_scale,
        weight_decay=cfg.weight_decay,
    )
    best_gate = None
    for step in range(1, cfg.decision_refine_steps + 1):
        batch = _batch_from_items(
            train_items, step * cfg.batch_size, cfg.batch_size,
            teacher.tok.pad_token_id
        )
        args = _model_args(batch, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=teacher.dtype, enabled=device.type == "cuda"
        ):
            tlog, tact = teacher.model(*args)

        entry = train_cache[(step - 1) % len(train_cache)]
        x = entry["x"].to(device, non_blocking=True)
        target = entry["y"].to(device, non_blocking=True)
        pos = _tree_to(entry["pos"], device)
        mask = _tree_to(entry["mask"], device)
        valid = entry["valid"].to(device, non_blocking=True)

        student.model.eval()
        replacement.train()
        replacement.out_drop.eval()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=teacher.dtype, enabled=device.type == "cuda"
        ):
            slog, sact = student.model(*args)
            pred, _ = replacement(x, position_embeddings=pos, attention_mask=mask)
            nmse, cosine = _functional_terms(pred, target, valid)
            local_loss = nmse + cfg.cosine_weight * (1.0 - cosine)
            dkl = _kl(
                slog, tlog, cfg.distill_temperature,
                batch["marker_mask"].to(device)
            )
            akl = _kl(sact, tact, cfg.distill_temperature)
            loss = (
                local_loss
                + cfg.decision_kl_weight * dkl
                + cfg.action_kl_weight * akl
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step == 1 or step % 5 == 0:
            print(
                f"decision-refine step={step:02d}/{cfg.decision_refine_steps} "
                f"nmse={nmse.detach().item():.4f} cos={cosine.detach().item():.4f} "
                f"dkl={dkl.detach().item():.4e} akl={akl.detach().item():.4e}"
            )

        if step % 10 == 0 or step == cfg.decision_refine_steps:
            local = _probe_metrics(replacement, probe_cache, device, teacher.dtype)
            if (
                local["nmse"] <= cfg.max_local_nmse
                and local["cosine"] >= cfg.min_local_cosine
            ):
                gate = evaluate_agent(
                    student, gate_cases, teacher_agent=teacher, label="student"
                )
                accepted, checks, drop = _accept(local, teacher_gate, gate, cfg)
                best_gate = (accepted, local, gate, checks, drop)
                if accepted:
                    return best_gate
    return best_gate


def _demo_and_latency(teacher, student):
    state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    }
    return (
        {"teacher": teacher.predict(state, questions), "student": student.predict(state, questions)},
        {
            "teacher": benchmark_latency(teacher, state, questions),
            "student": benchmark_latency(student, state, questions),
        },
    )


def _candidate_layers(model, cfg):
    layers = model.encoder.layers
    if cfg.target_all_attention and cfg.target_layers is not None:
        raise ValueError("Choose target_all_attention or target_layers, not both")
    if cfg.target_all_attention:
        candidates = list(range(len(layers)))
    elif cfg.target_layers is not None:
        candidates = list(dict.fromkeys(cfg.target_layers))
    else:
        candidates = [cfg.candidate_layer]
    if not candidates or any(not isinstance(i, int) or not 0 <= i < len(layers) for i in candidates):
        raise ValueError(f"Invalid candidate layers: {candidates}")
    for i in candidates:
        if str(layers[i].attention_type) not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported attention type at layer {i}: {layers[i].attention_type}")
    return candidates


def run_memory_fusion_v3(cfg: LayaMemoryFusionV3Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    out_dir = Path(cfg.output_dir) / "memory_fusion_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    import laya
    print("Loading Laya teacher:", cfg.model_id)
    teacher = laya.load(
        cfg.model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    teacher.model.eval().requires_grad_(False)
    try:
        teacher.model.encoder.config.reference_compile = False
    except Exception:
        pass

    student = copy.copy(teacher)
    student.model = copy.deepcopy(teacher.model).eval().requires_grad_(False)

    candidates = _candidate_layers(student.model, cfg)

    train_ds = _load_typed_split("train")
    test_ds = _load_typed_split("test")
    train_rows, gate_rows = _split_train_gate_rows(
        train_ds, cfg.gate_cases, cfg.seed
    )
    gate_cases = dataset_cases(gate_rows, None)
    final_cases = _stratified_test_cases(test_ds, cfg.final_cases, cfg.seed + 1)
    teacher_gate = evaluate_agent(teacher, gate_cases, label="teacher")
    print("Teacher gate accuracy:", round(teacher_gate["accuracy"], 4))
    print(
        "Gate workflows:",
        {k: v["n"] for k, v in teacher_gate["by_workflow"].items()},
    )
    print("Candidate layers:", candidates)

    items = build_training_items(
        teacher, train_rows, cfg.train_cases, cfg.train_max_len, cfg.seed
    )
    needed = (
        (cfg.train_cache_batches + cfg.probe_cache_batches) * cfg.batch_size
    )
    if len(items) < max(needed, 32):
        raise RuntimeError(
            f"need at least {needed} training items for the configured caches; got {len(items)}"
        )

    history = []
    for idx in candidates:
        print(f"\n=== Candidate layer {idx}: {student.model.encoder.layers[idx].attention_type} ===")
        train_cache = _build_teacher_cache(
            teacher, idx, items,
            batches=cfg.train_cache_batches,
            batch_size=cfg.batch_size,
            offset=0,
            label="train",
        )
        probe_offset = cfg.train_cache_batches * cfg.batch_size
        probe_cache = _build_teacher_cache(
            teacher, idx, items,
            batches=cfg.probe_cache_batches,
            batch_size=cfg.batch_size,
            offset=probe_offset,
            label="probe",
        )

        cache_bytes = sum(
            _tree_bytes(entry) for entry in train_cache + probe_cache
        )
        move_cache = cfg.cache_on_device and teacher.device.type == "cuda"
        if move_cache:
            free_bytes, _ = torch.cuda.mem_get_info(teacher.device)
            move_cache = cache_bytes < free_bytes * 0.4
            if not move_cache:
                print(f"Keeping {cache_bytes / 2**20:.0f} MiB cache on CPU to leave training headroom.")
        if move_cache:
            def _move_cache(cache):
                moved = []
                for entry in cache:
                    moved.append({
                        "x": entry["x"].to(teacher.device),
                        "y": entry["y"].to(teacher.device),
                        "pos": _tree_to(entry["pos"], teacher.device),
                        "mask": _tree_to(entry["mask"], teacher.device),
                        "valid": entry["valid"].to(teacher.device),
                    })
                return moved
            train_cache = _move_cache(train_cache)
            probe_cache = _move_cache(probe_cache)
            print("Teacher I/O cache moved to GPU for zero-copy functional fitting.")

        original = student.model.encoder.layers[idx].attn
        replacement = MemoryFusionV3Attention(
            teacher.model.encoder.layers[idx].attn,
            cfg.feature_dim,
            cfg.memory_rank,
            cfg.dilations,
        ).to(student.device)
        proj_dtype = teacher.model.encoder.layers[idx].attn.Wqkv.weight.dtype
        replacement.Wqkv.to(device=student.device, dtype=proj_dtype)
        replacement.Wo.to(device=student.device, dtype=proj_dtype)
        student.model.encoder.layers[idx].attn = replacement

        accepted = False
        gate = checks = drop = None
        for round_idx in range(1, cfg.max_rounds + 1):
            print(f"\n=== V3 layer {idx}: functional round {round_idx}/{cfg.max_rounds} ===")
            local, local_pass = _train_functional_round(
                replacement, train_cache, probe_cache, cfg,
                teacher.device, teacher.dtype, round_idx
            )
            history.append({
                "layer": idx,
                "round": round_idx,
                "stage": "functional",
                "local": local,
                "local_pass": local_pass,
            })
            torch.save({
                "layer": idx,
                "round": round_idx,
                "config": replacement.config_dict(),
                "state_dict": _state_cpu(replacement),
                "metrics": local,
            }, out_dir / f"layer_{idx}_round_{round_idx}.pt")
            print(
                f"round {round_idx} best: NMSE={local['nmse']:.4f} "
                f"cos={local['cosine']:.4f}"
            )
            if not local_pass:
                continue

            print("Local gate passed; running Laya decision gate...")
            gate = evaluate_agent(
                student, gate_cases, teacher_agent=teacher, label="student"
            )
            accepted, checks, drop = _accept(local, teacher_gate, gate, cfg)
            print(json.dumps({
                "accepted": accepted,
                "teacher_agreement": gate.get("teacher_agreement"),
                "mean_teacher_kl": gate.get("mean_teacher_kl"),
                "accuracy": gate.get("accuracy"),
                "accuracy_drop": drop,
                "checks": checks,
            }, indent=2))
            history.append({
                "layer": idx, "round": round_idx, "stage": "decision_gate",
                "local": local, "gate": gate, "checks": checks,
                "accuracy_drop": drop, "accepted": accepted,
            })
            if accepted:
                break

            if cfg.decision_refine_steps > 0:
                print("Local gate passed but decision gate failed; short decision refinement...")
                before_refine = _state_cpu(replacement)
                refined = _decision_refine(
                    teacher, student, replacement, cfg, items, train_cache,
                    probe_cache, gate_cases, teacher_gate
                )
                if refined is not None:
                    accepted, local, gate, checks, drop = refined
                    history.append({
                        "layer": idx,
                        "round": round_idx,
                        "stage": "decision_refine",
                        "local": local,
                        "gate": gate,
                        "checks": checks,
                        "accuracy_drop": drop,
                        "accepted": accepted,
                    })
                    if accepted:
                        break
                replacement.load_state_dict(before_refine)

        if not accepted:
            student.model.encoder.layers[idx].attn = original
            print(
                f"❌ layer {idx} did not pass strict gates; original Laya attention restored."
            )
        else:
            print(f"✅ accepted MemoryFusionV3 layer {idx}")

        replacement.requires_grad_(False)
        student.model.eval()
        # Persist progress after every candidate, and release its I/O cache.
        torch.save(adapter_payload(student.model, cfg, {"history": history}), out_dir / "progress.pt")
        del train_cache, probe_cache, replacement, original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    teacher_final = evaluate_agent(teacher, final_cases, label="teacher")
    student_final = evaluate_agent(
        student, final_cases, teacher_agent=teacher, label="memory_fusion_v3"
    )
    demo, latency = _demo_and_latency(teacher, student)
    latency["speedup"] = latency["teacher"]["median_ms"] / max(latency["student"]["median_ms"], 1e-9)
    accepted_layers = [
        i for i, layer in enumerate(student.model.encoder.layers)
        if isinstance(layer.attn, MemoryFusionV3Attention)
    ]
    trainable_params = sum(
        p.numel()
        for i in accepted_layers
        for p in student.model.encoder.layers[i].attn.parameters()
    )

    report = {
        "architecture": "memory_fusion_v3",
        "status": ("complete" if len(accepted_layers) == len(student.model.encoder.layers)
                   else "partial" if accepted_layers else "failed_no_accepted_layers"),
        "all_attention_replaced": len(accepted_layers) == len(student.model.encoder.layers),
        "all_targets_replaced": set(candidates).issubset(accepted_layers),
        "remaining_attention_layers": [i for i in range(len(student.model.encoder.layers)) if i not in accepted_layers],
        "gate_final_disjoint": True,
        "model_id": cfg.model_id,
        "config": asdict(cfg),
        "candidate_layers": candidates,
        "accepted_layers": accepted_layers,
        "replacement_parameters": trainable_params,
        "replacement_trainable_parameters": sum(p.numel() for p in student.model.parameters() if p.requires_grad),
        "history": history,
        "teacher_gate": teacher_gate,
        "teacher_final": teacher_final,
        "student_final": student_final,
        "latency": latency,
        "demo": demo,
        "methodology": {
            "teacher": cfg.model_id + " unchanged",
            "gate_source": "held-out typed-decisions train rows",
            "final_source": "typed-decisions official test rows",
            "teacher_cache": True,
            "probe_items": cfg.probe_cache_batches * cfg.batch_size,
            "full_student_forward_every_functional_step": False,
        },
        "notes": [
            "V3 is a native bidirectional design, not a forward/reverse wrapper around the causal MemoryFusion core.",
            "The local branch uses symmetric Q/K/V sparse softmax neighborhoods.",
            "The Hedgehog branch is bidirectional and bounded to the original radius for sliding layers.",
            "Forward and backward GDN2 memories use separate parameters.",
            "Each candidate is fitted from cached teacher attention I/O, then gated in the cumulative student.",
            "Fast functional fitting starts with vectorized local+Hedgehog+direct-V branches.",
            "Router/gain parameters use a higher LR than feature maps; Wo uses a conservative LR.",
            "Cosine weight increases automatically once NMSE enters the near-gate region.",
            "Cached batches are reshuffled each pass and can remain on GPU for zero-copy fitting.",
            "Recurrent GDN2 is opt-in for full layers only; the default uses vectorized branches.",
            "Full-model Laya decision evaluation is deferred until local fidelity passes.",
        ],
    }
    torch.save(adapter_payload(student.model, cfg, report), out_dir / "adapter.pt")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nAccepted layers:", accepted_layers)
    print("Final student metrics:")
    print(json.dumps(student_final, indent=2))
    print("Latency:", json.dumps(latency, indent=2))
    print("Saved:", out_dir / "adapter.pt")
    return teacher, student, report
