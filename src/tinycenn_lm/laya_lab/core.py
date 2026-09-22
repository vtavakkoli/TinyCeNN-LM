from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LayaLabConfig:
    architecture: str = "integrated_memory_v22"
    model_id: str = "convaiinnovations/laya"
    mode: str = "balanced"
    seed: int = 2026
    feature_dim: int = 64
    memory_rank: int = 32
    local_kernel: int = 5
    pdelta_conv_kernel: int = 4
    pdelta_chunk_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    training_steps: int | None = None
    min_teacher_agreement: float = 0.95
    max_mean_kl: float = 0.05
    max_accuracy_drop: float = 0.02
    max_local_nmse: float = 0.30
    min_local_cosine: float = 0.88
    output_dir: str = "/content/laya_tinycenn"

    def mode_settings(self) -> dict[str, int]:
        if self.architecture == "integrated_memory_v22":
            # Learned attention transfer needs substantially more question-level
            # sequences than the old 160-step run.  Keep these budgets local to
            # Integrated Memory so PDelta3/MemoryFusion retain their schedules.
            modes = {
                "smoke": dict(
                    train_cases=40, gate_cases=20, final_cases=40, steps=60,
                    batch_size=2, train_max_len=256, max_candidates=1,
                ),
                "balanced": dict(
                    train_cases=400, gate_cases=80, final_cases=200, steps=600,
                    batch_size=4, train_max_len=512, max_candidates=4,
                ),
                "extended": dict(
                    train_cases=1000, gate_cases=200, final_cases=400, steps=1600,
                    batch_size=4, train_max_len=768, max_candidates=6,
                ),
            }
        else:
            modes = {
                "smoke": dict(
                    train_cases=24, gate_cases=12, final_cases=40, steps=30,
                    batch_size=2, train_max_len=192, max_candidates=1,
                ),
                "balanced": dict(
                    train_cases=160, gate_cases=60, final_cases=160, steps=160,
                    batch_size=3, train_max_len=320, max_candidates=4,
                ),
                "extended": dict(
                    train_cases=600, gate_cases=160, final_cases=400, steps=400,
                    batch_size=4, train_max_len=512, max_candidates=8,
                ),
            }
        if self.mode not in modes:
            raise ValueError(f"mode must be one of {sorted(modes)}")
        return modes[self.mode]


ARCHITECTURES = {
    "integrated_memory_v22",
    "pdelta3_gdn2_clvr",
    "memory_fusion",
}


def _orthogonal_maps(heads: int, out_dim: int, in_dim: int) -> Tensor:
    maps = torch.empty(heads, out_dim, in_dim)
    for h in range(heads):
        nn.init.orthogonal_(maps[h])
    return maps


def _apply_modernbert_rope(q: Tensor, k: Tensor, position_embeddings):
    if position_embeddings is None:
        return q, k
    try:
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    except Exception:
        return q, k
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)


def _valid_tokens(attention_mask: Tensor | None, hidden_states: Tensor) -> Tensor:
    b, t = hidden_states.shape[:2]
    if attention_mask is None:
        return torch.ones(b, t, dtype=torch.bool, device=hidden_states.device)
    if attention_mask.ndim == 2:
        return attention_mask.bool()
    if attention_mask.ndim != 4:
        return torch.ones(b, t, dtype=torch.bool, device=hidden_states.device)
    if attention_mask.dtype == torch.bool:
        allowed = attention_mask
    else:
        allowed = attention_mask > -1e4
    valid_q = allowed.any(dim=-1).any(dim=1)
    valid_k = allowed.any(dim=-2).any(dim=1)
    valid = valid_q & valid_k
    if valid.shape[-1] != t:
        return torch.ones(b, t, dtype=torch.bool, device=hidden_states.device)
    return valid


def _depthwise_sequence_conv(x: Tensor, weight: Tensor, *, causal: bool = False) -> Tensor:
    b, h, t, d = x.shape
    y = x.permute(0, 1, 3, 2).reshape(b, h * d, t)
    k = int(weight.shape[-1])
    if causal:
        y = F.pad(y, (k - 1, 0))
    else:
        left = (k - 1) // 2
        right = k - 1 - left
        y = F.pad(y, (left, right))
    y = F.conv1d(y, weight, groups=h * d)
    return y.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


def _identity_conv_weight(heads: int, head_dim: int, kernel: int, *, causal: bool) -> Tensor:
    w = torch.zeros(heads * head_dim, 1, kernel)
    w[:, 0, -1 if causal else (kernel - 1) // 2] = 1.0
    return w


class BaseLayaReplacementAttention(nn.Module):
    """Drop-in replacement for ModernBertAttention."""

    architecture = "base"

    def __init__(self, original: nn.Module):
        super().__init__()
        self.config = original.config
        self.layer_idx = int(getattr(original, "layer_idx", -1))
        self.hidden_size = int(self.config.hidden_size)
        self.num_heads = int(self.config.num_attention_heads)
        self.head_dim = int(getattr(original, "head_dim", self.hidden_size // self.num_heads))
        self.Wqkv = copy.deepcopy(original.Wqkv)
        self.Wo = copy.deepcopy(original.Wo)
        self.out_drop = copy.deepcopy(original.out_drop)
        for p in self.Wqkv.parameters():
            p.requires_grad = False
        for p in self.Wo.parameters():
            p.requires_grad = False
        for p in self.out_drop.parameters():
            p.requires_grad = False
        self.last_core_output: Tensor | None = None

    def qkv(self, hidden_states: Tensor, position_embeddings=None):
        b, t, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_modernbert_rope(q, k, position_embeddings)
        return q, k, v

    def finish(self, core_out: Tensor, hidden_states: Tensor):
        self.last_core_output = core_out
        b, _, t, _ = core_out.shape
        flat = core_out.transpose(1, 2).reshape(b, t, self.hidden_size).to(hidden_states.dtype)
        return self.out_drop(self.Wo(flat)), None

    def trainable_core_parameters(self) -> list[nn.Parameter]:
        blocked = {id(p) for p in self.Wqkv.parameters()} | {id(p) for p in self.Wo.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in blocked]

    def config_dict(self) -> dict[str, Any]:
        return {"architecture": self.architecture, "layer_idx": self.layer_idx}
