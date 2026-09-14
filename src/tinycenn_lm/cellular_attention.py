"""Causal 1-D Cellular Attention for TinyCeNN-LM research experiments.

The layer keeps attention sparse and local at each cellular step, but changes
the neighborhood across steps. Power-of-two dilations create an exponentially
growing receptive field without constructing a T-by-T attention matrix.

This is a research reference implementation. It favors clarity and auditable
causality over fused-kernel speed.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


VARIANTS = (
    "cellular_local3",
    "cellular_dilated3",
    "cellular_dilated5",
    "cellular_multiscale5",
    "cellular_shifted8",
)


class CellularAttentionLayer(nn.Module):
    """Sparse causal attention applied recurrently over changing 1-D neighborhoods.

    Q/K/V are expected after the pretrained model's RoPE. Q/K/V/O projections
    stay outside this module and can remain frozen in the replacement benchmark.

    The recurrent value state is updated step by step:
        H^(0) = repeat_gqa(V)
        H^(s+1)_i = (1-g_s) H^(s)_i + g_s sum_j a_ij H^(s)_j

    where each j belongs to a small causal neighborhood. Dilated variants use
    d_s = 2^s (or a supplied schedule), reducing maximum information-path length
    from O(N) local propagation to O(log N) cellular steps.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        feature_dim: int = 64,
        variant: str = "cellular_dilated3",
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        shifted_window: int = 8,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if min(num_heads, num_kv_heads, head_dim, feature_dim, shifted_window) < 1:
            raise ValueError("all dimensions must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        dilations = tuple(int(d) for d in dilations)
        if not dilations or min(dilations) < 1:
            raise ValueError("dilations must contain positive integers")

        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.groups = self.num_heads // self.num_kv_heads
        self.variant = variant
        self.dilations = dilations
        self.shifted_window = int(shifted_window)

        q_base = torch.zeros(self.num_heads, self.feature_dim, self.head_dim)
        k_base = torch.zeros(self.num_kv_heads, self.feature_dim, self.head_dim)
        for h in range(self.num_heads):
            nn.init.orthogonal_(q_base[h])
        for h in range(self.num_kv_heads):
            nn.init.orthogonal_(k_base[h])
        self.wq = nn.Parameter(q_base)
        self.wk = nn.Parameter(k_base)

        self.state_q = nn.Parameter(torch.zeros(
            self.num_heads, self.feature_dim, self.head_dim
        ))
        self.state_q_gate = nn.Parameter(torch.full(
            (len(self.dilations), self.num_heads), -2.0
        ))

        max_neighbors = max(
            self._max_neighbors_for_variant(variant), self.shifted_window
        )
        self.relative_bias = nn.Parameter(torch.zeros(
            len(self.dilations), self.num_heads, max_neighbors
        ))
        self.log_temperature = nn.Parameter(torch.zeros(
            len(self.dilations), self.num_heads
        ))
        self.step_gate = nn.Parameter(torch.zeros(
            len(self.dilations), self.num_heads
        ))

        eye = torch.eye(self.head_dim).expand(self.num_heads, -1, -1).clone()
        self.out_proj = nn.Parameter(eye)
        self.final_gate = nn.Parameter(torch.full((self.num_heads,), 2.0))
        self.log_gain = nn.Parameter(torch.zeros(self.num_heads))

    @staticmethod
    def _max_neighbors_for_variant(variant: str) -> int:
        if variant in ("cellular_local3", "cellular_dilated3"):
            return 3
        if variant in ("cellular_dilated5", "cellular_multiscale5"):
            return 5
        if variant == "cellular_shifted8":
            return 8
        raise ValueError(variant)

    @property
    def config(self) -> dict:
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "variant": self.variant,
            "dilations": list(self.dilations),
            "shifted_window": self.shifted_window,
        }

    def _offsets(self, step: int) -> tuple[int, ...]:
        d = self.dilations[step]
        if self.variant == "cellular_local3":
            return (0, 1, 2)
        if self.variant == "cellular_dilated3":
            return (0, d, 2 * d)
        if self.variant == "cellular_dilated5":
            return (0, d, 2 * d, 3 * d, 4 * d)
        if self.variant == "cellular_multiscale5":
            return tuple(dict.fromkeys((0, 1, d, 2 * d, 4 * d)))
        if self.variant == "cellular_shifted8":
            return tuple(range(self.shifted_window))
        raise ValueError(self.variant)

    def _valid_mask(self, length: int, step: int, device) -> tuple[Tensor, Tensor]:
        offsets = torch.tensor(self._offsets(step), device=device, dtype=torch.long)
        pos = torch.arange(length, device=device, dtype=torch.long)
        index = pos[:, None] - offsets[None, :]
        valid = index >= 0

        if self.variant == "cellular_shifted8":
            shift = 0 if step % 2 == 0 else self.shifted_window // 2
            block_start = ((pos + shift) // self.shifted_window) * self.shifted_window - shift
            valid = valid & (index >= block_start[:, None])

        return index.clamp_min(0), valid

    def _features(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        qf = F.normalize(torch.einsum("bhtd,hfd->bhtf", q, self.wq), dim=-1)
        kf = F.normalize(torch.einsum("bhtd,hfd->bhtf", k, self.wk), dim=-1)
        return qf, kf.repeat_interleave(self.groups, dim=1)

    def _cellular_step(
        self,
        qf: Tensor,
        kf: Tensor,
        state: Tensor,
        step: int,
    ) -> Tensor:
        length = state.shape[2]
        index, valid = self._valid_mask(length, step, state.device)
        keys = kf[:, :, index, :]
        values = state[:, :, index, :]

        dynamic_q = torch.einsum("bhtd,hfd->bhtf", state, self.state_q)
        mix = self.state_q_gate[step].sigmoid()[None, :, None, None]
        query = F.normalize(qf + mix * dynamic_q, dim=-1)

        scores = torch.einsum("bhtf,bhtwf->bhtw", query, keys)
        scores = scores / math.sqrt(self.feature_dim)
        scores = scores * self.log_temperature[step].clamp(-3, 3).exp()[None, :, None, None]
        width = index.shape[1]
        scores = scores + self.relative_bias[step, :, :width][None, :, None, :]
        scores = scores.masked_fill(~valid[None, None, :, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        message = torch.einsum("bhtw,bhtwd->bhtd", weights, values)

        gate = self.step_gate[step].sigmoid()[None, :, None, None]
        return state + gate * (message - state)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("expected Q/K/V as [batch, heads, time, dim]")
        if k.shape != v.shape:
            raise ValueError("K and V must have the same shape")
        if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError("Q/K/V batch, time and head_dim must match")
        if q.shape[1] != self.num_heads or k.shape[1] != self.num_kv_heads:
            raise ValueError("Q/K head counts do not match layer configuration")
        if q.shape[-1] != self.head_dim or q.shape[2] < 1:
            raise ValueError("head_dim mismatch or empty sequence")

        q, k, v = (x.to(self.wq.dtype) for x in (q, k, v))
        qf, kf = self._features(q, k)
        base = v.repeat_interleave(self.groups, dim=1)
        state = base
        for step in range(len(self.dilations)):
            state = self._cellular_step(qf, kf, state, step)

        projected = torch.einsum("bhtd,hde->bhte", state, self.out_proj)
        final_gate = self.final_gate.sigmoid()[None, :, None, None]
        output = base + final_gate * (projected - base)
        return output * self.log_gain.clamp(-4, 4).exp()[None, :, None, None]

    def receptive_field_tokens(self) -> int:
        reach = 0
        for step in range(len(self.dilations)):
            reach += max(self._offsets(step))
        return reach + 1

    def max_score_pairs(self, context: int) -> int:
        total = 0
        device = self.wq.device
        for step in range(len(self.dilations)):
            _, valid = self._valid_mask(context, step, device)
            total += int(valid.sum().item())
        return total

    def max_neighbors_per_step(self) -> int:
        return max(len(self._offsets(step)) for step in range(len(self.dilations)))
