"""Feature-lab components for faster, stronger P-Delta2 attention replacements.

This module keeps the recurrent state bounded while testing three ingredients:
1) a chunk-vectorized curvature preconditioner (same recurrence as serial P-Delta2),
2) sparse dilated exact retrieval over logarithmic offsets, and
3) dual-timescale recurrent memories with query-dependent mixing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.research_layers import delta_recurrence, local_window_attention


@dataclass
class PDeltaState:
    memory: Tensor
    curvature: Tensor


@dataclass
class FeatureState:
    fast: PDeltaState
    slow: PDeltaState | None = None
    keys: Tensor | None = None
    values: Tensor | None = None


def precondition_reference(kp: Tensor, curvature: Tensor, alpha: Tensor, beta: Tensor,
                           log_x: Tensor, center: Tensor):
    """Tokenwise oracle for the diagonal curvature preconditioner."""
    writes = []
    for t in range(kp.shape[2]):
        kt = kp[:, :, t]
        r = (curvature + 1e-4).log() - center[None]
        s = r / (1.0 + r.abs())
        scale = torch.exp(-log_x[None] * s)
        numerator = scale * kt
        denominator = 1.0 + (kt * numerator).sum(-1, keepdim=True)
        writes.append(numerator / denominator.clamp_min(1e-4))
        curvature = alpha[None] * curvature + beta[None] * kt.square()
    return torch.stack(writes, dim=2), curvature


def precondition_chunked(kp: Tensor, curvature: Tensor, alpha: Tensor, beta: Tensor,
                         log_x: Tensor, center: Tensor, chunk_size: int = 32):
    """Vectorize curvature states inside bounded chunks; recurrent only across chunks."""
    writes = []
    for start in range(0, kp.shape[2], chunk_size):
        kc = kp[:, :, start:start + chunk_size]
        length = kc.shape[2]
        squared = kc.square()
        t = torch.arange(length, device=kp.device)
        j = torch.arange(length, device=kp.device)
        lag = t[:, None] - 1 - j[None, :]
        valid = lag >= 0
        weights = alpha[:, :, None, None].pow(lag.clamp_min(0)[None, None])
        weights = weights * valid[None, None]
        contribution = torch.einsum("bhjf,hftj->bhtf", squared, weights)
        contribution = contribution * beta[None, :, None, :]
        powers = alpha[:, :, None].pow(t[None, None])
        before = curvature[:, :, None, :] * powers.permute(0, 2, 1)[None] + contribution

        r = (before + 1e-4).log() - center[None, :, None, :]
        s = r / (1.0 + r.abs())
        scale = torch.exp(-log_x[None, :, None, :] * s)
        numerator = scale * kc
        denominator = 1.0 + (kc * numerator).sum(-1, keepdim=True)
        writes.append(numerator / denominator.clamp_min(1e-4))

        curvature = alpha[None] * before[:, :, -1] + beta[None] * squared[:, :, -1]
    return torch.cat(writes, dim=2), curvature


class PDelta2Core(nn.Module):
    """P-Delta2 recurrence with a chunk-vectorized diagonal preconditioner."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 96, chunk_size: int = 32,
                 forget_bias: float | None = None):
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if min(num_heads, num_kv_heads, head_dim, feature_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.groups = num_heads // num_kv_heads
        self.chunk_size = chunk_size

        base = torch.zeros(num_kv_heads, feature_dim, head_dim)
        for head in range(num_kv_heads):
            if feature_dim == head_dim:
                base[head] = torch.eye(head_dim)
            else:
                nn.init.orthogonal_(base[head])
        self.wk = nn.Parameter(base.clone())
        self.wq = nn.Parameter(base.repeat_interleave(self.groups, dim=0).clone())
        self.forget_w = nn.Parameter(torch.zeros(num_kv_heads, feature_dim, head_dim))
        default_bias = math.log(0.04 / 0.96) if forget_bias is None else forget_bias
        self.forget_b = nn.Parameter(torch.full((num_kv_heads, feature_dim), default_bias))
        self.erase_w = nn.Parameter(torch.zeros(num_kv_heads, feature_dim, head_dim))
        self.erase_b = nn.Parameter(torch.full((num_kv_heads, feature_dim), -1.0))
        self.write_w = nn.Parameter(torch.zeros(num_kv_heads, head_dim, head_dim))
        self.write_b = nn.Parameter(torch.full((num_kv_heads, head_dim), -1.0))
        self.pre_log_decay = nn.Parameter(torch.full((num_kv_heads, feature_dim), math.log(0.995)))
        self.pre_gain_logit = nn.Parameter(torch.full((num_kv_heads, feature_dim), math.log(0.12 / 0.88)))
        self.pre_range_raw = nn.Parameter(torch.zeros(num_kv_heads, 1))
        self.pre_center = nn.Parameter(torch.zeros(num_kv_heads, 1))
        self.log_gain = nn.Parameter(torch.zeros(num_heads))

    @staticmethod
    def _project(x, weight, bias):
        return torch.einsum("bhtd,hfd->bhtf", x, weight) + bias[None, :, None]

    def features(self, q, k, v):
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        qp = F.normalize(torch.einsum("bhtd,hfd->bhtf", qn, self.wq), dim=-1)
        kp = F.normalize(torch.einsum("bhtd,hfd->bhtf", kn, self.wk), dim=-1)
        log_decay = -0.25 * self._project(kn, self.forget_w, self.forget_b).sigmoid()
        erase = kp * self._project(kn, self.erase_w, self.erase_b).sigmoid()
        write_gate = self._project(F.normalize(v, dim=-1), self.write_w, self.write_b).sigmoid()
        return qp, kp, v * write_gate, erase, log_decay

    def precondition_parameters(self):
        alpha = self.pre_log_decay.clamp(math.log(0.98), math.log(0.9999)).exp()
        beta = self.pre_gain_logit.sigmoid()
        log_x = math.log(2.0) + self.pre_range_raw.sigmoid() * (math.log(8.0) - math.log(2.0))
        return alpha, beta, log_x, self.pre_center

    def precondition_keys(self, kp, curvature):
        return precondition_chunked(kp, curvature, *self.precondition_parameters(), self.chunk_size)

    def precondition_keys_reference(self, kp, curvature):
        return precondition_reference(kp, curvature, *self.precondition_parameters())

    def forward(self, q, k, v, state: PDeltaState | None = None, return_state: bool = False):
        q, k, v = (x.to(self.wq.dtype) for x in (q, k, v))
        if state is None:
            state = PDeltaState(
                memory=q.new_zeros(q.shape[0], self.num_kv_heads, self.feature_dim, self.head_dim),
                curvature=q.new_ones(q.shape[0], self.num_kv_heads, self.feature_dim),
            )
        qp, kp, z, erase, log_decay = self.features(q, k, v)
        kpre, curvature = self.precondition_keys(kp, state.curvature)
        output, memory = delta_recurrence(
            qp, kpre, z, erase, log_decay, state.memory, self.groups, self.chunk_size
        )
        output = output * self.log_gain.clamp(-4, 4).exp()[None, :, None, None]
        new_state = PDeltaState(memory, curvature)
        return (output, new_state) if return_state else output

    def recurrent_state_bytes(self, batch_size: int = 1):
        elements = self.num_kv_heads * self.feature_dim * self.head_dim
        elements += self.num_kv_heads * self.feature_dim
        return batch_size * elements * self.wq.element_size()


def dilated_sparse_attention(q: Tensor, k: Tensor, v: Tensor, offsets: Iterable[int], groups: int):
    """Exact causal softmax over a fixed set of logarithmic past offsets."""
    offsets = tuple(sorted(set(int(x) for x in offsets)))
    if not offsets or offsets[0] != 0 or min(offsets) < 0:
        raise ValueError("offsets must be non-negative and include 0")
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    total, length = k.shape[2], q.shape[2]
    prefix = total - length
    positions = prefix + torch.arange(length, device=q.device)
    off = torch.tensor(offsets, device=q.device)
    index = positions[:, None] - off[None, :]
    valid = index >= 0
    index = index.clamp_min(0)
    selected_k = k[:, :, index, :]
    selected_v = v[:, :, index, :]
    scores = (q.unsqueeze(-2) * selected_k).sum(-1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~valid[None, None], float("-inf"))
    weights = scores.softmax(-1)
    return (weights.unsqueeze(-1) * selected_v).sum(-2)


class FeaturePDelta2Layer(nn.Module):
    """Composable P-Delta2 experiment with dense/dilated retrieval and optional dual time scales."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 96, retrieval: str = "dense", window: int = 32,
                 offsets: Iterable[int] = (0, 1, 2, 4, 8, 16, 32),
                 dual_timescale: bool = False, chunk_size: int = 32):
        super().__init__()
        if retrieval not in {"none", "dense", "dilated"}:
            raise ValueError("retrieval must be none, dense, or dilated")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.groups = num_heads // num_kv_heads
        self.retrieval = retrieval
        self.window = window
        self.offsets = tuple(int(x) for x in offsets)
        self.dual_timescale = bool(dual_timescale)
        self.chunk_size = chunk_size

        if dual_timescale:
            fast_dim = feature_dim // 2
            slow_dim = feature_dim - fast_dim
            self.fast = PDelta2Core(num_heads, num_kv_heads, head_dim, fast_dim, chunk_size, -1.5)
            self.slow = PDelta2Core(num_heads, num_kv_heads, head_dim, slow_dim, chunk_size, -5.0)
            self.timescale_w = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.timescale_b = nn.Parameter(torch.zeros(num_heads))
        else:
            self.fast = PDelta2Core(num_heads, num_kv_heads, head_dim, feature_dim, chunk_size)
            self.slow = None
        if retrieval != "none":
            self.retrieval_w = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.retrieval_b = nn.Parameter(torch.full((num_heads,), -0.5))

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "retrieval": self.retrieval,
            "window": self.window,
            "offsets": list(self.offsets),
            "dual_timescale": self.dual_timescale,
            "chunk_size": self.chunk_size,
        }

    def _retrieval_keep(self):
        if self.retrieval == "dense":
            return max(0, self.window - 1)
        if self.retrieval == "dilated":
            return max(self.offsets)
        return 0

    def forward(self, q, k, v, state: FeatureState | None = None, return_state: bool = False,
                implementation: str = "chunk"):
        if implementation != "chunk":
            raise ValueError("feature lab supports the chunk implementation")
        fast_state = None if state is None else state.fast
        fast_out, new_fast = self.fast(q, k, v, fast_state, return_state=True)
        recurrent = fast_out
        new_slow = None
        if self.slow is not None:
            slow_state = None if state is None else state.slow
            slow_out, new_slow = self.slow(q, k, v, slow_state, return_state=True)
            gate = (
                torch.einsum("bhtd,hd->bht", F.normalize(q.float(), dim=-1), self.timescale_w)
                + self.timescale_b[None, :, None]
            ).sigmoid().unsqueeze(-1)
            recurrent = gate * fast_out + (1.0 - gate) * slow_out

        new_state = FeatureState(new_fast, new_slow)
        if self.retrieval != "none":
            keys = k.float() if state is None or state.keys is None else torch.cat((state.keys, k.float()), dim=2)
            values = v.float() if state is None or state.values is None else torch.cat((state.values, v.float()), dim=2)
            if self.retrieval == "dense":
                retrieved = local_window_attention(q.float(), keys, values, self.window, self.groups)
            else:
                retrieved = dilated_sparse_attention(q.float(), keys, values, self.offsets, self.groups)
            mix = (
                torch.einsum("bhtd,hd->bht", F.normalize(q.float(), dim=-1), self.retrieval_w)
                + self.retrieval_b[None, :, None]
            ).sigmoid().unsqueeze(-1)
            recurrent = mix * retrieved + (1.0 - mix) * recurrent
            keep = self._retrieval_keep()
            new_state.keys = keys[:, :, -keep:].clone() if keep else None
            new_state.values = values[:, :, -keep:].clone() if keep else None

        return (recurrent, new_state) if return_state else recurrent

    def recurrent_state_bytes(self, batch_size: int = 1):
        total = self.fast.recurrent_state_bytes(batch_size)
        if self.slow is not None:
            total += self.slow.recurrent_state_bytes(batch_size)
        keep = self._retrieval_keep()
        total += batch_size * 2 * keep * self.num_kv_heads * self.head_dim * self.fast.wq.element_size()
        return total

    def retrieval_pairs_per_token(self):
        if self.retrieval == "none":
            return 0
        if self.retrieval == "dense":
            return self.window
        return len(self.offsets)
