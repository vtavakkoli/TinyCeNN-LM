"""CeNN V5.1: query-conditioned local-memory routing and low-rank head calibration.

An experimental extension of CeNNDeltaCellV3, compatible with the existing
Qwen35HeadRescueV5 group-selective Qwen attention adapter. Does NOT substitute
native SDPA on the rescued heads, or support Transformers hybrid KV-cache.
Core state uses FP32 and causal fixed-width local attention. No fused kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cenn_deltacell_v3 import CeNNDeltaCellV3, DeltaCellV3State
from .research_layers import local_window_attention


class CeNNQueryRescueV51(CeNNDeltaCellV3):
    """Approximate GQA heads with delta-memory, learned local gate and q-only correction.

    The native projection stack is provided by Qwen35HeadRescueV5. Here only
    unrecovered KV groups are instantiated. Initializing from V5 checkpoint
    preserves trained projections and local mixing weights; new routing and
    correction parameters are initially neutral.

    ``state`` can be fed across arbitrary causal call boundaries; exact native
    Qwen cache-based generation remains unsupported by the external adapter.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 64, chunk_size: int = 32,
                 local_window: int = 64, correction_rank: int = 16,
                 query_router: bool = True, query_correction: bool = True):
        if local_window < 1:
            raise ValueError("V5.1 needs a positive local_window")
        if not 1 <= correction_rank <= head_dim:
            raise ValueError("correction_rank must be in [1,head_dim]")
        # Base produces only the matrix-memory readout. A separate local branch
        # below prevents V3's fixed mixing from being applied twice.
        super().__init__(num_heads, num_kv_heads, head_dim, feature_dim,
                         chunk_size, mode="none", local_window=0)
        self.query_window = int(local_window)
        self.correction_rank = int(correction_rank)
        self.query_router_enabled = bool(query_router)
        self.query_correction_enabled = bool(query_correction)
        # Same state key and initialization as the V5 DeltaCellV3 checkpoint.
        self.local_logits = nn.Parameter(torch.full((num_heads,), -1.5))
        self.router_w = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.router_bias = nn.Parameter(torch.zeros(num_heads))
        # Low-rank, query-only residual: zero-init output factor leaves warm
        # starts numerically identical to V5 Delta+Local immediately.
        self.corr_down = nn.Parameter(torch.empty(num_heads, head_dim, correction_rank))
        self.corr_up = nn.Parameter(torch.zeros(num_heads, correction_rank, head_dim))
        nn.init.normal_(self.corr_down, std=1 / math.sqrt(head_dim))
        self.corr_log_gain = nn.Parameter(torch.full((num_heads,), -1.5))

    @property
    def config(self):
        return dict(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, feature_dim=self.feature_dim,
                    chunk_size=self.chunk_size, local_window=self.query_window,
                    correction_rank=self.correction_rank,
                    query_router=self.query_router_enabled,
                    query_correction=self.query_correction_enabled)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: DeltaCellV3State | None = None,
                return_state: bool = False):
        if state is not None:
            if ((state.keys is None) != (state.values is None)):
                raise ValueError("local key/value cache must be present together")
            if state.keys is not None:
                n = int(state.keys.shape[2])
                required = (q.shape[0], self.num_kv_heads, n, self.head_dim)
                if (tuple(state.keys.shape) != required or tuple(state.values.shape) != required
                    or n > self.query_window - 1
                    or state.keys.device != q.device or state.values.device != q.device):
                    raise ValueError("invalid local key/value cache")
        # FP32 recurrence is preserved; no hidden attention over future tokens.
        memory_only = None if state is None else DeltaCellV3State(state.memory, state.position)
        recurrent, updated = super().forward(q, k, v, state=memory_only,
                                             return_state=True)
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            keys = kf if state is None or state.keys is None else torch.cat(
                (state.keys.float(), kf), dim=2)
            values = vf if state is None or state.values is None else torch.cat(
                (state.values.float(), vf), dim=2)
            local = local_window_attention(qf, keys, values,
                                           self.query_window, self.groups)
            logit = self.local_logits.float()[None, :, None]
            query_norm = F.normalize(qf, dim=-1)
            if self.query_router_enabled:
                logit = (logit + torch.einsum("bhtd,hd->bht", query_norm,
                                             self.router_w.float())
                         + self.router_bias.float()[None, :, None])
            mix = logit.sigmoid().unsqueeze(-1)
            out = (1 - mix) * recurrent.float() + mix * local
            if self.query_correction_enabled:
                down = torch.einsum("bhtd,hdr->bhtr", query_norm,
                                    self.corr_down.float())
                correction = torch.einsum("bhtr,hrd->bhtd", down,
                                          self.corr_up.float())
                out = out + self.corr_log_gain.float().clamp(-5, 2).exp()[
                    None, :, None, None] * correction
            keep = self.query_window - 1
            next_k = keys[:, :, -keep:].clone() if keep else keys[:, :, :0].clone()
            next_v = values[:, :, -keep:].clone() if keep else values[:, :, :0].clone()
            result_state = DeltaCellV3State(updated.memory, updated.position,
                                            next_k, next_v)
        return (out, result_state) if return_state else out


def warmstart_v5_delta_core(core: CeNNQueryRescueV51, old_state: dict) -> dict:
    """Load V5 delta checkpoint without silently hiding incompatible dimensions.

    The only missing parameters are the genuinely new V5.1 router/correction.
    V5 memory weights and trained local_logits must all be loaded intact.
    """
    expected_new = {"router_w", "router_bias", "corr_down", "corr_up", "corr_log_gain"}
    incompatible = core.load_state_dict(old_state, strict=False)
    if set(incompatible.missing_keys) != expected_new or incompatible.unexpected_keys:
        raise ValueError("V5 checkpoint mismatch: " + repr(incompatible))
    return {"loaded_from_v5": len(old_state), "new_parameters": sorted(expected_new)}
