"""CeNN V5.2: output-conditioned low-rank correction for partial Qwen head rescue.

The frozen Qwen projections/decoder and exact KV group are handled by
Qwen35HeadRescueV5. This core extends V5.1's causal recurrent + local readout
with a zero-initialized, *output-conditioned* low-rank adapter. It is a research
reference; no cached Qwen generation or fused kernels are provided.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cenn_head_rescue_v51 import CeNNQueryRescueV51
from .cenn_deltacell_v3 import DeltaCellV3State


class CeNNHeadRescueV52(CeNNQueryRescueV51):
    """Causal CeNN V5.1 with optional content-conditioned output calibration.

    The post adapter consumes only the current token's already-causal memory/local
    readout. Its zero-initialized output factor gives exact V5.1 warm-start parity
    at any context length *for the same local_window*.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 64, chunk_size: int = 32,
                 local_window: int = 64, correction_rank: int = 16,
                 query_router: bool = True, query_correction: bool = False,
                 post_rank: int = 16, post_correction: bool = True):
        if not 1 <= post_rank <= head_dim:
            raise ValueError('post_rank must be in [1,head_dim]')
        super().__init__(num_heads, num_kv_heads, head_dim, feature_dim,
                         chunk_size, local_window, correction_rank,
                         query_router, query_correction)
        self.post_rank = int(post_rank)
        self.post_correction_enabled = bool(post_correction)
        self.post_down = nn.Parameter(torch.empty(num_heads, head_dim, post_rank))
        self.post_up = nn.Parameter(torch.zeros(num_heads, post_rank, head_dim))
        self.post_log_gain = nn.Parameter(torch.full((num_heads,), -2.0))
        nn.init.normal_(self.post_down, std=1.0 / math.sqrt(head_dim))

    @property
    def config(self):
        return {**super().config, 'post_rank': self.post_rank,
                'post_correction': self.post_correction_enabled}

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: DeltaCellV3State | None = None,
                return_state: bool = False):
        out, updated = super().forward(q, k, v, state=state, return_state=True)
        if self.post_correction_enabled:
            with torch.autocast(device_type=q.device.type, enabled=False):
                x = F.normalize(out.float(), dim=-1)
                small = torch.einsum('bhtd,hdr->bhtr', x, self.post_down.float())
                small = F.silu(small)
                delta = torch.einsum('bhtr,hrd->bhtd', small, self.post_up.float())
                out = out.float() + self.post_log_gain.float().clamp(-6, 2).exp()[
                    None, :, None, None] * delta
        return (out, updated) if return_state else out


def warmstart_v51_to_v52(core: CeNNHeadRescueV52, v51_state: dict) -> dict:
    """Strictly load old V5.1 parameters, rejecting unexpected or missing keys."""
    missing_expected = {'post_down', 'post_up', 'post_log_gain'}
    incompatible = core.load_state_dict(v51_state, strict=False)
    if set(incompatible.missing_keys) != missing_expected or incompatible.unexpected_keys:
        raise ValueError(f'V5.1 checkpoint is incompatible: {incompatible}')
    return {'loaded_parameters': len(v51_state),
            'new_parameters': sorted(missing_expected)}