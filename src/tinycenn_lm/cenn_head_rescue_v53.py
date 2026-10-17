"""CeNN V5.3: query-memory cross-calibration for Qwen3.5 head rescue.

Research reference, not a speedup claim.  Preserves all V5.2 behavior and
pretrained original Qwen projections.  A zero-output-initialized cross-readout
adds a strictly pointwise, causal correction using the current query AND the
causally computed output.  Exact initialization parity allows controlled
warm-start ablations.  This is NOT fused and not a cached-Qwen generation core.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cenn_head_rescue_v52 import CeNNHeadRescueV52
from .cenn_deltacell_v3 import DeltaCellV3State


class CeNNHeadRescueV53(CeNNHeadRescueV52):
    """Output-conditioned cross correction and per-head output calibration.

    q: [B, Hq, T, D]; output has the same shape.  Uses no future tokens.
    Constructor flags support independent ablations, with identical module keys
    so V5.2 weights can be loaded strictly apart from the new V5.3 parameters.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 64, chunk_size: int = 32,
                 local_window: int = 128, correction_rank: int = 16,
                 query_router: bool = True, query_correction: bool = True,
                 post_rank: int = 16, post_correction: bool = True,
                 cross_rank: int = 16, cross_correction: bool = True,
                 output_calibration: bool = True):
        if not 1 <= cross_rank <= head_dim:
            raise ValueError('cross_rank must be in [1, head_dim]')
        super().__init__(num_heads, num_kv_heads, head_dim, feature_dim,
                         chunk_size, local_window, correction_rank, query_router,
                         query_correction, post_rank, post_correction)
        self.cross_rank = int(cross_rank)
        self.cross_correction_enabled = bool(cross_correction)
        self.output_calibration_enabled = bool(output_calibration)
        # Cross features: query and previously computed causal local/recurrent output.
        self.cross_down = nn.Parameter(torch.empty(num_heads, 2 * head_dim, cross_rank))
        self.cross_up = nn.Parameter(torch.zeros(num_heads, cross_rank, head_dim))
        self.cross_log_gain = nn.Parameter(torch.full((num_heads,), -2.0))
        self.output_log_scale = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        nn.init.normal_(self.cross_down, std=1.0 / math.sqrt(2 * head_dim))

    @property
    def config(self):
        return {**super().config,
                'cross_rank': self.cross_rank,
                'cross_correction': self.cross_correction_enabled,
                'output_calibration': self.output_calibration_enabled}

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: DeltaCellV3State | None = None,
                return_state: bool = False):
        out, new_state = super().forward(q, k, v, state=state, return_state=True)
        with torch.autocast(device_type=q.device.type, enabled=False):
            out = out.float()
            if self.cross_correction_enabled:
                cross = torch.cat([F.normalize(q.float(), dim=-1),
                                   F.normalize(out, dim=-1)], dim=-1)
                small = F.silu(torch.einsum('bhtd,hdr->bhtr', cross,
                                           self.cross_down.float()))
                delta = torch.einsum('bhtr,hrd->bhtd', small, self.cross_up.float())
                gain = self.cross_log_gain.float().clamp(-6, 2).exp()[None, :, None, None]
                out = out + gain * delta
            if self.output_calibration_enabled:
                scale = self.output_log_scale.float().clamp(-0.25, 0.25).exp()
                out = out * scale[None, :, None, :] + self.output_bias.float()[None, :, None, :]
        return (out, new_state) if return_state else out


def warmstart_v52_to_v53(core: CeNNHeadRescueV53, old_state: dict) -> dict:
    """Require ALL old keys while permitting only the five new V5.3 tensors."""
    expected = {'cross_down', 'cross_up', 'cross_log_gain',
                'output_log_scale', 'output_bias'}
    incompatible = core.load_state_dict(old_state, strict=False)
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError('V5.2 checkpoint incompatible with V5.3: ' + repr(incompatible))
    return {'loaded_keys': len(old_state), 'new_keys': sorted(expected)}
