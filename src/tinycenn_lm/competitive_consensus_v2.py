"""CCM-v2: bounded, causal multi-cell consensus readout for feasibility testing.

This is a fully independent neural layer, not a hybrid attention/DeltaNet module.
However, its log-mass cellular merge is mathematically equivalent to normalized
content-selective pooling over the *bounded cell tape* (i.e., local attention).
Consequently, it must NOT be advertised as a novel alternative to attention.

Use as a stronger conservative-control than hard winner-take-all CCM; optimize
cellular mixing only after held-out language and speed evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ConsensusConfig:
    hidden_size: int = 960
    slots: int = 128
    tag_width: int = 64
    cell_width: int = 256
    score_scale: float = 2.0
    max_score_scale: float = 10.0
    init_gain: float = 0.1

    def validate(self):
        if self.slots < 2 or self.slots & (self.slots - 1):
            raise ValueError("slots must be a power of two, >=2")
        if min(self.hidden_size, self.tag_width, self.cell_width) < 1:
            raise ValueError("widths must be >0")
        if not (0 < self.score_scale <= self.max_score_scale):
            raise ValueError("invalid score scale")
        if not 0 < self.init_gain <= 1:
            raise ValueError("init_gain must be in (0,1]")


@dataclass
class ConsensusState:
    tags: Tensor        # last at most `slots` *projected* keys
    values: Tensor      # last at most `slots` *projected* values


class CompetitiveConsensusCeNN(nn.Module):
    """Causal content-addressed cellular memory with log-mass conservative merges.

    No quadratic T-by-T matrix or native attention kernels; for fixed slots K,
    implementation costs O(batch*T*K*(F+V)) and uses O(T*K*(F+V)) temporary
    memory during prefill. The bounded tape means this cannot imitate arbitrary
    full-context attention for T>K. Multi-cell merging preserves a probability
    simplex but *is attention-equivalent over its currently resident cells*.
    """
    def __init__(self, config: ConsensusConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.hidden_size
        self.query_proj = nn.Linear(d, config.tag_width, bias=False)
        self.key_proj = nn.Linear(d, config.tag_width, bias=False)
        self.value_proj = nn.Linear(d, config.cell_width, bias=False)
        self.out_proj = nn.Linear(config.cell_width, d, bias=False)
        self.log_gain = nn.Parameter(torch.tensor(math.log(config.init_gain)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(config.score_scale)))

    @property
    def cell_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def _consensus(self, q: Tensor, keys: Tensor, values: Tensor, valid: Tensor) -> Tensor:
        """Shapes q[B,T,F], keys[B,T,K,F], values[B,T,K,C], valid[T,K]."""
        score = (F.normalize(q, dim=-1)[:, :, None, :]
                 * F.normalize(keys, dim=-1)).sum(-1)
        scale = self.log_scale.exp().clamp(max=self.config.max_score_scale)
        scores = (score * scale).masked_fill(~valid[None, :, :], -1e4)
        state = values
        # Every merge is a local pairwise conductance weighted by conserved mass.
        # logaddexp rather than max ensures diffuse content is not discarded.
        while state.shape[-2] > 1:
            l, r = scores[..., 0::2], scores[..., 1::2]
            g = torch.sigmoid(r-l)[..., None]
            state = (1-g)*state[..., 0::2, :] + g*state[..., 1::2, :]
            scores = torch.logaddexp(l, r)
        return state[..., 0, :]

    def _project(self, x: Tensor):
        x = x.float()
        return self.query_proj(x), self.key_proj(x), self.value_proj(x)

    def forward(self, hidden_states: Tensor, state: ConsensusState | None = None,
                *, return_state: bool = False):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("input must be [batch, tokens, hidden_size]")
        b,t,h = hidden_states.shape
        if not t:
            raise ValueError("empty sequence")
        q,k,v = self._project(hidden_states)
        if state is not None:
            if state.tags.ndim != 3 or state.values.ndim != 3:
                raise ValueError("invalid state ranks")
            if state.tags.shape[0] != b or state.values.shape[0] != b:
                raise ValueError("state batch mismatch")
            if state.tags.shape[-1] != self.config.tag_width or state.values.shape[-1] != self.config.cell_width:
                raise ValueError("state width mismatch")
            if state.tags.shape[1] != state.values.shape[1] or state.tags.shape[1] > self.config.slots:
                raise ValueError("state length mismatch")
            history = state.tags.shape[1]
            all_k = torch.cat((state.tags, k), dim=1)
            all_v = torch.cat((state.values, v), dim=1)
        else:
            history=0
            all_k,all_v=k,v
        slots=self.config.slots
        # Prefix padding and unfolding implement a causal cellular tape in
        # parallel for all queries. Streaming chunks see exact same resident
        # cells after projecting the saved states (tested explicitly).
        def windows(seq):
            padded=F.pad(seq.transpose(1,2), (slots-1,0))
            return padded.unfold(-1, slots,1).permute(0,2,3,1)
        all_k_win=windows(all_k)[:,history:history+t]
        all_v_win=windows(all_v)[:,history:history+t]
        idx=torch.arange(history,history+t,device=q.device)[:,None]
        offset=torch.arange(slots,device=q.device)[None,:]
        valid=idx-slots+1+offset>=0
        y=self._consensus(q, all_k_win, all_v_win, valid)
        y=self.out_proj(y)*self.log_gain.exp()
        y=y.to(hidden_states.dtype)
        if return_state:
            return y,ConsensusState(tags=all_k[:,-slots:],values=all_v[:,-slots:])
        return y
