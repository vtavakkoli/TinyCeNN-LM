"""Research adaptations of KDA and Gated DeltaNet-2 for frozen GQA projections.

Equations and deviations are documented in RESEARCH_LAYERS.md. This is an
independent PyTorch reference implementation, not the authors' fused kernels.
All state updates use float32 by default; no T-by-T matrix is constructed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

VARIANTS = ("cenn_kda", "cenn_delta2", "cenn_delta2_window")


@dataclass
class DeltaState:
    memory: Tensor
    keys: Tensor | None = None
    values: Tensor | None = None

    @property
    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size()
                   for x in (self.memory, self.keys, self.values) if x is not None)


def delta_recurrence(q, k, z, erase, log_decay, memory, groups, chunk_size=32):
    """Asymmetric delta update, with a triangular solve inside each bounded chunk.

    Sbar_t = diag(exp(log_decay_t)) S_{t-1}
    S_t = Sbar_t + k_t (z_t - erase_t^T Sbar_t)^T.
    q: [B,Hq,T,F]; other features: [B,Hkv,T,F]; z: [B,Hkv,T,Dv].
    The chunk solve is algebraically identical to the token recurrence.
    """
    if not 1 <= chunk_size <= 32:
        raise ValueError("chunk_size must be in [1,32] for the bounded-decay reference")
    outputs = []
    for start in range(0, q.shape[2], chunk_size):
        stop = min(start + chunk_size, q.shape[2])
        kc, ec, zc = k[:, :, start:stop], erase[:, :, start:stop], z[:, :, start:stop]
        decay = log_decay[:, :, start:stop].cumsum(dim=2).exp()
        write = kc / decay
        read = ec * decay
        n = stop - start
        interaction = torch.matmul(read, write.transpose(-1, -2))
        lower = interaction.tril(diagonal=-1)
        system = lower + torch.eye(n, device=q.device, dtype=q.dtype)
        residual = torch.linalg.solve_triangular(
            system, zc - torch.matmul(read, memory), upper=False, unitriangular=True
        )
        qr = q[:, :, start:stop] * decay.repeat_interleave(groups, dim=1)
        wr = write.repeat_interleave(groups, dim=1)
        rr = residual.repeat_interleave(groups, dim=1)
        old = torch.matmul(qr, memory.repeat_interleave(groups, dim=1))
        within = torch.matmul(qr, wr.transpose(-1, -2)).tril()
        outputs.append(old + torch.matmul(within, rr))
        memory = decay[:, :, -1, :, None] * (
            memory + torch.matmul(write.transpose(-1, -2), residual)
        )
    return torch.cat(outputs, dim=2), memory


def delta_token_reference(q, k, z, erase, log_decay, memory, groups):
    """Independent, slow tokenwise oracle used for numerical validation."""
    outputs = []
    for t in range(q.shape[2]):
        memory = log_decay[:, :, t].exp().unsqueeze(-1) * memory
        old = torch.einsum("bhf,bhfv->bhv", erase[:, :, t], memory)
        memory = memory + k[:, :, t, :, None] * (z[:, :, t] - old).unsqueeze(-2)
        outputs.append(torch.einsum(
            "bhf,bhfv->bhv", q[:, :, t], memory.repeat_interleave(groups, dim=1)
        ))
    return torch.stack(outputs, dim=2), memory


def local_window_attention(q, k, v, window, groups):
    """Exact causal softmax over at most window keys, including the current token.

    k/v may include up to window-1 cached prefix tokens. Unfold creates views;
    scores occupy [B,Hq,T,window], never [B,Hq,T,T].
    """
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    total, t = k.shape[2], q.shape[2]
    kw = F.pad(k, (0, 0, window - 1, 0)).unfold(2, window, 1)
    vw = F.pad(v, (0, 0, window - 1, 0)).unfold(2, window, 1)
    kw = kw[:, :, -t:].transpose(-1, -2)
    vw = vw[:, :, -t:].transpose(-1, -2)
    scores = torch.einsum("bhtd,bhtwd->bhtw", q, kw) / math.sqrt(q.shape[-1])
    positions = torch.arange(total - t, total, device=q.device)
    offsets = torch.arange(window, device=q.device) - window + 1
    valid = positions[:, None] + offsets[None, :] >= 0
    weights = scores.masked_fill(~valid[None, None], float("-inf")).softmax(-1)
    return torch.einsum("bhtw,bhtwd->bhtd", weights, vw)


class ResearchCeNNLayer(nn.Module):
    """A GQA-compatible memory array with learned local state updates.

    Inputs are frozen, post-RoPE Q/K and frozen V. K/V heads share memory exactly
    as indicated by num_kv_heads. Query feature maps remain head-specific.
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, feature_dim=64,
                 variant="cenn_delta2", window=32, chunk_size=32):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        if min(num_heads, num_kv_heads, head_dim, feature_dim, window) < 1:
            raise ValueError("dimensions and window must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if not 1 <= chunk_size <= 32:
            raise ValueError("chunk_size must be in [1,32]")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim, self.feature_dim = head_dim, feature_dim
        self.groups = num_heads // num_kv_heads
        self.variant, self.window, self.chunk_size = variant, window, chunk_size
        base = torch.zeros(num_kv_heads, feature_dim, head_dim)
        for h in range(num_kv_heads):
            if feature_dim == head_dim:
                base[h] = torch.eye(head_dim)
            else:
                nn.init.orthogonal_(base[h])
        self.wk = nn.Parameter(base.clone())
        self.wq = nn.Parameter(base.repeat_interleave(self.groups, dim=0).clone())
        self.forget_w = nn.Parameter(torch.zeros(num_kv_heads, feature_dim, head_dim))
        # log(alpha) in [-0.25,0], initially -0.01. Bounds keep chunk rescaling safe.
        self.forget_b = nn.Parameter(torch.full((num_kv_heads, feature_dim),
                                               math.log(0.04 / 0.96)))
        erase_dim = 1 if variant == "cenn_kda" else feature_dim
        self.erase_w = nn.Parameter(torch.zeros(num_kv_heads, erase_dim, head_dim))
        self.erase_b = nn.Parameter(torch.full((num_kv_heads, erase_dim), -1.0))
        if variant != "cenn_kda":
            self.write_w = nn.Parameter(torch.zeros(num_kv_heads, head_dim, head_dim))
            self.write_b = nn.Parameter(torch.full((num_kv_heads, head_dim), -1.0))
        self.log_gain = nn.Parameter(torch.zeros(num_heads))
        if variant == "cenn_delta2_window":
            self.mix_w = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.mix_b = nn.Parameter(torch.zeros(num_heads))

    @property
    def config(self):
        return dict(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, feature_dim=self.feature_dim,
                    variant=self.variant, window=self.window, chunk_size=self.chunk_size)

    @staticmethod
    def _project(x, weight, bias):
        return torch.einsum("bhtd,hfd->bhtf", x, weight) + bias[None, :, None]

    def features(self, q, k, v):
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        qp = F.normalize(torch.einsum("bhtd,hfd->bhtf", qn, self.wq), dim=-1)
        kp = F.normalize(torch.einsum("bhtd,hfd->bhtf", kn, self.wk), dim=-1)
        log_decay = -0.25 * self._project(kn, self.forget_w, self.forget_b).sigmoid()
        erase_gate = self._project(kn, self.erase_w, self.erase_b).sigmoid()
        if self.variant == "cenn_kda":
            write_gate = erase_gate
        else:
            write_gate = self._project(
                F.normalize(v, dim=-1), self.write_w, self.write_b
            ).sigmoid()
        return qp, kp, v * write_gate, kp * erase_gate, log_decay

    def forward(self, q, k, v, state=None, return_state=False, implementation="chunk"):
        if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
            raise ValueError("expected [batch,heads,time,dim] Q/K/V with equal K/V shapes")
        if (q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
                or q.shape[1] != self.num_heads or k.shape[1] != self.num_kv_heads
                or q.shape[-1] != self.head_dim or q.shape[2] < 1):
            raise ValueError("Q/K/V shapes do not match the layer configuration")
        q, k, v = (x.to(self.wq.dtype) for x in (q, k, v))
        b = q.shape[0]
        if state is None:
            state = DeltaState(q.new_zeros(b, self.num_kv_heads,
                                          self.feature_dim, self.head_dim))
        qp, kp, z, erase, log_decay = self.features(q, k, v)
        if implementation == "chunk":
            output, memory = delta_recurrence(
                qp, kp, z, erase, log_decay, state.memory, self.groups, self.chunk_size
            )
        elif implementation == "token":
            output, memory = delta_token_reference(
                qp, kp, z, erase, log_decay, state.memory, self.groups
            )
        else:
            raise ValueError("implementation must be chunk or token")
        output = output * self.log_gain.clamp(-4, 4).exp()[None, :, None, None]
        new_state = DeltaState(memory)
        if self.variant == "cenn_delta2_window":
            keys = k if state.keys is None else torch.cat((state.keys, k), dim=2)
            values = v if state.values is None else torch.cat((state.values, v), dim=2)
            local = local_window_attention(q, keys, values, self.window, self.groups)
            mix = (torch.einsum("bhtd,hd->bht", F.normalize(q, dim=-1), self.mix_w)
                   + self.mix_b[None, :, None]).sigmoid().unsqueeze(-1)
            output = mix * local + (1.0 - mix) * output
            keep = self.window - 1
            new_state.keys = keys[:, :, -keep:].contiguous() if keep else keys[:, :, :0]
            new_state.values = values[:, :, -keep:].contiguous() if keep else values[:, :, :0]
        return (output, new_state) if return_state else output

    def recurrent_state_bytes(self, batch_size=1):
        # Matrix and cache both use this module's floating-point dtype.
        elements = self.num_kv_heads * self.feature_dim * self.head_dim
        if self.variant == "cenn_delta2_window":
            elements += 2 * (self.window - 1) * self.num_kv_heads * self.head_dim
        return batch_size * elements * self.wq.element_size()


def softmax_reference(q, k, v, groups):
    return F.scaled_dot_product_attention(
        q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1),
        dropout_p=0.0, is_causal=True
    )
