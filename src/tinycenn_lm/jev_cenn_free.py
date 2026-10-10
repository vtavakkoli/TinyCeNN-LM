"""Jev-CeNN attention-free feasibility lab: non-attentional sequence mixers.

No dot product between token queries and keys; no scaled_dot_product_attention,
no token-to-token affinity, no token-softmax, no linearized/landmark attention.
All token mixing is via depthwise convolutions, fixed average pooling and
query-independent cumulative memory statistics (O(T * hidden)).

This is a research candidate, not a demonstrated high-fidelity conversion.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _valid_tokens(attention_mask: Tensor | None, x: Tensor) -> Tensor:
    b, t = x.shape[:2]
    if attention_mask is None:
        return torch.ones(b, t, device=x.device, dtype=torch.bool)
    if attention_mask.ndim == 2:
        return attention_mask.bool()
    if attention_mask.ndim != 4:
        raise ValueError(f"Expected B×T or B×1×T×T mask, got {attention_mask.shape}")
    allowed = (attention_mask if attention_mask.dtype == torch.bool
               else attention_mask > -1e4)
    return allowed.any(-1).any(1) & allowed.any(-2).any(1)


def _depthwise_conv(x: Tensor, weight: Tensor, dilation: int) -> Tensor:
    """Bidirectional depthwise group convolution. x: [B,H,T,D]."""
    b, h, t, d = x.shape
    length = int(weight.shape[-1])
    y = x.permute(0, 1, 3, 2).reshape(b, h*d, t)
    y = F.conv1d(y, weight.to(y.dtype),
                 padding=(length//2)*dilation,
                 dilation=dilation, groups=h*d)
    return y.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


def _kernel(channels: int, length: int) -> Tensor:
    # Smooth base kernel with small identity shortcut, not a trivial copy.
    result = torch.full((channels, 1, length), 0.50/length)
    result[:, 0, length//2] += 0.50
    return result


def _bidirectional_memory(x: Tensor, valid: Tensor) -> Tensor:
    """Prefix/suffix and whole-sequence averages; no Q·K or learned attention.

    Works on padded batches and gives every valid position access to context
    from the entire sequence. Calculations are O(T·C).
    """
    m = valid[:, None, :, None].to(x.dtype)
    z = x * m
    num_fwd = z.cumsum(2)
    den = m.cumsum(2).clamp_min(1)
    prefix = num_fwd / den
    suffix = (z.flip(2).cumsum(2) / m.flip(2).cumsum(2).clamp_min(1)).flip(2)
    all_mean = z.sum(2, keepdim=True) / m.sum(2, keepdim=True).clamp_min(1)
    return (0.25*prefix + 0.25*suffix + 0.50*all_mean) * m


def _hierarchical_mean(x: Tensor, valid: Tensor, stride: int = 32) -> Tensor:
    """Coarse fixed-stride window averages upsampled to per-token features."""
    b,h,t,d = x.shape
    c = h*d
    m = valid[:, None, :].to(x.dtype)
    u = x.permute(0,1,3,2).reshape(b,c,t) * m
    # Pool sum and valid count (not padded-token average).
    window = min(int(stride), max(1,t))
    numer = F.avg_pool1d(u, kernel_size=window, stride=window,
                         ceil_mode=True)
    denom = F.avg_pool1d(m, kernel_size=window, stride=window,
                         ceil_mode=True)
    average = numer / denom.clamp_min(1e-6)
    up = F.interpolate(average, size=t, mode="nearest")
    return up.reshape(b,h,d,t).permute(0,1,3,2) * valid[:,None,:,None]


class JevCeNNFreeMixer(nn.Module):
    """True attention-free candidate for a single ModernBERT attention module.

    Variants:
      - local: 7/33/65-token CeNN-style depthwise mixers
      - multiscale: adds a 129-token dilated branch
      - memory: multiscale plus symmetric cumulative memory and coarse pooling

    Frozen pretrained QKV and Wo initialization ensure fair teacher comparison.
    Q/K may be used as pointwise features, never QK similarity scores.
    """

    architecture = "jev_cenn_free_mixer_v1"
    VARIANTS = ("local", "multiscale", "memory")

    def __init__(self, original: nn.Module, *, variant="memory", coarse_stride=32):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant}; use {self.VARIANTS}")
        if not hasattr(original, "Wqkv") or not hasattr(original, "Wo"):
            raise TypeError("Expected ModernBERT attention with Wqkv and Wo")
        self.variant = variant
        self.coarse_stride = int(coarse_stride)
        if self.coarse_stride < 1:
            raise ValueError("coarse_stride must be >=1")
        self.layer_idx = int(getattr(original, "layer_id", -1))
        self.num_heads = int(original.num_heads)
        self.head_dim = int(original.head_dim)
        self.hidden_size = self.num_heads*self.head_dim
        self.original_local_attention = tuple(original.local_attention)
        self.Wqkv = copy.deepcopy(original.Wqkv)
        self.Wo = copy.deepcopy(original.Wo)
        self.out_drop = copy.deepcopy(original.out_drop)
        # No RoPE used: convolutions encode relative position explicitly;
        # original q/k enter only as pointwise gates, never token affinities.
        self.Wqkv.requires_grad_(False)
        width = self.hidden_size
        self.kernel7 = nn.Parameter(_kernel(width, 7))
        self.kernel33 = nn.Parameter(_kernel(width, 9))  # dilation 4
        self.kernel65 = nn.Parameter(_kernel(width, 9))  # dilation 8
        self.kernel129 = nn.Parameter(_kernel(width, 9)) # dilation 16
        self.local_gate = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.content_gain = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        branches = 3 if variant=="local" else (4 if variant=="multiscale" else 6)
        self.route_logits = nn.Parameter(torch.zeros(self.num_heads, branches))
        # Per-token branch routes without token-to-token similarity.
        self.route_weight = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, branches))
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    def forward(self, hidden_states: Tensor, attention_mask=None,
                sliding_window_mask=None, position_ids=None,
                position_embeddings=None, **kwargs):
        b,t,_ = hidden_states.shape
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:,None,:,None].to(hidden_states.dtype)
        qkv = self.Wqkv(hidden_states).reshape(
            b,t,3,self.num_heads,self.head_dim
        )
        q,k,v = qkv.transpose(3,1).unbind(2)
        # v is the pretrained output-relevant feature stream;
        # query/key are pointwise gates only (not attention).
        content = v + 0.10*torch.tanh(self.content_gain)[None,:,None,:]*(q+k-2*v)
        content = content*mask
        short = _depthwise_conv(content,self.kernel7,1)
        medium = _depthwise_conv(content,self.kernel33,4)
        long = _depthwise_conv(content,self.kernel65,8)
        branches = [short, medium, long]
        if self.variant != "local":
            branches.append(_depthwise_conv(content,self.kernel129,16))
        if self.variant == "memory":
            branches += [
                _bidirectional_memory(content,valid),
                _hierarchical_mean(content,valid,self.coarse_stride),
            ]
        # Route only across fixed branches, not sequence positions.
        pointwise = torch.tanh(q+k) * torch.sigmoid(
            self.local_gate[None,:,None,:]
        )
        dynamic = torch.einsum("bhtd,hdm->bhtm",pointwise,self.route_weight)
        weights = torch.sigmoid(
            self.route_logits[None,:,None,:] + dynamic
        )
        weights = weights/weights.sum(dim=-1,keepdim=True).clamp_min(1e-6)
        stacked = torch.stack(branches,dim=-2)  # [B,H,T,branches,D]
        mixed = (weights[...,None]*stacked).sum(-2)
        gain = 0.5 + torch.sigmoid(self.output_gain)[None,:,None,:]
        mixed = (mixed*gain)*mask
        out = self.Wo(mixed.transpose(1,2).reshape(b,t,self.hidden_size))
        return (self.out_drop(out),)

    def config_dict(self):
        return {
            "architecture": self.architecture,
            "layer_idx": self.layer_idx,
            "variant": self.variant,
            "coarse_stride": self.coarse_stride,
            "pretrained_qkv": True,
            "pretrained_wo": True,
            "local_receptive_fields": [7,33,65]
                + ([129] if self.variant!="local" else []),
            "global_memory": self.variant=="memory",
            "no_token_attention": True,
            "no_sdpa": True,
            "no_qk_similarity": True,
            "no_token_softmax": True,
            "linear_in_sequence_length": True,
        }
