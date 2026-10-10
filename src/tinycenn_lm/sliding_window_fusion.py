"""ModernBERT SlidingWindowFusion: exact finite-window SDPA plus gated CeNN-style local mixers.

An experiment in approximating the *sliding* ModernBERT layers (not the global
layers). It preserves pretrained QKV, RoPE and output projection. The 64-radius
configuration retains the teacher's 129-position local receptive field, but
remains windowed attention and is NOT an attention-free replacement.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def valid_tokens(mask: Tensor | None, hidden: Tensor) -> Tensor:
    b, t = hidden.shape[:2]
    if mask is None:
        return torch.ones(b, t, dtype=torch.bool, device=hidden.device)
    if mask.ndim == 2:
        return mask.bool()
    if mask.ndim == 4:
        allowed = mask if mask.dtype == torch.bool else mask > -1e4
        return (allowed.any(-1).any(1) & allowed.any(-2).any(1)).bool()
    raise ValueError(f"Unrecognized attention mask shape {tuple(mask.shape)}")


def _identity_depthwise(channels: int, kernel: int) -> Tensor:
    weight = torch.zeros(channels, 1, kernel)
    weight[:, 0, kernel // 2] = 1.0
    return weight


def local_sliding_sdpa(
    q: Tensor, k: Tensor, v: Tensor, valid: Tensor, *,
    radius: int, block_size: int = 64
) -> Tensor:
    """O(T * (block_size + 2*radius)) memory local SDPA, exact ±radius semantics.

    Padding keys are excluded, and invalid query outputs are zero. Use this
    kernel as an alternative to the teacher's full T-by-T additive mask.
    """
    b, h, t, d = q.shape
    w = int(radius)
    bs = int(block_size)
    if w < 0 or bs < 1:
        raise ValueError("radius must be >=0, block_size >=1")
    blocks = math.ceil(t / bs)
    tail = blocks * bs - t
    # [B,H,blocks,BS,D] and [B,H,blocks,BS+2R,D]
    qp = F.pad(q, (0, 0, 0, tail)).reshape(b, h, blocks, bs, d)
    def keys_in_blocks(x):
        return F.pad(x, (0, 0, w, w + tail)).unfold(
            2, bs + 2*w, bs
        ).transpose(-1, -2)
    kp, vp = keys_in_blocks(k), keys_in_blocks(v)
    # SDPA bool mask: True means this key is permitted.
    kv = F.pad(valid, (w, w + tail)).unfold(1, bs + 2*w, bs)
    qi = torch.arange(bs, device=q.device)[:, None]
    ki = torch.arange(bs + 2*w, device=q.device)[None, :]
    band = (ki >= qi) & (ki <= qi + 2*w)
    allowed = (band[None, None, None, :, :]
               & kv[:, None, :, None, :])
    def flatten_blocks(x):
        return x.transpose(1, 2).reshape(
            b * blocks, h, x.shape[-2], d
        )
    allowed = allowed.transpose(1, 2).reshape(
        b * blocks, 1, bs, bs + 2*w
    )
    out = F.scaled_dot_product_attention(
        flatten_blocks(qp), flatten_blocks(kp), flatten_blocks(vp),
        attn_mask=allowed, dropout_p=0.0
    )
    out = out.reshape(b, blocks, h, bs, d).transpose(1, 2)
    return (out.reshape(b, h, blocks*bs, d)[:, :, :t]
            * valid[:, None, :, None].to(out.dtype))


def grouped_value_conv(v: Tensor, weight: Tensor, dilation: int) -> Tensor:
    b, h, t, d = v.shape
    k = weight.shape[-1]
    padding = dilation * (k - 1) // 2
    z = v.permute(0, 1, 3, 2).reshape(b, h*d, t)
    z = F.conv1d(z, weight.to(z.dtype), padding=padding,
                 dilation=dilation, groups=h*d)
    return z.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


class SlidingWindowFusion(nn.Module):
    """Learned per-head, per-token mixture of exact local window and local convs.

    A ±64 window matches ModernBERT's default native local attention radius.
    For radii <64, the context is genuinely truncated; no hidden teacher-mask
    attention path is available. Multiscale convolutions may recover some context.

    Fixed pretrained QKV/RoPE, tiny Wo calibration, FP32-master route weights.
    """

    architecture = "sliding_window_fusion_v30"

    def __init__(self, original: nn.Module, *,
                 radius: int = 64, block_size: int = 64):
        super().__init__()
        if not hasattr(original, "local_attention"):
            raise TypeError("Expected original ModernBERT attention")
        a, z = tuple(original.local_attention)
        if (a, z) == (-1, -1) or a != z:
            raise ValueError("SlidingWindowFusion accepts sliding attention only")
        if not (0 <= radius <= a):
            raise ValueError(f"Candidate radius {radius} exceeds native ±{a}")
        self.config = original.config
        self.layer_idx = int(getattr(original, "layer_id", -1))
        self.num_heads = int(original.num_heads)
        self.head_dim = int(original.head_dim)
        self.hidden_size = self.num_heads * self.head_dim
        self.radius = int(radius)
        self.native_radius = int(a)
        self.block_size = int(block_size)
        self.Wqkv = copy.deepcopy(original.Wqkv)
        self.Wo = copy.deepcopy(original.Wo)
        self.out_drop = copy.deepcopy(original.out_drop)
        self.rotary_emb = copy.deepcopy(original.rotary_emb)
        self.Wqkv.requires_grad_(False)
        self.rotary_emb.requires_grad_(False)
        channels = self.num_heads * self.head_dim
        self.short_weight = nn.Parameter(_identity_depthwise(channels, 7))
        self.dilated_weight = nn.Parameter(_identity_depthwise(channels, 9))
        self.wide_weight = nn.Parameter(_identity_depthwise(channels, 9))
        # 99.7% nominal route mass on the exact local attention path.
        # A small but nonzero residual path can be learned when W is narrowed.
        self.route_logits = nn.Parameter(
            torch.tensor([4.0, -3.0, -3.0, -3.0]).repeat(self.num_heads, 1)
        )
        self.route_query = nn.Parameter(
            torch.zeros(self.num_heads, 4, self.head_dim)
        )
        self.local_log_temperature = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, hidden_states: Tensor, position_embeddings=None,
                attention_mask=None, sliding_window_mask=None,
                position_ids=None, **kwargs):
        b, t, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            b, t, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.transpose(3, 1).unbind(2)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        valid = valid_tokens(attention_mask, hidden_states)
        key_mask = valid[:, None, :, None].to(v.dtype)
        # Distinct from native dense mask: no T×T tensor is constructed.
        q, k, v = q * key_mask, k * key_mask, v * key_mask
        temp = self.local_log_temperature.clamp(-0.25, 0.25).exp()
        local = local_sliding_sdpa(
            q * temp[None, :, None, None], k, v, valid,
            radius=self.radius, block_size=self.block_size
        )
        c7 = grouped_value_conv(v, self.short_weight, dilation=1)
        c9 = grouped_value_conv(v, self.dilated_weight, dilation=4)
        cwide = grouped_value_conv(v, self.wide_weight, dilation=8)
        # Dynamic, per-token routing; all branches share the current true
        # input features, so no teacher outputs leak into the student.
        qnorm = F.normalize(q, dim=-1)
        dynamic = torch.einsum("bhtd,hkd->bhtk", qnorm, self.route_query)
        gate = F.softmax(
            self.route_logits.to(q.dtype)[None, :, None, :] + dynamic, -1
        )
        fused = (gate[..., 0:1]*local
                 + gate[..., 1:2]*c7
                 + gate[..., 2:3]*c9
                 + gate[..., 3:4]*cwide) * key_mask
        out = self.Wo(fused.transpose(1, 2).reshape(
            b, t, self.hidden_size
        ).to(hidden_states.dtype))
        return (self.out_drop(out),)

    def config_dict(self):
        return {
            "architecture": self.architecture,
            "layer_idx": self.layer_idx,
            "radius": self.radius,
            "native_radius": self.native_radius,
            "block_size": self.block_size,
            "window_positions": 2*self.radius+1,
            "teacher_window_positions": 2*self.native_radius+1,
            "uses_local_sdpa": True,
            "attention_free": False,
            "full_context_dense_attention": False,
            "receptive_field_conv": [7, 33, 65],
        }
