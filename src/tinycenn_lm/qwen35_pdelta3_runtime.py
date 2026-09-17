"""Lightweight Qwen3.5 PDelta3/CLVR inference runtime.

This module intentionally contains only the architecture/runtime pieces required
for inference and standalone checkpoint reconstruction. It avoids datasets,
training utilities, optimizers, and experiment code.
"""
from __future__ import annotations

import copy
import math
import weakref
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from .pdelta3_frontier import FrontierPDelta3Layer


@dataclass(frozen=True)
class QwenPDelta3CLVRConfig:
    feature_dim: int = 96
    local_window: int = 32
    chunk_size: int = 32
    conv_kernel: int = 4
    state_dtype: str = "fp16"
    variant: str = "conv4_gdn2_clvr_f96"
    local_gate_init: float = 0.72
    warm_start_previous_core: bool = True

    def validate(self, cfg):
        if self.feature_dim < 16 or self.local_window < 1 or self.conv_kernel < 1:
            raise ValueError("invalid PDelta3-CLVR dimensions")
        if not 1 <= self.chunk_size <= 32:
            raise ValueError("chunk_size must be in [1,32]")
        if self.state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if int(cfg.num_attention_heads) % int(cfg.num_key_value_heads):
            raise ValueError("attention heads must be divisible by KV heads")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


def _text_config(model):
    return getattr(model.config, "text_config", model.config)


def full_attention_layers(model):
    kinds = list(getattr(_text_config(model), "layer_types", []))
    if not kinds:
        raise RuntimeError("Qwen3.5 config does not expose layer_types")
    return [i for i, kind in enumerate(kinds) if kind == "full_attention"]


def _repeat_kv(x, groups):
    return x.repeat_interleave(groups, dim=1)


class QwenPDelta3CLVRAttention(nn.Module):
    """Qwen3.5 full-attention replacement: Local-W + recurrent PDelta3/GDN2."""

    def __init__(self, original, cfg, config, layer_idx, previous_attention=None):
        super().__init__()
        config.validate(cfg)
        self.config = cfg
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", self.hidden_size // self.num_heads))
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = float(getattr(cfg, "attention_dropout", 0.0))
        self.is_causal = True
        self.local_window = int(config.local_window)
        object.__setattr__(
            self,
            "_previous_attention_ref",
            weakref.ref(previous_attention) if previous_attention is not None else None,
        )

        self.q_proj = copy.deepcopy(original.q_proj)
        self.k_proj = copy.deepcopy(original.k_proj)
        self.v_proj = copy.deepcopy(original.v_proj)
        self.o_proj = copy.deepcopy(original.o_proj)
        self.q_norm = copy.deepcopy(original.q_norm)
        self.k_norm = copy.deepcopy(original.k_norm)

        self.core = FrontierPDelta3Layer(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            feature_dim=config.feature_dim,
            variant=config.variant,
            chunk_size=config.chunk_size,
            conv_kernel=config.conv_kernel,
            state_dtype=config.state_dtype,
        )
        init = min(max(float(config.local_gate_init), 1e-4), 1 - 1e-4)
        logit = math.log(init / (1 - init))
        self.local_gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.local_gate_b = nn.Parameter(torch.full((self.num_heads,), logit))
        self.last_value = None

    def _previous_attention(self):
        ref = object.__getattribute__(self, "_previous_attention_ref")
        return None if ref is None else ref()

    def _local_attention(self, q, k, v, attention_mask):
        kh, vh = _repeat_kv(k, self.groups), _repeat_kv(v, self.groups)
        scores = torch.matmul(q.float(), kh.float().transpose(-2, -1)) * self.scaling
        t = q.shape[-2]
        qi = torch.arange(t, device=q.device)[:, None]
        kj = torch.arange(t, device=q.device)[None, :]
        allowed = (kj <= qi) & (kj >= qi - self.local_window + 1)
        bias = torch.zeros((t, t), device=q.device, dtype=scores.dtype)
        bias.masked_fill_(~allowed, torch.finfo(scores.dtype).min)
        scores = scores + bias[None, None]
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                scores = scores + attention_mask[..., :t, :t].float()
            elif attention_mask.ndim == 2:
                key_mask = 1.0 - attention_mask[:, None, None, :t].float()
                scores = scores + key_mask * torch.finfo(scores.dtype).min
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout:
            probs = F.dropout(probs, p=self.attention_dropout)
        return torch.matmul(probs, vh.to(probs.dtype))

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            raise ValueError("PDelta3 research replacement requires use_cache=False")
        shape = hidden_states.shape[:-1]
        qg = self.q_proj(hidden_states).view(*shape, self.num_heads, self.head_dim * 2)
        q, out_gate = torch.chunk(qg, 2, dim=-1)
        out_gate = out_gate.reshape(*shape, self.num_heads * self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(*shape, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*shape, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        self.last_value = v.detach()
        previous = self._previous_attention()
        routed_v = None
        if previous is not None and previous.last_value is not None and previous.last_value.shape == v.shape:
            routed_v = previous.last_value.to(device=v.device, dtype=v.dtype)
        if routed_v is None:
            routed_v = v

        global_out = self.core(q, k, v, routed_v=routed_v)
        local_out = self._local_attention(q, k, v, attention_mask)
        gate = torch.sigmoid(
            torch.einsum("bhtd,hd->bht", q.float(), self.local_gate_w.float())
            + self.local_gate_b.float()[None, :, None]
        ).to(q.dtype)
        mixed = gate[..., None] * local_out + (1 - gate[..., None]) * global_out
        out = mixed.transpose(1, 2).contiguous().reshape(*shape, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(out_gate)
        return self.o_proj(out.to(hidden_states.dtype)), None

    @torch.no_grad()
    def local_gate_mean(self):
        return float(torch.sigmoid(self.local_gate_b.float()).mean())


def replace_full_attention_layers(model, config, indices):
    cfg = _text_config(model)
    allowed = set(full_attention_layers(model))
    backbone = getattr(getattr(model, "model", model), "language_model", getattr(model, "model", model))
    previous = None
    made = []
    for idx in sorted(int(x) for x in indices):
        if idx not in allowed:
            raise ValueError(f"layer {idx} is not full attention")
        layer = backbone.layers[idx]
        if isinstance(layer.self_attn, QwenPDelta3CLVRAttention):
            wrapper = layer.self_attn
            object.__setattr__(
                wrapper,
                "_previous_attention_ref",
                weakref.ref(previous) if previous is not None else None,
            )
        else:
            wrapper = QwenPDelta3CLVRAttention(layer.self_attn, cfg, config, idx, previous)
            layer.self_attn = wrapper
        previous = wrapper
        made.append(wrapper)
    model.config.use_cache = False
    return made
