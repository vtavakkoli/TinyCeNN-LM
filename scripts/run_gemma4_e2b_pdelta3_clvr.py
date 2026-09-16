#!/usr/bin/env python3
"""Gemma-4 E2B adapter for the proven Qwen3.5 sequential PDelta3-CLVR trainer.

Reuses the Qwen training/acceptance/checkpoint loop, but swaps in Gemma4 text
loading and a Gemma4-compatible attention replacement. Only the pre-KV-sharing
full-attention prefix [4, 9, 14] is eligible; layer 14 preserves Gemma's shared
KV handoff for later layers.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import get_token
from torch import nn
from transformers import AutoTokenizer, Gemma4ForCausalLM
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_qwen35_pdelta3_clvr_sequential as qtrain
from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
if not HF_TOKEN:
    raise RuntimeError("google/gemma-4-E2B requires Hugging Face access; login/set HF_TOKEN first.")


class AuthTokenizer:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        kwargs["token"] = HF_TOKEN
        return AutoTokenizer.from_pretrained(name, **kwargs)


class AuthGemma4ForCausalLM:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        kwargs["token"] = HF_TOKEN
        return Gemma4ForCausalLM.from_pretrained(name, **kwargs)


@dataclass(frozen=True)
class Gemma4PDelta3CLVRConfig:
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def eligible_full_attention_layers(model):
    cfg = text_config(model)
    boundary = int(cfg.num_hidden_layers) - int(getattr(cfg, "num_kv_shared_layers", 0))
    kinds = list(cfg.layer_types)
    return [i for i, kind in enumerate(kinds) if kind == "full_attention" and i < boundary]


def repeat_kv(x, groups):
    return x.repeat_interleave(groups, dim=1)


class Gemma4PDelta3CLVRAttention(nn.Module):
    def __init__(self, original, cfg, config, layer_idx, previous_attention=None):
        super().__init__()
        config.validate(cfg)
        if getattr(original, "is_kv_shared_layer", False):
            raise ValueError("Gemma4 pilot does not replace shared-KV layers")
        self.config = cfg
        self.layer_idx = int(layer_idx)
        self.layer_type = getattr(original, "layer_type", "full_attention")
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.head_dim = int(original.head_dim)
        self.num_kv_heads = int(original.k_proj.out_features // self.head_dim)
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = float(getattr(original, "scaling", 1.0))
        self.attention_dropout = float(getattr(original, "attention_dropout", 0.0))
        self.is_causal = bool(getattr(original, "is_causal", True))
        self.is_kv_shared_layer = False
        self.store_full_length_kv = bool(getattr(original, "store_full_length_kv", False))
        self.local_window = int(config.local_window)
        object.__setattr__(self, "_previous_attention_ref", weakref.ref(previous_attention) if previous_attention else None)

        self.q_proj = copy.deepcopy(original.q_proj)
        self.k_proj = copy.deepcopy(original.k_proj)
        self.v_proj = copy.deepcopy(original.v_proj)
        self.o_proj = copy.deepcopy(original.o_proj)
        self.q_norm = copy.deepcopy(original.q_norm)
        self.k_norm = copy.deepcopy(original.k_norm)
        self.v_norm = copy.deepcopy(original.v_norm)
        self.core = FrontierPDelta3Layer(
            self.num_heads, self.num_kv_heads, self.head_dim,
            feature_dim=config.feature_dim, variant=config.variant,
            chunk_size=config.chunk_size, conv_kernel=config.conv_kernel,
            state_dtype=config.state_dtype,
        )
        p = min(max(float(config.local_gate_init), 1e-4), 1 - 1e-4)
        self.local_gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.local_gate_b = nn.Parameter(torch.full((self.num_heads,), math.log(p / (1 - p))))
        self.last_value = None

    def _previous_attention(self):
        ref = object.__getattribute__(self, "_previous_attention_ref")
        return None if ref is None else ref()

    def _local(self, q, k, v, mask):
        kh, vh = repeat_kv(k, self.groups), repeat_kv(v, self.groups)
        scores = torch.matmul(q.float(), kh.float().transpose(-2, -1)) * self.scaling
        t = q.shape[-2]
        qi = torch.arange(t, device=q.device)[:, None]
        kj = torch.arange(t, device=q.device)[None, :]
        allowed = (kj <= qi) & (kj >= qi - self.local_window + 1)
        scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)
        if mask is not None:
            if mask.ndim == 4:
                m = mask[..., :t, :t]
                scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min) if m.dtype == torch.bool else scores + m.float()
            elif mask.ndim == 2:
                m = mask[:, None, None, :t]
                scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min) if m.dtype == torch.bool else scores + (1-m.float()) * torch.finfo(scores.dtype).min
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, vh.to(probs.dtype))

    def forward(self, hidden_states, position_embeddings, attention_mask=None, shared_kv_states=None,
                past_key_values=None, position_ids=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("research replacement requires use_cache=False")
        shape = hidden_states.shape[:-1]
        q = self.q_norm(self.q_proj(hidden_states).view(*shape, self.num_heads, self.head_dim))
        q = apply_rotary_pos_emb(q, *position_embeddings, unsqueeze_dim=2).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(*shape, self.num_kv_heads, self.head_dim))
        k = apply_rotary_pos_emb(k, *position_embeddings, unsqueeze_dim=2).transpose(1, 2)
        v = self.v_norm(self.v_proj(hidden_states).view(*shape, self.num_kv_heads, self.head_dim)).transpose(1, 2)

        # Required by Gemma4: layer 14 feeds the later shared full-attention layers.
        if self.store_full_length_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (k, v)

        self.last_value = v.detach()
        prev = self._previous_attention()
        routed = prev.last_value.to(v.device, v.dtype) if prev is not None and prev.last_value is not None and prev.last_value.shape == v.shape else v
        global_out = self.core(q, k, v, routed_v=routed)
        local_out = self._local(q, k, v, attention_mask)
        gate = torch.sigmoid(torch.einsum("bhtd,hd->bht", q.float(), self.local_gate_w.float()) + self.local_gate_b.float()[None,:,None]).to(q.dtype)
        out = gate[...,None] * local_out + (1-gate[...,None]) * global_out
        out = out.transpose(1,2).contiguous().reshape(*shape, self.num_heads * self.head_dim)
        return self.o_proj(out.to(hidden_states.dtype)), None

    @torch.no_grad()
    def local_gate_mean(self):
        return float(torch.sigmoid(self.local_gate_b.float()).mean())


def replace_layers(model, config, indices):
    cfg = text_config(model)
    allowed = set(eligible_full_attention_layers(model))
    previous = None
    made = []
    for idx in sorted(indices):
        if idx not in allowed:
            raise ValueError(f"layer {idx} is not an eligible Gemma4 pre-sharing full-attention layer")
        layer = model.model.layers[idx]
        if isinstance(layer.self_attn, Gemma4PDelta3CLVRAttention):
            wrapper = layer.self_attn
            object.__setattr__(wrapper, "_previous_attention_ref", weakref.ref(previous) if previous else None)
        else:
            original = layer.self_attn
            wrapper = Gemma4PDelta3CLVRAttention(original, cfg, config, idx, previous)
            wrapper.to(device=original.q_proj.weight.device)
            layer.self_attn = wrapper
        previous = wrapper
        made.append(wrapper)
    return made


def gemma_call_attention(module, hidden, kwargs):
    kw = dict(kwargs)
    kw.pop("cache_position", None)
    kw.setdefault("shared_kv_states", {})
    out = module(hidden, past_key_values=None, **kw)
    return out[0] if isinstance(out, (tuple, list)) else out


# Patch the proven sequential trainer in-place.
qtrain.AutoTokenizer = AuthTokenizer
qtrain.Qwen3_5ForCausalLM = AuthGemma4ForCausalLM
qtrain.QwenPDelta3CLVRConfig = Gemma4PDelta3CLVRConfig
qtrain.QwenPDelta3CLVRAttention = Gemma4PDelta3CLVRAttention
qtrain.full_attention_layers = eligible_full_attention_layers
qtrain.replace_full_attention_layers = replace_layers
qtrain.call_attention = gemma_call_attention

print("[TinyCeNN][GEMMA4 ADAPTER] base=google/gemma-4-E2B eligible full-attention layers=[4, 9, 14]", flush=True)

if __name__ == "__main__":
    raise SystemExit(qtrain.main())
