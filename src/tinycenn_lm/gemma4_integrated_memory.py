"""Gemma 4 text-backbone TinyCeNN integrated memory.

Initial research adapter for google/gemma-4-E2B. It replaces only independent,
non-KV-shared full-attention layers. Gemma 4's shared-KV producer layers and
shared-KV consumer layers are intentionally left untouched.
"""
import copy
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, repeat_kv

from .optimized_memory import OptimizedMemory

FORMAT = "gemma4-cenn-integrated-v1"


def native_dtype(device):
    if torch.device(device).type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def text_config(model):
    cfg = model.config
    return cfg.get_text_config(decoder=True) if hasattr(cfg, "get_text_config") else cfg


def text_backbone(model):
    root = getattr(model, "model", model)
    return getattr(root, "language_model", root)


def strip_multimodal_towers(model):
    """Drop unused vision/audio modules for text-only benchmarking before deepcopy."""
    root = getattr(model, "model", None)
    if root is not None:
        for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if hasattr(root, name):
                setattr(root, name, None)
    return model


class Gemma4IntegratedCache(DynamicCache):
    def __init__(self, config, memory_layers=()):
        super().__init__(config=config)
        self.memory_layers = frozenset(int(x) for x in memory_layers)
        self.memory_states = {}

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self.memory_layers:
            state = self.memory_states.get(layer_idx)
            return state.position if state is not None else 0
        return super().get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx):
        if layer_idx in self.memory_layers:
            q = int(query_length.shape[0]) if isinstance(query_length, torch.Tensor) else int(query_length)
            return self.get_seq_length(layer_idx) + q, 0
        return super().get_mask_sizes(query_length, layer_idx)

    @staticmethod
    def _bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, (list, tuple)):
            return sum(Gemma4IntegratedCache._bytes(x) for x in value)
        if isinstance(value, dict):
            return sum(Gemma4IntegratedCache._bytes(x) for x in value.values())
        return 0

    @property
    def nbytes(self):
        total = 0
        for layer in self.layers:
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                total += self._bytes(getattr(layer, name, None))
        total += sum(state.nbytes for state in self.memory_states.values())
        return total

    def reorder_cache(self, *args, **kwargs):
        if self.memory_states:
            raise NotImplementedError("Beam/batch cache reordering is unsupported for TinyCeNN states")
        return super().reorder_cache(*args, **kwargs)

    def crop(self, *args, **kwargs):
        if self.memory_states:
            raise NotImplementedError("Compressed TinyCeNN history cannot be cropped")
        return super().crop(*args, **kwargs)


class Gemma4IntegratedAttention(nn.Module):
    """Drop-in replacement for an independent Gemma 4 full-attention layer."""

    def __init__(self, original, core, layer_idx):
        super().__init__()
        if original.is_sliding:
            raise ValueError("TinyCeNN V1 only replaces Gemma 4 full-attention layers")
        if original.is_kv_shared_layer:
            raise ValueError("TinyCeNN V1 does not replace Gemma 4 KV-shared consumer layers")
        if original.store_full_length_kv:
            raise ValueError("TinyCeNN V1 does not replace the Gemma 4 shared-KV producer layer")
        self.original = original
        self.core = core
        self.layer_idx = int(layer_idx)
        for name in (
            "layer_type", "config", "head_dim", "num_key_value_groups", "scaling",
            "attention_dropout", "is_causal", "is_sliding", "sliding_window",
            "is_kv_shared_layer", "store_full_length_kv",
        ):
            setattr(self, name, getattr(original, name))

    @property
    def q_scale_for_core(self):
        # OptimizedMemory exact local attention internally divides q by sqrt(d).
        # Gemma 4 uses scaling=1.0, so rescale q to preserve native logits.
        return float(self.scaling) * math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        shared_kv_states=None,
        past_key_values=None,
        **kwargs,
    ):
        cache = past_key_values
        if cache is not None and not isinstance(cache, Gemma4IntegratedCache):
            raise TypeError("Use Gemma4IntegratedCache with a Gemma 4 TinyCeNN model")
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Train with use_cache=False")
        if shared_kv_states is None:
            shared_kv_states = {}

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        t = hidden_states.shape[1]
        cos, sin = position_embeddings

        q = self.original.q_proj(hidden_states).view(hidden_shape)
        q = self.original.q_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        k = self.original.k_proj(hidden_states).view(hidden_shape)
        v = self.original.v_proj(hidden_states).view(hidden_shape) if self.original.v_proj is not None else k
        k = self.original.k_norm(k)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        v = self.original.v_norm(v).transpose(1, 2)

        if self.core.variant == "transformer_readout":
            if cache is not None:
                k, v = cache.update(k, v, self.layer_idx)
            out = F.scaled_dot_product_attention(
                q,
                repeat_kv(k, self.num_key_value_groups),
                repeat_kv(v, self.num_key_value_groups),
                attn_mask=attention_mask,
                is_causal=attention_mask is None and t > 1,
                scale=float(self.scaling),
            )
            out = self.core.calibrate(out.float())
        else:
            q_core = q * self.q_scale_for_core
            if cache is not None:
                out, state = self.core(
                    q_core, k, v,
                    state=cache.memory_states.get(self.layer_idx),
                    return_state=True,
                    apply_readout=True,
                )
                cache.memory_states[self.layer_idx] = state
            else:
                out = self.core(q_core, k, v, apply_readout=True)

        flat = out.transpose(1, 2).reshape(*input_shape, -1).to(hidden_states.dtype)
        return self.original.o_proj(flat), None


def wrappers(model):
    return [
        layer.self_attn for layer in text_backbone(model).layers
        if isinstance(getattr(layer, "self_attn", None), Gemma4IntegratedAttention)
    ]


def full_attention_layers(model):
    return [
        i for i, layer in enumerate(text_backbone(model).layers)
        if getattr(layer.self_attn, "layer_type", None) == "full_attention"
    ]


def replaceable_full_attention_layers(model):
    result = []
    for i, layer in enumerate(text_backbone(model).layers):
        attn = layer.self_attn
        if (
            getattr(attn, "layer_type", None) == "full_attention"
            and not getattr(attn, "is_kv_shared_layer", False)
            and not getattr(attn, "store_full_length_kv", False)
        ):
            result.append(i)
    return result


def build_student(teacher, layers, variant="cenn_partition", features=64, block_size=32, sinks=4):
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    cfg = text_config(model)
    if cfg.model_type != "gemma4_text":
        raise ValueError(f"Expected gemma4_text, got {cfg.model_type}")
    tm = text_backbone(model)
    replaceable = set(replaceable_full_attention_layers(model))
    for index in layers:
        if index not in replaceable:
            raise ValueError(
                f"Gemma 4 layer {index} is not independently replaceable in V1; "
                f"safe layers are {sorted(replaceable)}"
            )
        original = tm.layers[index].self_attn
        h = original.q_proj.out_features // original.head_dim
        hk = original.k_proj.out_features // original.head_dim
        core = OptimizedMemory(
            h, hk, original.head_dim, features, variant, block_size, sinks
        ).to(original.q_proj.weight.device)
        tm.layers[index].self_attn = Gemma4IntegratedAttention(original, core, index)
    return model


@contextmanager
def inference_mode(model, compute_dtype="float32"):
    adapters = wrappers(model)
    previous = [a.core.compute_dtype for a in adapters]
    try:
        for adapter in adapters:
            adapter.core.compute_dtype = compute_dtype
        with torch.no_grad():
            yield model
    finally:
        for adapter, dtype in zip(adapters, previous):
            adapter.core.compute_dtype = dtype


def new_cache(model):
    memory_layers = [a.layer_idx for a in wrappers(model) if a.core.variant != "transformer_readout"]
    return Gemma4IntegratedCache(text_config(model), memory_layers)


def adapter_payload(model, metadata=None):
    return {
        "format": FORMAT,
        "metadata": metadata or {},
        "adapters": {
            str(a.layer_idx): {
                "config": a.core.config,
                "state_dict": {k: v.detach().cpu().clone() for k, v in a.core.state_dict().items()},
            }
            for a in wrappers(model)
        },
    }


def restore_student(teacher, payload):
    if payload["format"] != FORMAT:
        raise ValueError(f"Not a {FORMAT} checkpoint")
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    tm = text_backbone(model)
    replaceable = set(replaceable_full_attention_layers(model))
    for key, value in payload["adapters"].items():
        index = int(key)
        if index not in replaceable:
            raise ValueError(f"Checkpoint targets unsafe Gemma 4 layer {index}")
        original = tm.layers[index].self_attn
        core = OptimizedMemory(**value["config"]).to(original.q_proj.weight.device)
        core.load_state_dict(value["state_dict"])
        tm.layers[index].self_attn = Gemma4IntegratedAttention(original, core, index)
    return model


@torch.no_grad()
def greedy_generate(model, ids, tokens=32, stop_token_ids=()):
    if ids.shape[0] != 1 or ids.shape[1] < 1 or tokens < 1:
        raise ValueError("Use batch size one, a nonempty prompt, and positive tokens")
    stop_token_ids = set(int(x) for x in stop_token_ids)
    cache = new_cache(model)
    logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[:, -1]
    continuation = [logits.argmax(-1, keepdim=True)]
    for _ in range(tokens - 1):
        if int(continuation[-1].item()) in stop_token_ids:
            break
        logits = model(
            input_ids=continuation[-1], past_key_values=cache, use_cache=True
        ).logits[:, -1]
        continuation.append(logits.argmax(-1, keepdim=True))
    return torch.cat(continuation, dim=1), cache
