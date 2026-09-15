"""Jointly trainable V2 mixers with full-model causal caching for Llama/SmolLM2.

Unpadded, batch-one greedy decoding is the supported inference protocol. This
adapter deliberately does not implement beam search, cache cropping or offload.
"""
import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .optimized_memory import OptimizedMemory


def native_dtype(device):
    """Do not mistake emulated BF16 allocation support for native T4 arithmetic."""
    if torch.device(device).type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


class IntegratedCache(DynamicCache):
    def __init__(self, memory_layers=()):
        super().__init__()
        self.memory_layers = frozenset(memory_layers)
        self.memory_states = {}

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self.memory_layers:
            state = self.memory_states.get(layer_idx)
            return state.position if state is not None else 0
        return super().get_seq_length(layer_idx)

    def get_mask_sizes(self, cache_position, layer_idx):
        if layer_idx in self.memory_layers:
            return self.get_seq_length(layer_idx) + cache_position.shape[0], 0
        return super().get_mask_sizes(cache_position, layer_idx)

    @property
    def nbytes(self):
        tensors = [x for layer in self.layers for x in (layer.keys, layer.values)
                   if isinstance(x, torch.Tensor)]
        return sum(x.numel() * x.element_size() for x in tensors) + sum(
            state.nbytes for state in self.memory_states.values())

    def reorder_cache(self, *args, **kwargs):
        raise NotImplementedError("Use the supplied batch-one greedy decoder; beam search is unsupported")

    def crop(self, *args, **kwargs):
        raise NotImplementedError("Compressed history cannot be cropped; start a fresh cache")


class IntegratedAttention(nn.Module):
    def __init__(self, original, core, layer_idx):
        super().__init__()
        self.original, self.core, self.layer_idx = original, core, layer_idx
        self.register_buffer("fused_weight", None, persistent=False)

    def fuse(self, enabled=True):
        if not enabled:
            self.fused_weight = None
            return
        with torch.no_grad():
            weight = self.original.o_proj.weight.float().reshape(
                -1, self.core.num_heads, self.core.head_dim)
            self.fused_weight = torch.einsum("ohd,hkd->ohk", weight, self.core.readout.float()).reshape_as(
                self.original.o_proj.weight).to(self.original.o_proj.weight.dtype)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, **kwargs):
        if position_embeddings is None:
            raise ValueError("Llama rotary position embeddings are required")
        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is not None and not isinstance(cache, IntegratedCache):
            raise TypeError("Use IntegratedCache for this model")
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Train with use_cache=False")
        if self.fused_weight is not None and torch.is_grad_enabled():
            raise RuntimeError("Unfuse the readout before training")
        if attention_mask is not None:
            if attention_mask.ndim != 4 or bool((attention_mask[..., -1, :] < 0).any()):
                raise ValueError("Only unpadded causal blocks are supported")
        b, t, _ = hidden_states.shape
        h, hk, d = self.core.num_heads, self.core.num_kv_heads, self.core.head_dim
        q = self.original.q_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
        k = self.original.k_proj(hidden_states).view(b, t, hk, d).transpose(1, 2)
        v = self.original.v_proj(hidden_states).view(b, t, hk, d).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if self.core.variant == "transformer_readout":
            if cache is not None:
                k, v = cache.update(k, v, self.layer_idx)
            output = F.scaled_dot_product_attention(q, repeat_kv(k, h // hk), repeat_kv(v, h // hk),
                attn_mask=attention_mask, is_causal=attention_mask is None and t > 1)
            if self.fused_weight is None:
                output = self.core.calibrate(output.float())
        elif cache is not None:
            output, state = self.core(q, k, v, state=cache.memory_states.get(self.layer_idx),
                                     return_state=True, apply_readout=self.fused_weight is None)
            cache.memory_states[self.layer_idx] = state
        else:
            output = self.core(q, k, v, apply_readout=self.fused_weight is None)
        flat = output.transpose(1, 2).reshape(b, t, h * d).to(hidden_states.dtype)
        if self.fused_weight is None:
            return self.original.o_proj(flat), None
        return F.linear(flat, self.fused_weight, self.original.o_proj.bias), None


def wrappers(model):
    return [layer.self_attn for layer in model.model.layers
            if isinstance(layer.self_attn, IntegratedAttention)]


def build_student(teacher, layers, variant="cenn_partition", features=64, block_size=32, sinks=4):
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    config = model.config
    if config.model_type != "llama":
        raise ValueError("Only Llama-family models are supported")
    for index in layers:
        if not 0 <= index < len(model.model.layers):
            raise ValueError(f"Invalid layer {index}")
        original = model.model.layers[index].self_attn
        core = OptimizedMemory(config.num_attention_heads, config.num_key_value_heads,
                               config.hidden_size // config.num_attention_heads, features,
                               variant, block_size, sinks).to(original.q_proj.weight.device)
        model.model.layers[index].self_attn = IntegratedAttention(original, core, index)
    return model


@contextmanager
def inference_mode(model, compute_dtype="float32"):
    adapters = wrappers(model)
    previous = [a.core.compute_dtype for a in adapters]
    try:
        for a in adapters:
            a.core.compute_dtype = compute_dtype
            a.fuse()
        with torch.no_grad():
            yield model
    finally:
        for a, dtype in zip(adapters, previous):
            a.fuse(False)
            a.core.compute_dtype = dtype


def new_cache(model):
    return IntegratedCache(a.layer_idx for a in wrappers(model)
                           if a.core.variant != "transformer_readout")


def adapter_payload(model, metadata=None):
    return {"format": "smollm2-integrated-memory-v3", "metadata": metadata or {}, "adapters": {
        str(a.layer_idx): {"config": a.core.config,
                          "state_dict": {k: v.detach().cpu().clone() for k, v in a.core.state_dict().items()}}
        for a in wrappers(model)}}


def restore_student(teacher, payload):
    if payload["format"] != "smollm2-integrated-memory-v3":
        raise ValueError("Not an integrated memory checkpoint")
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    for key, value in payload["adapters"].items():
        index = int(key)
        original = model.model.layers[index].self_attn
        core = OptimizedMemory(**value["config"]).to(original.q_proj.weight.device)
        core.load_state_dict(value["state_dict"])
        model.model.layers[index].self_attn = IntegratedAttention(original, core, index)
    return model


@torch.no_grad()
def greedy_generate(model, ids, tokens=32):
    """Deterministic fixed-length generation; no early EOS for comparable timing."""
    if ids.shape[0] != 1 or ids.shape[1] < 1 or tokens < 1:
        raise ValueError("Use batch size one, a nonempty prompt, and positive tokens")
    cache = new_cache(model)
    output = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[:, -1]
    continuation = [output.argmax(-1, keepdim=True)]
    for _ in range(tokens - 1):
        output = model(input_ids=continuation[-1], past_key_values=cache, use_cache=True).logits[:, -1]
        continuation.append(output.argmax(-1, keepdim=True))
    return torch.cat(continuation, dim=1), cache
