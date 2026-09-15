"""Gemma 3 / FunctionGemma integrated TinyCeNN memory.

This module mirrors the SmolLM2 V3 experiment but respects Gemma 3's
Q/K RMS normalization and hybrid sliding/full-attention cache. It is intended
for replacing *full-attention* Gemma 3 text layers; the original sliding-window
layers remain untouched.
"""
import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, repeat_kv

from .optimized_memory import OptimizedMemory


FORMAT = "functiongemma-cenn-integrated-v1"


def native_dtype(device):
    """Use BF16 only when the GPU has native BF16 arithmetic."""
    if torch.device(device).type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


class Gemma3IntegratedCache(DynamicCache):
    """Gemma 3 hybrid cache plus bounded TinyCeNN states at replaced layers."""

    def __init__(self, config, memory_layers=()):
        super().__init__(config=config)
        self.memory_layers = frozenset(memory_layers)
        self.memory_states = {}

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self.memory_layers:
            state = self.memory_states.get(layer_idx)
            return state.position if state is not None else 0
        return super().get_seq_length(layer_idx)

    def get_mask_sizes(self, cache_position, layer_idx):
        if layer_idx in self.memory_layers:
            query_length = cache_position.shape[0] if isinstance(cache_position, torch.Tensor) else int(cache_position)
            return self.get_seq_length(layer_idx) + query_length, 0
        return super().get_mask_sizes(cache_position, layer_idx)

    @property
    def nbytes(self):
        tensors = [
            tensor
            for layer in self.layers
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None))
            if isinstance(tensor, torch.Tensor)
        ]
        return sum(x.numel() * x.element_size() for x in tensors) + sum(
            state.nbytes for state in self.memory_states.values()
        )

    def reorder_cache(self, *args, **kwargs):
        raise NotImplementedError("Use batch-one greedy decoding; beam-search cache reordering is unsupported")

    def crop(self, *args, **kwargs):
        raise NotImplementedError("Compressed TinyCeNN history cannot be cropped; start a fresh cache")

    def batch_repeat_interleave(self, *args, **kwargs):
        raise NotImplementedError("Batch expansion is unsupported for TinyCeNN memory states")

    def batch_select_indices(self, *args, **kwargs):
        raise NotImplementedError("Batch selection is unsupported for TinyCeNN memory states")


class Gemma3IntegratedAttention(nn.Module):
    """Full-attention Gemma 3 layer replaced by a TinyCeNN memory core.

    Gemma3DecoderLayer inspects attributes on ``self_attn`` before calling it
    (most importantly ``is_sliding`` to select local vs global RoPE).  Mirror
    the lightweight structural attributes of Gemma3Attention so replacing the
    module preserves the Transformers 4.57.x decoder contract.
    """

    def __init__(self, original, core, layer_idx):
        super().__init__()
        self.original = original
        self.core = core
        self.layer_idx = layer_idx

        # Structural Gemma3Attention API used by Gemma3DecoderLayer and by
        # attention tooling.  These are plain metadata values, not duplicate
        # module registrations; Q/K/V/O and RMSNorm modules remain under
        # ``self.original`` only.
        self.is_sliding = bool(original.is_sliding)
        self.config = original.config
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.scaling = original.scaling
        self.attention_dropout = original.attention_dropout
        self.is_causal = original.is_causal
        self.attn_logit_softcapping = original.attn_logit_softcapping
        self.sliding_window = original.sliding_window

        if self.is_sliding:
            raise ValueError("Gemma3IntegratedAttention only supports full-attention layers")

        self.register_buffer("fused_weight", None, persistent=False)

    def fuse(self, enabled=True):
        if not enabled:
            self.fused_weight = None
            return
        with torch.no_grad():
            weight = self.original.o_proj.weight.float().reshape(
                -1, self.core.num_heads, self.core.head_dim
            )
            self.fused_weight = torch.einsum(
                "ohd,hkd->ohk", weight, self.core.readout.float()
            ).reshape_as(self.original.o_proj.weight).to(self.original.o_proj.weight.dtype)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if position_embeddings is None:
            raise ValueError("Gemma 3 position_embeddings are required")
        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is not None and not isinstance(cache, Gemma3IntegratedCache):
            raise TypeError("Use Gemma3IntegratedCache with a FunctionGemma TinyCeNN model")
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Train with use_cache=False")
        if self.fused_weight is not None and torch.is_grad_enabled():
            raise RuntimeError("Unfuse the readout before training")

        if self.is_sliding:
            raise RuntimeError("TinyCeNN FunctionGemma V1 only supports replacing full-attention layers")

        if attention_mask is not None:
            if attention_mask.ndim != 4 or bool((attention_mask[..., -1, :] < 0).any()):
                raise ValueError("Only unpadded causal batches are supported")

        b, t, _ = hidden_states.shape
        h = self.core.num_heads
        hk = self.core.num_kv_heads
        d = self.core.head_dim

        q = self.original.q_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
        k = self.original.k_proj(hidden_states).view(b, t, hk, d).transpose(1, 2)
        v = self.original.v_proj(hidden_states).view(b, t, hk, d).transpose(1, 2)

        q = self.original.q_norm(q)
        k = self.original.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.core.variant == "transformer_readout":
            if cache is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = cache.update(k, v, self.layer_idx, cache_kwargs)
            output = F.scaled_dot_product_attention(
                q,
                repeat_kv(k, h // hk),
                repeat_kv(v, h // hk),
                attn_mask=attention_mask,
                is_causal=attention_mask is None and t > 1,
                scale=float(self.scaling),
            )
            if self.fused_weight is None:
                output = self.core.calibrate(output.float())
        elif cache is not None:
            output, state = self.core(
                q,
                k,
                v,
                state=cache.memory_states.get(self.layer_idx),
                return_state=True,
                apply_readout=self.fused_weight is None,
            )
            cache.memory_states[self.layer_idx] = state
        else:
            output = self.core(q, k, v, apply_readout=self.fused_weight is None)

        flat = output.transpose(1, 2).reshape(b, t, h * d).to(hidden_states.dtype)
        if self.fused_weight is None:
            return self.original.o_proj(flat), None
        return F.linear(flat, self.fused_weight, self.original.o_proj.bias), None


def wrappers(model):
    return [
        layer.self_attn
        for layer in model.model.layers
        if isinstance(layer.self_attn, Gemma3IntegratedAttention)
    ]


def full_attention_layers(model):
    return [
        i for i, layer in enumerate(model.model.layers)
        if not getattr(layer.self_attn, "is_sliding", False)
    ]


def build_student(teacher, layers, variant="cenn_partition", features=64, block_size=32, sinks=4):
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    config = model.config.get_text_config(decoder=True)
    if config.model_type != "gemma3_text":
        raise ValueError(f"Expected gemma3_text, got {config.model_type}")

    for index in layers:
        if not 0 <= index < len(model.model.layers):
            raise ValueError(f"Invalid layer {index}")
        original = model.model.layers[index].self_attn
        if getattr(original, "is_sliding", False):
            raise ValueError(
                f"Layer {index} is sliding_attention. FunctionGemma V1 intentionally replaces only full_attention layers."
            )
        # FunctionGemma uses query_pre_attn_scalar == head_dim (256), so its
        # native attention scale matches the OptimizedMemory softmax scale.
        # Keep this guard explicit for future Gemma3 checkpoints where that
        # assumption may not hold.
        expected_scale = original.head_dim ** -0.5
        if abs(float(original.scaling) - float(expected_scale)) > 1e-8:
            raise ValueError(
                f"Layer {index} uses attention scale {original.scaling}, but TinyCeNN currently expects {expected_scale}."
            )
        core = OptimizedMemory(
            config.num_attention_heads,
            config.num_key_value_heads,
            original.head_dim,
            features,
            variant,
            block_size,
            sinks,
        ).to(original.q_proj.weight.device)
        model.model.layers[index].self_attn = Gemma3IntegratedAttention(original, core, index)
    return model


@contextmanager
def inference_mode(model, compute_dtype="float32"):
    adapters = wrappers(model)
    previous = [a.core.compute_dtype for a in adapters]
    try:
        for adapter in adapters:
            adapter.core.compute_dtype = compute_dtype
            adapter.fuse()
        with torch.no_grad():
            yield model
    finally:
        for adapter, dtype in zip(adapters, previous):
            adapter.fuse(False)
            adapter.core.compute_dtype = dtype


def new_cache(model):
    memory_layers = [
        a.layer_idx for a in wrappers(model)
        if a.core.variant != "transformer_readout"
    ]
    return Gemma3IntegratedCache(model.config, memory_layers)


def adapter_payload(model, metadata=None):
    return {
        "format": FORMAT,
        "metadata": metadata or {},
        "adapters": {
            str(a.layer_idx): {
                "config": a.core.config,
                "state_dict": {
                    k: v.detach().cpu().clone() for k, v in a.core.state_dict().items()
                },
            }
            for a in wrappers(model)
        },
    }


def restore_student(teacher, payload):
    if payload["format"] != FORMAT:
        raise ValueError(f"Not a {FORMAT} checkpoint")
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    for key, value in payload["adapters"].items():
        index = int(key)
        original = model.model.layers[index].self_attn
        if getattr(original, "is_sliding", False):
            raise ValueError(f"Checkpoint attempts to replace sliding-attention layer {index}")
        core = OptimizedMemory(**value["config"]).to(original.q_proj.weight.device)
        core.load_state_dict(value["state_dict"])
        model.model.layers[index].self_attn = Gemma3IntegratedAttention(original, core, index)
    return model


@torch.no_grad()
def greedy_generate(model, ids, tokens=32, stop_token_ids=()):
    """Batch-one cached greedy generation for the hybrid FunctionGemma model."""
    if ids.shape[0] != 1 or ids.shape[1] < 1 or tokens < 1:
        raise ValueError("Use batch size one, a nonempty prompt, and positive tokens")
    stop_token_ids = set(int(x) for x in stop_token_ids)
    cache = new_cache(model)
    output = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[:, -1]
    continuation = []
    token = output.argmax(-1, keepdim=True)
    continuation.append(token)
    for _ in range(tokens - 1):
        if int(continuation[-1].item()) in stop_token_ids:
            break
        output = model(
            input_ids=continuation[-1], past_key_values=cache, use_cache=True
        ).logits[:, -1]
        continuation.append(output.argmax(-1, keepdim=True))
    return torch.cat(continuation, dim=1), cache
