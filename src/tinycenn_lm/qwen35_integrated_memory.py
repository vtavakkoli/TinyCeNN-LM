"""Qwen3.5 text-backbone TinyCeNN integrated memory.

Replaces selected full-attention layers in Qwen3.5 while preserving its native
Gated DeltaNet linear-attention layers and Qwen3.5's post-attention output gate.
"""
import copy
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
from .optimized_memory import OptimizedMemory

FORMAT = "qwen35-cenn-integrated-v1"


def native_dtype(device):
    if torch.device(device).type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def text_config(model):
    cfg = model.config
    return cfg.get_text_config(decoder=True) if hasattr(cfg, "get_text_config") else cfg


class Qwen35IntegratedCache(DynamicCache):
    def __init__(self, config, memory_layers=()):
        super().__init__(config=config)
        self.memory_layers = frozenset(int(x) for x in memory_layers)
        self.memory_states = {}

    def _memory_seq_length(self, layer_idx=None):
        if layer_idx is not None:
            state = self.memory_states.get(int(layer_idx))
            return int(state.position) if state is not None else 0
        return max((int(state.position) for state in self.memory_states.values()), default=0)

    def get_seq_length(self, layer_idx=0):
        """Return the real sequence length even when the first attention layer is CeNN.

        Qwen3.5 layer 0 is linear attention. Hugging Face's DynamicCache therefore
        redirects the default get_seq_length() call to the first full-attention
        cache layer. When that first full-attention layer (layer 3 in Qwen3.5-0.8B)
        is replaced by TinyCeNN, its DynamicLayer is intentionally never updated;
        the position lives in memory_states instead. Without this override, cached
        decoding repeatedly reports length 0 and reuses incorrect RoPE positions.
        """
        if layer_idx in self.memory_layers:
            return self._memory_seq_length(layer_idx)
        if layer_idx == 0:
            memory_length = self._memory_seq_length()
            try:
                native_length = int(super().get_seq_length(layer_idx))
            except (ValueError, StopIteration):
                native_length = 0
            return max(memory_length, native_length)
        return super().get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx):
        """Mirror get_seq_length() for causal-mask construction at CeNN layers."""
        if layer_idx in self.memory_layers:
            return self._memory_seq_length(layer_idx) + int(query_length), 0
        if layer_idx == 0 and self.memory_states:
            return self.get_seq_length(0) + int(query_length), 0
        return super().get_mask_sizes(query_length, layer_idx)

    @staticmethod
    def _bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, (list, tuple)):
            return sum(Qwen35IntegratedCache._bytes(x) for x in value)
        if isinstance(value, dict):
            return sum(Qwen35IntegratedCache._bytes(x) for x in value.values())
        return 0

    @property
    def nbytes(self):
        total = 0
        for layer in self.layers:
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                total += self._bytes(getattr(layer, name, None))
        total += sum(state.nbytes for state in self.memory_states.values())
        return total

    def reorder_cache(self, beam_idx):
        if self.memory_states:
            raise NotImplementedError("Beam/batch reordering is unsupported for TinyCeNN states")
        return super().reorder_cache(beam_idx)

    def crop(self, *args, **kwargs):
        if self.memory_states:
            raise NotImplementedError("Compressed TinyCeNN history cannot be cropped")
        return super().crop(*args, **kwargs)


class Qwen35IntegratedAttention(nn.Module):
    """Drop-in replacement for a Qwen3.5 full-attention module."""
    def __init__(self, original, core, layer_idx):
        super().__init__()
        self.original = original
        self.core = core
        self.layer_idx = int(layer_idx)
        self.config = original.config
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.scaling = original.scaling
        self.attention_dropout = original.attention_dropout
        self.is_causal = original.is_causal

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, **kwargs):
        if position_embeddings is None:
            raise ValueError("Qwen3.5 position_embeddings are required")
        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is not None and not isinstance(cache, Qwen35IntegratedCache):
            raise TypeError("Use Qwen35IntegratedCache with a Qwen3.5 TinyCeNN model")
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Train with use_cache=False")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        t = hidden_states.shape[1]

        query_states, gate = torch.chunk(
            self.original.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.original.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.original.k_norm(self.original.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.original.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attention_mask is not None:
            if attention_mask.ndim != 4 or bool((attention_mask[..., -1, :] < 0).any()):
                raise ValueError("Only unpadded causal batches are supported")

        if self.core.variant == "transformer_readout":
            if cache is not None:
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx)
            output = F.scaled_dot_product_attention(
                query_states,
                repeat_kv(key_states, self.num_key_value_groups),
                repeat_kv(value_states, self.num_key_value_groups),
                attn_mask=attention_mask,
                is_causal=attention_mask is None and t > 1,
                scale=float(self.scaling),
            )
            output = self.core.calibrate(output.float())
        elif cache is not None:
            output, state = self.core(
                query_states, key_states, value_states,
                state=cache.memory_states.get(self.layer_idx),
                return_state=True, apply_readout=True,
            )
            cache.memory_states[self.layer_idx] = state
        else:
            output = self.core(query_states, key_states, value_states, apply_readout=True)

        flat = output.transpose(1, 2).reshape(*input_shape, -1).to(hidden_states.dtype)
        # Important: Qwen3.5 gates after attention. Do not fold the TinyCeNN
        # readout into o_proj because that would not commute with this gate.
        flat = flat * torch.sigmoid(gate)
        return self.original.o_proj(flat), None


def wrappers(model):
    return [layer.self_attn for layer in model.model.layers
            if getattr(layer, "block_type", None) == "full_attention"
            and isinstance(getattr(layer, "self_attn", None), Qwen35IntegratedAttention)]


def full_attention_layers(model):
    return [i for i, layer in enumerate(model.model.layers)
            if getattr(layer, "block_type", None) == "full_attention"]


def build_student(teacher, layers, variant="cenn_partition", features=64, block_size=32, sinks=4):
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    cfg = text_config(model)
    if cfg.model_type != "qwen3_5_text":
        raise ValueError(f"Expected qwen3_5_text, got {cfg.model_type}")
    for index in layers:
        if not 0 <= index < len(model.model.layers):
            raise ValueError(f"Invalid layer {index}")
        layer = model.model.layers[index]
        if getattr(layer, "block_type", None) != "full_attention":
            raise ValueError(f"Layer {index} is not full_attention")
        original = layer.self_attn
        core = OptimizedMemory(
            cfg.num_attention_heads, cfg.num_key_value_heads, original.head_dim,
            features, variant, block_size, sinks,
        ).to(original.q_proj.weight.device)
        layer.self_attn = Qwen35IntegratedAttention(original, core, index)
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
    cfg = text_config(model)
    memory_layers = [a.layer_idx for a in wrappers(model) if a.core.variant != "transformer_readout"]
    return Qwen35IntegratedCache(cfg, memory_layers)


def adapter_payload(model, metadata=None):
    return {
        "format": FORMAT,
        "metadata": metadata or {},
        "adapters": {
            str(a.layer_idx): {
                "config": a.core.config,
                "state_dict": {k: v.detach().cpu().clone() for k, v in a.core.state_dict().items()},
            } for a in wrappers(model)
        },
    }


def restore_student(teacher, payload):
    if payload["format"] != FORMAT:
        raise ValueError(f"Not a {FORMAT} checkpoint")
    model = copy.deepcopy(teacher).eval().requires_grad_(False)
    for key, value in payload["adapters"].items():
        index = int(key)
        layer = model.model.layers[index]
        if getattr(layer, "block_type", None) != "full_attention":
            raise ValueError(f"Checkpoint targets non-full-attention layer {index}")
        original = layer.self_attn
        core = OptimizedMemory(**value["config"]).to(original.q_proj.weight.device)
        core.load_state_dict(value["state_dict"])
        layer.self_attn = Qwen35IntegratedAttention(original, core, index)
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
        logits = model(input_ids=continuation[-1], past_key_values=cache, use_cache=True).logits[:, -1]
        continuation.append(logits.argmax(-1, keepdim=True))
    return torch.cat(continuation, dim=1), cache
