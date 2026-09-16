from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from .memory_attention import MemoryAugmentedCellularLayer

DEFAULT_GEMMA4 = "google/gemma-4-E2B"
FORMAT = "gemma4-e2b-memory-fusion-all-attention-v1"


@dataclass(frozen=True)
class Gemma4MemoryFusionConfig:
    feature_dim: int = 32
    memory_rank: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    shifted_window: int = 8
    train_output_projection: bool = True

    def validate(self, model_config) -> None:
        config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        if getattr(config, "model_type", None) != "gemma4_text":
            raise ValueError(f"expected gemma4_text, got {getattr(config, 'model_type', None)!r}")
        if self.feature_dim < 4 or self.memory_rank < 4:
            raise ValueError("feature_dim and memory_rank must be >= 4")
        if not self.dilations or min(self.dilations) < 1:
            raise ValueError("dilations must be positive")
        if int(config.num_attention_heads) % int(config.num_key_value_heads):
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["dilations"] = list(self.dilations)
        return value

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma4MemoryFusionConfig":
        value = dict(data)
        value["dilations"] = tuple(value.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128)))
        return cls(**value)


def text_model(model: nn.Module) -> nn.Module:
    base = getattr(model, "model", model)
    return getattr(base, "language_model", base)


def all_attention_layers(model: nn.Module) -> list[int]:
    return list(range(len(text_model(model).layers)))


def _native_attention(module: nn.Module) -> nn.Module:
    if isinstance(module, DualGemma4Attention):
        return module.original
    return module


def _closest_kv_donor(backbone: nn.Module, layer_idx: int, layer_type: str) -> nn.Module:
    for idx in range(layer_idx, -1, -1):
        candidate = _native_attention(backbone.layers[idx].self_attn)
        if getattr(candidate, "layer_type", None) != layer_type:
            continue
        if hasattr(candidate, "k_proj") and hasattr(candidate, "v_proj"):
            return candidate
    raise RuntimeError(f"no Gemma 4 K/V donor found for layer {layer_idx} ({layer_type})")


class MemoryFusionGemma4Attention(nn.Module):
    """TinyCeNN Memory Fusion replacement for one Gemma 4 text attention layer.

    Gemma 4 E2B uses both sliding and full attention, different head dimensions,
    and shared-K/V layers late in the stack. Every replacement gets its own local
    K/V path. For native shared-K/V layers, that path is initialized from the
    closest preceding non-shared attention of the same type. This makes it possible
    to replace *all* 35 text-attention layers while keeping a stable starting point.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        kv_donor: nn.Module,
        model_config,
        config: Gemma4MemoryFusionConfig,
        layer_idx: int,
    ):
        super().__init__()
        config.validate(model_config)
        text_config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        self.layer_idx = int(layer_idx)
        self.config = getattr(original_attn, "config", text_config)
        self.layer_type = str(getattr(original_attn, "layer_type", text_config.layer_types[layer_idx]))
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = int(text_config.sliding_window) if self.is_sliding else None
        self.hidden_size = int(text_config.hidden_size)
        self.num_heads = int(text_config.num_attention_heads)
        self.head_dim = int(getattr(original_attn, "head_dim"))
        self.num_key_value_heads = int(text_config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_width = self.num_heads * self.head_dim
        self.scaling = float(getattr(original_attn, "scaling", 1.0))
        self.attention_dropout = float(getattr(original_attn, "attention_dropout", 0.0))
        self.is_causal = True
        self.is_kv_shared_layer = False
        self.store_full_length_kv = bool(getattr(original_attn, "store_full_length_kv", False))

        self.q_proj = copy.deepcopy(original_attn.q_proj)
        self.q_norm = copy.deepcopy(original_attn.q_norm)
        self.o_proj = copy.deepcopy(original_attn.o_proj)
        self.k_proj = copy.deepcopy(kv_donor.k_proj)
        self.v_proj = copy.deepcopy(kv_donor.v_proj)
        self.k_norm = copy.deepcopy(kv_donor.k_norm)
        self.v_norm = copy.deepcopy(kv_donor.v_norm)

        self.core = MemoryAugmentedCellularLayer(
            num_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            feature_dim=config.feature_dim,
            variant="cellular_memory_fusion",
            dilations=config.dilations,
            shifted_window=config.shifted_window,
            memory_rank=config.memory_rank,
        )
        self.last_core_output: Tensor | None = None

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings,
        attention_mask=None,
        shared_kv_states=None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        if past_key_values is not None:
            raise RuntimeError("Gemma 4 Memory Fusion V1 requires use_cache=False")
        if position_embeddings is None:
            raise ValueError("Gemma 4 position_embeddings are required")

        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_norm(v)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Preserve Gemma 4's native shared-K/V contract while only some layers have
        # been converted. Later untouched shared-K/V layers can still consume this.
        if self.store_full_length_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (k, v)

        core_out = self.core(q.float(), k.float(), v.float())
        self.last_core_output = core_out
        flat = core_out.transpose(1, 2).reshape(bsz, seq_len, self.attention_width)
        return self.o_proj(flat.to(hidden_states.dtype)), None


class DualGemma4Attention(nn.Module):
    """Keeps native and Memory Fusion attention in one model to avoid two 5B copies."""

    def __init__(self, original: nn.Module, memory: MemoryFusionGemma4Attention):
        super().__init__()
        self.original = original
        self.memory = memory
        self.mode = "original"
        for name in (
            "layer_idx", "layer_type", "is_sliding", "sliding_window", "head_dim",
            "num_key_value_groups", "scaling", "attention_dropout", "is_causal",
            "is_kv_shared_layer", "store_full_length_kv", "config",
        ):
            if hasattr(original, name):
                setattr(self, name, getattr(original, name))
        self.last_call: dict[str, object] = {}

    @staticmethod
    def _detach_tree(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, tuple):
            return tuple(DualGemma4Attention._detach_tree(v) for v in value)
        if isinstance(value, list):
            return [DualGemma4Attention._detach_tree(v) for v in value]
        if isinstance(value, dict):
            return {k: DualGemma4Attention._detach_tree(v) for k, v in value.items()}
        return value

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings,
        attention_mask=None,
        shared_kv_states=None,
        past_key_values=None,
        **kwargs,
    ):
        self.last_call = {
            "hidden_states": hidden_states.detach(),
            "position_embeddings": self._detach_tree(position_embeddings),
            "attention_mask": self._detach_tree(attention_mask),
            "shared_kv_states": self._detach_tree(shared_kv_states),
            "past_key_values": None,
        }
        target = self.original if self.mode == "original" else self.memory
        return target(
            hidden_states,
            position_embeddings,
            attention_mask,
            shared_kv_states,
            past_key_values=past_key_values,
            **kwargs,
        )


def ensure_dual_layer(model: nn.Module, config: Gemma4MemoryFusionConfig, layer_idx: int) -> DualGemma4Attention:
    config.validate(model.config)
    backbone = text_model(model)
    idx = int(layer_idx)
    layer = backbone.layers[idx]
    if isinstance(layer.self_attn, DualGemma4Attention):
        return layer.self_attn
    original = layer.self_attn
    donor = _closest_kv_donor(backbone, idx, str(original.layer_type))
    memory = MemoryFusionGemma4Attention(original, donor, model.config, config, idx)
    device = original.q_proj.weight.device
    dtype = original.q_proj.weight.dtype
    for module in (memory.q_proj, memory.k_proj, memory.v_proj, memory.o_proj, memory.q_norm, memory.k_norm, memory.v_norm):
        module.to(device=device, dtype=dtype)
    memory.core.to(device=device, dtype=torch.float32)
    layer.self_attn = DualGemma4Attention(original, memory)
    model.config.use_cache = False
    backbone.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return layer.self_attn


def ensure_dual_layers(model: nn.Module, config: Gemma4MemoryFusionConfig, layers: Iterable[int]) -> nn.Module:
    for idx in layers:
        ensure_dual_layer(model, config, int(idx))
    return model


def set_attention_mode(model: nn.Module, mode: str) -> None:
    if mode not in {"original", "memory"}:
        raise ValueError("mode must be 'original' or 'memory'")
    for layer in text_model(model).layers:
        if isinstance(layer.self_attn, DualGemma4Attention):
            layer.self_attn.mode = mode


def freeze_current_layer_only(model: nn.Module, layer_idx: int, *, train_output_projection: bool = True) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    module = text_model(model).layers[int(layer_idx)].self_attn
    if not isinstance(module, DualGemma4Attention):
        raise TypeError(f"layer {layer_idx} is not DualGemma4Attention")
    trainable: list[nn.Parameter] = []
    for p in module.memory.core.parameters():
        p.requires_grad = True
        trainable.append(p)
    if train_output_projection:
        for p in module.memory.o_proj.parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def structural_summary(model: nn.Module) -> dict[str, object]:
    backbone = text_model(model)
    dual = [i for i, layer in enumerate(backbone.layers) if isinstance(layer.self_attn, DualGemma4Attention)]
    kinds = list(backbone.config.layer_types)
    return {
        "memory_fusion_layers": dual,
        "remaining_attention_layers": [i for i in range(len(backbone.layers)) if i not in dual],
        "sliding_attention_layers": [i for i, kind in enumerate(kinds) if kind == "sliding_attention"],
        "full_attention_layers": [i for i, kind in enumerate(kinds) if kind == "full_attention"],
    }


def selected_memory_state(model: nn.Module, layers: Iterable[int]) -> dict[str, Tensor]:
    backbone = text_model(model)
    result: dict[str, Tensor] = {}
    for idx in (int(i) for i in layers):
        module = backbone.layers[idx].self_attn
        if not isinstance(module, DualGemma4Attention):
            raise TypeError(f"layer {idx} is not wrapped")
        for key, value in module.memory.state_dict().items():
            result[f"layers.{idx}.memory.{key}"] = value.detach().cpu()
    return result


def load_selected_memory_state(model: nn.Module, state: dict[str, Tensor], layers: Iterable[int]) -> None:
    backbone = text_model(model)
    for idx in (int(i) for i in layers):
        module = backbone.layers[idx].self_attn
        if not isinstance(module, DualGemma4Attention):
            raise TypeError(f"layer {idx} is not wrapped")
        prefix = f"layers.{idx}.memory."
        local = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        incompatible = module.memory.load_state_dict(local, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"layer {idx} checkpoint mismatch: missing={incompatible.missing_keys[:6]} "
                f"unexpected={incompatible.unexpected_keys[:6]}"
            )


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: Gemma4MemoryFusionConfig,
    base_model: str,
    replaced_layers: list[int],
    layer_reports: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected_memory_state(model, replaced_layers), output_dir / "gemma4_memory_fusion.pt")
    payload = {
        "format": FORMAT,
        "base_model": base_model,
        "replaced_layers": list(replaced_layers),
        "memory_fusion": config.to_dict(),
        "layer_reports": layer_reports or [],
        "metadata": metadata or {},
    }
    (output_dir / "gemma4_memory_fusion_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir
