from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

from .memory_attention import MemoryAugmentedCellularLayer

DEFAULT_FUNCTIONGEMMA = "vtava/functiongemma-270m-it-simple-tool-calling"
FORMAT = "functiongemma-memory-fusion-sequential-v1"


@dataclass(frozen=True)
class Gemma3MemoryFusionConfig:
    feature_dim: int = 32
    memory_rank: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    shifted_window: int = 8
    train_output_projection: bool = True

    def validate(self, model_config) -> None:
        config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        if getattr(config, "model_type", None) != "gemma3_text":
            raise ValueError(f"expected gemma3_text, got {getattr(config, 'model_type', None)!r}")
        if self.feature_dim < 4:
            raise ValueError("feature_dim must be >= 4")
        if self.memory_rank < 4:
            raise ValueError("memory_rank must be >= 4")
        if not self.dilations or min(self.dilations) < 1:
            raise ValueError("dilations must be positive")
        if int(config.num_attention_heads) % int(config.num_key_value_heads):
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["dilations"] = list(self.dilations)
        return value

    @classmethod
    def from_dict(cls, data: dict) -> "Gemma3MemoryFusionConfig":
        value = dict(data)
        value["dilations"] = tuple(value.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128)))
        return cls(**value)


class MemoryFusionGemma3Attention(nn.Module):
    """Gemma3 full-attention replacement using the TinyCeNN Memory Fusion core.

    The pretrained Q/K/V/O projections and Gemma3 Q/K RMS normalizers are copied
    exactly. Only original *full-attention* layers are supported in V1; the model's
    sliding-window layers remain untouched. Training and prompt checks use full
    prefixes with ``use_cache=False`` so the experiment is intentionally simple
    and auditable before adding a hybrid recurrent cache.
    """

    def __init__(self, original_attn: nn.Module, model_config, config: Gemma3MemoryFusionConfig, layer_idx: int):
        super().__init__()
        config.validate(model_config)
        text_config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        if bool(getattr(original_attn, "is_sliding", False)):
            raise ValueError("Memory Fusion V1 replaces only Gemma3 full-attention layers")

        self.layer_idx = int(layer_idx)
        self.config = getattr(original_attn, "config", text_config)
        self.is_sliding = False
        self.hidden_size = int(text_config.hidden_size)
        self.num_heads = int(text_config.num_attention_heads)
        self.num_key_value_heads = int(text_config.num_key_value_heads)
        self.head_dim = int(getattr(original_attn, "head_dim", text_config.head_dim))
        self.attention_width = self.num_heads * self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = float(getattr(original_attn, "scaling", self.head_dim ** -0.5))
        self.attention_dropout = float(getattr(original_attn, "attention_dropout", 0.0))
        self.is_causal = bool(getattr(original_attn, "is_causal", True))
        self.attn_logit_softcapping = getattr(original_attn, "attn_logit_softcapping", None)
        self.sliding_window = getattr(original_attn, "sliding_window", None)

        self.q_proj = copy.deepcopy(original_attn.q_proj)
        self.k_proj = copy.deepcopy(original_attn.k_proj)
        self.v_proj = copy.deepcopy(original_attn.v_proj)
        self.o_proj = copy.deepcopy(original_attn.o_proj)
        self.q_norm = copy.deepcopy(original_attn.q_norm)
        self.k_norm = copy.deepcopy(original_attn.k_norm)

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
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        past_key_value=None,
        use_cache: bool = False,
        cache_position=None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        if use_cache or past_key_values is not None or past_key_value is not None:
            raise RuntimeError("FunctionGemma Memory Fusion V1 currently requires use_cache=False")
        if position_embeddings is None:
            raise ValueError("Gemma3 position_embeddings are required")

        bsz, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[-1] != seq_len:
                raise ValueError("only unpadded full causal blocks are supported")
            if bool((attention_mask[..., -1, :] < -1e4).any()):
                raise ValueError("padded batches are not supported")

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        core_out = self.core(q.float(), k.float(), v.float())
        self.last_core_output = core_out
        # Gemma3 can use num_heads * head_dim != hidden_size (FunctionGemma does).
        # The original o_proj maps the attention width back to hidden_size.
        flat = core_out.transpose(1, 2).reshape(bsz, seq_len, self.attention_width)
        return self.o_proj(flat.to(hidden_states.dtype)), None


def full_attention_layers(model: nn.Module) -> list[int]:
    return [
        i for i, layer in enumerate(model.model.layers)
        if not bool(getattr(layer.self_attn, "is_sliding", False))
    ]


def replace_attention_layers(model: nn.Module, config: Gemma3MemoryFusionConfig, layer_indices: Iterable[int]) -> nn.Module:
    config.validate(model.config)
    for raw_idx in layer_indices:
        idx = int(raw_idx)
        layer = model.model.layers[idx]
        if isinstance(layer.self_attn, MemoryFusionGemma3Attention):
            continue
        old = layer.self_attn
        if bool(getattr(old, "is_sliding", False)):
            raise ValueError(f"layer {idx} is sliding_attention; V1 replaces only full_attention")
        device = old.q_proj.weight.device
        projection_dtype = old.q_proj.weight.dtype
        new = MemoryFusionGemma3Attention(old, model.config, config, idx)
        for module in (new.q_proj, new.k_proj, new.v_proj, new.o_proj, new.q_norm, new.k_norm):
            module.to(device=device, dtype=projection_dtype)
        new.core.to(device=device, dtype=torch.float32)
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def freeze_current_layer_only(model: nn.Module, layer_idx: int, *, train_output_projection: bool = True) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    module = model.model.layers[int(layer_idx)].self_attn
    if not isinstance(module, MemoryFusionGemma3Attention):
        raise TypeError(f"layer {layer_idx} is not MemoryFusionGemma3Attention")
    trainable: list[nn.Parameter] = []
    for p in module.core.parameters():
        p.requires_grad = True
        trainable.append(p)
    if train_output_projection:
        for p in module.o_proj.parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def freeze_all_memory_fusion(model: nn.Module, *, train_output_projection: bool = True) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for layer in model.model.layers:
        module = layer.self_attn
        if not isinstance(module, MemoryFusionGemma3Attention):
            continue
        for p in module.core.parameters():
            p.requires_grad = True
            trainable.append(p)
        if train_output_projection:
            for p in module.o_proj.parameters():
                p.requires_grad = True
                trainable.append(p)
    return trainable


def structural_summary(model: nn.Module) -> dict[str, object]:
    fusion = [
        i for i, layer in enumerate(model.model.layers)
        if isinstance(layer.self_attn, MemoryFusionGemma3Attention)
    ]
    full = [
        i for i, layer in enumerate(model.model.layers)
        if not isinstance(layer.self_attn, MemoryFusionGemma3Attention)
        and not bool(getattr(layer.self_attn, "is_sliding", False))
    ]
    sliding = [
        i for i, layer in enumerate(model.model.layers)
        if not isinstance(layer.self_attn, MemoryFusionGemma3Attention)
        and bool(getattr(layer.self_attn, "is_sliding", False))
    ]
    return {
        "memory_fusion_layers": fusion,
        "remaining_full_attention_layers": full,
        "sliding_attention_layers": sliding,
    }


def selected_attention_state(model: nn.Module, layers: Iterable[int]) -> dict[str, Tensor]:
    prefixes = tuple(f"model.layers.{int(i)}.self_attn." for i in layers)
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if prefixes and key.startswith(prefixes)
    }


def save_adapter(model: nn.Module, output_dir: str | Path, *, config: Gemma3MemoryFusionConfig, base_model: str, accepted_layers: list[int], metadata: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected_attention_state(model, accepted_layers), output_dir / "functiongemma_memory_fusion.pt")
    payload = {
        "format": FORMAT,
        "base_model": base_model,
        "accepted_layers": list(accepted_layers),
        "memory_fusion": config.to_dict(),
        "metadata": metadata or {},
    }
    (output_dir / "functiongemma_memory_fusion_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir
