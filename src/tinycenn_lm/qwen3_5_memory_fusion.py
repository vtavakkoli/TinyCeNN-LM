from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from .memory_attention import MemoryAugmentedCellularLayer

DEFAULT_QWEN35 = "Qwen/Qwen3.5-0.8B"
FORMAT = "qwen3.5-memory-fusion-sequential-v1"


@dataclass(frozen=True)
class Qwen35MemoryFusionConfig:
    feature_dim: int = 32
    memory_rank: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    shifted_window: int = 8
    train_output_projection: bool = True

    def validate(self, model_config) -> None:
        config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        if getattr(config, "model_type", None) != "qwen3_5_text":
            raise ValueError(f"expected qwen3_5_text, got {getattr(config, 'model_type', None)!r}")
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
    def from_dict(cls, data: dict) -> "Qwen35MemoryFusionConfig":
        value = dict(data)
        value["dilations"] = tuple(value.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128)))
        return cls(**value)


class MemoryFusionQwen35Attention(nn.Module):
    """Replace a Qwen3.5 *full-attention* anchor with TinyCeNN Memory Fusion.

    Qwen3.5 already contains native Gated DeltaNet linear-attention layers. This
    adapter deliberately leaves those layers untouched and only targets the
    original full-attention anchors. The pretrained Q/K/V/O projections, Q/K
    RMS normalizers, partial RoPE path, and Qwen attention-output gate are kept.

    V1 is a research/full-prefix implementation and requires ``use_cache=False``.
    """

    def __init__(self, original_attn: nn.Module, model_config, config: Qwen35MemoryFusionConfig, layer_idx: int):
        super().__init__()
        config.validate(model_config)
        text_config = model_config.get_text_config(decoder=True) if hasattr(model_config, "get_text_config") else model_config
        self.layer_idx = int(layer_idx)
        self.config = getattr(original_attn, "config", text_config)
        self.hidden_size = int(text_config.hidden_size)
        self.num_heads = int(text_config.num_attention_heads)
        self.num_key_value_heads = int(text_config.num_key_value_heads)
        self.head_dim = int(getattr(original_attn, "head_dim", text_config.head_dim))
        self.attention_width = self.num_heads * self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = float(getattr(original_attn, "scaling", self.head_dim ** -0.5))
        self.attention_dropout = float(getattr(original_attn, "attention_dropout", 0.0))
        self.is_causal = True
        self.layer_type = "full_attention"

        # Qwen3.5 q_proj emits both query and output-gate channels.
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
            raise RuntimeError("Qwen3.5 Memory Fusion V1 currently requires use_cache=False")
        if position_embeddings is None:
            raise ValueError("Qwen3.5 position_embeddings are required")

        bsz, seq_len, _ = hidden_states.shape
        q_and_gate = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim * 2
        )
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(bsz, seq_len, self.attention_width)

        q = self.q_norm(query).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(
                bsz, seq_len, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        core_out = self.core(q.float(), k.float(), v.float())
        self.last_core_output = core_out
        flat = core_out.transpose(1, 2).reshape(bsz, seq_len, self.attention_width)
        flat = flat.to(hidden_states.dtype) * torch.sigmoid(gate).to(hidden_states.dtype)
        return self.o_proj(flat), None


def text_model(model: nn.Module) -> nn.Module:
    """Return Qwen3.5's text decoder for text-only or multimodal wrappers."""
    base = getattr(model, "model", model)
    return getattr(base, "language_model", base)


def full_attention_layers(model: nn.Module) -> list[int]:
    backbone = text_model(model)
    return [
        i for i, kind in enumerate(backbone.config.layer_types)
        if kind == "full_attention"
    ]


def replace_attention_layers(model: nn.Module, config: Qwen35MemoryFusionConfig, layer_indices: Iterable[int]) -> nn.Module:
    config.validate(model.config)
    backbone = text_model(model)
    available = set(full_attention_layers(model))
    for raw_idx in layer_indices:
        idx = int(raw_idx)
        if idx not in available:
            raise ValueError(
                f"layer {idx} is not a Qwen3.5 full-attention anchor; available={sorted(available)}"
            )
        layer = backbone.layers[idx]
        if isinstance(layer.self_attn, MemoryFusionQwen35Attention):
            continue
        old = layer.self_attn
        device = old.q_proj.weight.device
        projection_dtype = old.q_proj.weight.dtype
        new = MemoryFusionQwen35Attention(old, model.config, config, idx)
        for module in (new.q_proj, new.k_proj, new.v_proj, new.o_proj, new.q_norm, new.k_norm):
            module.to(device=device, dtype=projection_dtype)
        # Keep the research memory core numerically stable in FP32.
        new.core.to(device=device, dtype=torch.float32)
        layer.self_attn = new
    model.config.use_cache = False
    backbone.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def freeze_current_layer_only(model: nn.Module, layer_idx: int, *, train_output_projection: bool = True) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    module = text_model(model).layers[int(layer_idx)].self_attn
    if not isinstance(module, MemoryFusionQwen35Attention):
        raise TypeError(f"layer {layer_idx} is not MemoryFusionQwen35Attention")
    trainable: list[nn.Parameter] = []
    for p in module.core.parameters():
        p.requires_grad = True
        trainable.append(p)
    if train_output_projection:
        for p in module.o_proj.parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def structural_summary(model: nn.Module) -> dict[str, object]:
    backbone = text_model(model)
    fusion = [
        i for i, layer in enumerate(backbone.layers)
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, MemoryFusionQwen35Attention)
    ]
    remaining_full = [
        i for i, kind in enumerate(backbone.config.layer_types)
        if kind == "full_attention" and i not in fusion
    ]
    linear = [
        i for i, kind in enumerate(backbone.config.layer_types)
        if kind == "linear_attention"
    ]
    return {
        "memory_fusion_layers": fusion,
        "remaining_full_attention_layers": remaining_full,
        "native_linear_attention_layers": linear,
    }


def selected_attention_state(model: nn.Module, layers: Iterable[int]) -> dict[str, Tensor]:
    backbone = text_model(model)
    result: dict[str, Tensor] = {}
    for idx in (int(i) for i in layers):
        module = backbone.layers[idx].self_attn
        for key, value in module.state_dict().items():
            result[f"layers.{idx}.self_attn.{key}"] = value.detach().cpu()
    return result


def load_selected_attention_state(model: nn.Module, state: dict[str, Tensor], layers: Iterable[int]) -> None:
    backbone = text_model(model)
    for idx in (int(i) for i in layers):
        prefix = f"layers.{idx}.self_attn."
        local = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        incompatible = backbone.layers[idx].self_attn.load_state_dict(local, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"layer {idx} checkpoint mismatch: missing={incompatible.missing_keys[:6]} "
                f"unexpected={incompatible.unexpected_keys[:6]}"
            )


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: Qwen35MemoryFusionConfig,
    base_model: str,
    accepted_layers: list[int],
    metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        selected_attention_state(model, accepted_layers),
        output_dir / "qwen35_memory_fusion.pt",
    )
    payload = {
        "format": FORMAT,
        "base_model": base_model,
        "accepted_layers": list(accepted_layers),
        "memory_fusion": config.to_dict(),
        "metadata": metadata or {},
    }
    (output_dir / "qwen35_memory_fusion_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return output_dir
