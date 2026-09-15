from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from .memory_attention import MemoryAugmentedCellularLayer

DEFAULT_SMOLLM2 = "HuggingFaceTB/SmolLM2-135M"


@dataclass(frozen=True)
class SmolMemoryFusionConfig:
    feature_dim: int = 32
    memory_rank: int = 48
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    shifted_window: int = 8
    train_output_projection: bool = True

    def validate(self, model_config) -> None:
        if self.feature_dim < 4:
            raise ValueError("feature_dim must be >= 4")
        if self.memory_rank < 4:
            raise ValueError("memory_rank must be >= 4")
        if not self.dilations or min(self.dilations) < 1:
            raise ValueError("dilations must be positive")
        if int(model_config.num_attention_heads) % int(model_config.num_key_value_heads):
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["dilations"] = list(self.dilations)
        return value

    @classmethod
    def from_dict(cls, data: dict) -> "SmolMemoryFusionConfig":
        value = dict(data)
        value["dilations"] = tuple(value.get("dilations", (1, 2, 4, 8, 16, 32, 64, 128)))
        return cls(**value)


class MemoryFusionLlamaAttention(nn.Module):
    """Drop-in no-cache Llama attention using TinyCeNN Memory Fusion.

    The pretrained Q/K/V/O projections are copied exactly. The new trainable core
    combines adaptive-MaxPool Cellular attention, a Hedgehog-style global linear
    path, and GDN2-style editable memory. The core stays in float32 for numerical
    stability while the surrounding SmolLM2 projections can remain bf16/fp16.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        model_config,
        config: SmolMemoryFusionConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        config.validate(model_config)
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.head_dim = int(getattr(model_config, "head_dim", self.hidden_size // self.num_heads))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = copy.deepcopy(original_attn.q_proj)
        self.k_proj = copy.deepcopy(original_attn.k_proj)
        self.v_proj = copy.deepcopy(original_attn.v_proj)
        self.o_proj = copy.deepcopy(original_attn.o_proj)

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

    def _apply_rope(self, q: Tensor, k: Tensor, position_embeddings) -> tuple[Tensor, Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for Memory Fusion attention")
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        cos, sin = position_embeddings
        return apply_rotary_pos_emb(q, k, cos, sin)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        past_key_value=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        if use_cache or past_key_values is not None or past_key_value is not None:
            raise RuntimeError("SmolLM2 Memory Fusion currently requires use_cache=False")
        bsz, seq_len, _ = hidden_states.shape

        if attention_mask is not None and attention_mask.ndim == 4:
            if attention_mask.shape[-1] != seq_len:
                raise ValueError("attention mask length mismatch")
            if bool((attention_mask[..., -1, :] < -1e4).any()):
                raise ValueError("padded batches are not supported by Memory Fusion attention")

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        core_out = self.core(q.float(), k.float(), v.float())
        self.last_core_output = core_out
        flat = core_out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        out = self.o_proj(flat.to(dtype=hidden_states.dtype))
        return out, None


def replace_attention_layers(
    model: nn.Module,
    config: SmolMemoryFusionConfig,
    layer_indices: Iterable[int],
) -> nn.Module:
    config.validate(model.config)
    for raw_idx in layer_indices:
        idx = int(raw_idx)
        layer = model.model.layers[idx]
        if isinstance(layer.self_attn, MemoryFusionLlamaAttention):
            continue
        old = layer.self_attn
        device = old.q_proj.weight.device
        projection_dtype = old.q_proj.weight.dtype
        new = MemoryFusionLlamaAttention(old, model.config, config, idx)
        for projection in (new.q_proj, new.k_proj, new.v_proj, new.o_proj):
            projection.to(device=device, dtype=projection_dtype)
        new.core.to(device=device, dtype=torch.float32)
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def replace_all_attention(model: nn.Module, config: SmolMemoryFusionConfig) -> nn.Module:
    return replace_attention_layers(model, config, range(int(model.config.num_hidden_layers)))


def freeze_for_group_calibration(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    train_output_projection: bool = True,
) -> list[nn.Parameter]:
    selected = {int(x) for x in layer_indices}
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, MemoryFusionLlamaAttention) and module.layer_idx in selected:
            for p in module.core.parameters():
                p.requires_grad = True
                trainable.append(p)
            if train_output_projection:
                for p in module.o_proj.parameters():
                    p.requires_grad = True
                    trainable.append(p)
    return trainable


def freeze_for_global_training(
    model: nn.Module,
    *,
    train_output_projection: bool = True,
) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, MemoryFusionLlamaAttention):
            for p in module.core.parameters():
                p.requires_grad = True
                trainable.append(p)
            if train_output_projection:
                for p in module.o_proj.parameters():
                    p.requires_grad = True
                    trainable.append(p)
    return trainable


def parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cores = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, MemoryFusionLlamaAttention)
        for p in m.core.parameters()
    )
    output_proj = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, MemoryFusionLlamaAttention)
        for p in m.o_proj.parameters()
    )
    return {
        "total": total,
        "trainable": trainable,
        "memory_fusion_cores": cores,
        "output_projections": output_proj,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def structural_summary(model: nn.Module) -> dict[str, int]:
    fusion = sum(isinstance(m, MemoryFusionLlamaAttention) for m in model.modules())
    transformer = sum(
        m.__class__.__name__ == "LlamaAttention"
        for m in model.modules()
        if not isinstance(m, MemoryFusionLlamaAttention)
    )
    return {"memory_fusion_layers": fusion, "transformer_attention_layers": transformer}


def structural_assertions(model: nn.Module) -> None:
    expected = int(model.config.num_hidden_layers)
    summary = structural_summary(model)
    if summary["memory_fusion_layers"] != expected:
        raise RuntimeError(f"expected {expected} Memory Fusion layers, found {summary['memory_fusion_layers']}")
    if summary["transformer_attention_layers"]:
        raise RuntimeError("Transformer self-attention remains in the final student")


def _attention_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".self_attn." in name:
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("no Memory Fusion attention state found")
    return state


def save_smollm2_memory_fusion(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: SmolMemoryFusionConfig,
    base_model: str = DEFAULT_SMOLLM2,
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_attention_state(model), output_dir / "smollm2_memory_fusion.pt")
    meta = {
        "format_version": 1,
        "architecture": "smollm2-memory-fusion",
        "base_model": base_model,
        "memory_fusion": config.to_dict(),
    }
    if extra_metadata:
        meta["training"] = extra_metadata
    (output_dir / "smollm2_memory_fusion_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return output_dir


def load_smollm2_memory_fusion_weights(model: nn.Module, student_dir: str | Path) -> nn.Module:
    path = Path(student_dir) / "smollm2_memory_fusion.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_attention_state(model))
    missing = [k for k in incompatible.missing_keys if k in expected]
    unexpected = [k for k in incompatible.unexpected_keys if k not in expected]
    if missing:
        raise RuntimeError(f"missing Memory Fusion keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected Memory Fusion keys: {unexpected[:8]}")
    return model


def build_smollm2_memory_fusion(student_dir: str | Path, *, device=None, dtype=None):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    meta = json.loads((student_dir / "smollm2_memory_fusion_config.json").read_text())
    if meta.get("architecture") != "smollm2-memory-fusion":
        raise ValueError("checkpoint is not SmolLM2 Memory Fusion")
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(meta["base_model"], **kwargs)
    if device is not None:
        model.to(device)
    config = SmolMemoryFusionConfig.from_dict(meta["memory_fusion"])
    replace_all_attention(model, config)
    load_smollm2_memory_fusion_weights(model, student_dir)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model
