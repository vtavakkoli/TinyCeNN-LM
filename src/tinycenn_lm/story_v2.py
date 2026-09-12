from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modeling import DEFAULT_BASE_MODEL, _get_decoder_layers
from .sharded_moe import (
    ShardedMoECeNNConfig,
    ShardedMoECeNNReplacementLayer,
    build_sharded_moe_student,
)


@dataclass(frozen=True)
class StoryV2Config:
    memory_rank: int = 32
    head_rank: int = 4

    def validate(self, hidden_size: int, vocab_size: int) -> None:
        if not 1 <= self.memory_rank <= hidden_size:
            raise ValueError("memory_rank must be in [1, hidden_size]")
        if not 1 <= self.head_rank <= min(hidden_size, vocab_size):
            raise ValueError("head_rank must be positive and <= hidden/vocab size")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StoryV2Config":
        return cls(**dict(data))


class CausalStoryMemory(nn.Module):
    """Cheap global prefix memory with no attention and no token-by-token Python loop.

    For position t we compute the cumulative mean of hidden states 0..t, then pass
    it through a small bottleneck adapter. The up projection is zero initialized,
    so upgrading a trained Story-v1 checkpoint is exactly function preserving.
    """

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        work = hidden_states.float()
        prefix_sum = work.cumsum(dim=1)
        denom = torch.arange(
            1,
            hidden_states.shape[1] + 1,
            device=hidden_states.device,
            dtype=work.dtype,
        ).view(1, -1, 1)
        prefix_mean = (prefix_sum / denom).to(dtype=dtype)
        return self.up(F.silu(self.down(prefix_mean)))


class StoryV2ReplacementLayer(nn.Module):
    def __init__(
        self,
        cenn: nn.Module,
        hidden_size: int,
        story_config: StoryV2Config,
    ) -> None:
        super().__init__()
        self.cenn = cenn
        self.story_memory = CausalStoryMemory(hidden_size, story_config.memory_rank)

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        if kwargs.get("use_cache", False):
            raise RuntimeError("TinyCeNN Story-v2 requires use_cache=False")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("TinyCeNN Story-v2 has no attention matrices")
        memory = self.story_memory(hidden_states)
        enriched = hidden_states + memory
        return hidden_states + self.cenn(enriched)


class LowRankLMHeadAdapter(nn.Module):
    """Frozen language-model head plus a tiny trainable low-rank story adapter."""

    def __init__(self, base_head: nn.Module, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.base_head = base_head
        for parameter in self.base_head.parameters():
            parameter.requires_grad = False
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    @property
    def weight(self):
        return getattr(self.base_head, "weight", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        base_logits = self.base_head(hidden_states)
        delta = self.up(F.silu(self.down(hidden_states)))
        return base_logits + delta


def upgrade_sharded_model_to_story_v2(model: nn.Module, story_config: StoryV2Config) -> nn.Module:
    hidden_size = int(model.config.hidden_size)
    vocab_size = int(model.config.vocab_size)
    story_config.validate(hidden_size, vocab_size)

    layers = _get_decoder_layers(model)
    replaced = 0
    for index, layer in enumerate(list(layers)):
        if isinstance(layer, ShardedMoECeNNReplacementLayer):
            layers[index] = StoryV2ReplacementLayer(layer.cenn, hidden_size, story_config)
            replaced += 1
    if replaced == 0:
        raise RuntimeError("no Sharded MoE-CeNN layer found to upgrade")

    base_head = model.get_output_embeddings()
    if base_head is None:
        raise RuntimeError("base model has no output embedding/head")
    if not isinstance(base_head, LowRankLMHeadAdapter):
        model.set_output_embeddings(
            LowRankLMHeadAdapter(
                base_head,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                rank=story_config.head_rank,
            )
        )
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def freeze_story_v2_interfaces(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, StoryV2ReplacementLayer):
            for parameter in module.cenn.parameters():
                parameter.requires_grad = True
            for parameter in module.story_memory.parameters():
                parameter.requires_grad = True
        elif isinstance(module, LowRankLMHeadAdapter):
            for parameter in module.down.parameters():
                parameter.requires_grad = True
            for parameter in module.up.parameters():
                parameter.requires_grad = True


def story_v2_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, CausalStoryMemory)
        for p in m.parameters()
    )
    head_adapter = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, LowRankLMHeadAdapter)
        for sub in (m.down, m.up)
        for p in sub.parameters()
    )
    return {
        "total": total,
        "trainable": trainable,
        "memory": memory,
        "head_adapter": head_adapter,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def _story_v2_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".cenn." in name or ".story_memory." in name:
            state[name] = tensor.detach().cpu()
        elif ".lm_head.down." in name or ".lm_head.up." in name:
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("no Story-v2 trainable state found")
    return state


def save_story_v2_student(
    model: nn.Module,
    output_dir: str | Path,
    *,
    story_config: StoryV2Config,
    sharded_config: ShardedMoECeNNConfig,
    base_model: str = DEFAULT_BASE_MODEL,
    layer_indices: Sequence[int] = (0,),
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_story_v2_state_dict(model), output_dir / "story_v2_student.pt")
    metadata = {
        "format_version": 1,
        "architecture": "tinycenn-story-v2-memory-head",
        "base_model": base_model,
        "layer_indices": list(layer_indices),
        "sharded_moe_cenn": sharded_config.to_dict(),
        "story_v2": story_config.to_dict(),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "story_v2_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir


def load_story_v2_weights(model: nn.Module, student_dir: str | Path) -> nn.Module:
    state = torch.load(Path(student_dir) / "story_v2_student.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_story_v2_state_dict(model))
    missing = [key for key in incompatible.missing_keys if key in expected]
    unexpected = [key for key in incompatible.unexpected_keys if key not in expected]
    if missing:
        raise RuntimeError(f"missing Story-v2 keys: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected Story-v2 keys: {unexpected}")
    return model


def build_story_v2_student(
    student_dir: str | Path,
    *,
    device=None,
    dtype=None,
    attn_implementation: str = "sdpa",
):
    from transformers import AutoModelForCausalLM
    from .sharded_moe import replace_transformer_with_sharded_moe_cenn

    student_dir = Path(student_dir)
    metadata = json.loads((student_dir / "story_v2_config.json").read_text())
    if metadata.get("architecture") != "tinycenn-story-v2-memory-head":
        raise ValueError("checkpoint is not a TinyCeNN Story-v2 model")

    kwargs = {"attn_implementation": attn_implementation}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    sharded_config = ShardedMoECeNNConfig.from_dict(metadata["sharded_moe_cenn"])
    replace_transformer_with_sharded_moe_cenn(
        model, sharded_config, tuple(metadata["layer_indices"])
    )
    story_config = StoryV2Config.from_dict(metadata["story_v2"])
    upgrade_sharded_model_to_story_v2(model, story_config)
    load_story_v2_weights(model, student_dir)

    move_kwargs = {}
    if device is not None:
        move_kwargs["device"] = device
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if move_kwargs:
        model.to(**move_kwargs)
    model.config.use_cache = False
    return model


def build_story_v2_from_story_v1(
    story_v1_dir: str | Path,
    *,
    story_config: StoryV2Config,
    device=None,
    dtype=None,
):
    model = build_sharded_moe_student(story_v1_dir, device=device, dtype=dtype)
    upgrade_sharded_model_to_story_v2(model, story_config)
    freeze_story_v2_interfaces(model)
    return model
