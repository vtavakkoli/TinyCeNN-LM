from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cenn import CeNNConfig, CausalDepthwiseNeighborhood, StableRMSNorm
from .modeling import DEFAULT_BASE_MODEL, _get_decoder_layers


@dataclass(frozen=True)
class ShardedMoECeNNConfig:
    """Parameter-neutral routed FFN built by partitioning one dense CeNN FFN.

    The original dense SwiGLU hidden dimension is split across ``num_shards``.
    Therefore all shard parameters together equal one dense FFN (plus a tiny
    router and one scalar route-mix parameter), rather than ``num_shards`` full
    copies of the FFN.
    """

    hidden_size: int = 192
    kernel_size: int = 3
    expansion: int = 4
    steps: int = 7
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    num_shards: int = 8
    top_k: int = 2
    router_noise_std: float = 1e-3

    @property
    def dense_inner(self) -> int:
        return self.hidden_size * self.expansion

    @property
    def shard_inner(self) -> int:
        return self.dense_inner // self.num_shards

    def validate(self) -> None:
        CeNNConfig(
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            expansion=self.expansion,
            steps=self.steps,
            dilations=self.dilations,
            rms_norm_eps=self.rms_norm_eps,
            dropout=self.dropout,
        ).validate()
        if self.num_shards < 2:
            raise ValueError("num_shards must be >= 2")
        if self.dense_inner % self.num_shards:
            raise ValueError(
                f"dense FFN inner size {self.dense_inner} must divide evenly by "
                f"num_shards={self.num_shards}"
            )
        if not 1 <= self.top_k <= self.num_shards:
            raise ValueError("top_k must be in [1, num_shards]")
        if self.router_noise_std < 0:
            raise ValueError("router_noise_std must be >= 0")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["dilations"] = list(self.dilations)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ShardedMoECeNNConfig":
        data = dict(data)
        if "dilations" in data:
            data["dilations"] = tuple(data["dilations"])
        return cls(**data)


class TopKShardRouter(nn.Module):
    def __init__(self, hidden_size: int, num_shards: int, top_k: int, noise_std: float) -> None:
        super().__init__()
        self.num_shards = num_shards
        self.top_k = top_k
        self.proj = nn.Linear(hidden_size, num_shards, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=noise_std)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        logits = self.proj(x).float()
        probs = F.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        assignment = F.one_hot(top_indices, num_classes=self.num_shards).float().sum(dim=-2)
        assignment = assignment / float(self.top_k)
        shard_fraction = assignment.mean(dim=(0, 1))
        probability_fraction = probs.mean(dim=(0, 1))
        load_balance = self.num_shards * torch.sum(shard_fraction * probability_fraction)
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return top_indices, top_weights.to(dtype=x.dtype), {
            "load_balance": load_balance,
            "z_loss": z_loss,
            "entropy": entropy,
            "shard_fraction": shard_fraction,
            "probability_fraction": probability_fraction,
        }


class ShardedSwiGLU(nn.Module):
    """One dense SwiGLU split into parameter-neutral channel shards.

    All shards are evaluated to reconstruct the complete dense FFN. Top-k routing
    produces a sparse, scaled estimate of that same FFN. A zero-initialized scalar
    learns how much of the sparse routed specialization to mix into the complete
    FFN. At initialization this module is exactly the original dense FFN.
    """

    def __init__(self, config: ShardedMoECeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        e = config.num_shards
        h = config.hidden_size
        s = config.shard_inner

        # Vectorized expert-shard weights. Across all 8 shards these contain
        # exactly the same number of parameters as one dense SwiGLU FFN.
        self.in_weight = nn.Parameter(torch.empty(e, 2 * s, h))
        self.out_weight = nn.Parameter(torch.empty(e, h, s))
        nn.init.kaiming_uniform_(self.in_weight, a=5**0.5)
        nn.init.zeros_(self.out_weight)

        self.router = TopKShardRouter(h, e, config.top_k, config.router_noise_std)
        self.route_mix = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(config.dropout)
        self.last_router_stats: dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        # [B,S,H] -> [B,S,E,2I]
        projected = torch.einsum("bsh,eih->bsei", x, self.in_weight)
        a, b = projected.chunk(2, dim=-1)
        hidden = F.silu(a) * b
        # [B,S,E,I] x [E,H,I] -> [B,S,E,H]
        shard_outputs = torch.einsum("bsei,ehi->bseh", hidden, self.out_weight)
        shard_outputs = self.dropout(shard_outputs)

        # Complete dense-FFN reconstruction from all disjoint shards.
        dense_full = shard_outputs.sum(dim=-2)

        top_idx, top_weight, stats = self.router(x)
        gather_index = top_idx.unsqueeze(-1).expand(*top_idx.shape, x.shape[-1])
        selected = torch.gather(shard_outputs, dim=-2, index=gather_index)
        routed = (selected * top_weight.unsqueeze(-1)).sum(dim=-2)

        # Scale the Top-k estimate to the full shard count. route_mix starts at
        # exactly zero, so warm-start output exactly equals the trained dense FFN.
        sparse_scaled = routed * (self.config.num_shards / float(self.config.top_k))
        route_delta = sparse_scaled - dense_full
        mixed = dense_full + self.route_mix * route_delta

        self.last_router_stats = {
            **stats,
            "route_mix": self.route_mix,
        }
        return mixed


class ShardedMoESharedCeNNCell(nn.Module):
    def __init__(self, config: ShardedMoECeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.norm = StableRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.neighborhood = CausalDepthwiseNeighborhood(config.hidden_size, config.kernel_size)
        self.ffn = ShardedSwiGLU(config)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, state: Tensor, dilation: int, step_scale: float) -> tuple[Tensor, dict[str, Tensor]]:
        x = self.norm(state)
        local = self.neighborhood(x, dilation=dilation)
        update = self.ffn(local)
        gate = torch.sigmoid(self.gate_proj(local))
        return state + step_scale * gate * update, self.ffn.last_router_stats


class FastShardedMoECeNNCore(nn.Module):
    def __init__(self, config: ShardedMoECeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.cell = ShardedMoESharedCeNNCell(config)
        self.last_router_stats: dict[str, Tensor] = {}

    def forward(self, hidden_states: Tensor) -> Tensor:
        initial = hidden_states
        state = hidden_states
        step_scale = self.config.steps ** -0.5
        scalar_accum: dict[str, Tensor] = {}
        shard_fraction = None
        probability_fraction = None
        for step in range(self.config.steps):
            dilation = self.config.dilations[step % len(self.config.dilations)]
            state, stats = self.cell(state, dilation=dilation, step_scale=step_scale)
            for key in ("load_balance", "z_loss", "entropy"):
                scalar_accum[key] = scalar_accum.get(key, stats[key].new_zeros(())) + stats[key]
            shard_fraction = stats["shard_fraction"] if shard_fraction is None else shard_fraction + stats["shard_fraction"]
            probability_fraction = stats["probability_fraction"] if probability_fraction is None else probability_fraction + stats["probability_fraction"]

        self.last_router_stats = {
            "load_balance": scalar_accum["load_balance"] / self.config.steps,
            "z_loss": scalar_accum["z_loss"] / self.config.steps,
            "entropy": scalar_accum["entropy"] / self.config.steps,
            "shard_fraction": shard_fraction / self.config.steps,
            "probability_fraction": probability_fraction / self.config.steps,
            "route_mix": self.cell.ffn.route_mix,
        }
        return state - initial

    @property
    def receptive_field(self) -> int:
        radius = sum(self.config.dilations[i % len(self.config.dilations)] for i in range(self.config.steps))
        return 1 + (self.config.kernel_size - 1) * radius


class ShardedMoECeNNReplacementLayer(nn.Module):
    def __init__(self, config: ShardedMoECeNNConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.cenn = FastShardedMoECeNNCore(config)
        if device is not None or dtype is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            self.cenn.to(**kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        if kwargs.get("use_cache", False):
            raise RuntimeError("Sharded MoE-CeNN student requires use_cache=False")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("Sharded MoE-CeNN student has no attention matrices")
        return hidden_states + self.cenn(hidden_states)


def replace_transformer_with_sharded_moe_cenn(
    model: nn.Module,
    config: ShardedMoECeNNConfig,
    layer_indices: Sequence[int] = (0,),
) -> nn.Module:
    layers = _get_decoder_layers(model)
    if config.hidden_size != int(model.config.hidden_size):
        raise ValueError("Sharded MoE-CeNN hidden size does not match base model")
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    for index in layer_indices:
        old = layers[index]
        reference = next((p for p in old.parameters() if p.is_floating_point()), None)
        layers[index] = ShardedMoECeNNReplacementLayer(
            config,
            device=reference.device if reference is not None else None,
            dtype=reference.dtype if reference is not None else None,
        )
    return model


def freeze_sharded_moe_interfaces(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, ShardedMoECeNNReplacementLayer):
            for parameter in module.parameters():
                parameter.requires_grad = True


def warmstart_sharded_moe_from_plain_cenn(model: nn.Module, plain_student_dir: str | Path) -> None:
    """Slice the trained dense CeNN FFN exactly across 8 routed shards."""
    state = torch.load(Path(plain_student_dir) / "cenn_student.pt", map_location="cpu", weights_only=True)
    layer = next((m for m in model.modules() if isinstance(m, ShardedMoECeNNReplacementLayer)), None)
    if layer is None:
        raise RuntimeError("Sharded MoE-CeNN replacement layer not found")

    # Locate source prefix independently of the wrapper path.
    source_prefix = next(
        (name.rsplit("in_proj.weight", 1)[0] for name in state if name.endswith(".cenn.cell.in_proj.weight")),
        None,
    )
    if source_prefix is None:
        raise RuntimeError("plain CeNN checkpoint does not contain dense FFN weights")

    source_in = state[source_prefix + "in_proj.weight"]
    source_out = state[source_prefix + "out_proj.weight"]
    source_norm = state[source_prefix + "norm.weight"]
    source_neighborhood = state[source_prefix + "neighborhood.weight"]
    source_gate_w = state[source_prefix + "gate_proj.weight"]
    source_gate_b = state[source_prefix + "gate_proj.bias"]

    cfg = layer.config
    inner = cfg.dense_inner
    shard = cfg.shard_inner
    with torch.no_grad():
        layer.cenn.cell.norm.weight.copy_(source_norm.to(layer.cenn.cell.norm.weight))
        layer.cenn.cell.neighborhood.weight.copy_(source_neighborhood.to(layer.cenn.cell.neighborhood.weight))
        layer.cenn.cell.gate_proj.weight.copy_(source_gate_w.to(layer.cenn.cell.gate_proj.weight))
        layer.cenn.cell.gate_proj.bias.copy_(source_gate_b.to(layer.cenn.cell.gate_proj.bias))
        for expert_id in range(cfg.num_shards):
            lo = expert_id * shard
            hi = lo + shard
            layer.cenn.cell.ffn.in_weight[expert_id, :shard].copy_(
                source_in[lo:hi].to(layer.cenn.cell.ffn.in_weight)
            )
            layer.cenn.cell.ffn.in_weight[expert_id, shard:].copy_(
                source_in[inner + lo : inner + hi].to(layer.cenn.cell.ffn.in_weight)
            )
            layer.cenn.cell.ffn.out_weight[expert_id].copy_(
                source_out[:, lo:hi].to(layer.cenn.cell.ffn.out_weight)
            )
        layer.cenn.cell.ffn.route_mix.zero_()


def sharded_router_stats(model: nn.Module) -> dict[str, Tensor]:
    layer = next((m for m in model.modules() if isinstance(m, ShardedMoECeNNReplacementLayer)), None)
    if layer is None or not layer.cenn.last_router_stats:
        raise RuntimeError("router statistics unavailable; run a forward pass first")
    return layer.cenn.last_router_stats


def _state_dict(model: nn.Module) -> dict[str, Tensor]:
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if ".cenn." in name}
    if not state:
        raise ValueError("no Sharded MoE-CeNN weights found")
    return state


def save_sharded_moe_student(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: ShardedMoECeNNConfig,
    base_model: str = DEFAULT_BASE_MODEL,
    layer_indices: Sequence[int] = (0,),
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict(model), output_dir / "sharded_moe_cenn_student.pt")
    metadata = {
        "format_version": 1,
        "architecture": "sharded-moe-cenn-top2-replacement",
        "base_model": base_model,
        "layer_indices": list(layer_indices),
        "sharded_moe_cenn": config.to_dict(),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "sharded_moe_student_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir


def load_sharded_moe_student_weights(model: nn.Module, student_dir: str | Path, *, map_location="cpu", strict: bool = True) -> nn.Module:
    state = torch.load(Path(student_dir) / "sharded_moe_cenn_student.pt", map_location=map_location, weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_state_dict(model))
    missing = [key for key in incompatible.missing_keys if key in expected]
    unexpected = [key for key in incompatible.unexpected_keys if key not in expected]
    if strict and missing:
        raise RuntimeError(f"missing Sharded MoE-CeNN keys: {missing}")
    if strict and unexpected:
        raise RuntimeError(f"unexpected Sharded MoE-CeNN keys: {unexpected}")
    return model


def build_sharded_moe_student(student_dir: str | Path, *, device=None, dtype=None, attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    metadata = json.loads((student_dir / "sharded_moe_student_config.json").read_text())
    if metadata.get("architecture") != "sharded-moe-cenn-top2-replacement":
        raise ValueError("checkpoint is not a Sharded MoE-CeNN Top-2 student")
    kwargs = {"attn_implementation": attn_implementation}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    config = ShardedMoECeNNConfig.from_dict(metadata["sharded_moe_cenn"])
    replace_transformer_with_sharded_moe_cenn(model, config, tuple(metadata["layer_indices"]))
    load_sharded_moe_student_weights(model, student_dir)
    move_kwargs = {}
    if device is not None:
        move_kwargs["device"] = device
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if move_kwargs:
        model.to(**move_kwargs)
    model.config.use_cache = False
    return model
