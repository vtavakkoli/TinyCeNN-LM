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
class MoECeNNConfig:
    hidden_size: int = 192
    kernel_size: int = 3
    expansion: int = 4
    steps: int = 7
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    num_experts: int = 8
    top_k: int = 2
    router_noise_std: float = 1e-3

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
        if self.num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.router_noise_std < 0:
            raise ValueError("router_noise_std must be >= 0")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["dilations"] = list(self.dilations)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MoECeNNConfig":
        data = dict(data)
        if "dilations" in data:
            data["dilations"] = tuple(data["dilations"])
        return cls(**data)


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, expansion: int, dropout: float = 0.0) -> None:
        super().__init__()
        inner = hidden_size * expansion
        self.in_proj = nn.Linear(hidden_size, inner * 2, bias=False)
        self.out_proj = nn.Linear(inner, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(F.silu(a) * b))


class Top2Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, noise_std: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=noise_std)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        logits = self.proj(x).float()
        probs = F.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        assignment = F.one_hot(top_indices, num_classes=self.num_experts).float().sum(dim=-2)
        assignment = assignment / float(self.top_k)
        expert_fraction = assignment.mean(dim=(0, 1))
        probability_fraction = probs.mean(dim=(0, 1))
        load_balance = self.num_experts * torch.sum(expert_fraction * probability_fraction)
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return top_indices, top_weights.to(dtype=x.dtype), {
            "load_balance": load_balance,
            "z_loss": z_loss,
            "entropy": entropy,
            "expert_fraction": expert_fraction,
            "probability_fraction": probability_fraction,
        }


class MoESharedCeNNCell(nn.Module):
    def __init__(self, config: MoECeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.norm = StableRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.neighborhood = CausalDepthwiseNeighborhood(config.hidden_size, config.kernel_size)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        nn.init.constant_(self.gate_proj.bias, -1.0)
        self.router = Top2Router(
            config.hidden_size, config.num_experts, config.top_k, config.router_noise_std
        )
        self.experts = nn.ModuleList(
            SwiGLUExpert(config.hidden_size, config.expansion, config.dropout)
            for _ in range(config.num_experts)
        )

    def forward(self, state: Tensor, dilation: int, step_scale: float) -> tuple[Tensor, dict[str, Tensor]]:
        x = self.norm(state)
        local = self.neighborhood(x, dilation=dilation)
        top_idx, top_weight, stats = self.router(local)

        flat = local.reshape(-1, local.shape[-1])
        idx_flat = top_idx.reshape(-1, self.config.top_k)
        weight_flat = top_weight.reshape(-1, self.config.top_k)
        update = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            selected = idx_flat.eq(expert_id)
            positions = selected.nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            rows = positions[:, 0]
            slots = positions[:, 1]
            expert_out = expert(flat.index_select(0, rows))
            weighted = expert_out * weight_flat[rows, slots].unsqueeze(-1)
            update = update.index_add(0, rows, weighted)
        update = update.view_as(local)
        gate = torch.sigmoid(self.gate_proj(local))
        return state + step_scale * gate * update, stats


class FastMoECeNNCore(nn.Module):
    def __init__(self, config: MoECeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.cell = MoESharedCeNNCell(config)
        self.last_router_stats: dict[str, Tensor] = {}

    def forward(self, hidden_states: Tensor) -> Tensor:
        initial = hidden_states
        state = hidden_states
        step_scale = self.config.steps ** -0.5
        accum: dict[str, Tensor] = {}
        expert_fraction = None
        probability_fraction = None
        for step in range(self.config.steps):
            dilation = self.config.dilations[step % len(self.config.dilations)]
            state, stats = self.cell(state, dilation=dilation, step_scale=step_scale)
            for key in ("load_balance", "z_loss", "entropy"):
                accum[key] = accum.get(key, stats[key].new_zeros(())) + stats[key]
            expert_fraction = stats["expert_fraction"] if expert_fraction is None else expert_fraction + stats["expert_fraction"]
            probability_fraction = stats["probability_fraction"] if probability_fraction is None else probability_fraction + stats["probability_fraction"]
        self.last_router_stats = {
            "load_balance": accum["load_balance"] / self.config.steps,
            "z_loss": accum["z_loss"] / self.config.steps,
            "entropy": accum["entropy"] / self.config.steps,
            "expert_fraction": expert_fraction / self.config.steps,
            "probability_fraction": probability_fraction / self.config.steps,
        }
        return state - initial

    @property
    def receptive_field(self) -> int:
        radius = sum(self.config.dilations[i % len(self.config.dilations)] for i in range(self.config.steps))
        return 1 + (self.config.kernel_size - 1) * radius


class MoECeNNReplacementLayer(nn.Module):
    def __init__(self, config: MoECeNNConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.cenn = FastMoECeNNCore(config)
        if device is not None or dtype is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            self.cenn.to(**kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        if kwargs.get("use_cache", False):
            raise RuntimeError("MoE-CeNN student requires use_cache=False")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("MoE-CeNN student has no attention matrices")
        return hidden_states + self.cenn(hidden_states)


def replace_transformer_with_moe_cenn(model: nn.Module, config: MoECeNNConfig, layer_indices: Sequence[int] = (0,)) -> nn.Module:
    layers = _get_decoder_layers(model)
    if config.hidden_size != int(model.config.hidden_size):
        raise ValueError("MoE-CeNN hidden size does not match base model")
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    for index in layer_indices:
        old = layers[index]
        reference = next((p for p in old.parameters() if p.is_floating_point()), None)
        layers[index] = MoECeNNReplacementLayer(
            config,
            device=reference.device if reference is not None else None,
            dtype=reference.dtype if reference is not None else None,
        )
    return model


def freeze_moe_student_interfaces(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, MoECeNNReplacementLayer):
            for parameter in module.parameters():
                parameter.requires_grad = True


def warmstart_moe_from_plain_cenn(model: nn.Module, plain_student_dir: str | Path) -> None:
    """Initialize shared dynamics and clone the trained dense FFN into every expert.

    With identical expert weights, Top-2 weighted routing initially reproduces the
    dense CeNN FFN output (weights sum to one), while tiny router noise allows the
    experts to specialize during training.
    """
    state = torch.load(Path(plain_student_dir) / "cenn_student.pt", map_location="cpu", weights_only=True)
    target = model.state_dict()
    copied = 0
    for target_name in list(target):
        if ".cenn.cell.experts." in target_name:
            suffix = target_name.split(".experts.", 1)[1].split(".", 1)[1]
            prefix = target_name.split(".cenn.cell.experts.", 1)[0] + ".cenn.cell."
            source_name = prefix + suffix
        elif any(part in target_name for part in (".cenn.cell.norm.", ".cenn.cell.neighborhood.", ".cenn.cell.gate_proj.")):
            source_name = target_name
        elif ".cenn." not in target_name and target_name in state:
            # v2 dense checkpoints may include adapted language interfaces.
            source_name = target_name
        else:
            continue
        if source_name in state and state[source_name].shape == target[target_name].shape:
            target[target_name].copy_(state[source_name].to(dtype=target[target_name].dtype))
            copied += 1
    if copied == 0:
        raise RuntimeError("could not map plain CeNN weights into MoE-CeNN model")
    model.load_state_dict(target, strict=False)
    model._cenn_interface_keys = tuple(name for name in state if ".cenn." not in name)


def moe_router_stats(model: nn.Module) -> dict[str, Tensor]:
    layer = next((m for m in model.modules() if isinstance(m, MoECeNNReplacementLayer)), None)
    if layer is None or not layer.cenn.last_router_stats:
        raise RuntimeError("router statistics unavailable; run a forward pass first")
    return layer.cenn.last_router_stats


def _moe_state_dict(model: nn.Module) -> dict[str, Tensor]:
    interfaces = set(getattr(model, "_cenn_interface_keys", ()))
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
             if ".cenn." in name or name in interfaces}
    if not state:
        raise ValueError("no MoE-CeNN weights found")
    return state


def save_moe_cenn_student(model: nn.Module, output_dir: str | Path, *, config: MoECeNNConfig, base_model: str = DEFAULT_BASE_MODEL, layer_indices: Sequence[int] = (0,), extra_metadata: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _moe_state_dict(model)
    torch.save(state, output_dir / "moe_cenn_student.pt")
    metadata = {
        "format_version": 2,
        "architecture": "moe-cenn-top2-replacement",
        "base_model": base_model,
        "layer_indices": list(layer_indices),
        "moe_cenn": config.to_dict(),
        "state_keys": sorted(state),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "moe_student_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def load_moe_cenn_student_weights(model: nn.Module, student_dir: str | Path, *, map_location="cpu", strict: bool = True) -> nn.Module:
    state = torch.load(Path(student_dir) / "moe_cenn_student.pt", map_location=map_location, weights_only=True)
    metadata = json.loads((Path(student_dir) / "moe_student_config.json").read_text())
    core_keys = {name for name in model.state_dict() if ".cenn." in name}
    expected = set(metadata.get("state_keys", core_keys)) | core_keys
    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected | state.keys() - model.state_dict().keys())
    if strict and missing:
        raise RuntimeError(f"missing MoE-CeNN keys: {missing}")
    if strict and unexpected:
        raise RuntimeError(f"unexpected MoE-CeNN keys: {unexpected}")
    model.load_state_dict(state, strict=False)
    model._cenn_interface_keys = tuple(name for name in state if ".cenn." not in name)
    return model


def build_moe_cenn_student(student_dir: str | Path, *, device=None, dtype=None, attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    metadata = json.loads((student_dir / "moe_student_config.json").read_text())
    if metadata.get("architecture") != "moe-cenn-top2-replacement":
        raise ValueError("checkpoint is not an MoE-CeNN Top-2 student")
    kwargs = {"attn_implementation": attn_implementation}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    config = MoECeNNConfig.from_dict(metadata["moe_cenn"])
    replace_transformer_with_moe_cenn(model, config, tuple(metadata["layer_indices"]))
    load_moe_cenn_student_weights(model, student_dir)
    move_kwargs = {}
    if device is not None:
        move_kwargs["device"] = device
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if move_kwargs:
        model.to(**move_kwargs)
    model.config.use_cache = False
    return model
