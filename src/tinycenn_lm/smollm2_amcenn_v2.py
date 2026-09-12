from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from .smollm2_amcenn import DEFAULT_SMOLLM2, ShardedTop2LlamaMLP, SmolAMCeNNConfig


@dataclass(frozen=True)
class SmolAMCeNNV2Config:
    feature_dim: int = 128
    num_shards: int = 8
    top_k: int = 2
    feature_seed: int = 2026
    eps: float = 1e-6
    antithetic_features: bool = True
    learnable_feature_correction: bool = True

    def validate(self, model_config) -> None:
        if self.feature_dim < 8:
            raise ValueError("feature_dim must be >= 8")
        if self.antithetic_features and self.feature_dim % 2:
            raise ValueError("antithetic feature_dim must be even")
        if self.num_shards < 2:
            raise ValueError("num_shards must be >= 2")
        if not 1 <= self.top_k <= self.num_shards:
            raise ValueError("top_k must be in [1, num_shards]")
        if int(model_config.intermediate_size) % self.num_shards:
            raise ValueError("intermediate_size must divide evenly by num_shards")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SmolAMCeNNV2Config":
        return cls(**dict(data))


class AdaptivePositiveSoftmaxFeatures(nn.Module):
    """Positive softmax-kernel features with a zero-init learnable correction.

    The base projection is sampled from N(0,I). In v2 we optionally use
    antithetic pairs (omega, -omega) to reduce estimator variance while keeping
    Gaussian marginals. delta_projection starts at zero, so training begins from
    the mathematical random-feature estimator and can adapt the finite feature
    basis to each pretrained layer's actual Q/K distribution.
    """

    def __init__(
        self,
        head_dim: int,
        feature_dim: int,
        seed: int,
        *,
        antithetic: bool = True,
        learnable_correction: bool = True,
    ) -> None:
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        if antithetic:
            half = torch.randn(feature_dim // 2, head_dim, generator=g)
            projection = torch.cat([half, -half], dim=0)
        else:
            projection = torch.randn(feature_dim, head_dim, generator=g)
        self.register_buffer("base_projection", projection, persistent=True)
        if learnable_correction:
            self.delta_projection = nn.Parameter(torch.zeros_like(projection))
        else:
            self.register_parameter("delta_projection", None)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.scale = head_dim ** -0.25
        self.log_norm = 0.5 * math.log(float(feature_dim))

    @property
    def projection(self) -> Tensor:
        if self.delta_projection is None:
            return self.base_projection
        return self.base_projection + self.delta_projection

    def forward(self, x: Tensor) -> Tensor:
        work = x.float() * self.scale
        projected = torch.einsum("...d,fd->...f", work, self.projection.float())
        norm = 0.5 * work.square().sum(dim=-1, keepdim=True)
        log_phi = (projected - norm - self.log_norm).clamp(min=-20.0, max=20.0)
        return torch.exp(log_phi)


class AMCeNNAttentionV2(nn.Module):
    """Causal finite-state approximation of softmax attention.

    S_t = S_{t-1} + phi(k_t) v_t^T
    z_t = z_{t-1} + phi(k_t)
    a_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)

    Unlike v1, the feature map is larger by default (m=128), variance reduced
    with antithetic features, and the finite basis can adapt through a zero-init
    delta projection. No T x T attention matrix is constructed.
    """

    def __init__(self, original_attn: nn.Module, model_config, config: SmolAMCeNNV2Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.eps = float(config.eps)
        self.layer_idx = int(layer_idx)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(original_attn.q_proj.weight)
            self.k_proj.weight.copy_(original_attn.k_proj.weight)
            self.v_proj.weight.copy_(original_attn.v_proj.weight)
            self.o_proj.weight.copy_(original_attn.o_proj.weight)

        self.features = AdaptivePositiveSoftmaxFeatures(
            self.head_dim,
            config.feature_dim,
            seed=config.feature_seed + self.layer_idx,
            antithetic=config.antithetic_features,
            learnable_correction=config.learnable_feature_correction,
        )
        self.last_state_norm = torch.tensor(0.0)

    def _apply_rope(self, q: Tensor, k: Tensor, position_embeddings) -> tuple[Tensor, Tensor]:
        if position_embeddings is None:
            return q, k
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            cos, sin = position_embeddings
            return apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            return q, k

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        if use_cache:
            raise RuntimeError("AM-CeNN v2 currently requires use_cache=False")
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        phi_q = self.features(q).transpose(1, 2)
        phi_k = self.features(k).transpose(1, 2)
        values = v.transpose(1, 2).float()

        kv_write = torch.einsum("btkf,btkd->btkfd", phi_k, values)
        state_s = kv_write.cumsum(dim=1)
        state_z = phi_k.cumsum(dim=1)

        state_s = state_s.repeat_interleave(self.num_key_value_groups, dim=2)
        state_z = state_z.repeat_interleave(self.num_key_value_groups, dim=2)
        numerator = torch.einsum("bthf,bthfd->bthd", phi_q, state_s)
        denominator = torch.einsum("bthf,bthf->bth", phi_q, state_z).unsqueeze(-1)
        out = numerator / denominator.clamp_min(self.eps)
        out = out.to(dtype=hidden_states.dtype).reshape(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)
        self.last_state_norm = state_s[:, -1].detach().float().norm().cpu()
        return out, None


def _moe_config(config: SmolAMCeNNV2Config) -> SmolAMCeNNConfig:
    return SmolAMCeNNConfig(
        feature_dim=max(8, config.feature_dim),
        num_shards=config.num_shards,
        top_k=config.top_k,
        feature_seed=config.feature_seed,
        eps=config.eps,
    )


def convert_all_ffns_to_sharded_top2(model: nn.Module, config: SmolAMCeNNV2Config) -> nn.Module:
    config.validate(model.config)
    moe_cfg = _moe_config(config)
    for layer in model.model.layers:
        if not isinstance(layer.mlp, ShardedTop2LlamaMLP):
            old = layer.mlp
            new = ShardedTop2LlamaMLP(old, model.config, moe_cfg)
            new.to(device=old.gate_proj.weight.device, dtype=old.gate_proj.weight.dtype)
            layer.mlp = new
    return model


def replace_attention_layers(
    model: nn.Module,
    config: SmolAMCeNNV2Config,
    layer_indices: Iterable[int],
) -> nn.Module:
    config.validate(model.config)
    for idx in layer_indices:
        layer = model.model.layers[int(idx)]
        if isinstance(layer.self_attn, AMCeNNAttentionV2):
            continue
        old = layer.self_attn
        new = AMCeNNAttentionV2(old, model.config, config, int(idx))
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def replace_all_smollm2_attention(model: nn.Module, config: SmolAMCeNNV2Config) -> nn.Module:
    return replace_attention_layers(model, config, range(int(model.config.num_hidden_layers)))


def freeze_for_group_calibration(model: nn.Module, layer_indices: Iterable[int]) -> list[nn.Parameter]:
    selected = {int(i) for i in layer_indices}
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, AMCeNNAttentionV2) and module.layer_idx in selected:
            for p in module.parameters():
                p.requires_grad = True
                trainable.append(p)
    return trainable


def freeze_for_global_training(model: nn.Module, *, train_router: bool = True) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, AMCeNNAttentionV2):
            for p in module.parameters():
                p.requires_grad = True
                trainable.append(p)
        elif isinstance(module, ShardedTop2LlamaMLP) and train_router:
            module.router.weight.requires_grad = True
            module.route_mix.requires_grad = True
            trainable.extend([module.router.weight, module.route_mix])
    return trainable


def v2_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    am = sum(p.numel() for m in model.modules() if isinstance(m, AMCeNNAttentionV2) for p in m.parameters())
    feature_delta = sum(
        m.features.delta_projection.numel()
        for m in model.modules()
        if isinstance(m, AMCeNNAttentionV2) and m.features.delta_projection is not None
    )
    routers = sum(
        m.router.weight.numel() + m.route_mix.numel()
        for m in model.modules()
        if isinstance(m, ShardedTop2LlamaMLP)
    )
    return {
        "total": total,
        "trainable": trainable,
        "amcenn_attention": am,
        "feature_delta": feature_delta,
        "router_and_mix": routers,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def _v2_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".self_attn." in name or ".mlp." in name:
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("no SmolLM2 AM-CeNN v2 state found")
    return state


def save_smollm2_amcenn_v2(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: SmolAMCeNNV2Config,
    base_model: str = DEFAULT_SMOLLM2,
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_v2_state(model), output_dir / "smollm2_amcenn_v2.pt")
    meta = {
        "format_version": 2,
        "architecture": "smollm2-amcenn-top2-v2",
        "base_model": base_model,
        "amcenn_v2": config.to_dict(),
    }
    if extra_metadata:
        meta["training"] = extra_metadata
    (output_dir / "smollm2_amcenn_v2_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return output_dir


def load_smollm2_amcenn_v2_weights(model: nn.Module, student_dir: str | Path) -> nn.Module:
    state = torch.load(Path(student_dir) / "smollm2_amcenn_v2.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_v2_state(model))
    missing = [k for k in incompatible.missing_keys if k in expected]
    unexpected = [k for k in incompatible.unexpected_keys if k not in expected]
    if missing:
        raise RuntimeError(f"missing AM-CeNN v2 keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected AM-CeNN v2 keys: {unexpected[:8]}")
    return model


def build_smollm2_amcenn_v2(student_dir: str | Path, *, device=None, dtype=None):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    meta = json.loads((student_dir / "smollm2_amcenn_v2_config.json").read_text())
    if meta.get("architecture") != "smollm2-amcenn-top2-v2":
        raise ValueError("checkpoint is not SmolLM2 AM-CeNN Top-2 v2")
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(meta["base_model"], **kwargs)
    config = SmolAMCeNNV2Config.from_dict(meta["amcenn_v2"])
    convert_all_ffns_to_sharded_top2(model, config)
    replace_all_smollm2_attention(model, config)
    load_smollm2_amcenn_v2_weights(model, student_dir)
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model
