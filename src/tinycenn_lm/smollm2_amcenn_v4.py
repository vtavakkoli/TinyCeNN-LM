from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .smollm2_amcenn import DEFAULT_SMOLLM2
from .smollm2_amcenn_v2 import AdaptivePositiveSoftmaxFeatures


@dataclass(frozen=True)
class LayerProfileV4:
    tier: str
    local_window: int
    feature_dim: int
    gate_init: float
    gate_cap: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SmolAMCeNNV4Config:
    """Layer-adaptive exact-local/anchor + recurrent-global attention.

    The default profile comes directly from the v3 calibration pattern:
    easy layers keep a smaller exact window, difficult layers get a larger exact
    window and a larger AM-CeNN feature map, and the two most difficult layers get
    a 96-token exact window plus 512 recurrent features.
    """

    easy_feature_dim: int = 256
    medium_feature_dim: int = 320
    hard_feature_dim: int = 384
    critical_feature_dim: int = 512
    easy_window: int = 32
    medium_window: int = 48
    hard_window: int = 64
    critical_window: int = 96
    anchor_tokens: int = 8
    feature_seed: int = 4040
    eps: float = 1e-6
    antithetic_features: bool = True
    learnable_feature_correction: bool = True
    gate_init: float = 0.03
    easy_gate_cap: float = 0.25
    medium_gate_cap: float = 0.20
    hard_gate_cap: float = 0.15
    critical_gate_cap: float = 0.10
    easy_layers: tuple[int, ...] = (0, 1, 2, 10, 11, 24, 29)
    hard_layers: tuple[int, ...] = (6, 7, 8, 14, 17, 23, 26)
    critical_layers: tuple[int, ...] = (18, 20)

    def validate(self, model_config) -> None:
        dims = (
            self.easy_feature_dim,
            self.medium_feature_dim,
            self.hard_feature_dim,
            self.critical_feature_dim,
        )
        if min(dims) < 16:
            raise ValueError("all v4 feature dimensions must be >= 16")
        if self.antithetic_features and any(dim % 2 for dim in dims):
            raise ValueError("antithetic v4 feature dimensions must be even")
        windows = (
            self.easy_window,
            self.medium_window,
            self.hard_window,
            self.critical_window,
        )
        if min(windows) < 1:
            raise ValueError("all v4 local windows must be >= 1")
        if self.anchor_tokens < 0:
            raise ValueError("anchor_tokens must be >= 0")
        caps = (
            self.easy_gate_cap,
            self.medium_gate_cap,
            self.hard_gate_cap,
            self.critical_gate_cap,
        )
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        if any(not 0.0 < cap < 1.0 for cap in caps):
            raise ValueError("all v4 gate caps must be in (0, 1)")
        if any(self.gate_init >= cap for cap in caps):
            raise ValueError("gate_init must be lower than every v4 gate cap")
        hidden = int(model_config.hidden_size)
        heads = int(model_config.num_attention_heads)
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    def profile_for_layer(self, layer_idx: int) -> LayerProfileV4:
        idx = int(layer_idx)
        if idx in self.critical_layers:
            return LayerProfileV4(
                "critical", self.critical_window, self.critical_feature_dim,
                self.gate_init, self.critical_gate_cap,
            )
        if idx in self.hard_layers:
            return LayerProfileV4(
                "hard", self.hard_window, self.hard_feature_dim,
                self.gate_init, self.hard_gate_cap,
            )
        if idx in self.easy_layers:
            return LayerProfileV4(
                "easy", self.easy_window, self.easy_feature_dim,
                self.gate_init, self.easy_gate_cap,
            )
        return LayerProfileV4(
            "medium", self.medium_window, self.medium_feature_dim,
            self.gate_init, self.medium_gate_cap,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SmolAMCeNNV4Config":
        values = dict(data)
        for key in ("easy_layers", "hard_layers", "critical_layers"):
            if key in values:
                values[key] = tuple(int(v) for v in values[key])
        return cls(**values)


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class AdaptiveHybridAMCeNNAttentionV4(nn.Module):
    """Exact anchors + exact local window + token-dependent recurrent memory.

    For query position t, exact softmax covers:
      * the first ``anchor_tokens`` tokens (causally masked), and
      * the most recent ``local_window`` tokens.

    AM-CeNN summarizes only the older middle tokens not already covered exactly.
    A token-dependent, per-head gate chooses how much of that recurrent memory to
    use. The gate is intrinsically capped per layer tier, so difficult layers can
    never over-rely on a poor recurrent approximation.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        model_config,
        config: SmolAMCeNNV4Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        config.validate(model_config)
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = int(layer_idx)
        self.anchor_tokens = int(config.anchor_tokens)
        self.eps = float(config.eps)
        self.profile = config.profile_for_layer(self.layer_idx)
        self.local_window = int(self.profile.local_window)
        self.feature_dim = int(self.profile.feature_dim)
        self.gate_cap = float(self.profile.gate_cap)
        self.tier = self.profile.tier

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
            self.feature_dim,
            seed=config.feature_seed + self.layer_idx,
            antithetic=config.antithetic_features,
            learnable_correction=config.learnable_feature_correction,
        )

        # gate = gate_cap * sigmoid(W h + b). W starts at zero, therefore v4
        # begins as a safe constant-gate model and can become token-adaptive.
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        initial_fraction = float(self.profile.gate_init) / self.gate_cap
        nn.init.constant_(self.gate_proj.bias, _logit(initial_fraction))

        self.last_mean_gate = torch.tensor(float(self.profile.gate_init))
        self.last_max_gate = torch.tensor(float(self.profile.gate_init))
        self.last_global_state_norm = torch.tensor(0.0)

    def _apply_rope(self, q: Tensor, k: Tensor, position_embeddings) -> tuple[Tensor, Tensor]:
        if position_embeddings is None:
            return q, k
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            cos, sin = position_embeddings
            return apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            return q, k

    def _exact_attention(self, q: Tensor, k: Tensor, v: Tensor, attention_mask) -> Tensor:
        k_heads = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_heads = v.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.einsum("bhtd,bhsd->bhts", q.float(), k_heads.float())
        scores = scores / math.sqrt(float(self.head_dim))

        seq_len = q.shape[-2]
        positions = torch.arange(seq_len, device=q.device)
        qpos = positions[:, None]
        kpos = positions[None, :]
        causal = kpos <= qpos
        local = kpos >= (qpos - self.local_window + 1)
        anchors = kpos < min(self.anchor_tokens, seq_len)
        allowed = causal & (local | anchors)
        scores = scores.masked_fill(~allowed.view(1, 1, seq_len, seq_len), float("-inf"))

        if torch.is_tensor(attention_mask):
            mask = attention_mask
            try:
                if mask.ndim == 4 and mask.shape[-2:] == (seq_len, seq_len):
                    scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
                elif mask.ndim == 2 and mask.shape[-1] == seq_len:
                    valid = mask.to(device=scores.device).bool().view(mask.shape[0], 1, 1, seq_len)
                    scores = scores.masked_fill(~valid, float("-inf"))
            except Exception:
                pass

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.einsum("bhts,bhsd->bhtd", probs, v_heads.float())

    def _global_middle_memory(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        phi_q = self.features(q).transpose(1, 2)   # B,T,H,F or B,T,KV,F after repeat below
        phi_k = self.features(k).transpose(1, 2)   # B,T,KV,F
        values = v.transpose(1, 2).float()          # B,T,KV,D

        writes = torch.einsum("btkf,btkd->btkfd", phi_k, values)
        if self.anchor_tokens > 0:
            writes = writes.clone()
            phi_k = phi_k.clone()
            cutoff = min(self.anchor_tokens, writes.shape[1])
            writes[:, :cutoff] = 0
            phi_k[:, :cutoff] = 0

        prefix_s = writes.cumsum(dim=1)
        prefix_z = phi_k.cumsum(dim=1)
        old_s = torch.zeros_like(prefix_s)
        old_z = torch.zeros_like(prefix_z)
        if q.shape[-2] > self.local_window:
            old_s[:, self.local_window:] = prefix_s[:, :-self.local_window]
            old_z[:, self.local_window:] = prefix_z[:, :-self.local_window]

        old_s_h = old_s.repeat_interleave(self.num_key_value_groups, dim=2)
        old_z_h = old_z.repeat_interleave(self.num_key_value_groups, dim=2)
        numerator = torch.einsum("bthf,bthfd->bthd", phi_q, old_s_h)
        denominator = torch.einsum("bthf,bthf->bth", phi_q, old_z_h).unsqueeze(-1)
        global_out = numerator / denominator.clamp_min(self.eps)
        valid = denominator > self.eps
        global_out = torch.where(valid, global_out, torch.zeros_like(global_out))
        return global_out.transpose(1, 2), valid.transpose(1, 2)

    def _token_gate(self, hidden_states: Tensor, global_valid: Tensor) -> Tensor:
        logits = F.linear(
            hidden_states.float(),
            self.gate_proj.weight.float(),
            self.gate_proj.bias.float(),
        )
        gate = self.gate_cap * torch.sigmoid(logits)
        gate = gate.transpose(1, 2).unsqueeze(-1)  # B,H,T,1
        return gate * global_valid.to(dtype=gate.dtype)

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
            raise RuntimeError("AM-CeNN v4 currently requires use_cache=False")

        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        exact_out = self._exact_attention(q, k, v, attention_mask)
        global_out, global_valid = self._global_middle_memory(q, k, v)
        gate = self._token_gate(hidden_states, global_valid)

        mixed = exact_out.float() + gate * (global_out.float() - exact_out.float())
        mixed = mixed.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(mixed.to(dtype=hidden_states.dtype))

        valid_gates = gate.detach().float()[global_valid.expand_as(gate)]
        if valid_gates.numel():
            self.last_mean_gate = valid_gates.mean().cpu()
            self.last_max_gate = valid_gates.max().cpu()
        else:
            self.last_mean_gate = torch.tensor(0.0)
            self.last_max_gate = torch.tensor(0.0)
        self.last_global_state_norm = global_out[:, :, -1].detach().float().norm().cpu()
        return out, None


def replace_attention_layers_v4(
    model: nn.Module,
    config: SmolAMCeNNV4Config,
    layer_indices: Iterable[int],
) -> nn.Module:
    config.validate(model.config)
    for idx in layer_indices:
        layer = model.model.layers[int(idx)]
        if isinstance(layer.self_attn, AdaptiveHybridAMCeNNAttentionV4):
            continue
        old = layer.self_attn
        new = AdaptiveHybridAMCeNNAttentionV4(old, model.config, config, int(idx))
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def replace_all_attention_v4(model: nn.Module, config: SmolAMCeNNV4Config) -> nn.Module:
    return replace_attention_layers_v4(model, config, range(int(model.config.num_hidden_layers)))


def freeze_for_v4_calibration(model: nn.Module, layer_indices: Iterable[int]) -> list[nn.Parameter]:
    selected = {int(i) for i in layer_indices}
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, AdaptiveHybridAMCeNNAttentionV4) or module.layer_idx not in selected:
            continue
        if module.features.delta_projection is not None:
            module.features.delta_projection.requires_grad = True
            trainable.append(module.features.delta_projection)
        for p in module.gate_proj.parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def v4_global_parameter_groups(
    model: nn.Module,
    *,
    memory_lr: float,
    qkvo_lr: float,
    weight_decay: float = 0.01,
) -> tuple[list[dict], list[nn.Parameter]]:
    for p in model.parameters():
        p.requires_grad = False
    memory: list[nn.Parameter] = []
    qkvo: list[nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, AdaptiveHybridAMCeNNAttentionV4):
            continue
        if module.features.delta_projection is not None:
            module.features.delta_projection.requires_grad = True
            memory.append(module.features.delta_projection)
        for p in module.gate_proj.parameters():
            p.requires_grad = True
            memory.append(p)
        for projection in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            projection.weight.requires_grad = True
            qkvo.append(projection.weight)
    groups = [
        {"params": memory, "lr": float(memory_lr), "weight_decay": float(weight_decay)},
        {"params": qkvo, "lr": float(qkvo_lr), "weight_decay": float(weight_decay)},
    ]
    return groups, [*memory, *qkvo]


def v4_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    modules = [m for m in model.modules() if isinstance(m, AdaptiveHybridAMCeNNAttentionV4)]
    feature_delta = sum(
        m.features.delta_projection.numel()
        for m in modules
        if m.features.delta_projection is not None
    )
    gate_params = sum(p.numel() for m in modules for p in m.gate_proj.parameters())
    return {
        "total": total,
        "trainable": trainable,
        "hybrid_attention": sum(p.numel() for m in modules for p in m.parameters()),
        "feature_delta": feature_delta,
        "token_gate": gate_params,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def v4_attention_stats(model: nn.Module) -> dict:
    modules = [m for m in model.modules() if isinstance(m, AdaptiveHybridAMCeNNAttentionV4)]
    if not modules:
        raise RuntimeError("AM-CeNN v4 attention modules unavailable")
    means = [float(m.last_mean_gate) for m in modules]
    maxima = [float(m.last_max_gate) for m in modules]
    by_tier: dict[str, list[float]] = {}
    layer_profiles = []
    for m, mean_gate in zip(modules, means):
        by_tier.setdefault(m.tier, []).append(mean_gate)
        layer_profiles.append({
            "layer": m.layer_idx,
            "tier": m.tier,
            "local_window": m.local_window,
            "feature_dim": m.feature_dim,
            "gate_cap": m.gate_cap,
            "mean_gate": mean_gate,
        })
    return {
        "mean_global_gate": sum(means) / len(means),
        "max_observed_gate": max(maxima),
        "mean_gate_by_tier": {k: sum(v) / len(v) for k, v in by_tier.items()},
        "layer_profiles": layer_profiles,
    }


def _v4_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".self_attn." in name:
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("no AM-CeNN v4 state found")
    return state


def save_smollm2_amcenn_v4(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: SmolAMCeNNV4Config,
    base_model: str = DEFAULT_SMOLLM2,
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_v4_state(model), output_dir / "smollm2_amcenn_v4.pt")
    metadata = {
        "format_version": 4,
        "architecture": "smollm2-amcenn-adaptive-v4",
        "base_model": base_model,
        "amcenn_v4": config.to_dict(),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "smollm2_amcenn_v4_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir


def load_smollm2_amcenn_v4_weights(model: nn.Module, student_dir: str | Path) -> nn.Module:
    state = torch.load(
        Path(student_dir) / "smollm2_amcenn_v4.pt",
        map_location="cpu",
        weights_only=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_v4_state(model))
    missing = [k for k in incompatible.missing_keys if k in expected]
    unexpected = [k for k in incompatible.unexpected_keys if k not in expected]
    if missing:
        raise RuntimeError(f"missing AM-CeNN v4 keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected AM-CeNN v4 keys: {unexpected[:8]}")
    return model


def build_smollm2_amcenn_v4(student_dir: str | Path, *, device=None, dtype=None):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    metadata = json.loads((student_dir / "smollm2_amcenn_v4_config.json").read_text())
    if metadata.get("architecture") != "smollm2-amcenn-adaptive-v4":
        raise ValueError("checkpoint is not SmolLM2 AM-CeNN adaptive v4")
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    config = SmolAMCeNNV4Config.from_dict(metadata["amcenn_v4"])
    replace_all_attention_v4(model, config)
    load_smollm2_amcenn_v4_weights(model, student_dir)
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model
