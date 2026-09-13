from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from .smollm2_amcenn import DEFAULT_SMOLLM2
from .smollm2_amcenn_v2 import AdaptivePositiveSoftmaxFeatures


@dataclass(frozen=True)
class SmolAMCeNNV3Config:
    """Hybrid exact-local + recurrent-global attention configuration."""

    feature_dim: int = 256
    local_window: int = 32
    feature_seed: int = 3030
    eps: float = 1e-6
    antithetic_features: bool = True
    learnable_feature_correction: bool = True
    global_gate_init: float = 0.05

    def validate(self, model_config) -> None:
        if self.feature_dim < 16:
            raise ValueError("feature_dim must be >= 16")
        if self.antithetic_features and self.feature_dim % 2:
            raise ValueError("antithetic feature_dim must be even")
        if self.local_window < 1:
            raise ValueError("local_window must be >= 1")
        if not 0.0 < self.global_gate_init < 1.0:
            raise ValueError("global_gate_init must be in (0, 1)")
        hidden = int(model_config.hidden_size)
        heads = int(model_config.num_attention_heads)
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SmolAMCeNNV3Config":
        return cls(**dict(data))


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class HybridLocalAMCeNNAttention(nn.Module):
    """Exact local causal attention plus gated recurrent AM-CeNN long memory.

    Recent tokens use exact softmax attention inside ``local_window``. Older tokens
    are summarized by the positive-feature recurrent state used by AM-CeNN. A
    per-head sigmoid gate, initialized close to zero, mixes the recurrent branch
    into the exact-local branch. Q/K/V/O weights are copied from the pretrained
    attention module.

    During calibration the copied projections stay frozen; only the finite-feature
    correction and global gate learn. Global distillation can later update Q/K/V/O
    with a much smaller learning rate.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        model_config,
        config: SmolAMCeNNV3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        config.validate(model_config)
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.local_window = int(config.local_window)
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
        self.global_gate_logit = nn.Parameter(
            torch.full((self.num_heads,), _logit(config.global_gate_init), dtype=torch.float32)
        )
        self.last_global_gate = torch.tensor(float(config.global_gate_init))
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

    def _local_exact(self, q: Tensor, k: Tensor, v: Tensor, attention_mask) -> Tensor:
        k_heads = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_heads = v.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.einsum("bhtd,bhsd->bhts", q.float(), k_heads.float())
        scores = scores / math.sqrt(float(self.head_dim))

        seq_len = q.shape[-2]
        positions = torch.arange(seq_len, device=q.device)
        qpos = positions[:, None]
        kpos = positions[None, :]
        local = (kpos <= qpos) & (kpos >= (qpos - self.local_window + 1))
        scores = scores.masked_fill(~local.view(1, 1, seq_len, seq_len), float("-inf"))

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

    def _global_old_memory(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        phi_q = self.features(q).transpose(1, 2)
        phi_k = self.features(k).transpose(1, 2)
        values = v.transpose(1, 2).float()

        writes = torch.einsum("btkf,btkd->btkfd", phi_k, values)
        full_s = writes.cumsum(dim=1)
        full_z = phi_k.cumsum(dim=1)

        old_s = torch.zeros_like(full_s)
        old_z = torch.zeros_like(full_z)
        if q.shape[-2] > self.local_window:
            old_s[:, self.local_window:] = full_s[:, :-self.local_window]
            old_z[:, self.local_window:] = full_z[:, :-self.local_window]

        old_s_h = old_s.repeat_interleave(self.num_key_value_groups, dim=2)
        old_z_h = old_z.repeat_interleave(self.num_key_value_groups, dim=2)
        numerator = torch.einsum("bthf,bthfd->bthd", phi_q, old_s_h)
        denominator = torch.einsum("bthf,bthf->bth", phi_q, old_z_h).unsqueeze(-1)
        global_out = numerator / denominator.clamp_min(self.eps)
        valid = denominator > self.eps
        global_out = torch.where(valid, global_out, torch.zeros_like(global_out))
        return global_out.transpose(1, 2), valid.transpose(1, 2)

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
            raise RuntimeError("AM-CeNN v3 currently requires use_cache=False")

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

        local_out = self._local_exact(q, k, v, attention_mask)
        global_out, global_valid = self._global_old_memory(q, k, v)

        gate = torch.sigmoid(self.global_gate_logit.float()).view(1, self.num_heads, 1, 1)
        effective_gate = gate * global_valid.to(dtype=gate.dtype)
        mixed = local_out.float() + effective_gate * (global_out.float() - local_out.float())
        mixed = mixed.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(mixed.to(dtype=hidden_states.dtype))

        self.last_global_gate = gate.detach().mean().cpu()
        self.last_global_state_norm = global_out[:, :, -1].detach().float().norm().cpu()
        return out, None


def replace_attention_layers_v3(
    model: nn.Module,
    config: SmolAMCeNNV3Config,
    layer_indices: Iterable[int],
) -> nn.Module:
    config.validate(model.config)
    for idx in layer_indices:
        layer = model.model.layers[int(idx)]
        if isinstance(layer.self_attn, HybridLocalAMCeNNAttention):
            continue
        old = layer.self_attn
        new = HybridLocalAMCeNNAttention(old, model.config, config, int(idx))
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        new.global_gate_logit.data = new.global_gate_logit.data.float()
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def replace_all_attention_v3(model: nn.Module, config: SmolAMCeNNV3Config) -> nn.Module:
    return replace_attention_layers_v3(model, config, range(int(model.config.num_hidden_layers)))


def freeze_for_v3_calibration(
    model: nn.Module,
    layer_indices: Iterable[int],
) -> list[nn.Parameter]:
    selected = {int(i) for i in layer_indices}
    for p in model.parameters():
        p.requires_grad = False

    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, HybridLocalAMCeNNAttention) and module.layer_idx in selected:
            if module.features.delta_projection is not None:
                module.features.delta_projection.requires_grad = True
                trainable.append(module.features.delta_projection)
            module.global_gate_logit.requires_grad = True
            trainable.append(module.global_gate_logit)
    return trainable


def v3_global_parameter_groups(
    model: nn.Module,
    *,
    main_lr: float,
    qkvo_lr: float,
    weight_decay: float = 0.01,
) -> tuple[list[dict], list[nn.Parameter]]:
    for p in model.parameters():
        p.requires_grad = False

    main: list[nn.Parameter] = []
    qkvo: list[nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, HybridLocalAMCeNNAttention):
            continue
        if module.features.delta_projection is not None:
            module.features.delta_projection.requires_grad = True
            main.append(module.features.delta_projection)
        module.global_gate_logit.requires_grad = True
        main.append(module.global_gate_logit)
        for projection in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            projection.weight.requires_grad = True
            qkvo.append(projection.weight)

    groups = [
        {"params": main, "lr": float(main_lr), "weight_decay": float(weight_decay)},
        {"params": qkvo, "lr": float(qkvo_lr), "weight_decay": float(weight_decay)},
    ]
    return groups, [*main, *qkvo]


def v3_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attention = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, HybridLocalAMCeNNAttention)
        for p in module.parameters()
    )
    feature_delta = sum(
        module.features.delta_projection.numel()
        for module in model.modules()
        if isinstance(module, HybridLocalAMCeNNAttention)
        and module.features.delta_projection is not None
    )
    gates = sum(
        module.global_gate_logit.numel()
        for module in model.modules()
        if isinstance(module, HybridLocalAMCeNNAttention)
    )
    return {
        "total": total,
        "trainable": trainable,
        "hybrid_attention": attention,
        "feature_delta": feature_delta,
        "global_gates": gates,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def v3_attention_stats(model: nn.Module) -> dict[str, float]:
    modules = [m for m in model.modules() if isinstance(m, HybridLocalAMCeNNAttention)]
    if not modules:
        raise RuntimeError("AM-CeNN v3 attention modules unavailable")
    gates = [float(torch.sigmoid(m.global_gate_logit.detach().float()).mean().cpu()) for m in modules]
    return {
        "mean_global_gate": sum(gates) / len(gates),
        "min_global_gate": min(gates),
        "max_global_gate": max(gates),
    }


def _v3_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".self_attn." in name:
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("no AM-CeNN v3 state found")
    return state


def save_smollm2_amcenn_v3(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: SmolAMCeNNV3Config,
    base_model: str = DEFAULT_SMOLLM2,
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_v3_state(model), output_dir / "smollm2_amcenn_v3.pt")
    metadata = {
        "format_version": 3,
        "architecture": "smollm2-amcenn-hybrid-v3",
        "base_model": base_model,
        "amcenn_v3": config.to_dict(),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "smollm2_amcenn_v3_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir


def load_smollm2_amcenn_v3_weights(model: nn.Module, student_dir: str | Path) -> nn.Module:
    state = torch.load(
        Path(student_dir) / "smollm2_amcenn_v3.pt",
        map_location="cpu",
        weights_only=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_v3_state(model))
    missing = [key for key in incompatible.missing_keys if key in expected]
    unexpected = [key for key in incompatible.unexpected_keys if key not in expected]
    if missing:
        raise RuntimeError(f"missing AM-CeNN v3 keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected AM-CeNN v3 keys: {unexpected[:8]}")
    return model


def build_smollm2_amcenn_v3(student_dir: str | Path, *, device=None, dtype=None):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    metadata = json.loads((student_dir / "smollm2_amcenn_v3_config.json").read_text())
    if metadata.get("architecture") != "smollm2-amcenn-hybrid-v3":
        raise ValueError("checkpoint is not SmolLM2 AM-CeNN hybrid v3")

    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    config = SmolAMCeNNV3Config.from_dict(metadata["amcenn_v3"])
    replace_all_attention_v3(model, config)
    load_smollm2_amcenn_v3_weights(model, student_dir)
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
        for module in model.modules():
            if isinstance(module, HybridLocalAMCeNNAttention):
                module.global_gate_logit.data = module.global_gate_logit.data.float()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model
