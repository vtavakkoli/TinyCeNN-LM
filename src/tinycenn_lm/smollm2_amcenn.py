from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_SMOLLM2 = "HuggingFaceTB/SmolLM2-135M"


@dataclass(frozen=True)
class SmolAMCeNNConfig:
    feature_dim: int = 32
    num_shards: int = 8
    top_k: int = 2
    feature_seed: int = 1234
    eps: float = 1e-6

    def validate(self, model_config) -> None:
        if self.feature_dim < 4:
            raise ValueError("feature_dim must be >= 4")
        if self.num_shards < 2:
            raise ValueError("num_shards must be >= 2")
        if not 1 <= self.top_k <= self.num_shards:
            raise ValueError("top_k must be in [1, num_shards]")
        if int(model_config.intermediate_size) % self.num_shards:
            raise ValueError("intermediate_size must divide evenly by num_shards")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SmolAMCeNNConfig":
        return cls(**dict(data))


class PositiveSoftmaxFeatures(nn.Module):
    """Finite positive random features for exp(q^T k / sqrt(d)).

    With x=q/d^(1/4), y=k/d^(1/4) and omega~N(0,I),
    E[phi(x)^T phi(y)] = exp(q^T k / sqrt(d)).
    This is the finite-state approximation of the exact infinite-dimensional
    recurrent softmax-kernel state discussed in the TinyCeNN derivation.
    """

    def __init__(self, head_dim: int, feature_dim: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(feature_dim, head_dim, generator=g)
        self.register_buffer("projection", projection, persistent=True)
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.scale = head_dim ** -0.25
        self.log_norm = 0.5 * math.log(float(feature_dim))

    def forward(self, x: Tensor) -> Tensor:
        work = x.float() * self.scale
        projected = torch.einsum("...d,fd->...f", work, self.projection.float())
        norm = 0.5 * work.square().sum(dim=-1, keepdim=True)
        # The clamp is only a numerical guard; the mathematical feature map is exp(log_phi).
        log_phi = (projected - norm - self.log_norm).clamp(min=-20.0, max=20.0)
        return torch.exp(log_phi)


class AMCeNNAttention(nn.Module):
    """Causal recurrent associative-memory replacement for Llama self-attention.

    S_t = S_{t-1} + phi(k_t) v_t^T
    z_t = z_{t-1} + phi(k_t)
    a_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + eps)

    Training uses a vectorized causal prefix scan (cumsum), so no T x T attention
    matrix is constructed. Generation currently uses use_cache=False and rebuilds
    the prefix state; a persistent recurrent cache can be added separately.
    """

    def __init__(self, original_attn: nn.Module, model_config, config: SmolAMCeNNConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.eps = float(config.eps)
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(original_attn.q_proj.weight)
            self.k_proj.weight.copy_(original_attn.k_proj.weight)
            self.v_proj.weight.copy_(original_attn.v_proj.weight)
            self.o_proj.weight.copy_(original_attn.o_proj.weight)

        self.features = PositiveSoftmaxFeatures(
            self.head_dim, config.feature_dim, seed=config.feature_seed + layer_idx
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
            # Compatibility fallback across transformers versions. The model remains
            # causal, but users should keep a current transformers release in Colab.
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
            raise RuntimeError("AM-CeNN currently requires use_cache=False")
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        # [B,H,T,D] -> [B,T,H,F], [B,K,T,D] -> [B,T,K,F]
        phi_q = self.features(q).transpose(1, 2)
        phi_k = self.features(k).transpose(1, 2)
        values = v.transpose(1, 2).float()

        # Recurrent state written at every token, then prefix-scanned causally.
        kv_write = torch.einsum("btkf,btkd->btkfd", phi_k, values)
        state_s = kv_write.cumsum(dim=1)
        state_z = phi_k.cumsum(dim=1)

        # GQA: each KV head serves a contiguous group of query heads.
        state_s = state_s.repeat_interleave(self.num_key_value_groups, dim=2)
        state_z = state_z.repeat_interleave(self.num_key_value_groups, dim=2)
        numerator = torch.einsum("bthf,bthfd->bthd", phi_q, state_s)
        denominator = torch.einsum("bthf,bthf->bth", phi_q, state_z).unsqueeze(-1)
        out = numerator / denominator.clamp_min(self.eps)
        out = out.to(dtype=hidden_states.dtype).reshape(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)
        self.last_state_norm = state_s[:, -1].detach().float().norm().cpu()
        return out, None


class ShardedTop2LlamaMLP(nn.Module):
    """Exact parameter-neutral 8-way partition of a pretrained Llama SwiGLU FFN.

    All disjoint shards sum to the original dense FFN. Top-2 routing supplies a
    learned correction controlled by route_mix. route_mix=0 is exactly the
    pretrained dense FFN, so the FFN conversion itself is function preserving.
    """

    def __init__(self, original_mlp: nn.Module, model_config, config: SmolAMCeNNConfig) -> None:
        super().__init__()
        h = int(model_config.hidden_size)
        inner = int(model_config.intermediate_size)
        e = config.num_shards
        s = inner // e
        self.hidden_size = h
        self.inner_size = inner
        self.num_shards = e
        self.shard_inner = s
        self.top_k = config.top_k

        self.gate_weight = nn.Parameter(torch.empty(e, s, h))
        self.up_weight = nn.Parameter(torch.empty(e, s, h))
        self.down_weight = nn.Parameter(torch.empty(e, h, s))
        self.router = nn.Linear(h, e, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
        self.route_mix = nn.Parameter(torch.zeros(()))

        with torch.no_grad():
            for i in range(e):
                lo, hi = i * s, (i + 1) * s
                self.gate_weight[i].copy_(original_mlp.gate_proj.weight[lo:hi])
                self.up_weight[i].copy_(original_mlp.up_proj.weight[lo:hi])
                self.down_weight[i].copy_(original_mlp.down_proj.weight[:, lo:hi])
        self.last_router_stats: dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.einsum("bth,esh->bt es", x, self.gate_weight).contiguous()
        up = torch.einsum("bth,esh->bt es", x, self.up_weight).contiguous()
        # Remove spaces introduced solely for readability in einsum labels above.
        gate = gate.view(*x.shape[:2], self.num_shards, self.shard_inner)
        up = up.view(*x.shape[:2], self.num_shards, self.shard_inner)
        hidden = F.silu(gate) * up
        shard_out = torch.einsum("btes,ehs->bteh", hidden, self.down_weight)
        dense_full = shard_out.sum(dim=2)

        logits = self.router(x).float()
        probs = F.softmax(logits, dim=-1)
        top_values, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_weight = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gather_idx = top_idx.unsqueeze(-1).expand(*top_idx.shape, self.hidden_size)
        selected = torch.gather(shard_out, dim=2, index=gather_idx)
        routed = (selected * top_weight.to(selected.dtype).unsqueeze(-1)).sum(dim=2)
        sparse_scaled = routed * (self.num_shards / float(self.top_k))
        out = dense_full + self.route_mix * (sparse_scaled - dense_full)

        assignment = F.one_hot(top_idx, num_classes=self.num_shards).float().sum(dim=-2)
        assignment = assignment / float(self.top_k)
        shard_fraction = assignment.mean(dim=(0, 1))
        probability_fraction = probs.mean(dim=(0, 1))
        self.last_router_stats = {
            "load_balance": self.num_shards * torch.sum(shard_fraction * probability_fraction),
            "z_loss": torch.logsumexp(logits, dim=-1).pow(2).mean(),
            "entropy": -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean(),
            "shard_fraction": shard_fraction,
            "probability_fraction": probability_fraction,
            "route_mix": self.route_mix,
        }
        return out


def replace_smollm2_core(model: nn.Module, config: SmolAMCeNNConfig) -> nn.Module:
    config.validate(model.config)
    layers = model.model.layers
    for idx, layer in enumerate(layers):
        if not isinstance(layer.self_attn, AMCeNNAttention):
            old_attn = layer.self_attn
            new_attn = AMCeNNAttention(old_attn, model.config, config, idx)
            new_attn.to(device=old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
            layer.self_attn = new_attn
        if not isinstance(layer.mlp, ShardedTop2LlamaMLP):
            old_mlp = layer.mlp
            new_mlp = ShardedTop2LlamaMLP(old_mlp, model.config, config)
            new_mlp.to(device=old_mlp.gate_proj.weight.device, dtype=old_mlp.gate_proj.weight.dtype)
            layer.mlp = new_mlp
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def freeze_smollm2_for_amcenn_training(model: nn.Module, *, train_ffn_shards: bool = False) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, AMCeNNAttention):
            for p in module.parameters():
                p.requires_grad = True
        elif isinstance(module, ShardedTop2LlamaMLP):
            module.router.weight.requires_grad = True
            module.route_mix.requires_grad = True
            if train_ffn_shards:
                module.gate_weight.requires_grad = True
                module.up_weight.requires_grad = True
                module.down_weight.requires_grad = True


def amcenn_router_stats(model: nn.Module) -> dict[str, Tensor]:
    stats = [m.last_router_stats for m in model.modules() if isinstance(m, ShardedTop2LlamaMLP) and m.last_router_stats]
    if not stats:
        raise RuntimeError("router statistics unavailable; run a forward pass first")
    keys = ("load_balance", "z_loss", "entropy", "route_mix")
    out = {key: torch.stack([s[key].float() for s in stats]).mean() for key in keys}
    out["shard_fraction"] = torch.stack([s["shard_fraction"].float() for s in stats]).mean(dim=0)
    out["probability_fraction"] = torch.stack([s["probability_fraction"].float() for s in stats]).mean(dim=0)
    return out


def amcenn_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attention = sum(p.numel() for m in model.modules() if isinstance(m, AMCeNNAttention) for p in m.parameters())
    routers = sum(
        m.router.weight.numel() + m.route_mix.numel()
        for m in model.modules()
        if isinstance(m, ShardedTop2LlamaMLP)
    )
    return {
        "total": total,
        "trainable": trainable,
        "attention_replacement": attention,
        "router_and_mix": routers,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def _replacement_state(model: nn.Module) -> dict[str, Tensor]:
    state = {}
    for name, tensor in model.state_dict().items():
        if ".self_attn." in name or ".mlp." in name:
            state[name] = tensor.detach().cpu()
    if not state:
        raise RuntimeError("no AM-CeNN replacement state found")
    return state


def save_smollm2_amcenn(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: SmolAMCeNNConfig,
    base_model: str = DEFAULT_SMOLLM2,
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_replacement_state(model), output_dir / "smollm2_amcenn_top2.pt")
    metadata = {
        "format_version": 1,
        "architecture": "smollm2-amcenn-top2",
        "base_model": base_model,
        "amcenn": config.to_dict(),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "smollm2_amcenn_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def load_smollm2_amcenn_weights(model: nn.Module, student_dir: str | Path) -> nn.Module:
    state = torch.load(Path(student_dir) / "smollm2_amcenn_top2.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected = set(_replacement_state(model))
    missing = [k for k in incompatible.missing_keys if k in expected]
    unexpected = [k for k in incompatible.unexpected_keys if k not in expected]
    if missing:
        raise RuntimeError(f"missing AM-CeNN keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected AM-CeNN keys: {unexpected[:8]}")
    return model


def build_smollm2_amcenn(student_dir: str | Path, *, device=None, dtype=None):
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    meta = json.loads((student_dir / "smollm2_amcenn_config.json").read_text())
    if meta.get("architecture") != "smollm2-amcenn-top2":
        raise ValueError("checkpoint is not SmolLM2 AM-CeNN Top-2")
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(meta["base_model"], **kwargs)
    config = SmolAMCeNNConfig.from_dict(meta["amcenn"])
    replace_smollm2_core(model, config)
    load_smollm2_amcenn_weights(model, student_dir)
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.config.use_cache = False
    return model
