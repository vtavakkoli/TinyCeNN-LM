from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class FlyEmbeddingV3Config:
    fly_nodes: int = 256
    graph_steps: int = 1
    graph_mix_init: float = 0.05
    max_residual_scale: float = 0.05
    dropout: float = 0.0

    def validate(self, hidden_size: int, adjacency: Tensor) -> None:
        if not 16 <= self.fly_nodes <= hidden_size * 2:
            raise ValueError("fly_nodes must be in [16, 2*hidden_size]")
        if adjacency.ndim != 2 or tuple(adjacency.shape) != (self.fly_nodes, self.fly_nodes):
            raise ValueError("adjacency must be [fly_nodes, fly_nodes]")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 < self.graph_mix_init < 1.0:
            raise ValueError("graph_mix_init must be in (0,1)")
        if not 0.0 < self.max_residual_scale <= 0.5:
            raise ValueError("max_residual_scale must be in (0,0.5]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


class FlyResidualCoreV3(nn.Module):
    """Small nonlinear graph residual operating on exact Qwen embeddings.

    Identity is guaranteed at installation because residual_scale_raw is
    initialized to zero. The residual branch is non-zero at initialization so
    the scalar gate receives a useful gradient immediately.
    """

    def __init__(
        self,
        hidden_size: int,
        config: FlyEmbeddingV3Config,
        adjacency: Tensor,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        config.validate(int(hidden_size), adjacency)
        self.hidden_size = int(hidden_size)
        self.fly_nodes = int(config.fly_nodes)
        self.graph_steps = int(config.graph_steps)
        self.max_residual_scale = float(config.max_residual_scale)
        self.dropout_p = float(config.dropout)

        # Keep the graph math in fp32 for stability; cast the resulting residual
        # back to the embedding dtype at the end.
        self.down = nn.Linear(
            self.hidden_size, self.fly_nodes, bias=False, device=device, dtype=torch.float32
        )
        self.up = nn.Linear(
            self.fly_nodes, self.hidden_size, bias=False, device=device, dtype=torch.float32
        )
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)

        self.residual_scale_raw = nn.Parameter(
            torch.zeros((), device=device, dtype=torch.float32)
        )
        self.graph_mix_logit = nn.Parameter(
            torch.tensor(_logit(config.graph_mix_init), device=device, dtype=torch.float32)
        )

        a = torch.as_tensor(adjacency, device=device, dtype=torch.float32)
        self.register_buffer("adjacency", a, persistent=True)

        # Feature-wise gain starts at one. It can specialize which hidden
        # directions receive Fly residual updates without touching base Qwen.
        self.feature_gain_raw = nn.Parameter(
            torch.zeros(self.hidden_size, device=device, dtype=torch.float32)
        )

    @property
    def residual_scale(self) -> Tensor:
        return self.max_residual_scale * torch.tanh(self.residual_scale_raw)

    @property
    def graph_mix(self) -> Tensor:
        return torch.sigmoid(self.graph_mix_logit)

    def graph_step(self, q: Tensor) -> Tensor:
        mix = self.graph_mix
        a = self.adjacency
        for _ in range(self.graph_steps):
            q = (1.0 - mix) * q + mix * torch.matmul(q, a.t())
        return q

    def residual(self, embedding: Tensor) -> Tensor:
        x = embedding.float()
        q = F.linear(x, self.down.weight)
        q = F.silu(q)
        q = self.graph_step(q)
        q = F.silu(q)
        if self.training and self.dropout_p:
            q = F.dropout(q, p=self.dropout_p)
        delta = F.linear(q, self.up.weight)
        gain = 1.0 + 0.10 * torch.tanh(self.feature_gain_raw)
        delta = delta * gain
        return delta.to(dtype=embedding.dtype)

    def forward(self, embedding: Tensor) -> Tensor:
        # At initialization residual_scale == 0 exactly, hence exact identity.
        return embedding + self.residual_scale.to(embedding.dtype) * self.residual(embedding)

    def parameter_stats(self, base_embedding_params: int) -> dict[str, int | float]:
        adapter = sum(p.numel() for p in self.parameters())
        return {
            "base_embedding_params_preserved": int(base_embedding_params),
            "adapter_trainable_params": int(adapter),
            "adapter_overhead_vs_embedding_pct": float(100.0 * adapter / max(base_embedding_params, 1)),
            "fly_nodes": self.fly_nodes,
            "graph_steps": self.graph_steps,
            "residual_scale": float(self.residual_scale.detach().cpu()),
            "graph_mix": float(self.graph_mix.detach().cpu()),
        }


class FlyEmbeddingV3(nn.Module):
    """Exact frozen Qwen embedding + learnable Fly residual adapter."""

    def __init__(self, base_embedding: nn.Module, core: FlyResidualCoreV3) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.core = core

    @property
    def weight(self):
        # Preserve compatibility with HF code that inspects input embedding weight.
        return self.base_embedding.weight

    @property
    def num_embeddings(self):
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.base_embedding.embedding_dim

    def forward(self, input_ids: Tensor) -> Tensor:
        base = self.base_embedding(input_ids)
        return self.core(base)


def install_fly_embedding_v3(
    model: nn.Module,
    config: FlyEmbeddingV3Config,
    adjacency: Tensor,
) -> nn.Module:
    """Wrap Qwen's existing embedding without altering its values or LM head."""
    old_embedding = model.model.embed_tokens
    reference = old_embedding.weight
    for p in old_embedding.parameters():
        p.requires_grad_(False)

    core = FlyResidualCoreV3(
        int(model.config.hidden_size),
        config,
        adjacency,
        device=reference.device,
        dtype=reference.dtype,
    )
    wrapper = FlyEmbeddingV3(old_embedding, core)
    model.fly_embedding_v3_core = core
    model.model.embed_tokens = wrapper
    return model


def freeze_qwen_train_fly_v3(model: nn.Module) -> list[Tensor]:
    for p in model.parameters():
        p.requires_grad_(False)
    core = model.fly_embedding_v3_core
    params = list(core.parameters())
    for p in params:
        p.requires_grad_(True)
    return params


@torch.no_grad()
def assert_fly_embedding_v3_identity(
    model: nn.Module,
    reference_embedding: nn.Module,
    token_ids: Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, float | bool]:
    if not isinstance(model.model.embed_tokens, FlyEmbeddingV3):
        raise RuntimeError("model input embedding is not FlyEmbeddingV3")
    if model.model.embed_tokens.base_embedding is not reference_embedding:
        raise RuntimeError("FlyEmbeddingV3 does not wrap the original Qwen embedding")

    base = reference_embedding(token_ids)
    adapted = model.model.embed_tokens(token_ids)
    max_abs = float((adapted.float() - base.float()).abs().max().item())
    exact = bool(torch.allclose(adapted, base, atol=atol, rtol=rtol))
    if not exact:
        raise RuntimeError(f"FlyEmbedding-v3 is not identity at initialization; max_abs={max_abs}")
    return {"exact_identity": exact, "max_abs_embedding_error": max_abs}


def assert_qwen35_fly_embedding_v3(model: nn.Module) -> None:
    core = getattr(model, "fly_embedding_v3_core", None)
    if not isinstance(core, FlyResidualCoreV3):
        raise RuntimeError("FlyResidualCoreV3 missing")
    emb = model.model.embed_tokens
    if not isinstance(emb, FlyEmbeddingV3):
        raise RuntimeError("input embedding is not FlyEmbeddingV3")
    if emb.core is not core:
        raise RuntimeError("FlyEmbeddingV3/core mismatch")


__all__ = [
    "FlyEmbeddingV3Config",
    "FlyResidualCoreV3",
    "FlyEmbeddingV3",
    "install_fly_embedding_v3",
    "freeze_qwen_train_fly_v3",
    "assert_fly_embedding_v3_identity",
    "assert_qwen35_fly_embedding_v3",
]
