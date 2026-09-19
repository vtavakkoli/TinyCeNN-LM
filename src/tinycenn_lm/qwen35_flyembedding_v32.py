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
class FlyEmbeddingV32Config:
    fly_nodes: int = 256
    graph_steps: int = 1
    graph_mix_init: float = 0.05
    max_residual_scale: float = 0.05
    gate_groups: int = 32
    gate_hidden: int = 64
    dropout: float = 0.0

    def validate(self, hidden_size: int, adjacency: Tensor) -> None:
        if not 16 <= self.fly_nodes <= hidden_size * 2:
            raise ValueError("fly_nodes must be in [16,2*hidden_size]")
        if adjacency.ndim != 2 or tuple(adjacency.shape) != (self.fly_nodes, self.fly_nodes):
            raise ValueError("adjacency must be [fly_nodes,fly_nodes]")
        if hidden_size % self.gate_groups != 0:
            raise ValueError("hidden_size must be divisible by gate_groups")
        if self.gate_hidden < 8:
            raise ValueError("gate_hidden must be >= 8")
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


class FlyResidualCoreV32(nn.Module):
    """Token-dependent grouped-gate Fly residual.

    Identity at initialization is exact because residual_scale_raw starts at 0.
    The gate is learned from each token embedding and controls groups of hidden
    dimensions independently.
    """

    def __init__(
        self,
        hidden_size: int,
        config: FlyEmbeddingV32Config,
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
        self.gate_groups = int(config.gate_groups)
        self.group_width = self.hidden_size // self.gate_groups
        self.dropout_p = float(config.dropout)

        self.down = nn.Linear(self.hidden_size, self.fly_nodes, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(self.fly_nodes, self.hidden_size, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.002)

        self.gate_down = nn.Linear(self.hidden_size, int(config.gate_hidden), bias=True, device=device, dtype=torch.float32)
        self.gate_up = nn.Linear(int(config.gate_hidden), self.gate_groups, bias=True, device=device, dtype=torch.float32)
        nn.init.normal_(self.gate_down.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate_down.bias)
        nn.init.normal_(self.gate_up.weight, mean=0.0, std=0.002)
        # Start group gates near 0.5; the global residual scale still guarantees exact identity.
        nn.init.zeros_(self.gate_up.bias)

        self.residual_scale_raw = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
        self.graph_mix_logit = nn.Parameter(
            torch.tensor(_logit(config.graph_mix_init), device=device, dtype=torch.float32)
        )

        a = torch.as_tensor(adjacency, device=device, dtype=torch.float32)
        self.register_buffer("adjacency", a, persistent=True)

    @property
    def residual_scale(self) -> Tensor:
        return self.max_residual_scale * torch.tanh(self.residual_scale_raw)

    @property
    def graph_mix(self) -> Tensor:
        return torch.sigmoid(self.graph_mix_logit)

    def _graph(self, q: Tensor) -> Tensor:
        mix = self.graph_mix
        a = self.adjacency
        for _ in range(self.graph_steps):
            q = (1.0 - mix) * q + mix * torch.matmul(q, a.t())
        return q

    def gate(self, embedding: Tensor) -> Tensor:
        x = embedding.float()
        g = F.silu(self.gate_down(x))
        g = torch.sigmoid(self.gate_up(g))
        # Expand group gates back to hidden size.
        return g.repeat_interleave(self.group_width, dim=-1)

    def residual(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        x = embedding.float()
        q = F.silu(F.linear(x, self.down.weight))
        q = self._graph(q)
        q = F.silu(q)
        if self.training and self.dropout_p:
            q = F.dropout(q, p=self.dropout_p)
        delta = F.linear(q, self.up.weight)
        gate = self.gate(embedding)
        delta = delta * gate
        return delta.to(dtype=embedding.dtype), gate

    def forward(self, embedding: Tensor) -> Tensor:
        delta, _ = self.residual(embedding)
        return embedding + self.residual_scale.to(embedding.dtype) * delta

    @torch.no_grad()
    def gate_stats(self, embedding: Tensor) -> dict[str, float]:
        g = self.gate(embedding)
        return {
            "gate_mean": float(g.mean().item()),
            "gate_std": float(g.std().item()),
            "gate_min": float(g.min().item()),
            "gate_max": float(g.max().item()),
        }

    def parameter_stats(self, base_embedding_params: int) -> dict[str, int | float]:
        adapter = sum(p.numel() for p in self.parameters())
        return {
            "base_embedding_params_preserved": int(base_embedding_params),
            "adapter_trainable_params": int(adapter),
            "adapter_overhead_vs_embedding_pct": float(100.0 * adapter / max(base_embedding_params, 1)),
            "fly_nodes": self.fly_nodes,
            "graph_steps": self.graph_steps,
            "gate_groups": self.gate_groups,
            "group_width": self.group_width,
            "residual_scale": float(self.residual_scale.detach().cpu()),
            "graph_mix": float(self.graph_mix.detach().cpu()),
        }


class FlyEmbeddingV32(nn.Module):
    def __init__(self, base_embedding: nn.Module, core: FlyResidualCoreV32) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.core = core

    @property
    def weight(self):
        return self.base_embedding.weight

    @property
    def num_embeddings(self):
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.base_embedding.embedding_dim

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.core(self.base_embedding(input_ids))


def install_fly_embedding_v32(model: nn.Module, config: FlyEmbeddingV32Config, adjacency: Tensor) -> nn.Module:
    old_embedding = model.model.embed_tokens
    reference = old_embedding.weight
    for p in old_embedding.parameters():
        p.requires_grad_(False)
    core = FlyResidualCoreV32(
        int(model.config.hidden_size),
        config,
        adjacency,
        device=reference.device,
        dtype=reference.dtype,
    )
    model.fly_embedding_v32_core = core
    model.model.embed_tokens = FlyEmbeddingV32(old_embedding, core)
    return model


def freeze_qwen_train_fly_v32(model: nn.Module) -> list[Tensor]:
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.fly_embedding_v32_core.parameters())
    for p in params:
        p.requires_grad_(True)
    return params


@torch.no_grad()
def assert_fly_embedding_v32_identity(
    model: nn.Module,
    reference_embedding: nn.Module,
    token_ids: Tensor,
) -> dict[str, float | bool]:
    if not isinstance(model.model.embed_tokens, FlyEmbeddingV32):
        raise RuntimeError("model input embedding is not FlyEmbeddingV32")
    base = reference_embedding(token_ids)
    adapted = model.model.embed_tokens(token_ids)
    max_abs = float((adapted.float() - base.float()).abs().max().item())
    exact = bool(torch.equal(adapted, base))
    if not exact:
        raise RuntimeError(f"FlyEmbedding-v3.2 identity failed; max_abs={max_abs}")
    return {"exact_identity": exact, "max_abs_embedding_error": max_abs}


def assert_qwen35_fly_embedding_v32(model: nn.Module) -> None:
    core = getattr(model, "fly_embedding_v32_core", None)
    if not isinstance(core, FlyResidualCoreV32):
        raise RuntimeError("FlyResidualCoreV32 missing")
    if not isinstance(model.model.embed_tokens, FlyEmbeddingV32):
        raise RuntimeError("input embedding is not FlyEmbeddingV32")
    if model.model.embed_tokens.core is not core:
        raise RuntimeError("FlyEmbeddingV32/core mismatch")


@torch.no_grad()
def materialize_fly_embedding_v32(model: nn.Module, *, chunk_rows: int = 4096) -> nn.Module:
    """Materialize the token-dependent v3.2 adapter into a standard embedding matrix."""
    assert_qwen35_fly_embedding_v32(model)
    wrapper = model.model.embed_tokens
    base = wrapper.base_embedding
    core = wrapper.core

    vocab = int(base.num_embeddings)
    hidden = int(base.embedding_dim)
    device = base.weight.device
    dtype = base.weight.dtype

    old_head = model.lm_head
    head_weight = old_head.weight.detach().clone()
    new_head = nn.Linear(
        hidden,
        int(head_weight.shape[0]),
        bias=getattr(old_head, "bias", None) is not None,
        device=device,
        dtype=head_weight.dtype,
    )
    new_head.weight.copy_(head_weight)
    if getattr(old_head, "bias", None) is not None:
        new_head.bias.copy_(old_head.bias.detach())

    effective = torch.empty(vocab, hidden, device="cpu", dtype=dtype)
    was_training = core.training
    core.eval()
    for lo in range(0, vocab, int(chunk_rows)):
        hi = min(lo + int(chunk_rows), vocab)
        ids = torch.arange(lo, hi, device=device, dtype=torch.long)
        effective[lo:hi].copy_(core(base(ids)).detach().cpu())
    core.train(was_training)

    new_embedding = nn.Embedding(vocab, hidden, device=device, dtype=dtype)
    new_embedding.weight.copy_(effective.to(device=device, dtype=dtype))
    model.model.embed_tokens = new_embedding
    model.lm_head = new_head
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    try:
        model._tied_weights_keys = {}
    except Exception:
        pass
    if hasattr(model, "fly_embedding_v32_core"):
        delattr(model, "fly_embedding_v32_core")
    return model


__all__ = [
    "FlyEmbeddingV32Config",
    "FlyResidualCoreV32",
    "FlyEmbeddingV32",
    "install_fly_embedding_v32",
    "freeze_qwen_train_fly_v32",
    "assert_fly_embedding_v32_identity",
    "assert_qwen35_fly_embedding_v32",
    "materialize_fly_embedding_v32",
]
