from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class FlyEmbeddingV33Config:
    latent_dim: int = 256
    fly_nodes: int = 256
    graph_steps: int = 1
    graph_mix: float = 0.05
    max_fly_scale: float = 0.05

    def validate(self, model_config) -> None:
        h = int(model_config.hidden_size)
        if not 32 <= self.latent_dim < h:
            raise ValueError(f"latent_dim must be in [32,{h-1}]")
        if self.fly_nodes < 16:
            raise ValueError("fly_nodes must be >= 16")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 <= self.graph_mix <= 1.0:
            raise ValueError("graph_mix must be in [0,1]")
        if not 0.0 < self.max_fly_scale <= 0.5:
            raise ValueError("max_fly_scale must be in (0,0.5]")

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def factorize_vocab_weight_v33(weight: Tensor, rank: int, chunk_rows: int = 4096) -> dict:
    """Low-rank initialization W ~= codes @ basis using the hidden-space Gram matrix."""
    vocab, hidden = map(int, weight.shape)
    gram = torch.zeros(hidden, hidden, device=weight.device, dtype=torch.float32)
    for lo in range(0, vocab, chunk_rows):
        x = weight[lo:min(lo+chunk_rows, vocab)].float()
        gram.addmm_(x.t(), x)
    evals, evecs = torch.linalg.eigh(gram)
    basis_cols = evecs[:, -rank:].contiguous()
    pos = evals.clamp_min(0)
    retained = float((pos[-rank:].sum()/pos.sum().clamp_min(1e-12)).item())

    codes = torch.empty(vocab, rank, dtype=weight.dtype, device="cpu")
    for lo in range(0, vocab, chunk_rows):
        hi = min(lo+chunk_rows, vocab)
        codes[lo:hi].copy_((weight[lo:hi].float() @ basis_cols).to(weight.dtype).cpu())
    basis = basis_cols.t().to(weight.dtype).cpu().contiguous()
    return {
        "codes": codes,
        "basis": basis,
        "retained_energy": retained,
        "original_params": vocab * hidden,
        "compact_params": vocab * rank + rank * hidden,
    }


class CompactFlyVocabV33(nn.Module):
    """Compact vocabulary core shared by input embedding and output head."""

    def __init__(self, vocab_size: int, hidden_size: int, cfg: FlyEmbeddingV33Config, *, device, dtype):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(cfg.latent_dim)
        self.fly_nodes = int(cfg.fly_nodes)
        self.graph_steps = int(cfg.graph_steps)
        self.graph_mix = float(cfg.graph_mix)
        self.max_fly_scale = float(cfg.max_fly_scale)

        self.codebook = nn.Embedding(self.vocab_size, self.latent_dim, device=device, dtype=dtype)
        self.basis = nn.Parameter(torch.empty(self.latent_dim, self.hidden_size, device=device, dtype=dtype))

        self.fly_down = nn.Linear(self.latent_dim, self.fly_nodes, bias=False, device=device, dtype=torch.float32)
        self.fly_up = nn.Linear(self.fly_nodes, self.latent_dim, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.fly_down.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fly_up.weight, mean=0.0, std=0.002)
        self.fly_scale_raw = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))

        # Deterministic small-world graph.
        a = torch.zeros(self.fly_nodes, self.fly_nodes, device=device, dtype=torch.float32)
        for i in range(self.fly_nodes):
            a[i, i] = 1.0
            for s in (1, 3, 7, 17):
                a[i, (i+s) % self.fly_nodes] = 1.0
                a[i, (i-s) % self.fly_nodes] = 1.0
        a /= a.sum(-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adjacency", a, persistent=True)

    @property
    def fly_scale(self) -> Tensor:
        return self.max_fly_scale * torch.tanh(self.fly_scale_raw)

    def _transform(self, z: Tensor) -> Tensor:
        q = F.silu(F.linear(z.float(), self.fly_down.weight))
        for _ in range(self.graph_steps):
            q = (1.0-self.graph_mix)*q + self.graph_mix*torch.matmul(q, self.adjacency.t())
        d = F.linear(F.silu(q), self.fly_up.weight)
        return z + self.fly_scale.to(z.dtype) * d.to(z.dtype)

    def embed(self, ids: Tensor) -> Tensor:
        z = self._transform(self.codebook(ids))
        return torch.matmul(z, self.basis)

    def logits(self, hidden: Tensor, chunk_size: int | None = None) -> Tensor:
        # Output head uses the effective compact vocabulary weights.
        latent = torch.matmul(hidden, self.basis.t())
        # Approximate adjoint with the base codebook projection. Training aligns
        # logits directly through CE/KL, so exact algebraic tying is not required.
        return F.linear(latent, self.codebook.weight)

    def effective_weight(self, chunk_rows: int = 4096) -> Tensor:
        rows = []
        for lo in range(0, self.vocab_size, chunk_rows):
            hi = min(lo+chunk_rows, self.vocab_size)
            ids = torch.arange(lo, hi, device=self.basis.device)
            rows.append(self.embed(ids).detach().cpu())
        return torch.cat(rows, dim=0)

    def stats(self, original_vocab_params: int) -> dict:
        current = sum(p.numel() for p in self.parameters())
        return {
            "compact_vocab_params": int(current),
            "original_vocab_params": int(original_vocab_params),
            "vocab_param_reduction_pct": float(100*(1-current/max(original_vocab_params,1))),
            "latent_dim": self.latent_dim,
            "fly_nodes": self.fly_nodes,
            "fly_scale": float(self.fly_scale.detach().cpu()),
        }


class ProgressiveFlyEmbeddingV33(nn.Module):
    def __init__(self, qwen_embedding: nn.Module, compact: CompactFlyVocabV33) -> None:
        super().__init__()
        self.qwen_embedding = qwen_embedding
        self.compact = compact
        self.register_buffer("beta", torch.tensor(0.0, dtype=torch.float32), persistent=True)

    @property
    def weight(self):
        # Compatibility while beta < 1. Final compact export removes this wrapper.
        return self.qwen_embedding.weight

    def set_beta(self, beta: float) -> None:
        self.beta.fill_(float(beta))

    def forward(self, ids: Tensor) -> Tensor:
        b = float(self.beta.item())
        if b <= 0.0:
            return self.qwen_embedding(ids)
        c = self.compact.embed(ids)
        if b >= 1.0:
            return c
        q = self.qwen_embedding(ids)
        return (1.0-b)*q + b*c


class ProgressiveFlyLMHeadV33(nn.Module):
    def __init__(self, qwen_head: nn.Module, compact: CompactFlyVocabV33) -> None:
        super().__init__()
        self.qwen_head = qwen_head
        self.compact = compact
        self.register_buffer("beta", torch.tensor(0.0, dtype=torch.float32), persistent=True)

    def set_beta(self, beta: float) -> None:
        self.beta.fill_(float(beta))

    def forward(self, hidden: Tensor) -> Tensor:
        b = float(self.beta.item())
        if b <= 0.0:
            return self.qwen_head(hidden)
        c = self.compact.logits(hidden)
        if b >= 1.0:
            return c
        q = self.qwen_head(hidden)
        return (1.0-b)*q + b*c


@torch.no_grad()
def initialize_compact_v33(compact: CompactFlyVocabV33, factorization: dict) -> None:
    compact.codebook.weight.copy_(factorization["codes"].to(compact.codebook.weight.device, compact.codebook.weight.dtype))
    compact.basis.copy_(factorization["basis"].to(compact.basis.device, compact.basis.dtype))


def install_fly_embedding_v33(model: nn.Module, cfg: FlyEmbeddingV33Config, factorization: dict) -> nn.Module:
    cfg.validate(model.config)
    qemb = model.model.embed_tokens
    qhead = model.lm_head
    for p in qemb.parameters(): p.requires_grad_(False)
    for p in qhead.parameters(): p.requires_grad_(False)

    ref = qemb.weight
    compact = CompactFlyVocabV33(
        int(model.config.vocab_size), int(model.config.hidden_size), cfg,
        device=ref.device, dtype=ref.dtype
    )
    initialize_compact_v33(compact, factorization)

    model.fly_embedding_v33_core = compact
    model.model.embed_tokens = ProgressiveFlyEmbeddingV33(qemb, compact)
    model.lm_head = ProgressiveFlyLMHeadV33(qhead, compact)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    return model


def set_progressive_beta_v33(model: nn.Module, beta: float) -> None:
    model.model.embed_tokens.set_beta(beta)
    model.lm_head.set_beta(beta)


def freeze_qwen_train_compact_v33(model: nn.Module) -> list[Tensor]:
    for p in model.parameters(): p.requires_grad_(False)
    params = list(model.fly_embedding_v33_core.parameters())
    for p in params: p.requires_grad_(True)
    return params


@torch.no_grad()
def export_compact_only_v33(model: nn.Module) -> nn.Module:
    """Drop the full Qwen input embedding/head from the active model path.

    The result still uses the Qwen transformer body, but its vocabulary
    representation is entirely CompactFlyVocabV33.
    """
    core = model.fly_embedding_v33_core

    class CompactEmbedding(nn.Module):
        def __init__(self, c):
            super().__init__()
            object.__setattr__(self, "_core", c)
        def forward(self, ids):
            return object.__getattribute__(self, "_core").embed(ids)

    class CompactHead(nn.Module):
        def __init__(self, c):
            super().__init__()
            object.__setattr__(self, "_core", c)
        def forward(self, h):
            return object.__getattribute__(self, "_core").logits(h)

    model.model.embed_tokens = CompactEmbedding(core)
    model.lm_head = CompactHead(core)
    return model


__all__ = [
    "FlyEmbeddingV33Config",
    "CompactFlyVocabV33",
    "ProgressiveFlyEmbeddingV33",
    "ProgressiveFlyLMHeadV33",
    "factorize_vocab_weight_v33",
    "install_fly_embedding_v33",
    "set_progressive_beta_v33",
    "freeze_qwen_train_compact_v33",
    "export_compact_only_v33",
]
