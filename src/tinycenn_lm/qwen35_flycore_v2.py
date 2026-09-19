from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen35_flyffn_v3 import FlyFFNV3Config, assert_qwen35_flyffn_v3


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class FlyVocabV2Config:
    latent_dim: int = 768
    fly_nodes: int = 256
    graph_steps: int = 1
    graph_mix_init: float = 0.10
    max_fly_scale: float = 0.10
    hot_token_count: int = 8192

    def validate(self, model_config, adjacency: Tensor) -> None:
        hidden = int(model_config.hidden_size)
        vocab = int(model_config.vocab_size)
        if not 64 <= self.latent_dim < hidden:
            raise ValueError(f"latent_dim must be in [64,{hidden-1}]")
        if int(adjacency.shape[0]) != self.fly_nodes or adjacency.ndim != 2:
            raise ValueError("Fly adjacency shape does not match fly_nodes")
        if not 0 <= self.hot_token_count < vocab:
            raise ValueError("hot_token_count must be in [0,vocab_size)")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 < self.graph_mix_init < 1.0:
            raise ValueError("graph_mix_init must be in (0,1)")
        if not 0.0 < self.max_fly_scale <= 0.5:
            raise ValueError("max_fly_scale must be in (0,0.5]")

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def factorize_embedding_weight_v2(
    weight: Tensor,
    latent_dim: int,
    chunk_rows: int = 4096,
) -> dict[str, Tensor | float]:
    if weight.ndim != 2:
        raise ValueError("embedding weight must be [vocab, hidden]")
    vocab, hidden = map(int, weight.shape)
    rank = int(latent_dim)
    if not 1 <= rank < hidden:
        raise ValueError(f"latent_dim={rank} must be < hidden_size={hidden}")

    device = weight.device
    gram = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
    for lo in range(0, vocab, int(chunk_rows)):
        hi = min(lo + int(chunk_rows), vocab)
        x = weight[lo:hi].float()
        gram.addmm_(x.t(), x)

    evals, evecs = torch.linalg.eigh(gram)
    top = evecs[:, -rank:].contiguous()
    pos = evals.clamp_min(0)
    retained = float((pos[-rank:].sum() / pos.sum().clamp_min(1e-12)).item())

    codes = torch.empty(vocab, rank, dtype=weight.dtype, device="cpu")
    sample_num = sample_den = 0.0
    budget = min(vocab, 8192)
    seen = 0

    for lo in range(0, vocab, int(chunk_rows)):
        hi = min(lo + int(chunk_rows), vocab)
        x = weight[lo:hi].float()
        z = x @ top
        codes[lo:hi].copy_(z.to(dtype=weight.dtype, device="cpu"))
        if seen < budget:
            n = min(hi - lo, budget - seen)
            xr = z[:n] @ top.t()
            ref = x[:n]
            sample_num += float((xr - ref).square().sum().item())
            sample_den += float(ref.square().sum().item())
            seen += n

    basis = top.t().to(dtype=weight.dtype, device="cpu").contiguous()
    return {
        "codes": codes,
        "basis": basis,
        "retained_energy": retained,
        "relative_reconstruction_mse": sample_num / max(sample_den, 1e-12),
        "original_params": int(vocab * hidden),
        "factorized_params": int(vocab * rank + rank * hidden),
    }


@torch.no_grad()
def choose_hot_tokens(
    token_batches,
    vocab_size: int,
    hot_token_count: int,
    special_ids: list[int] | None = None,
) -> Tensor:
    k = min(int(hot_token_count), int(vocab_size))
    if k <= 0:
        return torch.empty(0, dtype=torch.long)

    counts = torch.zeros(vocab_size, dtype=torch.int64)
    for ids in token_batches:
        flat = ids.detach().cpu().long().reshape(-1)
        flat = flat[(flat >= 0) & (flat < vocab_size)]
        counts += torch.bincount(flat, minlength=vocab_size)

    if special_ids:
        bonus = counts.max().clamp_min(1) + 1
        for tid in special_ids:
            if tid is not None and 0 <= int(tid) < vocab_size:
                counts[int(tid)] += bonus

    return torch.topk(counts, k=k, largest=True, sorted=True).indices


class FlyVocabCoreV2(nn.Module):
    """Adjoint-consistent tied low-rank vocabulary with Fly graph + hot-token residuals."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        config: FlyVocabV2Config,
        adjacency: Tensor,
        hot_token_ids: Tensor,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(config.latent_dim)
        self.fly_nodes = int(config.fly_nodes)
        self.graph_steps = int(config.graph_steps)
        self.max_fly_scale = float(config.max_fly_scale)

        self.codebook = nn.Embedding(
            self.vocab_size, self.latent_dim, device=device, dtype=dtype
        )
        self.basis = nn.Parameter(
            torch.empty(self.latent_dim, self.hidden_size, device=device, dtype=dtype)
        )

        # Linear Fly transform so the LM head can use the exact adjoint T^T.
        self.fly_down = nn.Linear(
            self.latent_dim, self.fly_nodes, bias=False, device=device, dtype=torch.float32
        )
        self.fly_up = nn.Linear(
            self.fly_nodes, self.latent_dim, bias=False, device=device, dtype=torch.float32
        )
        nn.init.normal_(self.fly_down.weight, mean=0.0, std=0.01)
        # Keep the transform exactly disabled through fly_scale=0, but give
        # fly_up a tiny non-zero initialization so fly_scale receives a gradient.
        nn.init.normal_(self.fly_up.weight, mean=0.0, std=0.002)
        self.fly_scale_raw = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
        self.graph_mix_logit = nn.Parameter(
            torch.tensor(_logit(config.graph_mix_init), device=device, dtype=torch.float32)
        )

        a = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
        self.register_buffer("adjacency", a, persistent=True)

        hot = hot_token_ids.detach().cpu().long().unique(sorted=True)
        self.register_buffer("hot_token_ids", hot.to(device=device), persistent=True)
        inverse = torch.full((self.vocab_size,), -1, dtype=torch.int32)
        if hot.numel():
            inverse[hot] = torch.arange(hot.numel(), dtype=torch.int32)
        self.register_buffer("hot_inverse", inverse.to(device=device), persistent=True)
        self.hot_residual = nn.Parameter(
            torch.zeros(hot.numel(), self.hidden_size, device=device, dtype=dtype)
        )

    @property
    def fly_scale(self) -> Tensor:
        return self.max_fly_scale * torch.tanh(self.fly_scale_raw.float())

    def _graph_forward(self, q: Tensor) -> Tensor:
        mix = torch.sigmoid(self.graph_mix_logit.float())
        a = self.adjacency.float()
        for _ in range(self.graph_steps):
            q = (1.0 - mix) * q + mix * torch.matmul(q, a.t())
        return q

    def _graph_adjoint(self, q: Tensor) -> Tensor:
        mix = torch.sigmoid(self.graph_mix_logit.float())
        a = self.adjacency.float()
        for _ in range(self.graph_steps):
            q = (1.0 - mix) * q + mix * torch.matmul(q, a)
        return q

    def transform(self, z: Tensor) -> Tensor:
        zf = z.float()
        q = F.linear(zf, self.fly_down.weight.float())
        q = self._graph_forward(q)
        delta = F.linear(q, self.fly_up.weight.float())
        return z + (self.fly_scale * delta).to(dtype=z.dtype)

    def transform_adjoint(self, z: Tensor) -> Tensor:
        zf = z.float()
        # If forward is z @ D^T @ S @ U^T, adjoint is z @ U @ S^T @ D.
        q = torch.matmul(zf, self.fly_up.weight.float())
        q = self._graph_adjoint(q)
        delta = torch.matmul(q, self.fly_down.weight.float())
        return z + (self.fly_scale * delta).to(dtype=z.dtype)

    def embed(self, input_ids: Tensor) -> Tensor:
        z = self.transform(self.codebook(input_ids))
        out = torch.matmul(z, self.basis)
        if self.hot_token_ids.numel():
            idx = self.hot_inverse[input_ids].long()
            mask = idx >= 0
            if mask.any():
                safe = idx.clamp_min(0)
                residual = self.hot_residual[safe]
                out = out + residual * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out

    def logits(self, hidden_states: Tensor) -> Tensor:
        latent = torch.matmul(hidden_states, self.basis.t())
        latent = self.transform_adjoint(latent)
        logits = F.linear(latent, self.codebook.weight)
        if self.hot_token_ids.numel():
            corr = F.linear(hidden_states, self.hot_residual)
            logits.index_add_(-1, self.hot_token_ids.long(), corr)
        return logits

    def effective_hot_weight(self, hot_index: Tensor | None = None) -> Tensor:
        if not self.hot_token_ids.numel():
            return torch.empty(
                0, self.hidden_size, device=self.basis.device, dtype=self.basis.dtype
            )
        if hot_index is None:
            token_ids = self.hot_token_ids
            residual = self.hot_residual
        else:
            hot_index = hot_index.long()
            token_ids = self.hot_token_ids[hot_index]
            residual = self.hot_residual[hot_index]
        z = self.transform(self.codebook(token_ids))
        return torch.matmul(z, self.basis) + residual

    def parameter_stats(self) -> dict[str, int | float]:
        original = self.vocab_size * self.hidden_size
        current = sum(p.numel() for p in self.parameters())
        return {
            "original_tied_vocab_params": int(original),
            "fly_vocab_trainable_params": int(current),
            "vocab_param_ratio": float(current / original),
            "vocab_param_reduction_pct": float(100.0 * (1.0 - current / original)),
            "latent_dim": self.latent_dim,
            "hot_token_count": int(self.hot_token_ids.numel()),
            "fly_scale": float(self.fly_scale.detach().cpu()),
        }


class FlyEmbeddingV2(nn.Module):
    def __init__(self, core: FlyVocabCoreV2) -> None:
        super().__init__()
        object.__setattr__(self, "_core_ref", core)

    @property
    def core(self) -> FlyVocabCoreV2:
        return object.__getattribute__(self, "_core_ref")

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.core.embed(input_ids)


class FlyLMHeadV2(nn.Module):
    def __init__(self, core: FlyVocabCoreV2) -> None:
        super().__init__()
        self.in_features = core.hidden_size
        self.out_features = core.vocab_size
        object.__setattr__(self, "_core_ref", core)

    @property
    def core(self) -> FlyVocabCoreV2:
        return object.__getattribute__(self, "_core_ref")

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.core.logits(hidden_states)


@torch.no_grad()
def initialize_fly_vocab_v2(
    core: FlyVocabCoreV2,
    factorization: dict[str, Tensor | float],
    teacher_weight: Tensor,
) -> None:
    codes = factorization["codes"]
    basis = factorization["basis"]
    if not isinstance(codes, Tensor) or not isinstance(basis, Tensor):
        raise TypeError("factorization must contain codes and basis tensors")

    core.codebook.weight.copy_(
        codes.to(core.codebook.weight.device, core.codebook.weight.dtype)
    )
    core.basis.copy_(basis.to(core.basis.device, core.basis.dtype))

    if core.hot_token_ids.numel():
        hot = core.hot_token_ids.to(teacher_weight.device)
        target = teacher_weight[hot].to(core.basis.device, core.basis.dtype)
        base = torch.matmul(
            core.codebook(core.hot_token_ids).to(core.basis.dtype),
            core.basis,
        )
        core.hot_residual.copy_(target - base)



def install_fly_embedding_v2(
    model: nn.Module,
    config: FlyVocabV2Config,
    adjacency: Tensor,
    hot_token_ids: Tensor,
    *,
    factorization: dict[str, Tensor | float],
    teacher_weight: Tensor,
) -> nn.Module:
    """Replace only the input embedding; keep Qwen's original LM head untouched."""
    config.validate(model.config, adjacency)
    old_embed = model.model.embed_tokens
    reference = old_embed.weight

    core = FlyVocabCoreV2(
        int(model.config.vocab_size),
        int(model.config.hidden_size),
        config,
        adjacency,
        hot_token_ids,
        device=reference.device,
        dtype=reference.dtype,
    )
    initialize_fly_vocab_v2(core, factorization, teacher_weight)

    model.fly_vocab_core_v2 = core
    model.model.embed_tokens = FlyEmbeddingV2(core)

    # The output projection must stay as the original Qwen head. Breaking the
    # embedding tie is intentional: the experiment compresses input embeddings
    # only, avoiding vocabulary-ranking / early-EOS drift in the LM head.
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    try:
        model._tied_weights_keys = {}
    except Exception:
        pass
    return model


def assert_qwen35_fly_embedding_v2(model: nn.Module) -> None:
    core = getattr(model, "fly_vocab_core_v2", None)
    if not isinstance(core, FlyVocabCoreV2):
        raise RuntimeError("FlyVocabCoreV2 missing")
    if not isinstance(model.model.embed_tokens, FlyEmbeddingV2):
        raise RuntimeError("input embedding is not FlyEmbeddingV2")
    if model.model.embed_tokens.core is not core:
        raise RuntimeError("FlyEmbeddingV2 does not reference fly_vocab_core_v2")
    if isinstance(model.lm_head, FlyLMHeadV2):
        raise RuntimeError("LM head was replaced; input-only mode must keep Qwen lm_head")


def install_fly_vocab_v2(
    model: nn.Module,
    config: FlyVocabV2Config,
    adjacency: Tensor,
    hot_token_ids: Tensor,
    *,
    factorization: dict[str, Tensor | float],
    teacher_weight: Tensor,
) -> nn.Module:
    config.validate(model.config, adjacency)
    old_embed = model.model.embed_tokens
    reference = old_embed.weight

    core = FlyVocabCoreV2(
        int(model.config.vocab_size),
        int(model.config.hidden_size),
        config,
        adjacency,
        hot_token_ids,
        device=reference.device,
        dtype=reference.dtype,
    )
    initialize_fly_vocab_v2(core, factorization, teacher_weight)

    model.fly_vocab_core_v2 = core
    model.model.embed_tokens = FlyEmbeddingV2(core)
    model.lm_head = FlyLMHeadV2(core)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    try:
        model._tied_weights_keys = {}
    except Exception:
        pass
    return model


def assert_qwen35_flycore_v2(model: nn.Module) -> None:
    assert_qwen35_flyffn_v3(model)
    core = getattr(model, "fly_vocab_core_v2", None)
    if not isinstance(core, FlyVocabCoreV2):
        raise RuntimeError("FlyVocabCoreV2 missing")
    if not isinstance(model.model.embed_tokens, FlyEmbeddingV2):
        raise RuntimeError("input embedding is not FlyEmbeddingV2")
    if not isinstance(model.lm_head, FlyLMHeadV2):
        raise RuntimeError("LM head is not FlyLMHeadV2")
    if model.model.embed_tokens.core is not core or model.lm_head.core is not core:
        raise RuntimeError("embedding/head do not share FlyVocabCoreV2")


def freeze_source_ffn_train_vocab(model: nn.Module) -> list[Tensor]:
    for p in model.parameters():
        p.requires_grad = False
    core = model.fly_vocab_core_v2
    params = [
        core.codebook.weight,
        core.basis,
        core.hot_residual,
        core.fly_down.weight,
        core.fly_up.weight,
        core.fly_scale_raw,
        core.graph_mix_logit,
    ]
    for p in params:
        p.requires_grad = True
    return params


__all__ = [
    "FlyVocabV2Config",
    "FlyVocabCoreV2",
    "FlyEmbeddingV2",
    "FlyLMHeadV2",
    "factorize_embedding_weight_v2",
    "choose_hot_tokens",
    "install_fly_embedding_v2",
    "install_fly_vocab_v2",
    "assert_qwen35_fly_embedding_v2",
    "assert_qwen35_flycore_v2",
    "freeze_source_ffn_train_vocab",
    "FlyFFNV3Config",
]
