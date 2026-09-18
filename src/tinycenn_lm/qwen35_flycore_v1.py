from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen35_flyffn_v3 import (
    FlyFFNV3Config,
    assert_qwen35_flyffn_v3,
    replace_all_ffns_with_fly_v3,
)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class FlyVocabConfig:
    """Shared low-rank FlyEmbedding + FlyLMHead configuration."""

    latent_dim: int = 384
    fly_nodes: int = 256
    graph_steps: int = 1
    graph_mix_init: float = 0.10

    def validate(self, model_config, adjacency: Tensor) -> None:
        hidden = int(model_config.hidden_size)
        vocab = int(model_config.vocab_size)
        if not 32 <= self.latent_dim < hidden:
            raise ValueError(f"latent_dim must be in [32,{hidden - 1}]")
        if self.fly_nodes < 32:
            raise ValueError("fly_nodes must be >= 32")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Fly adjacency must be square")
        if int(adjacency.shape[0]) != int(self.fly_nodes):
            raise ValueError(
                f"adjacency has {adjacency.shape[0]} nodes, expected {self.fly_nodes}"
            )
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 < self.graph_mix_init < 1.0:
            raise ValueError("graph_mix_init must be in (0,1)")
        if vocab < 1000:
            raise ValueError("unexpectedly small vocabulary")

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def factorize_embedding_weight(
    weight: Tensor,
    latent_dim: int,
    chunk_rows: int = 4096,
) -> dict[str, Tensor | float]:
    """Optimal uncentered rank-r approximation from E^T E, without materializing E in fp32."""
    if weight.ndim != 2:
        raise ValueError("embedding weight must be [vocab, hidden]")
    vocab, hidden = map(int, weight.shape)
    rank = int(latent_dim)
    if not 1 <= rank < hidden:
        raise ValueError(f"latent_dim={rank} must be < hidden_size={hidden}")

    device = weight.device
    gram = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
    total_sq = 0.0
    for lo in range(0, vocab, int(chunk_rows)):
        hi = min(lo + int(chunk_rows), vocab)
        x = weight[lo:hi].float()
        gram.addmm_(x.t(), x)
        total_sq += float(x.square().sum().item())

    evals, evecs = torch.linalg.eigh(gram)
    top = evecs[:, -rank:].contiguous()  # [hidden, rank]
    kept = evals[-rank:].clamp_min(0).sum()
    retained_energy = float((kept / evals.clamp_min(0).sum().clamp_min(1e-12)).item())

    codes = torch.empty(vocab, rank, dtype=weight.dtype, device="cpu")
    sample_err_num = 0.0
    sample_err_den = 0.0
    sample_budget = min(vocab, 8192)
    sample_done = 0

    for lo in range(0, vocab, int(chunk_rows)):
        hi = min(lo + int(chunk_rows), vocab)
        x = weight[lo:hi].float()
        z = x @ top
        codes[lo:hi].copy_(z.to(dtype=weight.dtype, device="cpu"))

        if sample_done < sample_budget:
            n = min(hi - lo, sample_budget - sample_done)
            xr = z[:n] @ top.t()
            ref = x[:n]
            sample_err_num += float((xr - ref).square().sum().item())
            sample_err_den += float(ref.square().sum().item())
            sample_done += n

    basis = top.t().to(dtype=weight.dtype, device="cpu").contiguous()  # [rank, hidden]
    relative_mse = sample_err_num / max(sample_err_den, 1e-12)

    return {
        "codes": codes,
        "basis": basis,
        "retained_energy": retained_energy,
        "relative_reconstruction_mse": float(relative_mse),
        "original_params": int(vocab * hidden),
        "factorized_params": int(vocab * rank + rank * hidden),
    }


class FlyVocabCore(nn.Module):
    """One shared latent vocabulary core used by both input embedding and LM head."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        config: FlyVocabConfig,
        adjacency: Tensor,
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

        self.codebook = nn.Embedding(
            self.vocab_size,
            self.latent_dim,
            device=device,
            dtype=dtype,
        )
        self.basis = nn.Parameter(
            torch.empty(self.latent_dim, self.hidden_size, device=device, dtype=dtype)
        )

        # Fly residual path. fly_up starts at zero, so the initial model is exactly
        # the low-rank SVD approximation before Fly adaptation starts.
        self.fly_down = nn.Linear(
            self.latent_dim, self.fly_nodes, bias=False, device=device, dtype=torch.float32
        )
        self.fly_up = nn.Linear(
            self.fly_nodes, self.latent_dim, bias=False, device=device, dtype=torch.float32
        )
        nn.init.normal_(self.fly_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fly_up.weight)

        self.graph_mix_logit = nn.Parameter(
            torch.tensor(_logit(config.graph_mix_init), device=device, dtype=torch.float32)
        )

        a = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
        self.register_buffer("adjacency", a, persistent=True)

        self.register_buffer("input_fly_enabled", torch.tensor(1, dtype=torch.int8), persistent=True)
        self.register_buffer("output_fly_enabled", torch.tensor(1, dtype=torch.int8), persistent=True)

    def _fly(self, z: Tensor) -> Tensor:
        z_dtype = z.dtype
        q = F.silu(F.linear(z.float(), self.fly_down.weight.float()))
        mix = torch.sigmoid(self.graph_mix_logit.float())
        a = self.adjacency.float()
        for _ in range(self.graph_steps):
            graph_q = torch.matmul(q, a.t())
            q = torch.tanh((1.0 - mix) * q + mix * graph_q)
        delta = F.linear(q, self.fly_up.weight.float())
        return z + delta.to(dtype=z_dtype)

    def embed(self, input_ids: Tensor) -> Tensor:
        z = self.codebook(input_ids)
        if bool(self.input_fly_enabled.item()):
            z = self._fly(z)
        return torch.matmul(z, self.basis)

    def logits(self, hidden_states: Tensor) -> Tensor:
        latent = torch.matmul(hidden_states, self.basis.t())
        if bool(self.output_fly_enabled.item()):
            latent = self._fly(latent)
        return F.linear(latent, self.codebook.weight)

    def parameter_stats(self) -> dict[str, int | float]:
        original = self.vocab_size * self.hidden_size
        current = sum(p.numel() for p in self.parameters())
        return {
            "original_tied_vocab_params": int(original),
            "fly_vocab_trainable_params": int(current),
            "vocab_param_ratio": float(current / original),
            "vocab_param_reduction_pct": float(100.0 * (1.0 - current / original)),
        }


class FlyEmbedding(nn.Module):
    def __init__(self, core: FlyVocabCore) -> None:
        super().__init__()
        object.__setattr__(self, "_core_ref", core)

    @property
    def core(self) -> FlyVocabCore:
        return object.__getattribute__(self, "_core_ref")

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.core.embed(input_ids)


class FlyLMHead(nn.Module):
    def __init__(self, core: FlyVocabCore) -> None:
        super().__init__()
        self.in_features = core.hidden_size
        self.out_features = core.vocab_size
        object.__setattr__(self, "_core_ref", core)

    @property
    def core(self) -> FlyVocabCore:
        return object.__getattribute__(self, "_core_ref")

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.core.logits(hidden_states)


@torch.no_grad()
def initialize_fly_vocab_from_factorization(
    core: FlyVocabCore,
    factorization: dict[str, Tensor | float],
) -> None:
    codes = factorization["codes"]
    basis = factorization["basis"]
    if not isinstance(codes, Tensor) or not isinstance(basis, Tensor):
        raise TypeError("factorization must contain tensor codes and basis")
    if tuple(codes.shape) != tuple(core.codebook.weight.shape):
        raise ValueError(
            f"code shape mismatch: {tuple(codes.shape)} vs {tuple(core.codebook.weight.shape)}"
        )
    if tuple(basis.shape) != tuple(core.basis.shape):
        raise ValueError(
            f"basis shape mismatch: {tuple(basis.shape)} vs {tuple(core.basis.shape)}"
        )
    core.codebook.weight.copy_(
        codes.to(device=core.codebook.weight.device, dtype=core.codebook.weight.dtype)
    )
    core.basis.copy_(basis.to(device=core.basis.device, dtype=core.basis.dtype))


def install_fly_vocab(
    model: nn.Module,
    config: FlyVocabConfig,
    adjacency: Tensor,
    *,
    factorization: dict[str, Tensor | float] | None = None,
) -> nn.Module:
    config.validate(model.config, adjacency)
    old_embed = model.model.embed_tokens
    reference = old_embed.weight

    core = FlyVocabCore(
        vocab_size=int(model.config.vocab_size),
        hidden_size=int(model.config.hidden_size),
        config=config,
        adjacency=adjacency,
        device=reference.device,
        dtype=reference.dtype,
    )
    if factorization is not None:
        initialize_fly_vocab_from_factorization(core, factorization)

    model.fly_vocab_core = core
    model.model.embed_tokens = FlyEmbedding(core)
    model.lm_head = FlyLMHead(core)

    # Prevent Transformers from trying to retie a non-existent full vocab matrix.
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    try:
        model._tied_weights_keys = {}
    except Exception:
        pass
    return model


def replace_qwen_with_flycore(
    model: nn.Module,
    ffn_config: FlyFFNV3Config,
    vocab_config: FlyVocabConfig,
    adjacency: Tensor,
    *,
    factorization: dict[str, Tensor | float] | None = None,
) -> nn.Module:
    replace_all_ffns_with_fly_v3(model, ffn_config, adjacency)
    install_fly_vocab(
        model,
        vocab_config,
        adjacency,
        factorization=factorization,
    )
    assert_qwen35_flycore(model)
    return model


def assert_qwen35_flycore(model: nn.Module) -> None:
    assert_qwen35_flyffn_v3(model)
    if not isinstance(model.model.embed_tokens, FlyEmbedding):
        raise RuntimeError("input embedding is not FlyEmbedding")
    if not isinstance(model.lm_head, FlyLMHead):
        raise RuntimeError("LM head is not FlyLMHead")
    if not isinstance(getattr(model, "fly_vocab_core", None), FlyVocabCore):
        raise RuntimeError("shared FlyVocabCore is missing")
    if model.model.embed_tokens.core is not model.fly_vocab_core:
        raise RuntimeError("FlyEmbedding is not sharing the registered FlyVocabCore")
    if model.lm_head.core is not model.fly_vocab_core:
        raise RuntimeError("FlyLMHead is not sharing the registered FlyVocabCore")


def fly_vocab_parameter_groups(
    model: nn.Module,
    *,
    code_lr: float,
    basis_lr: float,
    fly_lr: float,
    weight_decay: float = 0.01,
):
    core = model.fly_vocab_core
    for p in model.parameters():
        p.requires_grad = False

    core.codebook.weight.requires_grad = True
    core.basis.requires_grad = True
    for p in core.fly_down.parameters():
        p.requires_grad = True
    for p in core.fly_up.parameters():
        p.requires_grad = True
    core.graph_mix_logit.requires_grad = True

    fly_params = [
        *core.fly_down.parameters(),
        *core.fly_up.parameters(),
        core.graph_mix_logit,
    ]
    groups = [
        {"params": [core.codebook.weight], "lr": float(code_lr), "weight_decay": float(weight_decay)},
        {"params": [core.basis], "lr": float(basis_lr), "weight_decay": float(weight_decay)},
        {"params": fly_params, "lr": float(fly_lr), "weight_decay": float(weight_decay)},
    ]
    trainable = [core.codebook.weight, core.basis, *fly_params]
    return groups, trainable


__all__ = [
    "FlyVocabConfig",
    "FlyVocabCore",
    "FlyEmbedding",
    "FlyLMHead",
    "factorize_embedding_weight",
    "install_fly_vocab",
    "replace_qwen_with_flycore",
    "assert_qwen35_flycore",
    "fly_vocab_parameter_groups",
    "FlyFFNV3Config",
]
