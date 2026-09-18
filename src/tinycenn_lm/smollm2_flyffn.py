from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class FlyFFNConfig:
    """SmolLM2 FFN replacement using FlyWire-routed disjoint SwiGLU shards.

    Attention is intentionally untouched. Each original dense Llama/SmolLM2 MLP is
    split exactly across ``num_shards`` disjoint channel groups. In dense mode the
    shards sum to the original pretrained FFN exactly. Progressive calibration then
    switches groups of layers to a FlyWire-routed Top-k sparse estimate.
    """

    fly_nodes: int = 256
    router_rank: int = 64
    num_shards: int = 8
    top_k: int = 2
    graph_steps: int = 1
    graph_mix_init: float = 0.50
    router_noise_std: float = 5e-3

    def validate(self, model_config) -> None:
        h = int(model_config.hidden_size)
        inner = int(model_config.intermediate_size)
        if self.fly_nodes < 32:
            raise ValueError("fly_nodes must be >= 32")
        if not 8 <= self.router_rank <= h:
            raise ValueError("router_rank must be in [8, hidden_size]")
        if self.num_shards < 2:
            raise ValueError("num_shards must be >= 2")
        if inner % self.num_shards:
            raise ValueError(
                f"intermediate_size={inner} must be divisible by num_shards={self.num_shards}"
            )
        if not 1 <= self.top_k <= self.num_shards:
            raise ValueError("top_k must be in [1, num_shards]")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 < self.graph_mix_init < 1.0:
            raise ValueError("graph_mix_init must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class FlyWireShardRouter(nn.Module):
    """Low-rank hidden->FlyWire->shard router.

    The graph cannot be bypassed: after the low-rank projection, every routing
    decision passes through a convex content/graph mixture with a learnable graph
    fraction. The output projection starts with a small non-zero initialization so
    biological vs rewired topology can affect routing from the first update.
    """

    def __init__(
        self,
        hidden_size: int,
        adjacency: Tensor,
        router_rank: int,
        num_shards: int,
        graph_steps: int,
        graph_mix_init: float,
        router_noise_std: float,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        self.fly_nodes = int(adjacency.shape[0])
        self.graph_steps = int(graph_steps)
        self.num_shards = int(num_shards)
        self.register_buffer("adjacency", adjacency, persistent=True)

        self.down = nn.Linear(hidden_size, router_rank, bias=False)
        self.to_fly = nn.Linear(router_rank, self.fly_nodes, bias=False)
        self.out = nn.Linear(self.fly_nodes, num_shards, bias=False)
        self.graph_mix_logit = nn.Parameter(
            torch.tensor(_logit(graph_mix_init), dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.to_fly.weight, a=math.sqrt(5))
        nn.init.normal_(self.out.weight, mean=0.0, std=float(router_noise_std))

    def forward(self, x: Tensor) -> Tensor:
        z = F.silu(F.linear(x.float(), self.down.weight.float()))
        z = torch.tanh(F.linear(z, self.to_fly.weight.float()))
        a = self.adjacency.float()
        mix = torch.sigmoid(self.graph_mix_logit.float())
        for _ in range(self.graph_steps):
            graph_z = torch.matmul(z, a.t())
            z = torch.tanh((1.0 - mix) * z + mix * graph_z)
        return F.linear(z, self.out.weight.float())


class FlyShardedSwiGLU(nn.Module):
    """Function-preserving dense FFN that can be progressively switched to sparse Fly routing."""

    def __init__(
        self,
        original_mlp: nn.Module,
        model_config,
        config: FlyFFNConfig,
        adjacency: Tensor,
        layer_idx: int,
    ) -> None:
        super().__init__()
        config.validate(model_config)
        h = int(model_config.hidden_size)
        inner = int(model_config.intermediate_size)
        e = int(config.num_shards)
        s = inner // e

        self.layer_idx = int(layer_idx)
        self.hidden_size = h
        self.inner_size = inner
        self.num_shards = e
        self.shard_inner = s
        self.top_k = int(config.top_k)
        self.sparse_active = False

        # Across all shards, these are exactly the original gate/up/down parameters.
        self.gate_weight = nn.Parameter(torch.empty(e, s, h))
        self.up_weight = nn.Parameter(torch.empty(e, s, h))
        self.down_weight = nn.Parameter(torch.empty(e, h, s))
        with torch.no_grad():
            for i in range(e):
                lo, hi = i * s, (i + 1) * s
                self.gate_weight[i].copy_(original_mlp.gate_proj.weight[lo:hi])
                self.up_weight[i].copy_(original_mlp.up_proj.weight[lo:hi])
                self.down_weight[i].copy_(original_mlp.down_proj.weight[:, lo:hi])

        self.router = FlyWireShardRouter(
            hidden_size=h,
            adjacency=adjacency,
            router_rank=int(config.router_rank),
            num_shards=e,
            graph_steps=int(config.graph_steps),
            graph_mix_init=float(config.graph_mix_init),
            router_noise_std=float(config.router_noise_std),
        )
        self.output_scale = nn.Parameter(torch.ones(()))
        self.last_router_stats: dict[str, Tensor] = {}

    def set_sparse(self, enabled: bool) -> None:
        self.sparse_active = bool(enabled)

    def _all_shard_outputs(self, x: Tensor) -> Tensor:
        gate = torch.einsum("bth,esh->btes", x, self.gate_weight)
        up = torch.einsum("bth,esh->btes", x, self.up_weight)
        hidden = F.silu(gate) * up
        return torch.einsum("btes,ehs->bteh", hidden, self.down_weight)

    def forward(self, x: Tensor) -> Tensor:
        shard_out = self._all_shard_outputs(x)
        dense_full = shard_out.sum(dim=2)

        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_values, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_weight = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gather_idx = top_idx.unsqueeze(-1).expand(*top_idx.shape, self.hidden_size)
        selected = torch.gather(shard_out, dim=2, index=gather_idx)

        # Unbiased uniform Top-k estimator: if top weights are equal, this is
        # E/k * sum(selected). Using normalized weights therefore requires E,
        # not E/k, as the multiplier.
        sparse = (
            selected * top_weight.to(dtype=selected.dtype).unsqueeze(-1)
        ).sum(dim=2) * float(self.num_shards)
        sparse = self.output_scale.to(dtype=sparse.dtype) * sparse

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
            "graph_mix": torch.sigmoid(self.router.graph_mix_logit),
            "output_scale": self.output_scale,
        }
        return sparse if self.sparse_active else dense_full


def flyffn_modules(model: nn.Module) -> list[FlyShardedSwiGLU]:
    return [m for m in model.modules() if isinstance(m, FlyShardedSwiGLU)]


def replace_ffns_with_fly(
    model: nn.Module,
    config: FlyFFNConfig,
    adjacency: Tensor,
    layer_indices: list[int] | tuple[int, ...] | None = None,
) -> nn.Module:
    config.validate(model.config)
    if layer_indices is None:
        layer_indices = list(range(len(model.model.layers)))
    selected = set(int(i) for i in layer_indices)
    for idx, layer in enumerate(model.model.layers):
        if idx not in selected or isinstance(layer.mlp, FlyShardedSwiGLU):
            continue
        old = layer.mlp
        new = FlyShardedSwiGLU(old, model.config, config, adjacency, idx)
        new.to(device=old.gate_proj.weight.device, dtype=old.gate_proj.weight.dtype)
        # Keep routing network numerically stable in FP32.
        new.router.down.float()
        new.router.to_fly.float()
        new.router.out.float()
        new.router.adjacency.data = new.router.adjacency.data.float()
        new.router.graph_mix_logit.data = new.router.graph_mix_logit.data.float()
        new.output_scale.data = new.output_scale.data.float()
        layer.mlp = new
    return model


def set_sparse_layers(model: nn.Module, layer_indices=None, enabled: bool = True) -> None:
    selected = None if layer_indices is None else set(int(i) for i in layer_indices)
    for m in flyffn_modules(model):
        if selected is None or m.layer_idx in selected:
            m.set_sparse(enabled)


def sparse_layer_indices(model: nn.Module) -> list[int]:
    return [m.layer_idx for m in flyffn_modules(model) if m.sparse_active]


def assert_flyffn_replacement(model: nn.Module, expected: int | None = None) -> None:
    mods = flyffn_modules(model)
    if expected is None:
        expected = len(model.model.layers)
    if len(mods) != expected:
        raise RuntimeError(f"expected {expected} FlyFFN layers, found {len(mods)}")
    # This experiment must not replace attention.
    bad = [
        (i, type(layer.self_attn).__name__)
        for i, layer in enumerate(model.model.layers)
        if "Fly" in type(layer.self_attn).__name__ or "AMCeNN" in type(layer.self_attn).__name__
    ]
    if bad:
        raise RuntimeError(f"attention was modified unexpectedly: {bad[:4]}")


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def flyffn_parameter_groups(
    model: nn.Module,
    layer_indices,
    router_lr: float,
    shard_lr: float,
    weight_decay: float = 0.01,
):
    selected = set(int(i) for i in layer_indices)
    freeze_all(model)
    router_params, shard_params = [], []
    for m in flyffn_modules(model):
        if m.layer_idx not in selected:
            continue
        for p in m.router.parameters():
            p.requires_grad = True
            router_params.append(p)
        m.output_scale.requires_grad = True
        router_params.append(m.output_scale)
        for p in (m.gate_weight, m.up_weight, m.down_weight):
            p.requires_grad = True
            shard_params.append(p)
    groups = [
        {"params": router_params, "lr": float(router_lr), "weight_decay": float(weight_decay)},
        {"params": shard_params, "lr": float(shard_lr), "weight_decay": float(weight_decay)},
    ]
    return groups, [*router_params, *shard_params]


def flyffn_router_regularizer(model: nn.Module, layer_indices=None) -> Tensor:
    selected = None if layer_indices is None else set(int(i) for i in layer_indices)
    terms = []
    for m in flyffn_modules(model):
        if selected is not None and m.layer_idx not in selected:
            continue
        if not m.last_router_stats:
            continue
        s = m.last_router_stats
        # Keep experts used while discouraging pathological router logits.
        terms.append(0.01 * s["load_balance"] + 1e-4 * s["z_loss"])
    if not terms:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack([t.float() for t in terms]).mean()


def flyffn_stats(model: nn.Module) -> dict:
    mods = flyffn_modules(model)
    stats = [m.last_router_stats for m in mods if m.last_router_stats]
    out = {
        "flyffn_layers": len(mods),
        "sparse_active_layers": len([m for m in mods if m.sparse_active]),
    }
    if stats:
        for key in ("load_balance", "entropy", "graph_mix", "output_scale"):
            out[f"mean_{key}"] = float(torch.stack([s[key].detach().float() for s in stats]).mean().cpu())
    return out
