from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class FlyFFNV2Config:
    """Progressive SmolLM2 FFN conversion with FlyWire-routed sparse SwiGLU shards."""

    fly_nodes: int = 256
    router_rank: int = 64
    num_shards: int = 8
    graph_steps: int = 1
    graph_mix_init: float = 0.50
    router_noise_std: float = 5e-3
    anchor_every: int = 4

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
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 < self.graph_mix_init < 1.0:
            raise ValueError("graph_mix_init must be in (0, 1)")
        if self.anchor_every < 2:
            raise ValueError("anchor_every must be >= 2")

    def to_dict(self) -> dict:
        return asdict(self)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class SharedFlyGraph(nn.Module):
    """One FlyWire adjacency shared by all FlyFFN layers."""

    def __init__(self, adjacency: Tensor) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        self.register_buffer("adjacency", adjacency, persistent=True)


class FlyWireV2Router(nn.Module):
    """Low-rank hidden -> FlyWire -> shard router."""

    def __init__(
        self,
        hidden_size: int,
        shared_graph: SharedFlyGraph,
        router_rank: int,
        num_shards: int,
        graph_steps: int,
        graph_mix_init: float,
        router_noise_std: float,
    ) -> None:
        super().__init__()
        self.fly_nodes = int(shared_graph.adjacency.shape[0])
        self.graph_steps = int(graph_steps)
        self.num_shards = int(num_shards)
        object.__setattr__(self, "_shared_graph_ref", shared_graph)

        self.down = nn.Linear(hidden_size, router_rank, bias=False)
        self.to_fly = nn.Linear(router_rank, self.fly_nodes, bias=False)
        self.out = nn.Linear(self.fly_nodes, num_shards, bias=False)
        self.graph_mix_logit = nn.Parameter(
            torch.tensor(_logit(graph_mix_init), dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.to_fly.weight, a=math.sqrt(5))
        nn.init.normal_(self.out.weight, mean=0.0, std=float(router_noise_std))

    @property
    def adjacency(self) -> Tensor:
        return self._shared_graph_ref.adjacency

    def forward(self, x: Tensor) -> Tensor:
        z = F.silu(F.linear(x.float(), self.down.weight.float()))
        z = torch.tanh(F.linear(z, self.to_fly.weight.float()))
        a = self.adjacency.float()
        mix = torch.sigmoid(self.graph_mix_logit.float())
        for _ in range(self.graph_steps):
            graph_z = torch.matmul(z, a.t())
            z = torch.tanh((1.0 - mix) * z + mix * graph_z)
        return F.linear(z, self.out.weight.float())


class ProgressiveFlySwiGLU(nn.Module):
    """Exact dense FFN plus progressively mixed FlyWire sparse approximation."""

    def __init__(
        self,
        original_mlp: nn.Module,
        model_config,
        config: FlyFFNV2Config,
        shared_graph: SharedFlyGraph,
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

        self.gate_weight = nn.Parameter(torch.empty(e, s, h))
        self.up_weight = nn.Parameter(torch.empty(e, s, h))
        self.down_weight = nn.Parameter(torch.empty(e, h, s))
        with torch.no_grad():
            for i in range(e):
                lo, hi = i * s, (i + 1) * s
                self.gate_weight[i].copy_(original_mlp.gate_proj.weight[lo:hi])
                self.up_weight[i].copy_(original_mlp.up_proj.weight[lo:hi])
                self.down_weight[i].copy_(original_mlp.down_proj.weight[:, lo:hi])

        self.router = FlyWireV2Router(
            hidden_size=h,
            shared_graph=shared_graph,
            router_rank=int(config.router_rank),
            num_shards=e,
            graph_steps=int(config.graph_steps),
            graph_mix_init=float(config.graph_mix_init),
            router_noise_std=float(config.router_noise_std),
        )
        self.output_scale = nn.Parameter(torch.ones(()))
        # Per-shard calibration is applied only to the sparse branch.  Keeping it
        # out of the dense path preserves exact teacher equivalence at route_mix=0.
        self.shard_scale = nn.Parameter(torch.ones(e))
        # Start close to uniform top-k weighting.  This is substantially more
        # stable than immediately renormalizing potentially noisy router scores,
        # while still allowing training to learn confidence-sensitive weighting.
        self.route_weight_mix_logit = nn.Parameter(
            torch.tensor(_logit(0.10), dtype=torch.float32)
        )
        self.register_buffer("active_k_state", torch.tensor(e, dtype=torch.int64), persistent=True)
        self.register_buffer("route_mix_state", torch.tensor(0.0, dtype=torch.float32), persistent=True)
        self.last_router_stats: dict[str, Tensor] = {}

    @property
    def active_k(self) -> int:
        return int(self.active_k_state.item())

    @property
    def route_mix(self) -> float:
        return float(self.route_mix_state.item())

    def set_route_state(self, active_k: int, route_mix: float) -> None:
        k = int(active_k)
        mix = float(route_mix)
        if not 1 <= k <= self.num_shards:
            raise ValueError(f"active_k must be in [1,{self.num_shards}]")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("route_mix must be in [0,1]")
        self.active_k_state.fill_(k)
        self.route_mix_state.fill_(mix)

    def _dense_output(self, x: Tensor) -> Tensor:
        """Dense SwiGLU using the original matrix layout.

        A single dense GEMM sequence matches the source MLP numerics much more
        closely than summing independently accumulated shard outputs.
        """
        gate_w = self.gate_weight.reshape(self.inner_size, self.hidden_size)
        up_w = self.up_weight.reshape(self.inner_size, self.hidden_size)
        down_w = self.down_weight.permute(1, 0, 2).reshape(self.hidden_size, self.inner_size)
        hidden = F.silu(F.linear(x, gate_w)) * F.linear(x, up_w)
        return F.linear(hidden, down_w)

    def _selected_sparse_output(
        self,
        x: Tensor,
        top_idx: Tensor,
        top_weight: Tensor,
    ) -> Tensor:
        """Execute only selected shards.

        This avoids the previous quality-prototype behavior of evaluating every
        shard and discarding the unselected outputs.  The loop is over the small
        number of shards, while matrix multiplies run only on assigned tokens.
        """
        flat_x = x.reshape(-1, self.hidden_size)
        flat_idx = top_idx.reshape(-1, top_idx.shape[-1])
        flat_weight = top_weight.reshape(-1, top_weight.shape[-1])
        out = torch.zeros(
            flat_x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype
        )

        for shard in range(self.num_shards):
            token_pos, slot_pos = torch.where(flat_idx == shard)
            if token_pos.numel() == 0:
                continue
            xs = flat_x.index_select(0, token_pos)
            gate = F.linear(xs, self.gate_weight[shard])
            up = F.linear(xs, self.up_weight[shard])
            hidden = F.silu(gate) * up
            ys = F.linear(hidden, self.down_weight[shard])
            scale = self.shard_scale[shard].to(dtype=ys.dtype)
            weight = flat_weight[token_pos, slot_pos].to(dtype=ys.dtype).unsqueeze(-1)
            out.index_add_(0, token_pos, ys * (scale * weight))

        return out.reshape(*x.shape[:-1], self.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        mix = self.route_mix_state.to(device=x.device, dtype=torch.float32)
        k = self.active_k

        # Critical quality invariant: with routing disabled, this follows the
        # exact dense SwiGLU computation instead of a sum of shard GEMMs.
        if self.route_mix == 0.0:
            dense_full = self._dense_output(x)
            self.last_router_stats = {
                "load_balance": torch.tensor(1.0, device=x.device),
                "z_loss": torch.tensor(0.0, device=x.device),
                "entropy": torch.tensor(math.log(self.num_shards), device=x.device),
                "graph_mix": torch.sigmoid(self.router.graph_mix_logit),
                "output_scale": self.output_scale,
                "route_weight_mix": torch.sigmoid(self.route_weight_mix_logit),
                "route_mix": mix,
                "active_k": torch.tensor(float(k), device=x.device),
            }
            return dense_full

        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_values, top_idx = torch.topk(probs, k=k, dim=-1)

        # Pure probability renormalization tends to over-amplify early router
        # mistakes. Blend it with the unbiased uniform top-k estimator.
        confidence_weight = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        uniform_weight = torch.full_like(confidence_weight, 1.0 / float(k))
        confidence_mix = torch.sigmoid(self.route_weight_mix_logit.float())
        top_weight = uniform_weight + confidence_mix * (confidence_weight - uniform_weight)
        # Sum-of-shards estimator: E/k scaling is implicit via E * weights.
        sparse = self._selected_sparse_output(x, top_idx, top_weight) * float(self.num_shards)
        sparse = self.output_scale.to(dtype=sparse.dtype) * sparse

        if self.route_mix >= 0.999999:
            out = sparse
        else:
            dense_full = self._dense_output(x)
            out = dense_full + mix.to(dtype=dense_full.dtype) * (sparse - dense_full)

        assignment = F.one_hot(top_idx, num_classes=self.num_shards).float().sum(dim=-2)
        assignment = assignment / float(k)
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
            "route_weight_mix": confidence_mix,
            "route_mix": mix,
            "active_k": torch.tensor(float(k), device=x.device),
        }
        return out


def anchor_layer_indices(model: nn.Module, anchor_every: int) -> list[int]:
    return [i for i in range(len(model.model.layers)) if (i + 1) % int(anchor_every) == 0]


def flyffn_v2_modules(model: nn.Module) -> list[ProgressiveFlySwiGLU]:
    return [m for m in model.modules() if isinstance(m, ProgressiveFlySwiGLU)]


def replace_ffns_with_fly_v2(model: nn.Module, config: FlyFFNV2Config, adjacency: Tensor) -> nn.Module:
    config.validate(model.config)
    anchors = set(anchor_layer_indices(model, config.anchor_every))
    shared_graph = SharedFlyGraph(adjacency)
    reference = next(model.parameters())
    shared_graph.to(device=reference.device, dtype=torch.float32)
    model.flyffn_shared_graph = shared_graph

    for idx, layer in enumerate(model.model.layers):
        if idx in anchors:
            continue
        old = layer.mlp
        if isinstance(old, ProgressiveFlySwiGLU):
            continue
        new = ProgressiveFlySwiGLU(old, model.config, config, shared_graph, idx)
        new.to(device=old.gate_proj.weight.device, dtype=old.gate_proj.weight.dtype)
        new.router.down.float()
        new.router.to_fly.float()
        new.router.out.float()
        new.router.graph_mix_logit.data = new.router.graph_mix_logit.data.float()
        new.output_scale.data = new.output_scale.data.float()
        layer.mlp = new
    return model


def set_route_state(model: nn.Module, layer_indices=None, active_k: int | None = None, route_mix: float | None = None) -> None:
    selected = None if layer_indices is None else set(int(i) for i in layer_indices)
    for m in flyffn_v2_modules(model):
        if selected is not None and m.layer_idx not in selected:
            continue
        k = m.active_k if active_k is None else int(active_k)
        mix = m.route_mix if route_mix is None else float(route_mix)
        m.set_route_state(k, mix)


def assert_flyffn_v2_replacement(model: nn.Module, anchor_every: int) -> None:
    anchors = set(anchor_layer_indices(model, anchor_every))
    mods = flyffn_v2_modules(model)
    expected = len(model.model.layers) - len(anchors)
    if len(mods) != expected:
        raise RuntimeError(f"expected {expected} FlyFFN-v2 layers, found {len(mods)}")
    for idx, layer in enumerate(model.model.layers):
        if idx in anchors and isinstance(layer.mlp, ProgressiveFlySwiGLU):
            raise RuntimeError(f"anchor FFN layer {idx} was replaced")
        if idx not in anchors and not isinstance(layer.mlp, ProgressiveFlySwiGLU):
            raise RuntimeError(f"non-anchor layer {idx} is not FlyFFN-v2")
        if "Fly" in type(layer.self_attn).__name__ or "AMCeNN" in type(layer.self_attn).__name__:
            raise RuntimeError(f"attention modified unexpectedly at layer {idx}: {type(layer.self_attn).__name__}")


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def flyffn_v2_parameter_groups(model: nn.Module, layer_indices, router_lr: float, shard_lr: float, weight_decay: float = 0.01):
    selected = set(int(i) for i in layer_indices)
    freeze_all(model)
    router_params, shard_params = [], []
    for m in flyffn_v2_modules(model):
        if m.layer_idx not in selected:
            continue
        for p in m.router.parameters():
            p.requires_grad = True
            router_params.append(p)
        m.output_scale.requires_grad = True
        m.shard_scale.requires_grad = True
        m.route_weight_mix_logit.requires_grad = True
        router_params.extend([m.output_scale, m.shard_scale, m.route_weight_mix_logit])
        for p in (m.gate_weight, m.up_weight, m.down_weight):
            p.requires_grad = True
            shard_params.append(p)
    return (
        [
            {"params": router_params, "lr": float(router_lr), "weight_decay": float(weight_decay)},
            {"params": shard_params, "lr": float(shard_lr), "weight_decay": float(weight_decay)},
        ],
        [*router_params, *shard_params],
    )


def flyffn_v2_router_regularizer(model: nn.Module, layer_indices=None) -> Tensor:
    selected = None if layer_indices is None else set(int(i) for i in layer_indices)
    terms = []
    for m in flyffn_v2_modules(model):
        if selected is not None and m.layer_idx not in selected:
            continue
        if not m.last_router_stats or m.route_mix <= 0:
            continue
        s = m.last_router_stats
        terms.append(0.01 * s["load_balance"] + 1e-4 * s["z_loss"])
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack([t.float() for t in terms]).mean()


def flyffn_v2_stats(model: nn.Module) -> dict:
    mods = flyffn_v2_modules(model)
    stats = [m.last_router_stats for m in mods if m.last_router_stats]
    out = {
        "flyffn_v2_layers": len(mods),
        "mean_active_k": float(sum(m.active_k for m in mods) / max(len(mods), 1)),
        "mean_route_mix": float(sum(m.route_mix for m in mods) / max(len(mods), 1)),
        "fully_sparse_layers": len([m for m in mods if m.route_mix >= 0.999 and m.active_k < m.num_shards]),
    }
    if stats:
        for key in ("load_balance", "entropy", "graph_mix", "output_scale", "route_weight_mix"):
            vals = [s[key].detach().float() for s in stats if key in s]
            if vals:
                out[f"mean_{key}"] = float(torch.stack(vals).mean().cpu())
    return out


def routing_schedule(model: nn.Module) -> dict[int, dict[str, float | int]]:
    return {
        m.layer_idx: {"active_k": m.active_k, "route_mix": m.route_mix}
        for m in flyffn_v2_modules(model)
    }
