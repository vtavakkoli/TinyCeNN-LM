from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .smollm2_flyffn_v2 import (
    FlyFFNV2Config,
    ProgressiveFlySwiGLU,
    SharedFlyGraph,
    flyffn_v2_modules,
    flyffn_v2_parameter_groups,
    flyffn_v2_router_regularizer,
    flyffn_v2_stats,
    routing_schedule,
    set_route_state,
)


@dataclass(frozen=True)
class FlyFFNV3Config:
    """All-FFN FlyFFN-v3 configuration for Qwen3.5."""

    fly_nodes: int = 256
    router_rank: int = 96
    num_shards: int = 8
    graph_steps: int = 1
    graph_mix_init: float = 0.50
    router_noise_std: float = 5e-3

    def to_v2(self, num_layers: int) -> FlyFFNV2Config:
        # anchor_every > num_layers means the v2 replacement primitive
        # produces zero dense anchors while retaining the tested module.
        return FlyFFNV2Config(
            fly_nodes=self.fly_nodes,
            router_rank=self.router_rank,
            num_shards=self.num_shards,
            graph_steps=self.graph_steps,
            graph_mix_init=self.graph_mix_init,
            router_noise_std=self.router_noise_std,
            anchor_every=max(int(num_layers) + 1, 2),
        )

    def to_dict(self) -> dict:
        return {
            "fly_nodes": int(self.fly_nodes),
            "router_rank": int(self.router_rank),
            "num_shards": int(self.num_shards),
            "graph_steps": int(self.graph_steps),
            "graph_mix_init": float(self.graph_mix_init),
            "router_noise_std": float(self.router_noise_std),
            "replace_all_ffns": True,
            "dense_anchors": 0,
        }


def replace_all_ffns_with_fly_v3(
    model: nn.Module,
    config: FlyFFNV3Config,
    adjacency: Tensor,
) -> nn.Module:
    from .smollm2_flyffn_v2 import replace_ffns_with_fly_v2

    internal = config.to_v2(len(model.model.layers))
    replace_ffns_with_fly_v2(model, internal, adjacency)
    return model


def assert_qwen35_flyffn_v3(model: nn.Module) -> None:
    layers = model.model.layers
    mods = flyffn_v2_modules(model)

    if len(mods) != len(layers):
        raise RuntimeError(
            f"FlyFFN-v3 requires every FFN to be replaced: "
            f"expected {len(layers)}, found {len(mods)}"
        )

    for idx, layer in enumerate(layers):
        if not isinstance(layer.mlp, ProgressiveFlySwiGLU):
            raise RuntimeError(f"layer {idx} FFN is not FlyFFN-v3")

        for attr in ("self_attn", "linear_attn"):
            mixer = getattr(layer, attr, None)
            if mixer is None:
                continue
            name = type(mixer).__name__
            if "Fly" in name or "AMCeNN" in name:
                raise RuntimeError(
                    f"Qwen token mixer modified unexpectedly at layer {idx}: {name}"
                )


def all_fly_layer_indices(model: nn.Module) -> list[int]:
    return list(range(len(model.model.layers)))


__all__ = [
    "FlyFFNV3Config",
    "ProgressiveFlySwiGLU",
    "SharedFlyGraph",
    "replace_all_ffns_with_fly_v3",
    "assert_qwen35_flyffn_v3",
    "all_fly_layer_indices",
    "flyffn_v2_modules",
    "flyffn_v2_parameter_groups",
    "flyffn_v2_router_regularizer",
    "flyffn_v2_stats",
    "routing_schedule",
    "set_route_state",
]
