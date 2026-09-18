from __future__ import annotations

from torch import nn

from .smollm2_flyffn_v2 import (
    FlyFFNV2Config,
    ProgressiveFlySwiGLU,
    SharedFlyGraph,
    FlyWireV2Router,
    anchor_layer_indices,
    flyffn_v2_modules,
    flyffn_v2_parameter_groups,
    flyffn_v2_router_regularizer,
    flyffn_v2_stats,
    replace_ffns_with_fly_v2,
    routing_schedule,
    set_route_state,
)


def assert_qwen35_flyffn_v2(model: nn.Module, anchor_every: int = 4) -> None:
    """Verify that only Qwen3.5 FFNs changed and token mixers stayed untouched."""
    anchors = set(anchor_layer_indices(model, anchor_every))
    mods = flyffn_v2_modules(model)
    expected = len(model.model.layers) - len(anchors)
    if len(mods) != expected:
        raise RuntimeError(f"expected {expected} FlyFFN-v2 layers, found {len(mods)}")

    for idx, layer in enumerate(model.model.layers):
        is_fly = isinstance(layer.mlp, ProgressiveFlySwiGLU)
        if idx in anchors and is_fly:
            raise RuntimeError(f"anchor FFN layer {idx} was replaced")
        if idx not in anchors and not is_fly:
            raise RuntimeError(f"non-anchor FFN layer {idx} is not FlyFFN-v2")

        for attr in ("self_attn", "linear_attn"):
            mixer = getattr(layer, attr, None)
            if mixer is None:
                continue
            name = type(mixer).__name__
            if "Fly" in name or "AMCeNN" in name:
                raise RuntimeError(f"token mixer modified unexpectedly at layer {idx}: {name}")


__all__ = [
    "FlyFFNV2Config",
    "ProgressiveFlySwiGLU",
    "SharedFlyGraph",
    "FlyWireV2Router",
    "anchor_layer_indices",
    "flyffn_v2_modules",
    "flyffn_v2_parameter_groups",
    "flyffn_v2_router_regularizer",
    "flyffn_v2_stats",
    "replace_ffns_with_fly_v2",
    "routing_schedule",
    "set_route_state",
    "assert_qwen35_flyffn_v2",
]
