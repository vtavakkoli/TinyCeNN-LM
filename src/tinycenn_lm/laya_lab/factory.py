from __future__ import annotations
from torch import nn
from .core import LayaLabConfig, BaseLayaReplacementAttention
from .integrated import IntegratedMemoryV22Attention
from .fusion import MemoryFusionAttention
from .pdelta import PDelta3GDN2CLVRAttention

def make_replacement(original: nn.Module, cfg: LayaLabConfig) -> BaseLayaReplacementAttention:
    if cfg.architecture == "integrated_memory_v22":
        return IntegratedMemoryV22Attention(original, cfg.feature_dim, cfg.local_kernel)
    if cfg.architecture == "memory_fusion":
        return MemoryFusionAttention(original, cfg.feature_dim, cfg.memory_rank, cfg.local_kernel)
    if cfg.architecture == "pdelta3_gdn2_clvr":
        return PDelta3GDN2CLVRAttention(
            original,
            cfg.feature_dim,
            cfg.pdelta_conv_kernel,
            cfg.pdelta_chunk_size,
            cfg.local_kernel,
        )
    raise ValueError(f"unknown architecture {cfg.architecture!r}")

def replacement_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for m in model.modules() if isinstance(m, BaseLayaReplacementAttention)
        for p in m.trainable_core_parameters()
    )

def replaced_layers(model: nn.Module) -> list[int]:
    return [
        i for i, layer in enumerate(model.encoder.layers)
        if isinstance(layer.attn, BaseLayaReplacementAttention)
    ]

def choose_candidate_layers(model: nn.Module, max_candidates: int) -> list[int]:
    n = len(model.encoder.layers)
    indices = list(range(1, max(1, n - 1)))
    center = (n - 1) / 2.0
    indices.sort(key=lambda i: (
        abs(i - center),
        0 if getattr(model.encoder.layers[i], "attention_type", "") == "sliding_attention" else 1
    ))
    chosen = []
    seen_types: set[str] = set()
    for i in indices:
        typ = str(getattr(model.encoder.layers[i], "attention_type", "unknown"))
        if typ not in seen_types or len(chosen) >= 2:
            chosen.append(i)
            seen_types.add(typ)
        if len(chosen) >= max_candidates:
            break
    return chosen
