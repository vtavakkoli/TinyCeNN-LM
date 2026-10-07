"""V2.8 fast full-attention-first Laya conversion helpers."""
from __future__ import annotations

from torch import nn

from .laya_lab.pdelta import PDelta3GDN2CLVRAttention


def make_v28_full_replacement(
    original: nn.Module,
    *,
    feature_dim: int = 96,
    local_window: int = 32,
    chunk_size: int = 32,
) -> PDelta3GDN2CLVRAttention:
    """PDelta3-GDN2 + exact bidirectional LocalW for a full-attention layer."""
    replacement = PDelta3GDN2CLVRAttention(
        original,
        feature_dim=int(feature_dim),
        conv_kernel=4,
        chunk_size=int(chunk_size),
        local_kernel=5,
        local_window=int(local_window),
        local_gate_init=0.72,
    )
    # Preserve pretrained QKV, but allow conservative output-projection
    # calibration during local and joint recovery.
    for p in replacement.Wqkv.parameters():
        p.requires_grad = False
    for p in replacement.Wo.parameters():
        p.requires_grad = True
    return replacement


def split_v28_parameters(
    module: PDelta3GDN2CLVRAttention,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (core, Wo) trainable parameter groups without duplicates."""
    core, output, seen = [], [], set()
    for name, p in module.named_parameters():
        # Discovery must not depend on the current frozen/trainable state.
        # Accepted layers are frozen between local transfer and joint recovery,
        # then explicitly re-enabled by the notebook.
        if name.startswith(("Wqkv.", "out_drop.")):
            continue
        target = output if name.startswith("Wo.") else core
        if id(p) not in seen:
            seen.add(id(p))
            target.append(p)
    return core, output
