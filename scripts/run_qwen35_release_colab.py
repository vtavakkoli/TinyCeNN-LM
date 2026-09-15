#!/usr/bin/env python3
"""Device-safe launcher for Qwen3.5 verification and Hugging Face release."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
release_path = HERE / "qwen35_release.py"
if not release_path.exists():
    raise FileNotFoundError(release_path)

spec = importlib.util.spec_from_file_location("tinycenn_qwen35_release", release_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import Qwen3.5 release helper from {release_path}")
release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = release
spec.loader.exec_module(release)

_original_replace = release.replace_full_attention_layers


def _device_safe_replace(model, config, indices):
    wrappers = _original_replace(model, config, indices)
    fallback_device = next(model.parameters()).device
    for wrapper in wrappers:
        q_weight = getattr(getattr(wrapper, "q_proj", None), "weight", None)
        target_device = q_weight.device if q_weight is not None else fallback_device
        wrapper.to(device=target_device)
        wrong = [
            f"{name}:{param.device}"
            for name, param in wrapper.named_parameters()
            if param.device != target_device
        ]
        if wrong:
            raise RuntimeError(
                f"Qwen3.5 release replacement layer {getattr(wrapper, 'layer_idx', '?')} "
                f"expected device {target_device}, got {wrong[:8]}"
            )
    return wrappers


release.replace_full_attention_layers = _device_safe_replace
print("[TinyCeNN][QWEN35 RELEASE DEVICE] Replacement device handoff active.", flush=True)
release.main()
