#!/usr/bin/env python3
"""Device-safe, explicitly authenticated launcher for Qwen3.5 verification/HF release."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from huggingface_hub import get_token

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

# ---------------------------------------------------------------------------
# 1) Keep dynamically inserted recurrent modules on the model's device.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2) qwen35_release.py deliberately disables implicit HF-token use for public
#    model downloads. For *upload*, however, HfApi must receive the token
#    explicitly. Otherwise a notebook can be logged in while the child process
#    still gets a 401/permission error.
# ---------------------------------------------------------------------------
command = sys.argv[1] if len(sys.argv) > 1 else None
active_token = os.environ.get("HF_TOKEN") or get_token()

if command == "upload":
    if not active_token:
        raise RuntimeError(
            "No Hugging Face token is available to the upload subprocess. "
            "Run the Colab upload cell and log in with a WRITE token."
        )
    os.environ["HF_TOKEN"] = active_token

    _OriginalHfApi = release.HfApi

    def _authenticated_hf_api(*args, **kwargs):
        kwargs.setdefault("token", active_token)
        return _OriginalHfApi(*args, **kwargs)

    release.HfApi = _authenticated_hf_api
    print("[TinyCeNN][HF AUTH] Explicit WRITE-token handoff active.", flush=True)

release.main()
