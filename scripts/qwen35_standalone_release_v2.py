#!/usr/bin/env python3
"""Compatibility-fixed standalone release runner for Qwen3.5 TinyCeNN models.

This delegates the working MemoryFusion/CeNN packaging pipeline to
qwen35_standalone_release.py and replaces only the PDelta3 source loader.
Older PDelta3 checkpoints were saved with ``model.language_model.*`` names,
while current Qwen3_5ForCausalLM uses ``model.*`` names.  The base Qwen model
is loaded first, the PDelta3 layers are reconstructed, and then every compatible
checkpoint tensor is overlaid with an explicit prefix translation.

All custom PDelta3 tensors must be consumed; otherwise release is aborted.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

import qwen35_standalone_release as release
from tinycenn_lm.qwen35_pdelta3_runtime import replace_full_attention_layers

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def _compat_key(key: str) -> str:
    prefix = "model.language_model."
    if key.startswith(prefix):
        return "model." + key[len(prefix):]
    return key


def _load_checkpoint_state(path: Path, expected: set[str]):
    mapped = {}
    unmapped_language = []
    raw_custom = []

    with safe_open(str(path), framework="pt", device="cpu") as f:
        for raw_key in f.keys():
            if ".self_attn.core." in raw_key or ".self_attn.local_gate_" in raw_key:
                raw_custom.append(raw_key)
            key = _compat_key(raw_key)
            if key in expected:
                mapped[key] = f.get_tensor(raw_key)
            elif raw_key.startswith("model.language_model."):
                unmapped_language.append((raw_key, key))

    return mapped, raw_custom, unmapped_language


def load_pdelta3_compat(local: Path, source: str, device: torch.device):
    local = Path(local)
    weight = local / "model.safetensors"
    if not weight.exists():
        raise FileNotFoundError(weight)

    layers, cfg = release.infer_pdelta_config(local, source)
    _, dtype = release.device_dtype()

    print(f"PDelta3 compatibility load: base={BASE_MODEL} layers={layers} dtype={dtype}")
    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        token=False,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    replace_full_attention_layers(model, cfg, layers)

    expected = set(model.state_dict().keys())
    state, raw_custom, unmapped_language = _load_checkpoint_state(weight, expected)

    if unmapped_language:
        preview = "\n".join(f"  {a} -> {b}" for a, b in unmapped_language[:12])
        raise RuntimeError(
            "PDelta3 compatibility remap left language-model tensors unmapped:\n" + preview
        )

    if not raw_custom:
        raise RuntimeError("PDelta3 checkpoint contains no custom recurrent tensors")

    required_custom = {
        key
        for key in expected
        if any(
            key.startswith(f"model.layers.{i}.self_attn.")
            and (".core." in key or ".local_gate_" in key)
            for i in layers
        )
    }
    loaded_custom = required_custom.intersection(state.keys())
    missing_custom = sorted(required_custom - loaded_custom)
    if missing_custom:
        raise RuntimeError(
            "PDelta3 custom tensors were not fully mapped: " + str(missing_custom[:16])
        )

    # Overlay checkpoint tensors on the frozen official base.  This intentionally
    # preserves any old-wrapper-omitted frozen base fields (e.g. output-head aliases)
    # while requiring every trained/custom replacement tensor to come from checkpoint.
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError("Unexpected mapped PDelta3 tensors: " + str(unexpected[:16]))

    markers = ("A_log", "decay_w", "route_proj", "local_gate_w", "local_gate_b")
    for marker in markers:
        if not any(marker in key for key in loaded_custom):
            raise RuntimeError(f"Required PDelta3 marker was not loaded: {marker}")

    print(
        f"PDelta3 checkpoint overlay: mapped={len(state)} custom={len(loaded_custom)} "
        f"base_fallback={len(incompatible.missing_keys)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True, use_fast=True)
    return model.to(device).eval(), tokenizer, "PDelta3 old-wrapper compatibility reconstruction"


# Patch only PDelta3; MemoryFusion and CeNN Integrated keep the already validated path.
release.load_pdelta3_strict = load_pdelta3_compat


if __name__ == "__main__":
    raise SystemExit(release.main())
