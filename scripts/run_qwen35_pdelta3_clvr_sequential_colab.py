#!/usr/bin/env python3
"""Google-Drive-aware Colab launcher for Qwen3.5 PDelta3-CLVR experiments.

Besides the Transformers/Qwen3.5 preflight, this launcher makes dynamically
created PDelta3 replacement modules device-safe. Qwen is moved to CUDA before
replacement, while FrontierPDelta3Layer is constructed on CPU by default; the
replacement must therefore be moved to the original attention device before its
first forward pass.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _qwen35_preflight() -> None:
    try:
        installed = version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Transformers is not installed. Run the Setup cell in the Qwen3.5 Colab first."
        ) from exc

    if _version_tuple(installed) < (5, 2, 0):
        raise RuntimeError(
            f"Qwen3.5 requires a newer Transformers build; found {installed}. "
            "Run the updated Setup cell, which installs transformers==5.17.0."
        )

    try:
        from transformers import Qwen3_5ForCausalLM  # noqa: F401
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"Transformers {installed} is installed but Qwen3.5 APIs are unavailable. "
            "Re-run the updated Setup cell before training."
        ) from exc

    print(f"[TinyCeNN][QWEN35 PREFLIGHT] transformers={installed} APIs=OK", flush=True)


def _arg_value(name: str) -> str | None:
    try:
        idx = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None


_qwen35_preflight()

output_dir = _arg_value("--output-dir")
on_drive = False
if output_dir:
    try:
        resolved = Path(output_dir).expanduser().resolve()
        drive_root = Path("/content/drive").resolve()
        on_drive = resolved == drive_root or drive_root in resolved.parents
    except Exception:
        on_drive = False

if on_drive:
    os.environ["TINYCENN_PARENT_BACKUP_ACTIVE"] = "1"
    os.environ["TINYCENN_PERSISTENCE_BACKEND"] = "google-drive"
    print(
        "[TinyCeNN][PERSISTENCE] Google Drive checkpointing active; "
        "redundant mandatory HF backup disabled.",
        flush=True,
    )

target = Path(__file__).with_name("train_qwen35_pdelta3_clvr_sequential.py")
if not target.exists():
    raise FileNotFoundError(target)

# Import the trainer as a module rather than executing a second copy with runpy.
# This lets all trainer paths (fresh replacement, resume, in-progress resume) use
# the same corrected replacement function.
spec = importlib.util.spec_from_file_location("tinycenn_qwen35_trainer", target)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import Qwen3.5 trainer from {target}")
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)

_original_replace = trainer.replace_full_attention_layers


def _device_safe_replace(model, config, indices):
    wrappers = _original_replace(model, config, indices)
    fallback_device = next(model.parameters()).device
    for wrapper in wrappers:
        q_weight = getattr(getattr(wrapper, "q_proj", None), "weight", None)
        target_device = q_weight.device if q_weight is not None else fallback_device

        # Move the newly created recurrent core and local-gate parameters to the
        # same device without changing their intentionally mixed dtypes.
        wrapper.to(device=target_device)

        wrong = [
            f"{name}:{param.device}"
            for name, param in wrapper.named_parameters()
            if param.device != target_device
        ]
        if wrong:
            raise RuntimeError(
                f"Qwen3.5 replacement layer {getattr(wrapper, 'layer_idx', '?')} "
                f"has parameters on the wrong device; expected {target_device}: {wrong[:8]}"
            )
    return wrappers


trainer.replace_full_attention_layers = _device_safe_replace
print(
    "[TinyCeNN][QWEN35 DEVICE] Dynamic PDelta3 replacements follow the Qwen layer device.",
    flush=True,
)

raise SystemExit(trainer.main())
