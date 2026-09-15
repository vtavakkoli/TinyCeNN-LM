#!/usr/bin/env python3
"""Google-Drive-aware Colab launcher for Qwen3.5 PDelta3-CLVR experiments."""
from __future__ import annotations

import os, runpy, sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _arg_value(name: str) -> str | None:
    try:
        idx = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None


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
runpy.run_path(str(target), run_name="__main__")
