#!/usr/bin/env python3
"""Google-Drive-aware Colab launcher for SmolLM2 PDelta3-CLVR sequential training."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _arg_value(name: str) -> str | None:
    try:
        idx = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None


output_dir = _arg_value("--output-dir")
if output_dir:
    try:
        resolved = Path(output_dir).expanduser().resolve()
        drive_root = Path("/content/drive").resolve()
        on_drive = resolved == drive_root or drive_root in resolved.parents
    except Exception:
        on_drive = False
else:
    on_drive = False

if on_drive:
    os.environ["TINYCENN_PARENT_BACKUP_ACTIVE"] = "1"
    os.environ["TINYCENN_PERSISTENCE_BACKEND"] = "google-drive"
    print(
        "[TinyCeNN][PERSISTENCE] Google Drive checkpointing active; "
        "redundant mandatory HF backup disabled for this Colab run.",
        flush=True,
    )

target = Path(__file__).with_name("train_smollm2_pdelta3_clvr_sequential.py")
if not target.exists():
    raise FileNotFoundError(target)

runpy.run_path(str(target), run_name="__main__")
