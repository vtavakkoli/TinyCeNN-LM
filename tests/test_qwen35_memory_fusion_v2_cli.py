from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_qwen35_v2_cli_help():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["TINYCENN_PARENT_BACKUP_ACTIVE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_qwen35_memory_fusion_sequential_v2.py"), "--help"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Sequential Qwen3.5 Memory Fusion" in completed.stdout
