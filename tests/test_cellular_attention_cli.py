"""Regression tests for Cellular Attention command-line launch behavior."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_cellular_attention_colab import command_for


REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "scripts" / "benchmark_cellular_attention.py"


def test_launcher_uses_module_execution():
    command = command_for(
        BENCHMARK,
        REPO / "result" / "dummy",
        layers="18",
        variants="cellular_dilated3",
        seed=2026,
        config={"context": 64},
    )
    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "scripts.benchmark_cellular_attention",
    ]


def test_benchmark_module_imports_from_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_cellular_attention", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Cellular Attention" in result.stdout
