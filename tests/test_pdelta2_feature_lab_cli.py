import subprocess
import sys
from pathlib import Path


def test_feature_lab_cli_starts_from_absolute_script_path():
    """Match the Colab launch mode: absolute script path with repo as cwd."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_pdelta2_feature_lab.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "--profile" in result.stdout
    assert "--output-dir" in result.stdout
