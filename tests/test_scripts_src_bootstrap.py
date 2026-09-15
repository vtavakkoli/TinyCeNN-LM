from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_scripts_package_bootstraps_src_layout_without_site_packages():
    repo = Path(__file__).resolve().parents[1]
    code = r'''
import sys
from pathlib import Path
repo = Path.cwd().resolve()
assert str(repo / "src") not in sys.path
import scripts
assert str(repo / "src") in sys.path, sys.path
print(str(repo / "src"))
'''
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert str(repo / "src") in completed.stdout
