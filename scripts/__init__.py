"""Training and evaluation entry points for TinyCeNN-LM.

The repository uses a ``src/`` package layout.  Notebook runtimes sometimes add
only the repository root to ``sys.path`` after an editable install, while the
already-running Python process has not re-read the editable ``.pth`` file yet.
In that situation ``from scripts import ...`` works but the imported script can
fail immediately on ``import tinycenn_lm``.

Make the local source tree discoverable whenever the ``scripts`` package is
imported.  This is intentionally tiny and idempotent, and fixes Colab imports
without requiring a kernel restart.
"""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SRC_TEXT = str(_SRC_ROOT)
if _SRC_ROOT.is_dir() and _SRC_TEXT not in sys.path:
    sys.path.insert(0, _SRC_TEXT)
