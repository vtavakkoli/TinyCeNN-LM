from __future__ import annotations

import os
import sys


def configure_live_console() -> None:
    """Prefer immediate stdout/stderr visibility for notebook-launched trainers.

    Existing Colab notebooks launch ``train_*.py`` with ``subprocess.run``. Setting
    PYTHONUNBUFFERED before child interpreters start and making the current process
    line-buffered keeps progress messages visible as they are produced instead of
    appearing in a large block at the end of a run.
    """
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except (TypeError, ValueError, OSError):
            # Some notebook stream wrappers do not expose every TextIO option.
            try:
                reconfigure(line_buffering=True)
            except (TypeError, ValueError, OSError):
                pass
