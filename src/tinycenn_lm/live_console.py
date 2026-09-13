from __future__ import annotations

import atexit
import os
import sys
import time
from pathlib import Path

_STATUS_INSTALLED = False
_PROCESS_FAILED = False
_PROCESS_STARTED = 0.0
_PROCESS_NAME = ""


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


def _install_training_process_status() -> None:
    global _STATUS_INSTALLED, _PROCESS_STARTED, _PROCESS_NAME
    if _STATUS_INSTALLED:
        return

    script = Path(sys.argv[0]).name
    if not (script.lower().startswith("train_") and script.lower().endswith(".py")):
        return

    _STATUS_INSTALLED = True
    _PROCESS_STARTED = time.monotonic()
    _PROCESS_NAME = script[:-3]
    print(f"[TinyCeNN][PROCESS START] {_PROCESS_NAME}", flush=True)

    original_excepthook = sys.excepthook

    def status_excepthook(exc_type, exc_value, traceback):
        global _PROCESS_FAILED
        _PROCESS_FAILED = True
        elapsed = _format_elapsed(time.monotonic() - _PROCESS_STARTED)
        print(
            f"[TinyCeNN][PROCESS FAILED] {_PROCESS_NAME} after {elapsed}: "
            f"{exc_type.__name__}: {exc_value}",
            flush=True,
        )
        original_excepthook(exc_type, exc_value, traceback)

    sys.excepthook = status_excepthook

    def final_status() -> None:
        elapsed = _format_elapsed(time.monotonic() - _PROCESS_STARTED)
        if _PROCESS_FAILED:
            print(f"[TinyCeNN][PROCESS END] {_PROCESS_NAME} failed after {elapsed}", flush=True)
        else:
            print(f"[TinyCeNN][PROCESS DONE] {_PROCESS_NAME} completed in {elapsed}", flush=True)

    atexit.register(final_status)


def configure_live_console() -> None:
    """Prefer immediate stdout/stderr visibility for notebook-launched trainers.

    Existing Colab notebooks launch ``train_*.py`` with ``subprocess.run``. Setting
    PYTHONUNBUFFERED before child interpreters start and making the current process
    line-buffered keeps progress messages visible as they are produced instead of
    appearing in a large block at the end of a run.

    Trainer processes also emit their own PROCESS START/DONE/FAILED markers. This
    covers notebooks that have not imported ``tinycenn_lm`` in the parent kernel
    before launching the trainer.
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

    _install_training_process_status()
