from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .colab_live_backup import _retry_required, _upload_live_required, output_dir_from_command
from .hf_persistence import redact_secrets, utc_run_id

_INSTALLED = False


def _is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or Path("/content").exists())


def _training_script_name() -> str | None:
    name = Path(sys.argv[0]).name
    if name.lower().startswith("train_") and name.lower().endswith(".py"):
        return name[:-3]
    return None


def _parent_backup_is_active(script: str, root: Path) -> bool:
    """Detect a recent parent-wrapper run so the child does not duplicate uploads."""
    backup_root = root / ".colab_live_backup"
    if not backup_root.exists():
        return False
    now = time.time()
    for meta in backup_root.glob("*/run_status.json"):
        try:
            if now - meta.stat().st_mtime > 10 * 60:
                continue
            data = json.loads(meta.read_text(encoding="utf-8"))
            if data.get("status") != "running":
                continue
            command = " ".join(str(x) for x in data.get("command", []))
            if script in command:
                return True
        except Exception:
            continue
    return False


class _TeeStream:
    def __init__(self, primary, log_file):
        self._primary = primary
        self._log = log_file

    def write(self, text):
        result = self._primary.write(text)
        self._primary.flush()
        self._log.write(text)
        self._log.flush()
        return result

    def flush(self):
        self._primary.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._primary, name)


def install_direct_training_backup(*, interval_seconds: int = 180) -> bool:
    """Mandatory trainer-side HF backup when no parent Colab wrapper is active.

    This is a safety fallback for notebooks that launch ``train_*.py`` before the
    notebook kernel imports ``tinycenn_lm``. Normal notebooks are backed up by the
    parent wrapper; this function detects that active run and stays out of the way.
    """
    global _INSTALLED
    if _INSTALLED or not _is_colab():
        return _INSTALLED

    script = _training_script_name()
    if script is None:
        return False

    root = Path.cwd().resolve()
    if _parent_backup_is_active(script, root):
        print("[TinyCeNN][BACKUP] parent mandatory backup detected; child fallback not needed.", flush=True)
        _INSTALLED = True
        return True

    try:
        from huggingface_hub import HfApi, get_token
    except Exception as exc:
        raise RuntimeError(
            "Mandatory Hugging Face backup requires huggingface_hub. Install it before training."
        ) from exc

    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Mandatory Hugging Face backup is enabled. Add HF_TOKEN to Colab Secrets, "
            "run the Hugging Face login cell, then start training."
        )

    api = HfApi(token=token)
    identity = _retry_required(api.whoami, label="trainer-side Hugging Face authentication")
    user = identity["name"]
    repo_id = f"{user}/TinyCeNN-LM-Colab-Backups"
    _retry_required(
        lambda: api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True),
        label="trainer-side private backup repository preflight",
    )

    run_id = utc_run_id(f"{script}-direct"[:32])
    output_dir = output_dir_from_command(sys.argv, cwd=root)
    live_root = root / ".colab_live_backup" / run_id
    live_root.mkdir(parents=True, exist_ok=True)
    log_path = live_root / "train.log"
    meta_path = live_root / "run_status.json"
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [redact_secrets(str(x)) for x in sys.argv],
        "output_dir": str(output_dir) if output_dir else None,
        "status": "running",
        "backup_policy": "mandatory-fail-closed-direct-trainer-fallback",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _retry_required(
        lambda: api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(meta_path),
            path_in_repo=f"runs/{run_id}/run_status.json",
            commit_message=f"Start mandatory direct backup {run_id}",
        ),
        label="trainer-side initial metadata upload",
    )

    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_handle)
    sys.stderr = _TeeStream(sys.stderr, log_handle)

    print(
        f"[TinyCeNN][BACKUP REQUIRED][DIRECT] "
        f"https://huggingface.co/{repo_id}/tree/main/runs/{run_id}",
        flush=True,
    )
    print(
        f"[TinyCeNN][BACKUP POLICY][DIRECT] mandatory sync every {interval_seconds}s; "
        "process exits if backup cannot be persisted after retries",
        flush=True,
    )

    stop_event = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                _upload_live_required(
                    api,
                    repo_id=repo_id,
                    run_id=run_id,
                    output_dir=output_dir,
                    log_file=log_path,
                )
                print("[TinyCeNN][BACKUP OK][DIRECT] live state persisted.", flush=True)
            except Exception as exc:
                print(
                    f"[TinyCeNN][BACKUP FATAL][DIRECT] {exc}. "
                    "Stopping training because remote safety cannot be guaranteed.",
                    flush=True,
                )
                os._exit(74)

    worker = threading.Thread(target=heartbeat, name="tinycenn-hf-backup", daemon=True)
    worker.start()

    def finalize() -> None:
        stop_event.set()
        metadata["status"] = "process-exit"
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_seconds"] = time.monotonic() - started
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            _upload_live_required(
                api,
                repo_id=repo_id,
                run_id=run_id,
                output_dir=output_dir,
                log_file=log_path,
            )
            _retry_required(
                lambda: api.upload_file(
                    repo_id=repo_id,
                    repo_type="model",
                    path_or_fileobj=str(meta_path),
                    path_in_repo=f"runs/{run_id}/run_status.json",
                    commit_message=f"Finish mandatory direct backup {run_id}",
                ),
                label="trainer-side final status upload",
            )
            if output_dir is not None and output_dir.exists():
                _retry_required(
                    lambda: api.upload_folder(
                        repo_id=repo_id,
                        repo_type="model",
                        folder_path=str(output_dir),
                        path_in_repo=f"runs/{run_id}/checkpoint",
                        ignore_patterns=[".hf_run_archive/**", ".hf_live_redacted/**"],
                        commit_message=f"Final direct checkpoint backup {run_id}",
                    ),
                    label="trainer-side final checkpoint upload",
                )
            print("[TinyCeNN][BACKUP COMPLETE][DIRECT] final backup committed.", flush=True)
        except Exception as exc:
            print(
                f"[TinyCeNN][BACKUP FATAL][DIRECT] final backup failed: {exc}",
                flush=True,
            )
            try:
                log_handle.flush()
            finally:
                os._exit(75)

    atexit.register(finalize)
    _INSTALLED = True
    return True
