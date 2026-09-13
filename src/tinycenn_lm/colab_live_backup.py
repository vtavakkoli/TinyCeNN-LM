from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .hf_persistence import redact_secrets, utc_run_id

_SMALL_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md", ".log", ".yaml", ".yml"}


def is_tinycenn_training_command(cmd: Any) -> bool:
    if not isinstance(cmd, (list, tuple)):
        return False
    parts = [str(x) for x in cmd]
    for part in parts:
        name = Path(part).name.lower()
        if name.startswith("train_") and name.endswith(".py"):
            return True
    return False


def training_script_name(cmd: list[str] | tuple[str, ...]) -> str:
    for part in cmd:
        name = Path(str(part)).name
        if name.lower().startswith("train_") and name.lower().endswith(".py"):
            return name[:-3]
    return "training"


def output_dir_from_command(cmd: list[str] | tuple[str, ...], cwd: str | Path | None = None) -> Path | None:
    parts = [str(x) for x in cmd]
    for i, part in enumerate(parts):
        if part == "--output-dir" and i + 1 < len(parts):
            path = Path(parts[i + 1])
            if not path.is_absolute() and cwd is not None:
                path = Path(cwd) / path
            return path
        if part.startswith("--output-dir="):
            path = Path(part.split("=", 1)[1])
            if not path.is_absolute() and cwd is not None:
                path = Path(cwd) / path
            return path
    return None


def _small_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _SMALL_SUFFIXES
        and p.stat().st_size <= 10 * 1024 * 1024
        and ".hf_run_archive" not in p.parts
        and ".hf_live_redacted" not in p.parts
    ]


def _retry_required(
    action: Callable[[], Any],
    *,
    label: str,
    attempts: int = 3,
    delay_seconds: float = 5.0,
) -> Any:
    """Run one mandatory Hugging Face operation with bounded retries."""
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return action()
        except Exception as exc:  # network/API failures must fail closed after retries
            last_error = exc
            print(
                f"[TinyCeNN][BACKUP] {label} failed "
                f"(attempt {attempt}/{attempts}): {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(
        f"Mandatory Hugging Face backup failed during {label} after {attempts} attempts"
    ) from last_error


def _upload_live_required(
    api,
    *,
    repo_id: str,
    run_id: str,
    output_dir: Path | None,
    log_file: Path,
    include_checkpoint: bool = True,
) -> None:
    """Persist the live log and reports, optionally mirroring checkpoint files."""
    if log_file.exists():
        clean_log = log_file.with_name("train_redacted.log")
        clean_log.write_text(
            redact_secrets(log_file.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
        _retry_required(
            lambda: api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(clean_log),
                path_in_repo=f"runs/{run_id}/train.log",
                commit_message=f"Live backup {run_id}",
            ),
            label="live log upload",
        )

    if output_dir is None or not output_dir.exists():
        return

    # Upload small human-readable artifacts separately after redaction.
    for path in _small_files(output_dir):
        rel = path.relative_to(output_dir)
        upload_path = path
        try:
            clean_dir = output_dir / ".hf_live_redacted"
            dst = clean_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(
                redact_secrets(path.read_text(encoding="utf-8", errors="replace")),
                encoding="utf-8",
            )
            upload_path = dst
        except Exception:
            upload_path = path

        _retry_required(
            lambda p=upload_path, r=rel: api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(p),
                path_in_repo=f"runs/{run_id}/artifacts/{r.as_posix()}",
                commit_message=f"Live results {run_id}",
            ),
            label=f"artifact upload: {rel.as_posix()}",
        )

    if not include_checkpoint:
        return

    # Mandatory periodic checkpoint mirror. This is disabled when the output
    # directory already existed before the run, because otherwise stale weights from
    # an earlier execution can trigger a large upload unrelated to the current run.
    checkpoint_files = [
        p for p in output_dir.rglob("*")
        if p.is_file()
        and ".hf_run_archive" not in p.parts
        and ".hf_live_redacted" not in p.parts
    ]
    if checkpoint_files:
        _retry_required(
            lambda: api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(output_dir),
                path_in_repo=f"runs/{run_id}/checkpoint",
                ignore_patterns=[".hf_run_archive/**", ".hf_live_redacted/**"],
                commit_message=f"Live checkpoint backup {run_id}",
            ),
            label="live checkpoint mirror",
        )


def _backup_metadata(cmd, output_dir: Path | None, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [redact_secrets(str(x)) for x in cmd],
        "output_dir": str(output_dir) if output_dir else None,
        "status": "running",
        "backup_policy": "mandatory-fail-closed",
    }


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


def install_colab_training_backup(*, interval_seconds: int = 180) -> bool:
    """Stream TinyCeNN Colab training and require a private Hugging Face backup.

    Every ``subprocess.run([... train_*.py ...])`` call in Colab is converted to a
    line-streaming ``Popen`` execution. Before the child process starts, a valid
    Hugging Face login and a writable private ``TinyCeNN-LM-Colab-Backups`` repo are
    required. Live logs and reports are mirrored periodically, along with checkpoints
    when the output directory is new for the current run. If a mandatory sync fails
    after retries, training is terminated. A successful run must finish by uploading
    its complete checkpoint. A failed trainer still uploads its log/status promptly,
    but does not block on a large or stale checkpoint directory.
    """
    if not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or Path("/content").exists()):
        return False
    if getattr(subprocess.run, "_tinycenn_live_backup", False):
        return True

    original_run = subprocess.run
    original_popen = subprocess.Popen

    def run_with_backup(cmd, *args, **kwargs):
        if not is_tinycenn_training_command(cmd):
            return original_run(cmd, *args, **kwargs)

        unsupported = {"input", "capture_output", "stdout", "stderr", "timeout"} & set(kwargs)
        if unsupported or args:
            return original_run(cmd, *args, **kwargs)

        cwd = kwargs.pop("cwd", None)
        requested_env = kwargs.pop("env", None)
        check = bool(kwargs.pop("check", False))
        if kwargs:
            return original_run(cmd, cwd=cwd, env=requested_env, check=check, **kwargs)

        child_env = os.environ.copy()
        if requested_env is not None:
            child_env.update({str(k): str(v) for k, v in requested_env.items()})
        child_env["PYTHONUNBUFFERED"] = "1"

        script = training_script_name(cmd)
        run_id = utc_run_id(script[:32])
        output_dir = output_dir_from_command(cmd, cwd=cwd)
        output_preexisting = bool(output_dir is not None and output_dir.exists())
        work_root = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        live_root = work_root / ".colab_live_backup" / run_id
        live_root.mkdir(parents=True, exist_ok=True)
        log_file = live_root / "train.log"
        meta_file = live_root / "run_status.json"
        metadata = _backup_metadata(cmd, output_dir, run_id)
        metadata["output_dir_preexisting"] = output_preexisting
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Mandatory preflight. Do not spend GPU time unless the remote safety target
        # is authenticated, private, writable, and has accepted the run metadata.
        try:
            from huggingface_hub import HfApi, get_token
        except Exception as exc:
            raise RuntimeError(
                "Mandatory Hugging Face backup requires huggingface_hub. "
                "Install it before training."
            ) from exc

        token = get_token() or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "Mandatory Hugging Face backup is enabled. Log in first or provide "
                "HF_TOKEN (in Colab: add HF_TOKEN to Secrets and run the login cell)."
            )

        api = HfApi(token=token)
        identity = _retry_required(api.whoami, label="Hugging Face authentication")
        user = identity["name"]
        backup_repo = f"{user}/TinyCeNN-LM-Colab-Backups"
        _retry_required(
            lambda: api.create_repo(
                backup_repo,
                repo_type="model",
                private=True,
                exist_ok=True,
            ),
            label="private backup repository preflight",
        )
        _retry_required(
            lambda: api.upload_file(
                repo_id=backup_repo,
                repo_type="model",
                path_or_fileobj=str(meta_file),
                path_in_repo=f"runs/{run_id}/run_status.json",
                commit_message=f"Start mandatory backup {run_id}",
            ),
            label="initial backup metadata upload",
        )

        command_text = " ".join(shlex.quote(str(x)) for x in cmd)
        print("\n" + "=" * 88, flush=True)
        print(f"[TinyCeNN][START] {script}", flush=True)
        print(f"[TinyCeNN][COMMAND] {command_text}", flush=True)
        if output_dir is not None:
            print(f"[TinyCeNN][OUTPUT] {output_dir}", flush=True)
        if output_preexisting:
            print(
                "[TinyCeNN][BACKUP] output directory already exists; periodic backup "
                "will save logs/reports only and avoid re-uploading stale checkpoint files.",
                flush=True,
            )
        print(f"[TinyCeNN][LOCAL LOG] {log_file}", flush=True)
        print(
            f"[TinyCeNN][BACKUP REQUIRED] "
            f"https://huggingface.co/{backup_repo}/tree/main/runs/{run_id}",
            flush=True,
        )
        print(
            f"[TinyCeNN][BACKUP POLICY] mandatory sync every {interval_seconds}s; "
            "training aborts if backup cannot be persisted after retries",
            flush=True,
        )
        print("[TinyCeNN][LIVE] streaming training output...", flush=True)
        print("-" * 88, flush=True)

        started = time.monotonic()
        proc = original_popen(
            cmd,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_sync = started
        backup_failure: Exception | None = None

        with log_file.open("a", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                now = time.monotonic()
                if now - last_sync >= interval_seconds:
                    try:
                        _upload_live_required(
                            api,
                            repo_id=backup_repo,
                            run_id=run_id,
                            output_dir=output_dir,
                            log_file=log_file,
                            include_checkpoint=not output_preexisting,
                        )
                        print(
                            f"[TinyCeNN][BACKUP OK] live state persisted at "
                            f"{_format_elapsed(now - started)}",
                            flush=True,
                        )
                        last_sync = now
                    except Exception as exc:
                        backup_failure = exc
                        print(
                            f"[TinyCeNN][BACKUP FATAL] {exc}. Terminating training to protect the run.",
                            flush=True,
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        break

        returncode = proc.wait()
        elapsed = time.monotonic() - started

        if backup_failure is not None:
            metadata["status"] = "failed-backup"
            metadata["returncode"] = returncode
            metadata["elapsed_seconds"] = elapsed
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["backup_error"] = redact_secrets(str(backup_failure))
            meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            # Best effort only for the failure marker; the triggering sync already
            # proved that the remote is unavailable.
            try:
                api.upload_file(
                    repo_id=backup_repo,
                    repo_type="model",
                    path_or_fileobj=str(meta_file),
                    path_in_repo=f"runs/{run_id}/run_status.json",
                    commit_message=f"Backup failure {run_id}",
                )
            except Exception:
                pass
            raise RuntimeError(
                "Training was terminated because mandatory Hugging Face backup could not be maintained."
            ) from backup_failure

        metadata["status"] = "completed" if returncode == 0 else "failed-training"
        metadata["returncode"] = returncode
        metadata["elapsed_seconds"] = elapsed
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Always preserve the run's log, small reports and final status. On failure,
        # do not upload the checkpoint directory: it may be a stale fixed-name folder
        # from an earlier run and was the source of long post-crash Colab hangs.
        _upload_live_required(
            api,
            repo_id=backup_repo,
            run_id=run_id,
            output_dir=output_dir,
            log_file=log_file,
            include_checkpoint=False,
        )
        _retry_required(
            lambda: api.upload_file(
                repo_id=backup_repo,
                repo_type="model",
                path_or_fileobj=str(meta_file),
                path_in_repo=f"runs/{run_id}/run_status.json",
                commit_message=f"Finish mandatory backup {run_id}",
            ),
            label="final status upload",
        )

        if returncode == 0 and output_dir is not None and output_dir.exists():
            print("[TinyCeNN][BACKUP] uploading successful final checkpoint...", flush=True)
            _retry_required(
                lambda: api.upload_folder(
                    repo_id=backup_repo,
                    repo_type="model",
                    folder_path=str(output_dir),
                    path_in_repo=f"runs/{run_id}/checkpoint",
                    ignore_patterns=[".hf_run_archive/**", ".hf_live_redacted/**"],
                    commit_message=f"Final checkpoint backup {run_id}",
                ),
                label="final checkpoint upload",
            )
        elif returncode != 0:
            print(
                "[TinyCeNN][BACKUP] trainer failed; log/status backed up, large checkpoint upload skipped.",
                flush=True,
            )

        print("-" * 88, flush=True)
        print(
            f"[TinyCeNN][BACKUP COMPLETE] private Hugging Face backup committed for {run_id}",
            flush=True,
        )
        if returncode == 0:
            print(f"[TinyCeNN][DONE] {script} completed in {_format_elapsed(elapsed)}", flush=True)
        else:
            print(
                f"[TinyCeNN][FAILED] {script} exited with code {returncode} after {_format_elapsed(elapsed)}",
                flush=True,
            )
        print("=" * 88 + "\n", flush=True)

        completed = subprocess.CompletedProcess(cmd, returncode)
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, cmd)
        return completed

    run_with_backup._tinycenn_live_backup = True
    subprocess.run = run_with_backup
    return True