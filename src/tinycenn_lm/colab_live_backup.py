from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        if p.is_file() and p.suffix.lower() in _SMALL_SUFFIXES and p.stat().st_size <= 10 * 1024 * 1024
        and ".hf_run_archive" not in p.parts
    ]


def _safe_upload_live(api, *, repo_id: str, run_id: str, output_dir: Path | None, log_file: Path) -> None:
    try:
        if log_file.exists():
            clean_log = log_file.with_name("train_redacted.log")
            clean_log.write_text(
                redact_secrets(log_file.read_text(encoding="utf-8", errors="replace")),
                encoding="utf-8",
            )
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(clean_log),
                path_in_repo=f"runs/{run_id}/train.log",
                commit_message=f"Live backup {run_id}",
            )
        if output_dir is not None:
            for path in _small_files(output_dir):
                rel = path.relative_to(output_dir)
                upload_path = path
                if path.suffix.lower() in {".txt", ".md", ".log", ".json", ".jsonl", ".yaml", ".yml", ".csv"}:
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
                api.upload_file(
                    repo_id=repo_id,
                    repo_type="model",
                    path_or_fileobj=str(upload_path),
                    path_in_repo=f"runs/{run_id}/artifacts/{rel.as_posix()}",
                    commit_message=f"Live results {run_id}",
                )
    except Exception as exc:
        print(f"[TinyCeNN][BACKUP] warning: {exc}", flush=True)


def _backup_metadata(cmd, output_dir: Path | None, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [redact_secrets(str(x)) for x in cmd],
        "output_dir": str(output_dir) if output_dir else None,
        "status": "running",
    }


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


def install_colab_training_backup(*, interval_seconds: int = 180) -> bool:
    """Stream every TinyCeNN Colab training process and optionally back it up.

    Existing notebooks use ``subprocess.run([... train_*.py ...])``. In Colab this
    wrapper converts those calls to a line-streaming ``Popen`` execution so trainer
    metrics are visible immediately. The child receives ``PYTHONUNBUFFERED=1`` and
    Colab prints explicit START/LIVE/DONE/FAILED status markers.

    If a Hugging Face token is available, the previous private live-backup behavior
    remains enabled. Missing/invalid Hub credentials no longer disable live console
    streaming; only the backup part is skipped.
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
        work_root = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        live_root = work_root / ".colab_live_backup" / run_id
        live_root.mkdir(parents=True, exist_ok=True)
        log_file = live_root / "train.log"
        meta_file = live_root / "run_status.json"
        metadata = _backup_metadata(cmd, output_dir, run_id)
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        api = None
        backup_repo = None
        try:
            from huggingface_hub import HfApi, get_token

            token = get_token()
            if token:
                api = HfApi(token=token)
                user = api.whoami()["name"]
                backup_repo = f"{user}/TinyCeNN-LM-Colab-Backups"
                api.create_repo(backup_repo, repo_type="model", private=True, exist_ok=True)
                try:
                    api.upload_file(
                        repo_id=backup_repo,
                        repo_type="model",
                        path_or_fileobj=str(meta_file),
                        path_in_repo=f"runs/{run_id}/run_status.json",
                        commit_message=f"Start backup {run_id}",
                    )
                except Exception as exc:
                    print(f"[TinyCeNN][BACKUP] initial metadata upload warning: {exc}", flush=True)
            else:
                print("[TinyCeNN][BACKUP] no Hugging Face token; live console remains enabled.", flush=True)
        except Exception as exc:
            print(f"[TinyCeNN][BACKUP] disabled for this run: {exc}", flush=True)
            api = None
            backup_repo = None

        command_text = " ".join(shlex.quote(str(x)) for x in cmd)
        print("\n" + "=" * 88, flush=True)
        print(f"[TinyCeNN][START] {script}", flush=True)
        print(f"[TinyCeNN][COMMAND] {command_text}", flush=True)
        if output_dir is not None:
            print(f"[TinyCeNN][OUTPUT] {output_dir}", flush=True)
        print(f"[TinyCeNN][LOCAL LOG] {log_file}", flush=True)
        if backup_repo:
            print(
                f"[TinyCeNN][BACKUP] https://huggingface.co/{backup_repo}/tree/main/runs/{run_id}",
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
        with log_file.open("a", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                now = time.monotonic()
                if api is not None and backup_repo is not None and now - last_sync >= interval_seconds:
                    _safe_upload_live(
                        api,
                        repo_id=backup_repo,
                        run_id=run_id,
                        output_dir=output_dir,
                        log_file=log_file,
                    )
                    last_sync = now

        returncode = proc.wait()
        elapsed = time.monotonic() - started
        metadata["status"] = "completed" if returncode == 0 else "failed"
        metadata["returncode"] = returncode
        metadata["elapsed_seconds"] = elapsed
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if api is not None and backup_repo is not None:
            _safe_upload_live(
                api,
                repo_id=backup_repo,
                run_id=run_id,
                output_dir=output_dir,
                log_file=log_file,
            )
            try:
                api.upload_file(
                    repo_id=backup_repo,
                    repo_type="model",
                    path_or_fileobj=str(meta_file),
                    path_in_repo=f"runs/{run_id}/run_status.json",
                    commit_message=f"Finish backup {run_id}",
                )
                if output_dir is not None and output_dir.exists():
                    api.upload_folder(
                        repo_id=backup_repo,
                        repo_type="model",
                        folder_path=str(output_dir),
                        path_in_repo=f"runs/{run_id}/checkpoint",
                        ignore_patterns=[".hf_run_archive/**", ".hf_live_redacted/**"],
                        commit_message=f"Backup checkpoint {run_id}",
                    )
            except Exception as exc:
                print(f"[TinyCeNN][BACKUP] final upload warning: {exc}", flush=True)

        print("-" * 88, flush=True)
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
