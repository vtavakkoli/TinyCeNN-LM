from __future__ import annotations

import json
import os
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
            clean_log.write_text(redact_secrets(log_file.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
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
                        dst.write_text(redact_secrets(path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
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
        print(f"TinyCeNN live HF backup warning: {exc}")


def _backup_metadata(cmd, output_dir: Path | None, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [redact_secrets(str(x)) for x in cmd],
        "output_dir": str(output_dir) if output_dir else None,
        "status": "running",
    }


def install_colab_training_backup(*, interval_seconds: int = 180) -> bool:
    """Mirror TinyCeNN Colab training logs/results to a private Hugging Face repo.

    Only `subprocess.run([... train_*.py ...])` calls are wrapped. Other subprocess calls are untouched.
    The private repo is `<HF user>/TinyCeNN-LM-Colab-Backups`. Small reports/configs/logs are mirrored
    periodically. When training exits normally, the emitted checkpoint directory is uploaded as well.
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

        try:
            from huggingface_hub import HfApi, get_token
            token = get_token()
            if not token:
                return original_run(cmd, **kwargs)
            api = HfApi(token=token)
            user = api.whoami()["name"]
            backup_repo = f"{user}/TinyCeNN-LM-Colab-Backups"
            api.create_repo(backup_repo, repo_type="model", private=True, exist_ok=True)
        except Exception as exc:
            print(f"TinyCeNN live backup disabled for this run: {exc}")
            return original_run(cmd, **kwargs)

        cwd = kwargs.pop("cwd", None)
        env = kwargs.pop("env", None)
        check = bool(kwargs.pop("check", False))
        if kwargs:
            return original_run(cmd, cwd=cwd, env=env, check=check, **kwargs)

        script = training_script_name(cmd)
        run_id = utc_run_id(script[:32])
        output_dir = output_dir_from_command(cmd, cwd=cwd)
        live_root = Path("/content/TinyCeNN-LM/.colab_live_backup") / run_id
        live_root.mkdir(parents=True, exist_ok=True)
        log_file = live_root / "train.log"
        meta_file = live_root / "run_status.json"
        metadata = _backup_metadata(cmd, output_dir, run_id)
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            api.upload_file(
                repo_id=backup_repo, repo_type="model", path_or_fileobj=str(meta_file),
                path_in_repo=f"runs/{run_id}/run_status.json", commit_message=f"Start backup {run_id}",
            )
        except Exception as exc:
            print(f"TinyCeNN live HF backup warning: {exc}")

        print(f"TinyCeNN live backup: https://huggingface.co/{backup_repo}/tree/main/runs/{run_id}")
        proc = original_popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_sync = 0.0
        with log_file.open("a", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                now = time.monotonic()
                if now - last_sync >= interval_seconds:
                    _safe_upload_live(api, repo_id=backup_repo, run_id=run_id, output_dir=output_dir, log_file=log_file)
                    last_sync = now
        returncode = proc.wait()
        metadata["status"] = "completed" if returncode == 0 else "failed"
        metadata["returncode"] = returncode
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _safe_upload_live(api, repo_id=backup_repo, run_id=run_id, output_dir=output_dir, log_file=log_file)
        try:
            api.upload_file(
                repo_id=backup_repo, repo_type="model", path_or_fileobj=str(meta_file),
                path_in_repo=f"runs/{run_id}/run_status.json", commit_message=f"Finish backup {run_id}",
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
            print(f"TinyCeNN final HF backup warning: {exc}")

        completed = subprocess.CompletedProcess(cmd, returncode)
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, cmd)
        return completed

    run_with_backup._tinycenn_live_backup = True
    subprocess.run = run_with_backup
    return True
