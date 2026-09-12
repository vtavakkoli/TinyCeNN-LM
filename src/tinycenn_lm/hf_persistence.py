from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
_SMALL_ARTIFACT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md", ".log", ".yaml", ".yml"}


def utc_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def redact_secrets(text: str) -> str:
    return _TOKEN_RE.sub("hf_REDACTED", text)


def _json_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*.json")
        if p.is_file() and p.stat().st_size <= 5 * 1024 * 1024
    )


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_reports(folder: str | Path) -> dict[str, Any]:
    folder = Path(folder)
    reports: dict[str, Any] = {}
    for path in _json_files(folder):
        name = path.name.lower()
        if any(key in name for key in ("report", "result", "metric", "summary", "config")):
            value = _load_json(path)
            if value is not None:
                reports[str(path.relative_to(folder))] = value
    return reports


def _first_value(reports: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    def walk(value: Any) -> Any | None:
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] not in (None, ""):
                    return value[key]
            for child in value.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None

    return walk(reports)


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalars(child, name))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value
    return out


def _interesting_metrics(reports: dict[str, Any]) -> list[tuple[str, Any]]:
    wanted = (
        "status", "stop_reason", "seen_tokens", "updates", "context_length",
        "feature_dim", "num_shards", "top_k", "trainable", "trainable_percent",
        "last_training_ce", "last_distillation_kl", "best.student_ce", "best.teacher_ce",
        "teacher_gap_recovery_fraction", "mean_route_mix", "mean_router_entropy",
        "elapsed_minutes", "peak_vram_gib", "evaluation_performed",
    )
    flat: dict[str, Any] = {}
    for report in reports.values():
        if isinstance(report, dict):
            flat.update(_flatten_scalars(report))
    rows: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for want in wanted:
        for key, value in flat.items():
            if key == want or key.endswith("." + want):
                label = want.split(".")[-1]
                if label not in seen:
                    rows.append((label, value))
                    seen.add(label)
                break
    return rows


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.3e}"
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_model_card(
    folder: str | Path,
    *,
    title: str | None = None,
    architecture: str | None = None,
    base_model: str | None = None,
    source_repo: str = "https://github.com/vtavakkoli/TinyCeNN-LM",
    extra_notes: str | None = None,
) -> str:
    folder = Path(folder)
    reports = collect_reports(folder)
    inferred_arch = architecture or _first_value(reports, ("architecture", "task")) or "TinyCeNN-LM experiment"
    inferred_base = base_model or _first_value(reports, ("base_model", "teacher_model"))
    dataset = _first_value(reports, ("dataset",))
    dataset_config = _first_value(reports, ("dataset_config",))
    display_title = title or folder.name.replace("-", " ")

    tags = ["tinycenn", "cenn", "language-modeling", "text-generation", "research"]
    arch_lower = str(inferred_arch).lower()
    if "distill" in arch_lower:
        tags.append("knowledge-distillation")
    if "moe" in arch_lower or _first_value(reports, ("num_shards", "num_experts")):
        tags.append("mixture-of-experts")
    if "amcenn" in arch_lower or "attention-free" in arch_lower:
        tags.extend(["attention-free", "linear-attention", "recurrent-memory"])
    if "story" in arch_lower:
        tags.append("story-generation")

    yaml_lines = ["---", "library_name: transformers", "pipeline_tag: text-generation"]
    if inferred_base:
        yaml_lines.append(f"base_model: {inferred_base}")
    if dataset:
        yaml_lines.append("datasets:")
        yaml_lines.append(f"- {dataset}")
    yaml_lines.append("tags:")
    for tag in dict.fromkeys(tags):
        yaml_lines.append(f"- {tag}")
    yaml_lines.append("---")

    metrics = _interesting_metrics(reports)
    if metrics:
        lines = ["| Metric | Value |", "|---|---:|"]
        lines += [f"| `{name}` | {_format_value(value)} |" for name, value in metrics]
        metric_table = "\n".join(lines)
    else:
        metric_table = "No structured training report was found in this upload."

    report_files = sorted(reports)
    files_text = "\n".join(f"- `{name}`" for name in report_files) or "- No JSON report files detected."
    dataset_text = str(dataset) if dataset else "Not recorded"
    if dataset_config:
        dataset_text += f" (`{dataset_config}`)"

    limitations = (
        "This is a research checkpoint. Metrics saved here are the metrics produced by the corresponding "
        "training notebook/script; unless explicitly marked as held-out evaluation, they should not be treated "
        "as publication-grade benchmark results. Generation quality can differ substantially from the base model."
    )
    notes = f"\n## Notes\n\n{extra_notes.strip()}\n" if extra_notes else ""
    return redact_secrets("\n".join(yaml_lines) + f"\n\n# {display_title}\n\n"
        f"Research artifact from **TinyCeNN-LM**. Architecture: `{inferred_arch}`.\n\n"
        "## Architecture\n\n"
        f"- Architecture/run type: `{inferred_arch}`\n"
        f"- Base model: `{inferred_base or 'not recorded'}`\n"
        f"- Dataset: `{dataset_text}`\n"
        f"- Source code: {source_repo}\n\n"
        "## Latest saved results\n\n"
        f"{metric_table}\n\n"
        "The Hugging Face repository keeps timestamped run artifacts under `runs/`. This preserves training "
        "reports, configs and run metadata independently of the temporary Colab filesystem.\n\n"
        "## Saved experiment files\n\n"
        f"{files_text}\n\n"
        "## Reproducibility\n\n"
        "Run the matching notebook from the TinyCeNN-LM repository. Colab notebooks use a Hugging Face write "
        "token from the `HF_TOKEN` Colab Secret; tokens should never be pasted into notebook source.\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
        f"{notes}\n"
        "## Citation\n\n"
        "If you use this experimental checkpoint, cite the TinyCeNN-LM repository and the upstream base model.\n"
    )


def write_run_manifest(
    folder: str | Path,
    *,
    run_id: str,
    notebook: str | None = None,
    repo_id: str | None = None,
) -> Path:
    folder = Path(folder)
    reports = collect_reports(folder)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "notebook": notebook,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "reports": sorted(reports),
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    path = folder / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _copy_small_artifacts(folder: Path, archive_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.is_relative_to(archive_dir.parent):
                continue
        except AttributeError:
            pass
        if path.suffix.lower() not in _SMALL_ARTIFACT_SUFFIXES:
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        rel = path.relative_to(folder)
        dst = archive_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = path.read_text(encoding="utf-8")
            dst.write_text(redact_secrets(text), encoding="utf-8")
        except Exception:
            shutil.copy2(path, dst)
        copied.append(str(rel))
    return copied


def _prepare_run_archive(folder: Path, *, repo_id: str, run_id: str, notebook: str | None = None) -> Path:
    archive_dir = folder / ".hf_run_archive" / run_id
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_small_artifacts(folder, archive_dir)
    latest = {
        "run_id": run_id,
        "repo_id": repo_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": notebook,
        "artifacts": copied,
    }
    (archive_dir / "latest_run.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return archive_dir


def persist_hf_run(
    *,
    api,
    repo_id: str,
    folder_path: str | Path,
    token: str | None = None,
    title: str | None = None,
    architecture: str | None = None,
    base_model: str | None = None,
    notebook: str | None = None,
    run_id: str | None = None,
    commit_message: str | None = None,
    upload_model_files: bool = True,
) -> dict[str, Any]:
    """Persist a completed Colab run and a timestamped result archive to Hugging Face."""
    from huggingface_hub import HfApi

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(folder)
    run_id = run_id or utc_run_id(folder.name[:24] or "run")
    client = api if api is not None else HfApi(token=token)
    client.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    card = build_model_card(folder, title=title, architecture=architecture, base_model=base_model)
    (folder / "README.md").write_text(card, encoding="utf-8")
    write_run_manifest(folder, run_id=run_id, notebook=notebook, repo_id=repo_id)

    if upload_model_files:
        client.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(folder),
            commit_message=commit_message or f"Publish TinyCeNN run {run_id}",
            ignore_patterns=[".hf_run_archive/**"],
        )

    archive_dir = _prepare_run_archive(folder, repo_id=repo_id, run_id=run_id, notebook=notebook)
    latest_file = archive_dir / "latest_run.json"
    client.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(archive_dir),
        path_in_repo=f"runs/{run_id}", commit_message=f"Archive TinyCeNN results {run_id}",
    )
    client.upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=str(latest_file),
        path_in_repo="runs/latest_run.json",
        commit_message=f"Update latest TinyCeNN run pointer to {run_id}",
    )
    return json.loads(latest_file.read_text(encoding="utf-8"))


def _looks_like_tinycenn_folder(folder: Path) -> bool:
    if "tinycenn" in str(folder).lower() or "smollm2-amcenn" in str(folder).lower():
        return True
    names = {p.name.lower() for p in folder.iterdir()} if folder.exists() and folder.is_dir() else set()
    return any("cenn" in name or "story_v2" in name or "sharded" in name for name in names)


def install_colab_hf_upload_enhancer() -> bool:
    """Enhance existing notebook HfApi.upload_folder calls without duplicating notebook code.

    In Colab, TinyCeNN notebooks already import tinycenn_lm before publishing. This wrapper regenerates a
    report-backed README and archives small result files under runs/<timestamp>/ whenever a TinyCeNN checkpoint
    folder is uploaded. Outside Colab it is a no-op.
    """
    if not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or Path("/content").exists()):
        return False
    try:
        from huggingface_hub import HfApi
    except Exception:
        return False
    if getattr(HfApi.upload_folder, "_tinycenn_enhanced", False):
        return True

    original_upload_folder = HfApi.upload_folder
    original_upload_file = HfApi.upload_file

    def enhanced_upload_folder(self, *args, **kwargs):
        folder_value = kwargs.get("folder_path")
        repo_id = kwargs.get("repo_id")
        repo_type = kwargs.get("repo_type", "model")
        if folder_value is None and len(args) >= 1:
            folder_value = args[0]
        if repo_id is None and len(args) >= 2:
            repo_id = args[1]
        folder = Path(folder_value) if folder_value else None
        should_enhance = (
            repo_type == "model" and folder is not None and folder.exists() and folder.is_dir()
            and repo_id and _looks_like_tinycenn_folder(folder)
            and ".hf_run_archive" not in str(folder)
        )
        if not should_enhance:
            return original_upload_folder(self, *args, **kwargs)

        run_id = utc_run_id(folder.name[:24] or "run")
        reports = collect_reports(folder)
        architecture = _first_value(reports, ("architecture", "task"))
        base_model = _first_value(reports, ("base_model", "teacher_model"))
        (folder / "README.md").write_text(
            build_model_card(folder, title=str(repo_id).split("/")[-1], architecture=architecture, base_model=base_model),
            encoding="utf-8",
        )
        write_run_manifest(folder, run_id=run_id, repo_id=str(repo_id))
        ignore = list(kwargs.get("ignore_patterns") or [])
        if ".hf_run_archive/**" not in ignore:
            ignore.append(".hf_run_archive/**")
        kwargs["ignore_patterns"] = ignore
        result = original_upload_folder(self, *args, **kwargs)

        try:
            archive_dir = _prepare_run_archive(folder, repo_id=str(repo_id), run_id=run_id)
            original_upload_folder(
                self, repo_id=repo_id, repo_type="model", folder_path=str(archive_dir),
                path_in_repo=f"runs/{run_id}", commit_message=f"Archive TinyCeNN results {run_id}",
            )
            original_upload_file(
                self, repo_id=repo_id, repo_type="model",
                path_or_fileobj=str(archive_dir / "latest_run.json"),
                path_in_repo="runs/latest_run.json",
                commit_message=f"Update latest TinyCeNN run pointer to {run_id}",
            )
        except Exception as exc:
            print(f"TinyCeNN HF result archive warning: {exc}")
        return result

    enhanced_upload_folder._tinycenn_enhanced = True
    HfApi.upload_folder = enhanced_upload_folder
    return True


def fingerprint_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
