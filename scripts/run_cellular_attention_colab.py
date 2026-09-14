#!/usr/bin/env python3
"""Reliable Colab launcher for the Cellular Attention benchmark.

Runs an end-to-end smoke preflight in a fresh Python process, then starts the
requested scientific profile in another fresh process. Child stdout/stderr is
streamed live and copied to a durable log so Colab never hides the real error.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_VARIANTS = (
    "cellular_local3,cellular_dilated3,cellular_dilated5,"
    "cellular_multiscale5,cellular_shifted8"
)

PROFILES = {
    "smoke": {
        "context": 64,
        "test-contexts": "64,128",
        "feature-dims": "24",
        "dilations": "1,2,4,8,16,32",
        "train-documents": 8,
        "validation-documents": 4,
        "test-documents": 4,
        "steps": 12,
        "lm-steps": 2,
        "eval-every": 6,
        "shifted-window": 8,
    },
    "balanced": {
        "context": 256,
        "test-contexts": "256,512",
        "feature-dims": "32,64",
        "dilations": "1,2,4,8,16,32,64,128",
        "train-documents": 64,
        "validation-documents": 12,
        "test-documents": 24,
        "steps": 300,
        "lm-steps": 40,
        "eval-every": 50,
        "shifted-window": 8,
    },
    "extended": {
        "context": 512,
        "test-contexts": "512,1024",
        "feature-dims": "64,96",
        "dilations": "1,2,4,8,16,32,64,128,256",
        "train-documents": 192,
        "validation-documents": 32,
        "test-documents": 64,
        "steps": 900,
        "lm-steps": 100,
        "eval-every": 100,
        "shifted-window": 16,
    },
}

PREFLIGHT = {
    "context": 64,
    "test-contexts": "64",
    "feature-dims": "24",
    "dilations": "1,2,4,8,16,32",
    "train-documents": 4,
    "validation-documents": 2,
    "test-documents": 2,
    "steps": 2,
    "lm-steps": 1,
    "eval-every": 1,
    "shifted-window": 8,
}


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def command_for(benchmark: Path, output_dir: Path, *, layers: str, variants: str,
                seed: int, config: dict) -> list[str]:
    # Launch as a module from the repository root. Executing the file directly
    # makes sys.path[0] point at /scripts, which breaks imports such as
    # `from scripts.benchmark_cenn_research_layers import ...` on Colab.
    command = [
        sys.executable, "-u", "-m", "scripts.benchmark_cellular_attention",
        "--layers", layers,
        "--variants", variants,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    for key, value in config.items():
        command.extend([f"--{key}", str(value)])
    return command


def stream(command: list[str], *, cwd: Path, log_file: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Keep the repository importable in child processes regardless of the
    # parent notebook's working directory or editable-install behavior.
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(cwd) if not existing_pythonpath
        else str(cwd) + os.pathsep + existing_pythonpath
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("Command:", subprocess.list2cmdline(command), flush=True)
    print("Log:", log_file, flush=True)

    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="balanced")
    parser.add_argument("--layers", default="18")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-root", default="result/cellular-attention-colab")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    benchmark = repo / "scripts" / "benchmark_cellular_attention.py"
    if not benchmark.exists():
        raise FileNotFoundError(benchmark)

    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_file = root / "last_run.json"

    run_meta = {
        "status": "starting",
        "profile": args.profile,
        "layers": args.layers,
        "variants": args.variants,
        "seed": args.seed,
        "python": sys.version,
    }
    write_json(status_file, run_meta)

    if not args.skip_preflight:
        preflight_id = utc_id()
        preflight_dir = root / f"preflight-{preflight_id}"
        preflight_log = root / f"preflight-{preflight_id}.log"
        preflight_cmd = command_for(
            benchmark,
            preflight_dir,
            layers=args.layers.split(",")[0],
            variants="cellular_dilated3",
            seed=args.seed,
            config=PREFLIGHT,
        )
        print("\n=== Cellular Attention full-pipeline preflight ===", flush=True)
        code = stream(preflight_cmd, cwd=repo, log_file=preflight_log)
        run_meta.update({
            "preflight_output_dir": str(preflight_dir),
            "preflight_log": str(preflight_log),
            "preflight_exit_code": code,
        })
        if code != 0:
            run_meta["status"] = "preflight_failed"
            run_meta["error_tail"] = tail(preflight_log)
            write_json(status_file, run_meta)
            print("\nPRECHECK FAILED. Last log lines:\n", tail(preflight_log), flush=True)
            return code or 1
        print("\nPRECHECK PASSED.", flush=True)
        if args.preflight_only:
            run_meta["status"] = "preflight_completed"
            write_json(status_file, run_meta)
            return 0

    run_id = utc_id()
    output_dir = root / f"{args.profile}-{run_id}"
    log_file = root / f"{args.profile}-{run_id}.log"
    command = command_for(
        benchmark,
        output_dir,
        layers=args.layers,
        variants=args.variants,
        seed=args.seed,
        config=PROFILES[args.profile],
    )

    print(f"\n=== Cellular Attention {args.profile} run ===", flush=True)
    code = stream(command, cwd=repo, log_file=log_file)
    run_meta.update({
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "exit_code": code,
        "status": "completed" if code == 0 else "failed",
    })
    if code != 0:
        run_meta["error_tail"] = tail(log_file)
    write_json(status_file, run_meta)

    if code != 0:
        print("\nRUN FAILED. Last log lines:\n", tail(log_file), flush=True)
        return code or 1

    print("\nRUN COMPLETED")
    print("Results:", output_dir)
    print("Status:", status_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
