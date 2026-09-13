#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

MODELS = (
    "tinycenn-lm",
    "tinycenn-distill",
    "tinycenn-rigorous-continue",
    "tinycenn-optimized-continue",
    "tinycenn-moe-top2",
    "tinycenn-sharedffn-top2",
    "tinycenn-story-antirepeat",
    "tinycenn-story-v2",
    "smollm2-amcenn-top2",
    "smollm2-amcenn-top2-v2",
)

DEFAULT_SOURCE = {
    "tinycenn-rigorous-continue": "vtava/TinyCeNN-LM-Distilled",
    "tinycenn-optimized-continue": "vtava/TinyCeNN-LM-Distilled-v2",
    "tinycenn-moe-top2": "vtava/TinyCeNN-LM-Distilled-v2",
    "tinycenn-sharedffn-top2": "vtava/TinyCeNN-LM-Distilled-v2",
    "tinycenn-story-antirepeat": "vtava/TinyCeNN-LM-Sharded-MoE-Top2",
    "tinycenn-story-v2": "vtava/TinyCeNN-LM-Story-AntiRepeat",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Train, test, and collect one TinyCeNN-LM notebook-equivalent experiment"
    )
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--result-root", default="result")
    p.add_argument("--source", help="Warm-start local directory or Hugging Face repo; overrides notebook default")
    p.add_argument("--smoke", action="store_true", help="Use a small training budget for a quick end-to-end test")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--force", action="store_true", help="Allow reusing an existing result directory")
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> str:
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            lines.append(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    return "".join(lines)


def resolve_source(value: str | None, default: str | None) -> str | None:
    source = value or default
    if source is None:
        return None
    local = Path(source).expanduser()
    if local.exists():
        return str(local.resolve())
    from huggingface_hub import snapshot_download
    print(f"Downloading warm-start model from Hugging Face: {source}")
    return snapshot_download(
        repo_id=source,
        repo_type="model",
        token=os.environ.get("HF_TOKEN") or None,
    )


def notebook_train_command(model: str, out: Path, source: str | None, smoke: bool) -> list[str]:
    py = sys.executable
    if model == "tinycenn-lm":
        return [
            py, "scripts/train_adapter.py",
            "--max-tokens", "100000" if smoke else "1000000",
            "--context-length", "256", "--batch-size", "8", "--grad-accum", "4",
            "--steps", "4", "--learning-rate", "0.002",
            "--eval-every", "25", "--eval-batches", "8", "--eval-batch-size", "4",
            "--output-dir", str(out),
        ]
    if model == "tinycenn-distill":
        return [
            py, "scripts/train_distill.py",
            "--max-tokens", "100000" if smoke else "10000000",
            "--context-length", "256", "--batch-size", "4", "--grad-accum", "4",
            "--learning-rate", "1e-3", "--steps", "7", "--dilations", "1,2,4,8,16,32,64",
            "--temperature", "2.0", "--ce-weight", "1.0", "--kl-weight", "1.0",
            "--hidden-weight", "0.25", "--output-dir", str(out),
        ]
    if model == "tinycenn-rigorous-continue":
        assert source
        return [
            py, "scripts/train_distill_rigorous.py", "--resume-student-dir", source,
            "--max-tokens", "100000" if smoke else "30000000",
            "--context-length", "256", "--batch-size", "4", "--grad-accum", "4",
            "--learning-rate", "3e-4", "--steps", "7", "--dilations", "1,2,4,8,16,32,64",
            "--temperature", "2.0", "--ce-weight", "1.0", "--kl-weight", "1.0", "--hidden-weight", "0.25",
            "--shuffle-buffer", "4096", "--eval-batches", "8" if smoke else "64",
            "--eval-batch-size", "4", "--eval-every", "25" if smoke else "250",
            "--target-gap-recovery", "0.90", "--output-dir", str(out),
        ]
    if model == "tinycenn-optimized-continue":
        assert source
        return [
            py, "scripts/train_distill_rigorous.py", "--resume-student-dir", source,
            "--train-interfaces", "all", "--interface-lr-scale", "0.05",
            "--lr-schedule", "wsd", "--decay-ratio", "0.20",
            "--kl-final-weight", "0.25", "--hidden-final-weight", "0.0",
            "--max-tokens", "100000" if smoke else "30000000",
            "--context-length", "256", "--batch-size", "4", "--grad-accum", "4",
            "--learning-rate", "3e-4", "--steps", "7", "--dilations", "1,2,4,8,16,32,64",
            "--temperature", "2.0", "--ce-weight", "1.0", "--kl-weight", "1.0", "--hidden-weight", "0.25",
            "--shuffle-buffer", "4096", "--eval-batches", "8" if smoke else "64",
            "--eval-batch-size", "4", "--eval-every", "25" if smoke else "250",
            "--target-gap-recovery", "0.90", "--output-dir", str(out),
        ]
    if model == "tinycenn-moe-top2":
        assert source
        return [
            py, "scripts/train_moe_distill.py", "--warmstart-plain-dir", source,
            "--max-tokens", "100000" if smoke else "30000000",
            "--context-length", "256", "--batch-size", "4", "--grad-accum", "4",
            "--learning-rate", "2e-4", "--steps", "7", "--dilations", "1,2,4,8,16,32,64",
            "--num-experts", "8", "--top-k", "2", "--router-aux-weight", "0.01",
            "--router-z-weight", "0.001", "--shuffle-buffer", "4096",
            "--eval-batches", "8" if smoke else "64", "--eval-batch-size", "4",
            "--eval-every", "25" if smoke else "250", "--output-dir", str(out),
        ]
    if model == "tinycenn-sharedffn-top2":
        assert source
        return [
            py, "scripts/train_sharded_moe_distill.py", "--warmstart-plain-dir", source,
            "--max-tokens", "100000" if smoke else "20000000",
            "--max-runtime-minutes", "5" if smoke else "50",
            "--context-length", "256", "--batch-size", "4", "--grad-accum", "4",
            "--learning-rate", "2e-4", "--num-shards", "8", "--top-k", "2",
            "--eval-batches", "8" if smoke else "64", "--eval-batch-size", "4",
            "--eval-every", "25" if smoke else "500", "--shuffle-buffer", "4096",
            "--output-dir", str(out),
        ]
    if model == "tinycenn-story-antirepeat":
        assert source
        return [
            py, "scripts/train_story_antirepeat.py", "--source-dir", source, "--output-dir", str(out),
            "--max-tokens", "100000" if smoke else "20000000",
            "--max-runtime-minutes", "5" if smoke else "45", "--learning-rate", "1e-4",
            "--repeat-weight", "0.20", "--repeat-window", "32",
            "--router-aux-weight", "0.0005", "--router-z-weight", "0.0001",
            "--log-every", "20", "--save-every", "1000",
        ]
    if model == "tinycenn-story-v2":
        assert source
        return [
            py, "scripts/train_story_v2.py", "--source-dir", source, "--output-dir", str(out),
            "--max-tokens", "100000" if smoke else "15000000",
            "--max-runtime-minutes", "5" if smoke else "45", "--learning-rate", "7e-5",
            "--repeat-weight", "0.08", "--memory-rank", "32", "--head-rank", "4",
            "--max-trainable-params", "650000",
        ]
    if model == "smollm2-amcenn-top2":
        return [
            py, "scripts/train_smollm2_amcenn.py", "--base-model", "HuggingFaceTB/SmolLM2-135M",
            "--output-dir", str(out), "--context-length", "128", "--feature-dim", "32",
            "--max-tokens", "100000" if smoke else "2000000",
            "--max-runtime-minutes", "5" if smoke else "45",
        ]
    if model == "smollm2-amcenn-top2-v2":
        return [
            py, "scripts/train_smollm2_amcenn_v2.py", "--base-model", "HuggingFaceTB/SmolLM2-135M",
            "--output-dir", str(out), "--context-length", "128", "--feature-dim", "128",
            "--group-size", "5", "--calibration-steps", "3" if smoke else "30",
            "--final-max-tokens", "100000" if smoke else "500000",
            "--max-runtime-minutes", "5" if smoke else "60",
        ]
    raise ValueError(model)


def extract_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def main() -> None:
    args, extra = parse_args()
    repo = Path(__file__).resolve().parents[1]
    result_dir = (repo / args.result_root / args.model).resolve()
    model_dir = result_dir / "model"
    best_dir = result_dir / "model-best"

    if result_dir.exists() and any(result_dir.iterdir()) and not args.force:
        raise SystemExit(f"Result directory already exists: {result_dir}. Use --force to reuse it.")
    result_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_source(args.source, DEFAULT_SOURCE.get(args.model))
    train_cmd = notebook_train_command(args.model, model_dir, source, args.smoke) + extra
    manifest = {
        "model": args.model,
        "source": args.source or DEFAULT_SOURCE.get(args.model),
        "resolved_source": source,
        "result_dir": str(result_dir),
        "model_dir": str(model_dir),
        "train_command": train_cmd,
        "smoke": args.smoke,
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_logged(train_cmd, result_dir / "train.log", repo)

    selected = best_dir if best_dir.exists() else model_dir
    manifest["selected_checkpoint"] = str(selected)
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_eval:
        eval_cmd = [
            sys.executable, "scripts/evaluate_local.py", "--model", args.model,
            "--model-dir", str(selected), "--output", str(result_dir / "evaluation.json"),
            "--max-new-tokens", str(args.max_new_tokens),
        ]
        run_logged(eval_cmd, result_dir / "eval.log", repo)

        rigorous_cmd = None
        if args.model in {"tinycenn-rigorous-continue", "tinycenn-optimized-continue"}:
            rigorous_cmd = [
                sys.executable, "scripts/eval_distilled.py", "--student-dir", str(selected),
                "--ce-tolerance", "0.02",
            ]
        elif args.model == "tinycenn-sharedffn-top2":
            rigorous_cmd = [
                sys.executable, "scripts/eval_sharded_moe.py", "--student-dir", str(selected),
                "--ce-tolerance", "0.02",
            ]
        if rigorous_cmd:
            text = run_logged(rigorous_cmd, result_dir / "rigorous_eval.log", repo)
            parsed = extract_json_object(text)
            if parsed is not None:
                (result_dir / "rigorous_evaluation.json").write_text(
                    json.dumps(parsed, indent=2), encoding="utf-8"
                )

    print(f"\nCompleted: {args.model}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
