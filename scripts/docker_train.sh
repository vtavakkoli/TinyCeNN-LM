#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " TinyCeNN-LM automated training + health check"
echo "============================================================"

python - <<'PY'
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA GPU is not visible inside the container. "
        "Install/configure NVIDIA Container Toolkit or Docker Desktop WSL2 GPU support."
    )
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"vram_gib={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
PY

args=(
  --base-model "${BASE_MODEL:-arnir0/Tiny-LLM}"
  --dataset "${DATASET:-HuggingFaceFW/fineweb}"
  --dataset-config "${DATASET_CONFIG:-sample-10BT}"
  --output-dir "${OUTPUT_DIR:-/workspace/checkpoints/tinycenn-base}"
  --max-tokens "${MAX_TOKENS:-1000000}"
  --context-length "${CONTEXT_LENGTH:-256}"
  --batch-size "${BATCH_SIZE:-4}"
  --grad-accum "${GRAD_ACCUM:-8}"
  --steps "${CENN_STEPS:-4}"
  --learning-rate "${LEARNING_RATE:-0.002}"
  --eval-every "${EVAL_EVERY:-25}"
  --eval-batches "${EVAL_BATCHES:-8}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}"
  --health-min-improvement "${HEALTH_MIN_IMPROVEMENT:-0.002}"
  --divergence-factor "${DIVERGENCE_FACTOR:-1.25}"
)

if [[ "${FAIL_ON_NO_IMPROVEMENT:-0}" == "1" ]]; then
  args+=(--fail-on-no-improvement)
fi
if [[ "${NO_COMPILE:-0}" == "1" ]]; then
  args+=(--no-compile)
fi

python scripts/train_adapter.py "${args[@]}"

report="${OUTPUT_DIR:-/workspace/checkpoints/tinycenn-base}/training_report.json"
echo
echo "Training finished. Health report:"
python - "$report" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
print(json.dumps({
    "status": report["status"],
    "seen_tokens": report["seen_tokens"],
    "initial_eval_loss": round(report["initial_eval_loss"], 4),
    "best_eval_loss": round(report["best_eval_loss"], 4),
    "best_perplexity": round(report["best_perplexity"], 2),
    "relative_best_improvement_percent": round(report["relative_best_improvement"] * 100, 3),
    "best_update": report["best_update"],
    "elapsed_minutes": round(report["elapsed_seconds"] / 60, 2),
    "peak_vram_gib": round(report["peak_vram_gib"], 2),
}, indent=2))

if report["status"] == "diverged":
    raise SystemExit(2)
PY

echo "============================================================"
echo " Model files: ${OUTPUT_DIR:-/workspace/checkpoints/tinycenn-base}"
echo " Best adapter: ${OUTPUT_DIR:-/workspace/checkpoints/tinycenn-base}-best"
echo "============================================================"
