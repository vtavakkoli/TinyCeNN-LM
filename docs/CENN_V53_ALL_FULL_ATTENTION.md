# Qwen3.5-0.8B CeNN V5.3: all six full-attention layers

## Scope and honesty

Qwen3.5-0.8B has 24 decoder layers: full-attention layers **3, 7, 11, 15, 19 and 23** and 18 other gated-delta/linear-attention layers. The V5.3 `Qwen35HeadRescueV5` adapter is compatible **only** with the six full-attention layers. This experiment sequentially installs a V5.3 adapter at **each of those six locations**; it does not replace the remaining 18 layers. `REPLACEMENT_MODE='one_group'` preserves four native query heads per eligible layer, while `'both_groups'` approximates all eight full-attention query heads. Neither supports production Qwen KV cache or `.generate()`.

## Why progressive fitting?

Independent layer-wise matching ignores distribution shift when preceding attention modules are replaced. Before training layer *i*, the notebook runs the prefix already replaced and captures the **native target attention module**'s inputs and outputs on those shifted inputs. Only the current CeNN core trains; all original Qwen Q/K/V, RoPE, output projection and non-target layers remain frozen. After each stage it measures the **cumulative model**, not merely a single isolated layer.

## Training protocol

1. Install/validate `transformers==5.17.0`, use pinned original Qwen model revision and Git SHA, use `use_cache=False` and unpadded WikiText-2 windows.
2. Compute native full-model teacher next-token NLL and top-1 logits on validation.
3. For each of `[3,7,11,15,19,23]`, capture teacher mixer inputs/outputs under the *currently installed* prefix; train a new CeNNDeltaCellV3 baseline; initialize CeNN V5.3 via strict V5→V5.1→V5.2→V5.3 state transfer (new residual adapters initialize as identity); fine-tune the layer under normalized MSE and cosine discrepancy.
4. Evaluate cumulative native/modified NLL, teacher top-1 agreement and alternating paired GPU prefill latency. Record independent per-stage quality/deployment gate decisions. By default `FORCE_ALL_SIX=True` so experiments cover all targets **even if gates fail**, never misreport this as a passed model.
5. On completion, do a one-time internal diagnostic on a disjoint WikiText-2 *train tail*. This is **not** an independent external or blinded test; prior experiments already examined WikiText-2 test split.
6. Package adapter-only checkpoints, `training_steps.jsonl`, `training.log`, cumulative stage CSV, config, machine-readable final audit, plot and provenance checksums in a ZIP and **automatically initiate the Colab download**. The export also occurs on an exception during the training/audit phase, preserving partial logs/checkpoints.

## Model quality thresholds

Per-stage: local cosine ≥ 0.95, teacher top-1 agreement ≥ 0.98, next-token ΔNLL ≤ 0.10 (nats per position); deployment also requires paired prefill speedup > 1.05×. These are experimental gates, not scientific evidence of comparable downstream capabilities. `FORCE_ALL_SIX` is a diagnostic mode which intentionally ignores gate failures when deciding whether to continue to the next layer. A native-vs-six-layer cumulative comparison is the final output.

## Known constraints

No actual GPU result is baked into this notebook. The PyTorch gated-delta recurrence uses FP32 operations and reference triangular solves rather than a fused CUDA/Triton kernel; six-layer full-context inference may be **slower** than native Qwen. No continuous batched decoding, variable-length padding, audio/image pathways, or Qwen hybrid cache. Automatic downloading requires a running Colab browser; allow its download prompt. A runtime disconnect may destroy local state unless separately persisted.

## Launch

`https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_V53_All6_FullAttention_Sequential_Colab.ipynb`
