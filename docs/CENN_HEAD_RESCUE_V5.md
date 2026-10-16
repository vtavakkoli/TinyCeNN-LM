# CeNN V5 — Selective full-attention head rescue for Qwen3.5-0.8B

**Status:** experimental research module and notebook. **No real Qwen V5 GPU training or inference result has been run yet.** Native attention and GQA grouping are retained for accuracy-sensitive pretrained heads, while the remaining heads use CeNN memory with limited causal local attention. This is *not* a claim of architectural novelty or speedup.

## Why V5 follows the uploaded V4 experiment

A user-run V4 quick experiment (Qwen3.5-0.8B, full-attention layer 11, 128-token blocks, 80 shared plus 80 candidate steps, seed 2031) reported:

| V4 variant | Local cosine | Validation ΔNLL | Validation whole-model speedup |
|---|---:|---:|---:|
| V3.1 delta + Local32 | 0.879789 | +0.010813 | 0.9822× |
| V3 adaptive + Local32 | 0.880016 | +0.010609 | 0.9861× |
| V4 graph static + Local32 | 0.880406 | +0.010770 | 0.9726× |
| V4 graph adaptive + Local32 | 0.884557 | +0.010527 | 0.9677× |
| V4 graph shuffled + Local32 | 0.886318 | +0.010271 | 0.9688× |

No candidate passed the 0.95 cosine threshold or teacher agreement >=0.98. The validation-selected shuffled model had held-out ΔNLL +0.006447, teacher agreement 0.955381, paired speedup 0.980748×. Frozen graph-topology permutation lowered cosine by only 0.001802 and improved NLL by 0.000392 (not significant without more repeats). **We cannot attribute V4 progress to the learned graph.**

## Proposed intervention

For each full-attention layer, let Qwen provide its frozen Q/K/V, QK norm, RoPE, output gate, output projection, and unchanged surrounding decoder normalization/residual/MLP. Qwen3.5-0.8B has two KV groups in full attention, each serving four query heads. Select a fixed subset `exact_groups` at the *KV group* level:

* **Exact group:** run native causal PyTorch SDPA on its four pretrained query heads and one repeated KV head.
* **Approximate group:** instantiate CeNNDeltaCellV3 or CeNNMemoryGraphV4 **only for that group's query heads and KV state**; do not compute the CeNN recurrence on the exact group.
* Concatenate output head tensors in the original pretrained head order, apply Qwen's query gate and `o_proj`.

`exact_groups=()` reproduces a full-CeNN replacement; `exact_groups=(0,)` and `(1,)` isolate which pretrained head group is hardest; `exact_groups=(0,1)` is a **functional oracle** equivalent to native Qwen attention (with `core=None`). The all-exact oracle is never allowed to claim a faster CeNN result.

### V5 controls

1. Delta + Local32, feature dimension 32 (matches the V4 control).
2. Delta + Local64, feature dimension 64.
3. Delta + Local64, feature dimension 96 (capacity control).
4. Exact group 0, Delta + Local64 group 1 (selective rescue).
5. Exact group 1, Delta + Local64 group 0 (symmetric rescue).
6. Exact group 0, MemoryGraph V4 + Local64 group 1 (graph control).
7. Exact group 1, MemoryGraph V4 + Local64 group 0 (symmetric graph control).
8. Exact all-head parity oracle (no training and no eligibility).

All trainable replacements use identical training steps and the same train split; model selection is done on validation. The notebook does a **single held-out test audit after selection**. Full next-token NLL and original-model prediction agreement are measured at all valid token positions, plus paired native/student prefill GPU timing. The `PROFILE='quick'` settings (256 tokens and 140 training steps) are for a screening pass; `research` raises the workload to 512 tokens and 600 steps but does not replace multi-seed evaluation.

## Research success criteria

Local output cosine >=0.95, teacher token agreement >=0.98, validation NLL delta <=+0.10. A deployment acceptance additionally needs held-out NLL within the selected gate and a paired whole-model prefill speedup >1.05x. These are user-chosen heuristic thresholds, not scientific significance tests. Native Qwen remains the recommended model until all required checks pass. Additional tests: multiple independent seeds, 1024–4096+ context, state memory, and separate cached-token decode speed once an integrated cache exists.

## Critical limitations

- Only **Qwen3.5 full-attention layers** are compatible. Do not graft into the different Gated DeltaNet linear-attention layers.
- Prefill `use_cache=False` only; no `.generate()` KV caching, no beam search or multimodal training.
- Fixed, unpadded sequences only; user-supplied attention masks are restricted to standard full-prefix causal masks.
- Exact head-group rescue retains quadratic attention and may **increase** latency at short contexts; Torch SDPA on fully exact Qwen is already well-optimized.
- The delta and graph recurrences remain eager FP32 Python chunk loops with triangular solves; Triton fusion and inference kernels need separate validation before speed claims.
- Existing V4 weights trained with two KV groups do not load unmodified into a one-group V5 core; this is a new local distillation experiment, not a warmstart from the old ZIP.
- The exact-attention pattern resembles hybrid attention, which has prior art; the selective rescue idea alone is not an academic novelty claim.

## Files

* `src/tinycenn_lm/cenn_head_rescue_v5.py` — reusable pretrained-Qwen head selector / attention adapter.
* `tests/test_cenn_head_rescue_v5.py` — CPU exact-parity, causality, gradient and failure-mode tests.
* `notebooks/Qwen35_08B_CeNN_HeadRescue_V5_Quality_First_Colab.ipynb` — Google Colab Qwen3.5 experiment.

Run tests: `pytest -q tests/test_cenn_head_rescue_v5.py`.

## Colab reproducibility improvements

The notebook imports V5 directly from the cloned repository rather than embedding another copy. It adds symmetric group-1 graph rescue to permit paired GQA-group comparisons and uses a SHA256 run fingerprint (source commit, model revision, hyperparameters, candidate definitions) to resume only fully completed candidates with matching checkpoints. On a modified run, it retrains rather than silently reusing stale checkpoints.
