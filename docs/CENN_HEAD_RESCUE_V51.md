# CeNN V5.1 — Qwen3.5-0.8B no-upload, quality-first head rescue

## Status

**Experimental research notebook, not a validated release.** The Colab now runs from a fresh NVIDIA GPU runtime without uploading any checkpoint ZIP or connecting Google Drive. Neither training nor GPU speedups are pre-computed or claimed. It downloads the public `Qwen/Qwen3.5-0.8B` model and WikiText-2, then trains a compatible V5-style checkpoint automatically before performing V5.1 comparisons.

[Open the V5.1 Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V51_Warmstart_Colab.ipynb)

## Why V5.1?

In the earlier V5 run, the `rescue_g0_delta_f64_w64` candidate preserved native full attention for Qwen's KV group 0 while replacing group 1 with a compact CeNN DeltaCell. Its validation mixer cosine was approximately 0.934, below the 0.95 gate, and its held-out end-to-end latency was slower than native. V5.1 explores query-specific local-memory routing, low-rank correction and joint adaptation to recover fidelity. Those earlier empirical numbers are **historical context**; the corrected no-upload notebook **does not reuse their trained weights** and therefore may produce different results.

## Automatic baseline and common initialization

1. Clone the repository from `main`, install the editable package, record Git SHA and the V5.1 module SHA-256, and run the V5.1 CPU unit tests. The notebook requires a CUDA GPU for model training.
2. Resolve and pin the public Qwen3.5-0.8B model revision for the current run, verify 24 decoder layers with full-attention layer 11, and check the all-exact full-attention wrapper against native Qwen.
3. Construct WikiText-2 training-prefix windows and validation windows. Capture frozen layer-11 native attention inputs, outputs and RoPE tensors once. The disjoint training-tail diagnostic is not evaluated during selection.
4. Train a **new** V5-compatible `CeNNDeltaCellV3` baseline with 64 memory features, a 64-token local window and native exact attention on KV group 0; keep Qwen projections, normalizations, residuals and FFNs frozen. Select the best baseline checkpoint on the **validation mixer objective**.
5. Save the auto-trained baseline under `/content/qwen35_cenn_head_rescue_v51/checkpoints/v5_auto_L11_delta_f64_w64.pt`. Reuse that local checkpoint on repeated executions only if its fingerprint matches model revision, repository revision, calibration data hashes, seed, and training hyperparameters. Otherwise retrain. This is **not** an automatic download of the previous V5 result checkpoint.
6. Train or evaluate five V5.1 variants initialized from exactly the same baseline state, with new router/correction parameters initially neutral: `v5_frozen`, `v51_router_only`, `v51_correction_only`, `v51_router_correction`, `v51_joint`.
7. Optional end-to-end native-teacher KL refinement operates only on calibration/training input, and optional `v51_joint_plus_kl` can be restored by the audit path if selected.
8. Rank candidates on validation (full-token next-token NLL, teacher agreement, local cosine), then audit the predeclared disjoint tail of the WikiText-2 **training partition**, along with alternating paired native/student prefill timings. Report all candidates, checkpoints, configuration hashes, selected architecture, and final acceptance in a downloadable output ZIP.

## Scientific safeguards and acceptance

- The original Qwen model remains frozen. Full-attention layer 11 group 0 uses native SDPA; the approximate group 1 uses the trainable CeNN core only. No cache integration, padding, multimodal inference, beam search, or fused GPU kernel is implemented.
- Validation thresholds are mixer cosine ≥0.95, next-token teacher agreement ≥0.98 and ΔNLL ≤0.10. Final acceptance additionally requires matching fidelity and **>1.05× paired full-model prefill speedup** on the disjoint diagnostic. A failed model is not recommended for deployment even if its language-model loss remains close to native.
- The diagnostic uses the distant *training* tail beginning at token 250,000, **not an independent external test set**; model pretraining may have encountered this content. Additional datasets, seeds, multiple GPUs and long-context/cached-decoding benchmarks are necessary for publication-quality claims.
- Native Qwen GPU timing is not assumed to be slower than CeNN; Python recurrent loops and SDPA may defeat attempted acceleration at short sequence lengths.

## Files and tests

- `src/tinycenn_lm/cenn_head_rescue_v51.py`: reusable CeNN V5.1 core and strict warm-start adapter (`warmstart_v5_delta_core`).
- `src/tinycenn_lm/cenn_head_rescue_v5.py`: frozen native/approximate GQA head-group wrapper.
- `notebooks/Qwen35_08B_CeNN_HeadRescue_V51_Warmstart_Colab.ipynb`: **no-upload** auto-bootstrap GPU research runner.
- `tests/test_cenn_head_rescue_v51.py`: numerical/streaming and warm-start tests.
- `tests/test_qwen35_v51_no_upload_colab.py`: static notebook contract tests preventing regression to mandatory ZIP upload and checking cell syntax, bootstrap staging and optional KL restoration.

Run local static checks with `pytest -q tests/test_qwen35_v51_no_upload_colab.py` and run the full CPU numerical suite with `pytest -q tests/test_cenn_head_rescue_v51.py`.