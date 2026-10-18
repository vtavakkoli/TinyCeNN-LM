# Delta-CeNN V5.4: Quality-gated Qwen3.5-0.8B multi-layer experiment

The preceding V5.3 all-24 ZIP **completed** in about 286 s, not after eight steps: `DELTA_STEPS=8` was the deliberately short *per-linear-layer* schedule. Six full-attention layers used 24 seed + 32 V5.3 calibration steps each. `FORCE_ALL_24=True` installed all candidates despite all failing the quality gates. Final audit gave ΔNLL +4.1311 and 0.1737× end-to-end prefill relative to native Qwen. The full experiment is thus an unsuccessful performance/quality substitution, not a successful standalone model.

## What changes

1. Quick profile: Delta-CeNN calibration 80 maximum steps, full-attention seed 60, V5.3 full-attention 96; research profile 320/240/400. Steps are maxima: patience-based early stopping can end runs early.
2. The best available checkpoint is compared against **step-zero**, and checkpoint state is restored after fitting. This avoids systematic loss of initial Delta-CeNN parity due to an unstable learning rate or a bad last step.
3. `FORCE_ALL_24=False`: candidate checkpoints are still trained and exported, but harmful replacements are automatically rejected, preserving the original native layer in the final assembled model. Forced all-layer substitution remains available for explicit diagnostics only.
4. Adds descriptive **real-token fast evaluation** on later validation windows: gold next-token NLL, gold next-token accuracy, native top-1 agreement, and two short unpadded greedy chat probes using `use_cache=False` (this is **not** efficient cached generation or an external blind test).
5. Persists trained adapter checkpoints, stepwise logs, stage summaries, exact source/model fingerprints and fast eval artifacts; browser ZIP download after the last evaluation.
6. Optional Hugging Face **research adapter** export. No HF upload is claimed unless the notebook runs on a GPU with an account write-token. Repo defaults to `vtava/Qwen35-08B-DeltaCeNN-V54-All24-Research`. A research-mode upload is labeled *research only* and distinguishes selected and rejected candidate weights. A standard `AutoModel.from_pretrained` call does not reconstruct the custom layers; use `load_delta_cenn_bundle`.

## Reproducibility and limits

- Only Qwen3.5-0.8B's 18 linear-attention and six full-attention **mixer** modules are candidates; frozen Qwen projections, norms, residuals and FFNs are retained. `one_group` retains 4 native attention heads for each full-attention layer.
- Original Delta-CeNN uses FP32 eager recurrent loops and a separate local-attention component for full attention. No fused CUDA kernel, no Qwen hybrid KV cache, no padding/multimodal and no efficient `.generate()`.
- Real native-vs-student prefill speed is measured; no speed claim is presumed. Quality-gated install can legitimately result in **zero** CeNN layers installed; do not report that as success.
- The old WikiText-2 test data has been previously consulted; new later-validation fast eval and disjoint-train-tail audit are **diagnostic**, not new blinded datasets.
- HF upload cannot succeed with the read-only `vtava` account connector alone. A Colab-side HF_TOKEN with **write permission** is required. If unavailable, the local ZIP contains the adapter artifact and records `hf_upload_status.json`.

## Launch

`https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_DeltaCeNN_V54_All24_QualityGated_FastEval_Colab.ipynb`

## Tests

`pytest -q tests/test_delta_cenn_v54_io.py`
