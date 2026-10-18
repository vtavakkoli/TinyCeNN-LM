# Qwen3.5-0.8B: 24-layer Delta-CeNN + CeNN V5.3 experiment

## Ground truth: the prior six-layer run failed

The supplied `qwen35_cenn_v53_all6_results.zip` records that all six full-attention layers (3,7,11,15,19,23) were **forced** to install, despite failing per-stage quality gates. By the sixth substitution, validation native-teacher agreement was **0.87844** and cumulative prefill speedup **0.79589×**. The final internal diagnostic agreement was **0.83711**, ΔNLL **+0.08959**, and speedup **0.79968×**. This is **not** evidence that a six-layer CeNN model is accurate or faster.

## Next question

Can a separate, Qwen-compatible **Delta-CeNN** replace the 18 *linear_attention* (Gated DeltaNet) blocks while CeNN V5.3 continues to replace six *full_attention* blocks? Distinct native APIs MUST be respected.

- Full-attention layers: `Qwen35HeadRescueV5` plus `CeNNHeadRescueV53`. The wrapper preserves Qwen Q/K/V, RoPE, output gate, and (in `one_group` mode) half the exact attention heads.
- Linear-attention layers: `Qwen35DeltaCeNNLinear` plus `DeltaCeNNRule` in `src/tinycenn_lm/delta_cenn_qwen35.py`. Reuses pretrained depthwise causal QKV convolution, Q/K/V, decay/β and Z-gating projections, output GatedRMSNorm and out-projection. **Does not call native `torch_chunk_gated_delta_rule`**. Implements the gated delta update explicitly in fp32, optionally adding positive bounded nearest-neighbor state mixing every `cell_interval` tokens.
- `feature_dim=None` retains Qwen's original key head dimension and is the numerical parity reference. A smaller feature dimension is an experimental compression and should be evaluated independently, not assumed accurate. A full-rank no-cell control can reproduce the tokenwise update without cellular interactions.
- The unoptimized reference is *likely slower* than fused native Qwen. No GPU speedup claim is made. Both model families currently reject hybrid cache-based decode and padding.

## Experimental protocol

The notebook `notebooks/Qwen35_08B_DeltaCeNN_V53_All24_Progressive_Colab.ipynb` runs in layer order 0–23. Each candidate is trained against teacher mixer inputs/outputs **after earlier candidate layers have been installed**, using the frozen original mixer at the current layer. It compares the resulting *cumulative* model with fully native Qwen validation logits and paired GPU prefill latency. The native baseline restores **both** full and delta modules.

The `quick` profile is a short feasibility test. Forcing all 24 is diagnostic only; quality-gate failures are recorded, not hidden. Successful deployment would require local cosine ≥0.95, validation and diagnostic teacher agreement ≥0.98, ΔNLL ≤0.10, paired prefill speedup >1.05×, and all 24 stages meeting accuracy gates. Interpret timings under eager fp32 recurrence cautiously; validate multiple seeds and GPUs before publishing claims.

Training uses WikiText-2 train prefix, selection uses validation, and the post-selection diagnostic uses a **disjoint train tail**. This is not an independent external benchmark and is not the already consulted WikiText-2 test split.

## Automatically exported evidence

The `finally` block creates `qwen35_delta_cenn_all24_results.zip` and asks Colab to download it when training ends or raises a catchable exception. It includes: every-step `training_steps.jsonl`, `training.log`, per-layer checkpoints, `cumulative_layer_results.csv`, `config.json`, `run_status.json`, an artifact SHA-256 manifest, optional plot, `error_traceback.txt` on failure, and `final_audit.json` when completed. A browser may block automatic downloads; fetch the resulting ZIP from Colab's file browser in that case.

## How to run

1. Open https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_DeltaCeNN_V53_All24_Progressive_Colab.ipynb
2. Select an NVIDIA GPU and run all cells; **no uploads**.
3. Start with `PROFILE='quick'`, `REPLACEMENT_MODE='one_group'`, `DELTA_FEATURE_DIM=None`.
4. Review cumulative validation stats and diagnostic audit. Use `PROFILE='research'`, multiple seeds, other tasks, and fused kernels as a *separate* larger validation campaign if promising.

## Limitations

- Native Gated DeltaNet cached decoding and convolution state are not supported by the substitution adapter; `generate()` is not verified.
- The recurrence is eager PyTorch per token, not fused CUDA/Triton.
- The Qwen convolution must use `silu` or `gelu` as coded; a model architecture change raises a clear exception.
- The native-attention parity gate may be backend/dtype sensitive; failing it stops the experiment rather than hiding mismatches.
- Artifact auto-download requires Colab/browser cooperation; a terminated runtime cannot guarantee export.


## Parity fix (native chunk vs token recurrence)

The initial all-24 Colab failed on `delta_parity=0.234375` with an arbitrary
`max_abs<0.15` threshold. The reported Hugging Face `HF_TOKEN` warning is
nonfatal for public Qwen downloads; optional causal-conv/FLA fallbacks impact
performance and kernel numerical order, not download correctness.

The notebook now evaluates **two separate adapters** before training:

- `native_backend_oracle=True, cells=False`: executes Qwen's original
  `torch_chunk_gated_delta_rule` behind the Delta-CeNN projection/gate adapter.
  This is **diagnostic only**, never a substitute for the trainable Delta-CeNN
  recurrence or evidence of new-model speed. It must match native logits within
  the strict adapter parity requirement.
- `native_backend_oracle=False, cells=False`: executes the genuine tokenwise
  FP32 Delta-CeNN recurrence. Its logit drift against the original chunkwise
  implementation is measured with maximum absolute error, normalized MSE, cosine,
  and teacher top-1 agreement, and written to `parity_diagnostics.json`. It
  is an approximation comparison, not a bit-exact interoperability oracle.

All trainable `cells=True` candidates unconditionally use `DeltaCeNNRule` and
are evaluated against strict local, model-level and speed gates. A nonzero
token recurrence discrepancy is not silently accepted as model success.
