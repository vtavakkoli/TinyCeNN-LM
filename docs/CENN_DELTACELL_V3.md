# CeNN-DeltaCell V3: cellular-coupled gated delta memory (research prototype)

**Status:** experimental implementation and CPU-numerical tests; not a validated Qwen3.5 checkpoint, not an established novel result, and not a GPU speedup claim. The Colab runner produces actual GPU measurements only when executed.

## Motivation

Prior Qwen3.5-0.8B CeNN-X2 middle-layer screening did not satisfy local-activation fidelity gates. Enlarging the number of unrelated recurrent memory banks yielded little evidence of benefit. V3 instead tests a **controlled new transition on the persistent recurrent memory state**, retaining the pretrained projections of Qwen's full-attention layer.

## Core

- **Query/key/value:** original Qwen3.5 full-attention projections, query gate, QK RMSNorm and RoPE are reused by `Qwen35DeltaCellAttention`; original output projection and unchanged decoder residual/MLP remain intact.
- **Delta update:** reuse the independently tested `delta_recurrence` reference implementation, using fp32 recurrent state and bounded chunks up to 32 tokens. The key/value memory is `[batch, KV_heads, feature_dim, value_head_dim]`. Query-to-KV groups are preserved.
- **Cellular transition at fixed *global* chunk boundaries:** after tokens numbered `K, 2K, ...`, mix each feature-cell with its left and right neighbors. Boundaries are tracked in `DeltaCellV3State.position`, guaranteeing that arbitrary causal streaming call boundaries produce the same results as one full pass.
- **Three ablation modes:** `none` (pure gated delta, no diffusion), `fixed` (fixed convex coefficients 1/4, 1/2, 1/4), and `adaptive` (coefficients predicted from the contents of the recurrent state). The local branch is independently switchable using `local_window=0` or a bounded `local_window=32` readout.

For a cell index `f`, when a global chunk has been processed:

\[
\tilde M_{b,f}= w^-_{b} M_{b,f-1}+w^0_{b} M_{b,f}+w^+_{b} M_{b,f+1},
\quad w^-+w^0+w^+=1,\;w^\cdot\ge0.
\]

\[
M^\mathrm{new}_{b,f}=(1-\lambda)M_{b,f}+\lambda\tilde M_{b,f},
\qquad 0<\lambda<1.
\]

Zero boundary cells are used outside the feature-grid range. `adaptive` sets weights by a learned softmax router on the memory's per-cell summary. This is **memory-cell diffusion**, not a depthwise convolution over input tokens and not a change to the native Qwen's pretrained FFN.

## Proposed scientific distinction

The falsifiable hypothesis is that **content-adaptive local coupling of recurrent *memory cells*** improves restoration of pretrained attention outputs or content retention versus an identical gated delta recurrence. A demonstration must isolate the router from alternative changes, measure native Qwen and existing PDelta3-GDN2-CLVR/Local32 baselines, and hold parameter, training and data budgets constant. Convex mixing and delta memories each have substantial prior art; this combination should not be described as original in an academic submission without a careful literature review and independent ablation results.

## Files

- `src/tinycenn_lm/cenn_deltacell_v3.py`: core `CeNNDeltaCellV3`, state `DeltaCellV3State`, and pretrained-projection `Qwen35DeltaCellAttention` adapter.
- `tests/test_cenn_deltacell_v3.py`: causal and streaming numerical tests, ablations, gradients and Qwen-shaped adapter interface. Run `pytest -q tests/test_cenn_deltacell_v3.py`.
- `notebooks/Qwen35_08B_CeNN_DeltaCell_V3_Cellular_Memory_Colab.ipynb`: GPU-only full Qwen3.5 text experiment, validation ranking and independent test audit.

## Constraints and limitations

1. **The module only supports `use_cache=False` in the Qwen adapter.** The standalone core supports streaming states; integrating those with Qwen3.5's hybrid cache and rotary position bookkeeping is future work. `.generate()` cached decode is unsupported.
2. Only Qwen3.5 **full-attention** layers (3,7,11,15,19,23 in 0.8B) are valid targets. Gated DeltaNet layers have a different forward contract and are intentionally untouched.
3. Fixed unpadded causal text sequences only; no masked padding, multimodal inputs, beam search or packed variable-length training. The adapter checks shape and the causal mask on supplied inputs. The notebook supplies fixed unpadded token blocks.
4. The recurrent state and triangular solves are in fp32 even when Qwen uses BF16 or FP16. `torch.linalg.solve_triangular` and Python chunk loops are **reference operations, not fused kernels**. A future Triton/CUDA scan kernel must be independently tested for numerical parity and measured on the same GPU before claiming acceleration.
5. The local branch retains at most `window-1` K/V frames in standalone streaming mode, with no dynamic cache integration. The matrix state's memory consumption is approximately `4 * batch * KV_heads * feature_dim * value_dim` bytes, plus local window cache when enabled.
6. Teacher-output fit alone is insufficient. Acceptance gates demand local cosine, held-out next-token agreement, NLL nonregression, and *measured* full-model latency gain; if none passes, retain the native Qwen module.
7. The notebook's quick mode is screening, not publication-quality statistical evidence. Increase sequence lengths, validation/test examples and seeds; compare to native and pre-established CeNN/PDelta3 references; report full test results once only.
