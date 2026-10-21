# CeNN V5.2 — Fidelity Recovery on Qwen3.5-0.8B

**Status:** Experimental research implementation. No verified Qwen GPU speedup or improved model quality is claimed. Python reference memory recurrence only; not a cached-generation implementation.

## Motivation from observed V5.1 run

The user's completed 2026-10-17 V5.1 quick-profile archive reports:

| Variant | Validation cosine | Validation teacher agreement | Paired prefill speedup |
|---|---:|---:|---:|
| V5 baseline frozen | 0.934107 | 0.965850 | 0.974× |
| V5.1 router only | 0.949837 | 0.971242 | 0.969× |
| V5.1 router + correction | 0.952282 | 0.968627 | 0.973× |
| V5.1 joint | 0.957484 | 0.971405 | 0.971× |

The validation winner, router only, failed the acceptance gate; on the 6,120-decision diagnostic audit, its teacher agreement was 0.970098, ΔNLL +0.001474 and speedup 0.9533×. Accordingly, V5.2 prioritizes fidelity and explicitly penalizes latency regressions rather than presenting quality improvements as performance wins.

## Architecture

Qwen3.5-0.8B layer 11 remains a native full-attention layer except for **KV head-group 1**, which is approximated. The existing `Qwen35HeadRescueV5` preserves exact attention for KV group 0, original pretrained Q/K/V, QK normalization, RoPE, output gate, output projection, decoder residual and FFN. The new `CeNNHeadRescueV52` extends `CeNNQueryRescueV51` with a low-rank, output-conditioned correction:

\[
\widetilde y_t = y_t + \exp(g) \operatorname{SiLU}(\operatorname{norm}(y_t)D)U.
\]

`D` is initialized randomly, `U` is initialized to zero, making the initial mapping identical to V5.1 at the **same local window**. This is *not* long-range attention recovery; it is a causal pointwise correction conditioned on the existing recurrent/local output. No new projection biases or parameters in native Qwen are trainable. V5.1 q-only correction and query router remain separate controls.

## Controlled V5.2 Colab variants

1. `v5_frozen`: fresh automatically calibrated V5 delta + Local64 baseline.
2. `v51_router_w64`: V5.1 query-router only (matched baseline control).
3. `v52_post_w64`: new post-readout correction only.
4. `v52_router_post_w64`: query router plus post-readout correction.
5. `v52_router_post_w128`: same architecture with doubled local retrieval horizon.
6. `v52_joint_w128`: jointly fine-tuned V5.1 + post correction at window 128.

Optionally, a seventh model `v52_joint_w128_plus_kl` receives *train-only* native-logit KL distillation and is re-ranked by validation metrics.

## Scientific protocol and gates

- **No upload:** pin Hugging Face model checkpoint SHA, clone repository, automatically calibrate an independent V5-compatible baseline on WikiText-2 training windows. Local checkpoints are only reused if the fingerprint matches.
- Match candidate training batches and retain the best checkpoint on the validation partition. Full-prefix Qwen-level NLL and top-1 agreement are computed at every next-token position.
- Prior V5.1 experiments consulted WikiText-2 test and a train tail at token 250,000. V5.2's final diagnostic audit uses a disjoint train tail beginning at token **1,000,000**; this is **not an independent external benchmark** and remains subject to dataset reuse bias.
- Acceptance requires validation cosine ≥0.95, agreement ≥0.98, ΔNLL ≤0.10; final speedup >1.05× and audit agreement ≥0.98 with NLL constraint. Candidates failing quality gates are diagnostic only; keep native Qwen for production.
- Compare latency with alternating native/student GPU measurements and report medians and IQR. Faster eager FP32 recurrence on short sequences is *not* assumed.

## Files

- `src/tinycenn_lm/cenn_head_rescue_v52.py` — reusable causal V5.2 core, strict V5.1 warm-start.
- `tests/test_cenn_head_rescue_v52.py` — CPU state, causal, gradient, parity and checkpoint-contract tests.
- `notebooks/Qwen35_08B_CeNN_HeadRescue_V52_FidelityRecovery_Colab.ipynb` — end-to-end research runner, no file upload.

## Known limitations

Only text-only unpadded full-prefix inference with `use_cache=False` is supported by the original Qwen V5 wrapper. The standalone CeNN core can stream state, but that state is **not integrated into the Qwen hybrid KV cache**. No vision, beam search, variable padding, fused Triton kernel or independent multi-model evaluation is supplied. GPU performance claims require running the notebook on actual NVIDIA hardware, and novelty requires comparative literature review and research ablations.