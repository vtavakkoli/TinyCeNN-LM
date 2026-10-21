# CeNN Head Rescue V5.3: Cross-calibrated Qwen3.5 mixer recovery

## Empirical motivation

The uploaded V5.2 **quick** run on Qwen3.5-0.8B layer 11 completed. `v52_joint_w128` reached local cosine **0.96415**, validation next-token teacher agreement **0.97467** (5,965/6,120) and ΔNLL **+0.00245**. The selected `v52_router_post_w64` failed the 0.98 teacher agreement gate, and the prefill timing audit reported **0.9737×** native speed. No candidate achieved a validated speedup. These are user-provided results, not GPU tests of V5.3.

## Change in V5.3

`CeNNHeadRescueV53` extends the repository's V5.2 recurrent/local readout with two *pointwise causal* calibration mechanisms on the **non-rescued** Qwen attention KV group:

1. **Query–output cross readout.** Concatenate `normalize(query)` and `normalize(causal_recurrent_plus_local_output)` and project through a small per-head SiLU low-rank residual path. The output factor starts at zero, guaranteeing exact warm-start behavior at any identical context/window.
2. **Per-head-channel output calibration.** A trainable log-scale initialized to zero and additive bias initialized to zero. The calibration path begins at identity and cannot introduce future-token dependence.

Qwen's original Q/K/V projections, Q/K normalization, RoPE, output gate and output projection are frozen and reused by `Qwen35HeadRescueV5`. Original KV group 0 remains exact SDPA. Group 1 uses the CeNN V5.3 core. The model still **does not implement the Qwen hybrid KV cache**, masked padding or a fused recurrent CUDA kernel.

## Controlled experiments

- `v52_frozen_w128`: no new training, serves as direct V5.2 reference.
- `v53_cross_w128`: train only the new low-rank cross path and per-channel calibration.
- `v53_router_cross_w128`: also train local/global query gating.
- `v53_joint_w64`, `v53_joint_w128`, `v53_joint_w256`: train the full approximate group with different local retrieval widths.
- Optional `v53_joint_w128_plus_kl`: uses native Qwen teacher logits on **train-only** inputs and must requalify on validation. Disabled by default, because V5.2's KL did not improve its validation teacher agreement.

The notebook automatically trains its V5-compatible Delta+Local64 baseline, then a V5.2 joint-W128 reference. All V5.3 variants warm-start from that locally trained V5.2 checkpoint. There is **no file upload, no Google Drive dependency, no claim of recovering exact V5.2 results from the supplied ZIP**. Fingerprints prevent silent local checkpoint re-use after code/model/config changes.

## Evaluation and acceptance

Models are ranked on validation; only after selection is a predeclared disjoint WikiText-2 **train tail** from token offset 1,250,000 accessed. This is an internal diagnostic and not an independent blinded external test; WikiText-2 test was consulted in earlier experiments and is not used here. Validation/diagnostic gates: local cosine ≥0.95, top-1 teacher agreement ≥0.98, next-token ΔNLL ≤+0.10. Passing models additionally require paired end-to-end prefill speedup >1.05×, measured on the same GPU. If gates fail, retain native Qwen attention. Long-context streaming, cached autoregressive decode, BF16/FP16 parity, multi-seed replication and external tasks remain future work.

## Validate

```
pytest -q tests/test_cenn_head_rescue_v53.py tests/test_qwen35_v53_notebook_contract.py
```

## Launch

`https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V53_CrossCalibrated_Colab.ipynb`

The reference module lives at `src/tinycenn_lm/cenn_head_rescue_v53.py`; the Colab does not duplicate its source. The notebook requires an NVIDIA CUDA runtime and automatically exports local checkpoints/metrics as a ZIP download. CPU tests do not establish GPU correctness or real-model improvement.
