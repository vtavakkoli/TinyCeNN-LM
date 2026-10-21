# Delta-CeNN V5.6 — controlled four-way ablation on Qwen3.5-0.8B

## Evidence motivating this iteration

From the **user-provided** V5.5 quick/research experiment (`qwen35_delta_cenn_all24_results (3).zip`): all 24 substituted blocks were installed under `forced_all_24=true`, audit teacher-agreement was approximately **91.76%**, audit ΔNLL approximately **+0.00932**, and measured end-to-end prefill speedup **0.788×**. This **failed the quality/speed acceptance criteria**. The single-layer chunk-cell experiment was **1.225× faster than V5.4 tokenwise CeNN**, but approximately 1.5% slower than native Qwen. Two short generation prompts were reasonable sanity checks but not an external benchmark.

It is essential not to interpret 24 completed training stages as 24 quality-accepted substitutions. The V5.5 chunk-cell rule calls **Qwen's original chunkwise gated-delta implementation** between trainable CeNN state couplings. It is not an independently accelerated recurrence or a completely new language-model backbone.

## V5.6 architecture

- **Arm A — native:** Original frozen Qwen3.5-0.8B, all 24 decoder blocks.
- **Arm B — 18 Delta-CeNN:** Replace the 18 `linear_attention` blocks using V5.5 chunk-cell adapters, while retaining the six original full-attention blocks at `[3,7,11,15,19,23]`.
- **Arm C — quality-gated hybrid:** Start from B; individually train all six full-attention V5.3 candidates but install only those passing the cumulative teacher agreement, ΔNLL, local cosine, incremental NLL, and relative speed gates.
- **Arm D — forced 24 (diagnostic only):** Temporarily test all trained full-attention candidates on top of B, restoring the C architecture immediately after each measurement. D is ineligible for final selection.

The original Qwen QKV projections, convolution, output gate, feed-forward and all non-target computations are frozen. V5.3 attention candidates in `one_group` mode retain four exact query heads and approximate the other four; the six full-attention blocks are **not** fully replaced unless `both_groups` is selected.

## Metrics and selection

All arms are evaluated on identical WikiText-2 validation token windows using next-token NLL, gold-token accuracy, teacher top-1 agreement, and paired uncached end-to-end prefill latency. The validation-only selection utility in `cenn_v56_fourway.py` chooses between A, B, or C: B must meet agreement >=0.98, ΔNLL <=0.02 and prefill speedup >=0.90; C must improve fidelity materially without incurring excessive slowdown. Separately, each full-attention candidate must respect incremental ΔNLL <=0.010 and at most 5% relative speed loss in addition to quality gates. A separate **train-tail diagnostic** is performed after selection only, and cannot be used to re-rank candidates.

No speedup or real-world language-equivalence claim is made without independent GPU benchmarks, cached decode support, larger tasks, different seeds and external validation. This notebook has no official `.generate()` integration because the hybrid cache is not implemented.

## Reproduction and artifacts

Open [V5.6 four-way Google Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_DeltaCeNN_V56_FourWay_QualityGated_Colab.ipynb). Run all cells with an NVIDIA GPU. No user ZIP upload or Drive mount is needed. The notebook downloads the public Qwen base model and dataset and starts with `PROFILE='quick'`. Results are saved under `/content/qwen35_delta_cenn_v56_fourway_results/` and exported as a ZIP containing stage optimizer logs, training logs, all candidate checkpoints, four-way validation/audit CSVs, selection JSON, final audit, fast real-model evaluation, research-adapter manifest, provenance and checksums.

**Implementation limitations:** Native Qwen chunked delta core is reused; CeNN adds optional periodic spatial mixing. Speed depends strongly on hardware and availability of `causal_conv1d`/`flash-linear-attention` optimized backends. Colab training has not been run by the authoring environment; CPU tests establish only module/notebook contracts.
