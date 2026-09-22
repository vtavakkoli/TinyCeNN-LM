# CeNNMixer-v4 context and generation repair

## Diagnosis of the supplied run

The run explicitly reports `seq_len=64`, `quick_smoke=true`, and exploration mode. Its alpha=0.7 result still mixes 30% of the original layer and is not a complete replacement. The reported 48.63% is a parameter reduction in one selected mixer, not the whole 0.8B model or measured latency. At alpha=0.55, the previous smoke gate accepted 86.5% token agreement despite weak generation. At alpha=0 the old checkpoint ordering could rank floating-point differences in an identity-model KL ahead of actual mixer improvement.

## Changes

- Both notebook locations now run the same branch/code, with quality mode and exploration disabled by default.
- 128/256/512-token curriculum, contiguous source windows instead of ten randomly joined paragraphs, and validation at every length. `--seq-len` is the maximum training window, not a promise of the base model's maximum context.
- Smoke mode preserves explicit context and thresholds. It cannot produce `strict_quality_gate=true`.
- Alpha-zero selection prioritizes local mixer fidelity. All stages receive a minimum training budget, even when a lightly blended model initially passes.
- Lower default learning rate; after warmup, distribution/hidden-state losses have more weight than local mixer reconstruction. Teacher feedback on student-generated suffixes runs every four steps, alternating corpus prefixes and synthetic retrieval chat prefixes. Chat templates disable thinking for these short answer tasks.
- Vocabulary losses are computed in checkpointed token chunks to reduce FP32 softmax activation memory. Offline backward completes before on-policy forward/backward.
- Worst-context numerical metrics, CE gap, teacher-relative exact factual/retrieval answers, nonempty output, and repetition checks control progression. Token-set Jaccard is descriptive, never a quality score. This is a small development suite, not a broad instruction-following benchmark.
- A separate Wikitext test split and fresh retrieval keys are evaluated after full replacement; smoke runs cannot certify final quality. The general prompts remain development prompts and are not held-out instruction data.
- Failed/zero-update stages remain checkpointed; resume re-enters the specified stage instead of skipping a failed stage. Resume restores weights, not optimizer/RNG/history.
- Retains the runtime branch's FP32 recurrent state, short-prefill convolution buffer padding, finite-loss checks, and gradients through frozen original layers.

## Research basis and scope

1. [GKD, ICLR 2024](https://arxiv.org/abs/2306.13649): student-generated prefixes address the mismatch between teacher-forced training and autoregressive inference. Here this motivates more frequent on-policy suffix distillation; we do not claim to reproduce its full training setup.
2. [LoLCATs, ICLR 2025](https://github.com/HazyResearch/lolcats): transfer of mixer behavior followed by language-model adaptation motivates separating warmup from output-focused adaptation. This patch does not implement LoRA or reproduce LoLCATs.
3. [HALO/HypeNet, January 2026](https://arxiv.org/abs/2601.22156): converted models require explicit long-context assessment; short-window similarity is insufficient. 512 tokens here is a practical baseline, not long-context certification.
4. [Retrieval-aware distillation, February 2026](https://arxiv.org/abs/2602.11374): motivates retrieval probes and retaining Qwen's existing full-attention layers. This patch does not implement retrieval-head selection.
5. [Gated DeltaNet-2, May 2026](https://arxiv.org/abs/2605.22791): decoupled erase/write is a relevant future architecture ablation. It is intentionally not silently substituted into v4: its published gains do not establish gains for this small compressed student. Current v4 still uses the original coupled delta update.

## Running

Run notebook cells 1–7 with a fresh output directory. Defaults use contexts 128/256/512. For a larger-context experiment, set `SEQ_LEN=1024` and `CONTEXT_LENGTHS='128,256,512,1024'` together. Expect increased runtime and memory; there is no automatic reduction that would mislabel the evaluated context. The Python recurrent scan is a correctness implementation, not a fused high-throughput kernel.

If alpha=0 cannot meet local reconstruction thresholds after its budget, inspect the learning curves and consider larger associative dimensions or more data/updates. Do not loosen final gates merely to reach alpha=1. Both models remain resident during training. Full pretrained Qwen convergence, Colab GPU memory, generation quality, and speed require an actual GPU run; CPU regressions cannot establish these outcomes.

## Validation performed

18 CPU regression tests pass with PyTorch and Transformers 5.17.0. They cover full-sequence/cached equivalence (including short prefill), real tiny-Qwen gradients and removal of the original mixer, chunked-versus-dense loss values and gradients, context curriculum bounds, generation-gate failure, warmup checkpoint ordering, synchronized executable notebooks, and a tiny-Qwen runner integration through offline/on-policy optimization and final evaluation. The integration fixture bypasses quality thresholds to exercise the finalization path; it does not claim model quality.
