# CCM-v2: conservative multi-cell consensus — SmolLM2-360M-Instruct research Colab

[**Open research Colab**](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/research/competitive-cenn-feasibility/notebooks/CCM_v2_SmolLM2_360M_Consensus_Chat_Colab.ipynb)

## Motivation / grounded benchmark

The preceding CCM-v1 synthetic one-hot lookup converged on **sharp** attention-like recall (16 cells: `test_mse=9.6196e-7`, 32 cells: `test_mse=4.2893e-6`, at 400 training steps). But the actual **original first attention module** in `HuggingFaceTB/SmolLM2-360M-Instruct` was diffuse. From user-supplied `ccm_cenn_feasibility_results.zip` (3 textual probes per context, first 15 heads; teacher only, no candidate training):

| Mean metric | T=128 | T=256 |
|---|---:|---:|
| Top-1 probability mass | 0.1036 | 0.0754 |
| Top-4 mass | 0.2339 | 0.1661 |
| Top-8 mass | 0.3460 | 0.2436 |
| Top-16 mass | 0.5063 | 0.3563 |
| Effective context size | 56.0 | 109.8 |
| Oracle top-16 output relative MSE | 0.0788 | 0.1297 |

This is evidence **against** winner-take-all routing on first-layer natural-language attention. The oracle numbers are computed on pre-output-projection attention values and do **not** bound whole-model NLL.

## Architecture

`src/tinycenn_lm/competitive_consensus_v2.py`: causal finite tape with `slots` cells (128 by default); separate learned input-derived query/tag/value projections; local mass-conserving pairwise tree:

\[
\begin{aligned}
 s_{j,t} &= \gamma\operatorname{cos}(q_t,k_j),\\
 g &= \sigma(s_{right}-s_{left}),\\
 v_{parent} &= (1-g)v_{left}+gv_{right},\\
 s_{parent} &= \operatorname{logaddexp}(s_{left},s_{right}).
\end{aligned}
\]

**Mathematical caveat:** this evaluates an exact softmax-weighted sum over its **resident cells**, just through local log-mass merges. Hence it is not evidence of original novelty or a fundamentally non-attentional mechanism. It does not use a standard Transformer attention block, native SDPA, or a DeltaNet/LSTM path inside the replaced layer. The rest of SmolLM2 remains native and the experiment is **hybrid at full-model level**. Fixed `slots` restricts effective history; unlike full attention it cannot access evicted tokens.

The PyTorch research implementation vectorizes prefill via causal unfolded cellular windows (no token Python loop); its O(B·T·slots·width) cost and temporary footprint may be **worse** than optimized native attention on a T4. It is not a fused Triton kernel. Numerical parity to an explicit windowed softmax reference is tested on CPU; causal chunking, grads, BF16 output and adapter rollback are also tested.

## Evaluation protocol

The runner `scripts/benchmark_ccm_consensus_smol360.py`:

1. Load pinned `HuggingFaceTB/SmolLM2-360M-Instruct` (revision `a10cc1512eabd3dde888204e902eca88bddb4951`, Transformers `4.57.6`). Keep every pretrained parameter frozen.
2. Cache identical native first-attention input/output vectors on disjoint WikiText-2 TRAIN and VALIDATION windows; train CCM through local relative MSE + cosine.
3. Optionally refine on TRAIN with native-vs-student logit KL plus small gold CE propagated through the frozen rest of the LLM. This does not train native attention.
4. Choose the best CCM checkpoint strictly by **full-model DEV NLL** (not local cosine, speed, chat demo, or test).
5. Load WikiText-2 TEST only after selection. Compare full-model next-token NLL, KL, teacher token agreement, gold accuracy and local fidelity. Benchmark paired CUDA events for **whole-model prefill**; do not imply token decode speed.
6. Run five deterministic side-by-side chat questions on native and candidate, with exactly the same tokenization/chat template, max output length and no-cache greedy decoding. Additional free-text custom prompt cell is provided.
7. Save best experimental adapter, JSON & CSV training logs, test report, chats and PNG charts, automatically download ZIP. Never publish student model based on qualitative chat examples.

Acceptance only if DEV/TEST ΔNLL ≤ +0.02 nats/token, teacher token agreement ≥0.98, DEV local cosine ≥0.95, and measured full-model speedup >1.0×. If any condition fails, **retain native layer**. Chat answers are qualitative; their exact-match rate is not a factual answer-quality metric. Cached `.generate()` remains unsupported.

### Profiles

| Profile | length | train blocks | dev blocks | test blocks | local steps | end-to-end steps |
|---|---:|---:|---:|---:|---:|---:|
| smoke | 64 | 8 | 4 | 4 | 8 | 0 |
| quick (default) | 128 | 48 | 12 | 16 | 120 | 12 |
| research | 256 | 192 | 32 | 48 | 600 | 64 |

Run `pytest -q tests/test_competitive_consensus_v2.py tests/test_ccm_consensus_smol360.py` for CPU checks. The actual Hugging Face model was not downloaded nor trained during notebook preparation, so neither accuracy nor speed results from CCM-v2 are claimed here.
