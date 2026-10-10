# Jev-CeNN attention-free feasibility study

**Notebook:** [Open in Google Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_Attention_Free_Global_Sliding_Feasibility_Colab.ipynb)

This is an experimental research lab answering a narrower question than an
all-30-layer conversion: **can ONE original global layer and ONE original
sliding layer be replaced by a truly attention-free CeNN mixer while reaching
≥0.99 held-out output cosine?**

## Architecture variants

| Variant | Token-mixing paths | Scope |
|---|---|---|
| `local` | Parallel depthwise 7-, 33-, 65-token receptive fields | Bounded local |
| `multiscale` | Same plus 129-token dilated receptive field | Wider local |
| `memory` | Same plus prefix/suffix/global cumulative averages and coarse pooling | Global information |

Every candidate copies the pretrained QKV and output projection of the specific
teacher layer. Q and K are permitted *only as pointwise gating features*. The
new model contains **no** original QK dot-products, token-to-token affinity
matrices, SDPA, linear attention, landmark attention, or tokenwise softmax.
A fixed number of cellular branches are normalized at each token, which is
**not attention over sequence positions**. Convolutions and summary
statistics scale linearly with sequence length. The modules are CeNN-inspired
learned cellular/depthwise mixing, rather than complete classical continuous-
time Cellular Neural Networks.

## Experiment protocol

- Frozen teacher: `convaiinnovations/laya-typed-decisions`, pinned
  `transformers==4.57.6`, typed-decisions `train` cases only.
- Representative original **global layer 15** and original **sliding layer 16**.
- Exactly the same cached original-teacher inputs/outputs and training budget
  across all 3 architectures for each selected layer.
- Cases split into training, independent development probe, independent blind
  audit and downstream end-to-end development quality.
- Architecture and checkpoint selection use *probe output quality only* and
  probe-based latency. The blind audit is never used to select a candidate.
- Metrics: valid-token mean cosine, NMSE, native/student GPU attention-module
  timings, one-layer downstream teacher agreement and labeled accuracy.
- The official TEST partition is **not** used or consulted. There is no
  automatic whole-model replacement or Hugging Face publication in this lab.

## Success criteria

We report explicit **PASS** or **NOT REACHED** for:

1. Blind-audit local mean output cosine >= 0.99.
2. Blind-audit local NMSE <= 0.025.
3. Independent end-to-end held-out teacher decision agreement >= 0.99.

The broader hypothesis that a full 30-layer attention-free Jev-CeNN would
maintain 0.99 teacher agreement **is not proven by success on individual layers**.

## Execution

1. Use a Google Colab GPU runtime; run from the first cell.
2. `SCREEN_STEPS=250` controls the number of optimizer updates for each of
   the six candidate/layer pairs.
3. `REFINE_WINNER=True` optionally trains only the selected architecture
   longer (`REFINE_STEPS=600`) without changing the audit split.
4. Change `OUTPUT_DIR` to persistent Google Drive if checkpoints need to
   survive runtime resets.
5. Read the printed global and sliding feasibility verdicts and
   `jev_cenn_attention_free_feasibility.json` side by side, along with the
   actual speedup ratio (faster means >1.0).

No 0.99 results are asserted before real GPU Colab execution.
