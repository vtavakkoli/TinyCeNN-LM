# Laya V4.8 — SVD delta-shared attention (experimental)

V4.8 is a fidelity-first follow-up to V4.7. V4.7 tied the two local
attention layers' Wqkv and Wo projection weights exactly; its best 120-
decision PROBE checkpoint reached only **85.0% agreement** with the original
V4.6 teacher and **76.67% gold accuracy**, compared to the teacher's
**87.5%** on the same probe. (Source: experiment training_progress.json.)

## Architecture

Keep the original pretrained 28-layer ModernBERT encoder with its existing
FULL → LOCAL(128) → LOCAL(128) pattern. Each pair has distinct FFNs, layer
norms, residuals and position embeddings. Keep the original V4.6 64-context-
token marker-aware decision head. The **first** local layer retains its
pretrained QKV and output projection weights, while the **second** uses:

    W_second = W_first + A @ B

where the factors A/B are initialized by truncated, randomized SVD of the
original pretrained W_second − W_first. Distillation trains only these
factor matrices; the original encoder weights and score head remain frozen.

- Sweep ranks 32, 64, 128 and 256 on case-disjoint PROBE examples.
- Prefer the smallest rank passing quality constraints; if none passes,
  choose the least-divergent rank for an **unvalidated** training run.
- Select residual weights using PROBE only, then perform an untouched AUDIT.
- Require >=99% teacher agreement, <=1.5 percentage-point gold loss,
  <=0.003 JS divergence, at least 100MB actual FP32 checkpoint reduction
  and no >5% minimum latency regression across benchmark shapes.
- Export a fully standalone model with all encoder, scorer, tokenizer and
  adaptation weights. The offline loader is
  [modeling_laya_v48.py](modeling_laya_v48.py).
- Upload only after all gates pass, to
  `vtava/Laya-v48-svd-delta-shared`; do not modify V4.6.

**Important:** Sharing the base projection saves parameters; computing
the residual A/B adds operations. There is no guaranteed speedup, and
quality recovery is unknown until the Colab is executed.

## Run

[Open V4.8 Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V48_SVD_Delta_Shared_Attention_Colab.ipynb).

Uses [vtava/Laya-1.2-speedup-standalone](https://huggingface.co/vtava/Laya-1.2-speedup-standalone)
as frozen reference model. GPU required for training and benchmarking.

The exported model can run offline after its *own* model files have been
downloaded. Use the provided `load_v48_model(model_dir, device="cuda")`;
the source teacher model is not needed during inference.
