# Laya V5.0 — All-28-layer brute-force hybrid SVD experiment

[![Open V5.0 Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V50_All28_BruteForce_Hybrid_SVD_Colab.ipynb)

V5.0 is the next experiment after V4.9's successful **selective projection
compression**. The supplied V4.9 logs contain a passing step-120
PROBE checkpoint: **119/120 (99.17%) teacher agreement, 86.67% gold accuracy,
JS divergence 0.000553**. V4.9 also selected seven second-local attention
projections with approximately **35.13 million bytes predicted FP32 storage
reduction**. These results are *PROBE only* and do not establish holdout
accuracy, checkpoint bytes or runtime speed.

## Brute-force architecture search

All **28** pretrained ModernBERT blocks are examined, and the following
candidate projection families are available:

- 28 × attention `Wqkv` and `Wo`: **56 own-weight SVD families**
- 28 × MLP `Wi` and `Wo`: **56 own-weight SVD families**
- 9 local-layer pairs × `Wqkv` and `Wo`: **18 shared-delta SVD families**

Total: **130 projection families**, and up to **520 trials** at ranks
32, 64, 128 and 256. Candidates with no positive weight-storage savings
are skipped. The pretrained schedule FULL / LOCAL(128) / LOCAL(128) and the
validated 64-context-token decision head remain unchanged.

**own-weight compression:** `W ≈ A @ B`

**paired local-attention delta compression:**
`W_second ≈ W_first + A @ B`

Only **selected** projection matrices are replaced; all other original
encoder projections, norms, FFNs, type embeddings, and decision-head
parameters remain exactly as pretrained. Teacher/student distillation
updates the selected A/B matrices only.

## Validation, safety and reproducibility

- Teacher: `vtava/Laya-1.2-speedup-standalone`, revision recorded.
- Case-disjoint TRAIN / PROBE / AUDIT sampling, using the same SEED=2047
  as V4.7, V4.8 and V4.9 for comparison.
- Each candidate is individually screened using a 24-decision PROBE subset.
- Only combinations that pass the entire 120-decision PROBE are greedily
  selected (>=99% teacher agreement, <=1.5pp gold drop, JS<=0.003).
- Train selected A/B matrices for up to 400 steps and choose best PROBE
  checkpoint; one final 200-decision AUDIT is untouched until selection.
- Measure paired CUDA and wall latency and actual FP32 checkpoint byte savings.
  Requires >=20MB saving and no >5% regression on either minimum speed ratio.
- Save full offline standalone weights, tokenizer, encoder config and this
  exact loader; assert local-reloaded logits and predictions match.
- Publish to `vtava/Laya-v50-all28-bruteforce` only if all gates pass,
  using the Colab user's explicit HF write authorization.

## Reports

- `all28_bruteforce_scan.json` — each tested projection/rank and score
- `greedy_combinations.json` — accepted/rejected combinations
- `training_progress.json` — distillation and PROBE selection
- `v50_report.json` — final metrics, actual checkpoint size, audit and gates

**Limits:** The search exhausts the *specified* matrix families and ranks, not
all possible model architectures. Hundreds of candidate trials can overfit
PROBE; the independent AUDIT mitigates but does not eliminate this concern.
SVD rank compression saves parameters but can add matrix operations and does
not guarantee faster inference. GPU success is not yet verified.

Standalone runtime: [modeling_laya_v50.py](modeling_laya_v50.py).
