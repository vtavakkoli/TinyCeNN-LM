# Laya V5.2 — Supervised correction of teacher errors in a compressed model

[![Run V5.2 in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V52_Gold_Aware_Supervised_Colab.ipynb)

## Observed V5.1 results (user's uploaded files)

- **Teacher agreement: 120 / 120 = 100%** on the reused PROBE.
- **Student gold accuracy: 105 / 120 = 87.5%**.
- **Teacher gold accuracy: 105 / 120 = 87.5%**.
- **Gold errors in each model: 15 / 120**.
- Mean prediction distribution JS divergence: **0.002730755**.
- V5.1 early stop: **step 0**, reason `perfect_probe_at_initialization`.
- Compression: **20 selected attention/FFN projections** and **130,596,864
  theoretical FP32 parameter bytes** removed. This is *not measured model size*.
- The uploaded V5.1 files do not include an independently evaluated final
  test metric, GPU benchmark, measured full checkpoint size, or a model upload.

This means V5.1 did solve **teacher fidelity on the PROBE**, but it did
not demonstrate improved task/gold correctness. A loss composed only of
`KL(student || teacher)` cannot encourage the student to contradict a wrong
teacher decision. “Perfect teacher agreement” is **not 100% gold accuracy**.

## V5.2's new experiment

Rebuild the selected compression from a fixed teacher revision. Train against
the dataset's **gold labels** using mostly hard-label supervised loss, plus
soft gold targets and a smaller teacher-consistency loss. By default, train
the final scorer and the question-type embeddings, not all 28 pretrained
layers. Optionally train selected compressed SVD factors for more capacity.

Training and selection use disjoint subsets of the dataset's train split:

- 500 cases for supervised training
- 120 historical PROBE cases, diagnostic only
- 120 additional cases, used repeatedly for **gold validation and checkpoint
  selection**. These were previously part of V5.0's audit, so they are
  **not a never-observed cross-study holdout**.
- 200 gold-labeled decisions from the separate dataset `test` split,
  evaluated after freezing the model. Although the test split is separate,
  it may have been seen in previous experiment reports.

The experiment **stops only after a material gold-label accuracy gain**, not
after 100% agreement with teacher argmax choices.

Important implementation detail: label smoothing is computed only across
**valid** answer options; smoothing across padded action markers with
`-1e9` logits would create an invalid enormous cross-entropy loss.
An executable numeric regression test is included.

The final report distinguishes corrected teacher mistakes from newly introduced
mistakes, including by question type. Publication is blocked unless **both**
gold-validation and separate test-split gold accuracy exceed the teacher by at
least 1 percentage point (configurable), the standalone full checkpoint is at
least 20MB smaller, and offline model reload parity passes. GPU latency is
measured and disclosed but not automatically assumed faster.

Conditional, separate, **private** destination:
`vtava/Laya-v52-gold-aware-standalone`. No previously uploaded model is
modified unless the user explicitly opts to replace the V5.2 destination.

**Status**: Code committed; GPU outcomes and Hugging Face model publication
are not yet verified. Failure to improve gold accuracy is a valid result
and prevents a claim of better task performance.

[Core standalone runtime](../laya_v50_all28/modeling_laya_v50.py).
