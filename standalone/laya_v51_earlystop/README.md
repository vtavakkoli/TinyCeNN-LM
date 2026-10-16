# Laya V5.1 — Publication-ready, perfect-decision early-stop experiment

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V51_All28_EarlyStop_Publish_Colab.ipynb)

## What the actual V5.0 reports show

User-uploaded `all28_bruteforce_scan.json` completed all **130 matrix families** and
**520 individual rank candidates**. The `greedy_combinations.json` report
selected **20 matrices**, predicting **130,596,864 bytes** of FP32
parameter-storage savings (not measured standalone checkpoint bytes).

The uploaded `training_progress (4).json` shows:

| Step | Teacher agreement | Gold accuracy | JS | Quality gate |
|---:|---:|---:|---:|:---|
| 40 | 99.17% | 88.33% | 0.001152 | Pass |
| **120** | **100%** | **87.50%** | 0.001060 | **Pass; first perfect teacher match** |
| 160 | 100% | 87.50% | 0.000920 | Pass |
| 200 | 100% | 87.50% | 0.000882 | Pass |
| 240 | 99.17% | 86.67% | 0.000757 | Pass, but teacher agreement regressed |
| 280–400 | 97.5%–98.33% | Varies | Varies | Fail |

V5.0 prioritized reducing JS among already passing checkpoints and therefore
could select step 240 over step 200 despite losing a perfect teacher match.

## What V5.1 actually does

- **No 520-trial rerun.** It uses the exact 20 selected projection specifications
  from V5.0 and rebuilds their randomized SVD factors using V5.0's
  deterministic family-specific seed.
- Freezes the original 28-layer ModernBERT encoder *except* selected low-rank
  A/B residuals, as in V5.0. V4.6's marker-aware decision head remains intact.
- Defaults to evaluating all 120 PROBE decisions every **10 steps**.
- Stops at the **first twice-confirmed 120/120 top-1 teacher match**, provided
  there is no gold-accuracy drop and JS <=0.003. Training cap: **240** steps.
- Saves the best adapter factors as `best_selected_adapters.safetensors` and
  JSON metadata, supporting safe restart when the original model revision
  and selected-spec fingerprint match.
- Restores the first perfect checkpoint before the holdout assessment.
- Uses `LocalLLaMA/typed-decisions` **separate test split** for the 200-decision
  release audit; this differs from earlier development/audit subsets of train.
- Measures synchronized GPU and wall-clock speed using test-split batches.
- Exports the **entire model**, tokenizer, encoder config and proven
  `modeling_laya_v50.py` offline inference implementation into one directory.
- Keeps Hugging Face publication conditional on validation of holdout
  teacher agreement, gold loss, JS, at least 20MB actual FP32 checkpoint
  savings and offline loader parity. A strict speed nonregression requirement
  is optional; slower compression must never be called a speedup.

The output model is **vtava/Laya-v51-all28-earlystop-standalone**, private by
default. The Colab requires the user's own Hugging Face write authorization
and does not overwrite V4.6, V5.0 or other existing models automatically.

### Important limits

`100%` means **matching the teacher's top-1 choice across 120 PROBE
decisions**, not a perfect probabilistic match or 100% independent gold accuracy.
The repeated PROBE is heavily reused from earlier experiments. Even a new
separate test split is still a limited evaluation of only 200 decisions.
A smaller FP32 checkpoint does not necessarily improve runtime.

The source and exported checkpoint have **not been GPU-tested in this assistant
session**. Run the Colab before claiming any V5.1 performance or publication.

Runtime: [../laya_v50_all28/modeling_laya_v50.py](../laya_v50_all28/modeling_laya_v50.py).
