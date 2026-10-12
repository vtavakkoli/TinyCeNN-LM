# Jev-CeNN W512 — Selective 3→1 replacement with sealed evaluation

[Launch Google Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_W512_Selective_3to1_Heldout_Colab.ipynb)

## Why this experiment exists

The uploaded all-encoder forced nine-triple compression report showed:
- 28→10 active encoder blocks and ~34% fewer parameters;
- final teacher agreement **65.625%** and student gold accuracy **57.5%** (teacher **88.75%**);
- final B1 at 172 tokens **1.029×** faster, B2 at 335 tokens **0.965×**;
- final joint best model remained **step 0**;
- teacher-input local cosine as high as 0.991 near layers 24–26 but degraded real-prefix cosine (~0.925), revealing severe accumulated drift.

These are grounds for an exploratory **SELECTIVE** alternative, not proof that W512 or CeNN 3→1 is deployable.

## Architecture

For each of nine global/sliding/sliding triples, propose one complete
\`JevCeNNGroupSuperLayer\` using the existing exact W512 native-SDPA stage,
dilated W45/stride23 stage, multiscale CeNN and rank-96 corrective feedforward.
The two subsequent complete teacher blocks become bypasses only if the **whole
current encoder** meets strict acceptance gates. Otherwise the three native
original blocks are restored. The final original encoder layer 27 stays native.

**No group is guaranteed acceptance**. Keeping all 28 native blocks when none
passes the gate is a valid negative experiment, not a bug.

## Four case-isolated partitions (all from the benchmark's TRAIN split)

The source \`LocalLLaMA/typed-decisions\`, config \`all\`, has 1,200 TRAIN cases.
By default, shuffle fixed seed and allocate 700 original training cases,
80 development-probe cases, 140 development-gate cases, and 200 **sealed
holdout** cases. This is 1,120 unique cases; additional cases remain unused.
The original benchmark official TEST split is never used.

Case IDs AND exact \`state\` values are checked for cross-partition overlap.
All questions from a case stay in exactly one partition:

- **TRAIN**: streaming teacher targets + current student-prefix inputs,
  no fixed 64-decision replay loop. Each training sampler reshuffles
  without replacement over all available TRAIN decision examples.
- **DEV-PROBE**: initial 3→1 weights, teacher/prefix cosine and normalized
  mean squared error. Checkpoints selected here.
- **DEV-GATE**: full changed-model KD/gold updates assessed and each
  candidate definitively accepted or rolled back; repeated testing
  here induces model-selection bias.
- **HOLDOUT**: only materialized after the final set of accepted triples is
  frozen. Used exactly once for independent report of decisions, gold,
  final encoder feature fidelity and paired GPU timing, NOT for
  checkpoint selection, gate relaxation, or hyperparameter tuning.

The independent HOLDOUT here means **independent of this experiment's optimizer
and selection**. These are still cases from the dataset TRAIN split:
the original source checkpoint may already have been fitted on that split.
Neither statistical uncertainty nor source-model pretraining history
permits claiming official benchmark TEST generalization.

## Training

1. Freeze the source teacher, the student task head, and all original
   encoder modules. Each proposed superlayer starts with weights derived
   from the native teacher triple.
2. Stage A trains multiscale cellular convolution against the original
   first-layer residual, on continuously sampled TRAIN cases.
3. Stage B trains the W512+W45/stride23 superlayer against the three-block
   teacher output but uses **real current student-prefix inputs every step**.
   It also distills teacher first-layer states as an intermediate signal.
4. The candidate is installed into \`student_eval\` (two complete layers
   bypassed), then only that newly installed superlayer trains against
   the full teacher's decision KL/ranking, gold label CE and final encoder
   hidden-state alignment. All previously accepted groups stay frozen.
5. An objective gate checks **real-prefix local cosine ≥0.99**, real-prefix
   NMSE ≤0.025, **cumulative DEV teacher agreement ≥0.99**, incremental
   agreement loss ≤1 percentage point and both teacher-relative and
   baseline-relative gold losses ≤1.5 percentage points.
6. Compare actual current full-student GPU latency before/after the
   proposal; a \`REQUIRE_CANDIDATE_SPEEDUP\` switch can make speed
   a hard gate. By default speed is reported, not a hard filter,
   because accuracy-first exploration is desirable.
7. If any gate fails, restore **all three original complete layers**.
   Retain metrics and failure reasons without installing the candidate.
   If accepted, save the stage-B + full-model-selected weights.
8. Only at the end, evaluate the frozen structure on the independent
   HOLDOUT and benchmark B1/B2 at short/long padded sequence lengths,
   alternating teacher/student CUDA-event measurements.

## Default experimental settings

\`\`\`python
LOCAL_WINDOW = 512
DILATION_STRIDE = 23
DILATED_RADIUS = 22
CORRECTION_RANK = 96

TRAIN_CASES = 700
DEV_CASES = 220
PROBE_CASES = 80
HOLDOUT_CASES = 200

CENN_STEPS_PER_GROUP = 60
STEPS_PER_GROUP = 450
FEEDBACK_STEPS_PER_GROUP = 100

TARGET_COSINE = 0.99
TARGET_NMSE = 0.025
TARGET_AGREEMENT = 0.99
REQUIRE_CANDIDATE_SPEEDUP = False
USE_FP16_AMP = False
\`\`\`

This is more expensive than static cache reuse, especially on Colab T4,
because it makes fresh frozen-teacher and current-student forwards during
training. It is intentionally a fidelity-first experiment; even W512 with
GPU-native SDPA need not outperform optimized native ModernBERT attention.
FP32 is used by default for numerical stability.

## Output contracts

- \`jev_cenn_selective_PROGRESS.json\` (rejected/accepted proposals,
  development metrics, checkpoint paths),
- \`ACCEPTED_group_XX_YY_W512.pt\` only for committed candidates,
- \`jev_cenn_selective_W512_trials.csv\` (every candidate and rejection reason),
- \`jev_cenn_selective_W512_report.json\` (sealed holdout and all benchmarks),
- \`jev_cenn_selective_W512_quality.png\`,
- \`jev_cenn_selective_W512_EXPERIMENTAL_NOT_DEPLOYABLE.pt\` unless
  holdout quality, speed and nonzero actual compression gates all pass.

**Never overwrite the previously published Hugging Face Jev_CeNN model.**
No official TEST requests or Hub publishing are provided.

## Scientific caveats

- Full global connectivity does not imply exact function or optimizer convergence.
- Selection on DEV-GATE across nine candidates can overfit that particular
  set. A separate HOLDOUT is essential, and repeated tuning after seeing
  HOLDOUT would invalidate its independence.
- Binary accuracy/agreement from a few hundred decisions is noisy;
  several questions from one case are correlated.
- Incomplete compression is preferred to 3→1 quality collapse.
- Per-token cosine is necessary but cannot substitute for downstream
  agreement and gold accuracy.
- CUDA timings, 0.99 agreement and number of accepted candidates remain
  **unmeasured until the user runs Colab on a real GPU**.
