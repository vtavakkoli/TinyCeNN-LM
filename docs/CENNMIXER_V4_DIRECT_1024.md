# CeNNMixer-v4 Direct-1024

## Goal

Replace one Qwen3.5-0.8B linear-attention mixer with a smaller CeNNMixer-v4 block and train the replacement **directly**, without teacher/student alpha interpolation.

The quality target is function approximation of the original layer first, followed by end-to-end agreement of the frozen language model.

## Why alpha was removed

With an interpolated output

```
y = (1 - alpha) * y_teacher + alpha * y_student
```

small alpha makes model-level KL, hidden-state error, CE, and top-1 agreement look artificially good even when the student mixer is poor. At alpha=0 the language model is exactly the teacher regardless of the replacement quality.

The direct experiment instead uses

```
y = y_student
```

from update 1. The frozen native mixer is evaluated only to produce a local target.

Because only layer 0 is replaced, teacher and student receive the same input at the replaced layer. This makes local function matching well-defined and avoids target drift from earlier converted layers.

## Compact Gated-Delta recurrence

The associative state follows the same recurrence family as Qwen3.5 Gated DeltaNet.

For token t:

```
g_t = -exp(A) * softplus(a_t + dt)
Sminus_t = exp(g_t) * S_(t-1)

q_t = l2norm(q_t) / sqrt(d_k)
k_t = l2norm(k_t)
beta_t = sigmoid(b_t)

prediction_t = k_t^T Sminus_t
error_t = v_t - prediction_t
S_t = Sminus_t + beta_t * k_t * error_t^T
o_t = q_t^T S_t
```

The associative output is then gated by a zero-centered RMS normalization and projected back to the model hidden size.

## Teacher-aligned spectral initialization

Qwen3.5-0.8B uses 16 linear-attention key/value heads with 128-dimensional key and value heads. CeNNMixer-v4 Direct-1024 keeps the 16-head partition but compresses each head to 64 key and 64 value dimensions.

For each head h, let `Wq_h` and `Wk_h` be native query/key projection blocks. We form

```
C_qk = Wq_h Wq_h^T + Wk_h Wk_h^T
```

and take the top 64 eigenvectors as an orthonormal projector `P_qk`.

Then

```
Wq_student = P_qk Wq_teacher
Wk_student = P_qk Wk_teacher
```

uses the same subspace for q and k, which is important because the recurrent read/write rule depends on their inner-product geometry.

Likewise, for native value and output-gate projections:

```
C_vz = Wv_h Wv_h^T + Wz_h Wz_h^T
```

and its top-64 projector `P_v` initializes

```
Wv_student = P_v Wv_teacher
Wz_student = P_v Wz_teacher
```

If the compressed state coordinate is `y_s = P_v y_t`, then `y_t ~= P_v^T y_s`. Therefore the native output projection block `Wo_h` is composed as

```
Wo_student_h = Wo_teacher_h P_v^T
```

The beta projection, decay projection, `A_log`, and `dt_bias` are copied directly because the number of heads is preserved.

Depthwise convolution kernels are initialized by energy-weighted aggregation under the same spectral projectors.

## CeNN correction branch

The local CeNN branch is deliberately a correction rather than the primary initialization.

Neighbor coupling uses a normalized symmetric graph operator so learned coupling cannot arbitrarily amplify the state. Each temporal state uses a contractive EMA update:

```
s_t = d * s_(t-1) + (1-d) * gate_t * candidate_t,  0 < d < 1
```

Three time scales are used. The local branch gain starts small while the spectrally initialized Gated-Delta branch starts at unit gain.

## Direct loss

The local mixer objective combines:

- relative output MSE,
- cosine error,
- temporal-difference errors at lags 1, 4, 16, and 64,
- pooled errors at widths 8, 32, and 128,
- log-RMS amplitude error,
- final-128-token tail error.

The end-to-end objective combines:

- forward KL,
- reverse KL,
- top-k logit-rank matching,
- top-1 margin matching,
- hidden-state relative MSE,
- hidden temporal-difference MSE,
- a small next-token CE term.

The local/global balance is adaptive. When mixer MSE is high, local function matching gets more weight. As the replacement approaches the native mixer, more weight shifts to end-to-end behavior.

## Context policy

Training uses a 1024-token window from update 1. There is no short-context curriculum.

Validation checks 256, 512, and 1024 token windows. The worst context controls reported quality.

Expensive vocabulary-wide distillation losses are evaluated on evenly spaced token positions while direct mixer and hidden-state losses use the full sequence.

## Default compact dimensions

| Quantity | Native Qwen3.5-0.8B linear mixer | CeNNMixer-v4 Direct-1024 |
|---|---:|---:|
| Value heads | 16 | 16 |
| Key dimension/head | 128 | 64 |
| Value dimension/head | 128 | 64 |
| Recurrent state floats | 262,144 | 65,536 |
| Recurrent state reduction | - | 4x |
| Mixer parameters | 10,543,264 | about 8,584,449 |
| Mixer parameter reduction | - | about 18.6% |

This version intentionally gives back some of the earlier parameter reduction to improve the probability of converging closely to the native function.

## Files

- `src/tinycenn_lm/qwen35_cennmixer_v4_core.py`
- `src/tinycenn_lm/qwen35_cennmixer_v4.py`
- `src/tinycenn_lm/qwen35_cennmixer_v4_train.py`
- `scripts/run_qwen35_cennmixer_v4.py`
- `cenn-v4-context-quality/cennmixer-v4-runtime-training/notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb`

## Important limitation

The compact 64-dimensional per-head state is a lossy approximation of the native 128-dimensional state. Spectral initialization makes the approximation principled, but it cannot mathematically guarantee exact equivalence for arbitrary sequences. The notebook therefore measures direct layer error, temporal behavior, end-to-end distributions, retrieval, and final generation after the native mixer is removed.
