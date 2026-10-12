# Jev-CeNN: Global → Window → Global → Window (600 steps)

[**Run experimental Google Colab**](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_GlobalLocal_x2_600Step_Colab.ipynb)

## Why change the architecture

User's previous selective W512 experiment for teacher layers 0–2 showed the
feature-distillation probe cosine rising from **0.81652 to 0.96557 at step 450**.
During full-model task recovery, agreement was **92.9167% at step 1**,
then **92.0833% at steps 20 and 40**, with a worse gold score. It did
not meet ≥0.99 teacher agreement. Simply extending the previous two-pass
architecture is not assured to fix the information deficit.

## New candidate

The user run achieved local cosines **0.99150** at group 0–2 and
**0.99652** at group 3–5 but the corresponding best recorded recovery
teacher agreements were only **97.5%** and **98.75%**, respectively.
This motivates a more stable training progression, without implying
the 0.99 task-agreement gate has been met. Starting with four full passes
may cost more FLOPs during the initial 100 updates. It is intended to
make global contextual learning easier, not guarantee lower wall-clock time.

Original 3 teacher blocks: \`Global+FFN → Sliding+FFN → Sliding+FFN\`.

Proposed **ONE** candidate replacing the entire triple:

\`\`\`text
Input hidden states
  -> full-context QK^T+softmax+V          # pass 1; all valid tokens
  -> windowed W129 QK^T+softmax+V       # pass 2; +/-64 tokens
  -> multiscale CeNN residual + FFN
  -> full-context QK^T+softmax+V        # pass 3; all valid tokens
  -> windowed W129 QK^T+softmax+V       # pass 4; +/-64 tokens
  -> multiscale CeNN residual + FFN
  -> rank-64 zero-initialized correction
Output hidden states
\`\`\`

Each QKV/rotary/out stage is initialized from a cloned original global
or sliding-attention teacher module. The repeated global stage is
an *independently trainable clone*. CeNN is applied after both full+sliding
pairs; two full FFNs come from the original sliding blocks.

**Global means all non-padded key tokens**, including edges: this corrects
the W512 centered-window convention that failed to connect token 0 with
token 511. Both global and sliding attention use fused-capable native
PyTorch SDPA, not Python K/V gather.

**Warning:** four attentions + two FFNs may be *slower and/or larger* than
three original transformer blocks. This is an accuracy-oriented ablation,
not a proven compression in FLOPs. It still reduces active encoder *block
depth* from three to one for each accepted group. The notebook measures
real latency and total parameters; acceptance optionally requires >1x speed.

## Faster training: 600 total updates maximum per candidate

\`\`\`python
TOTAL_OPTIMIZER_STEPS = 600
LOCAL_DISTILL_STEPS = 520
FULL_MODEL_RECOVERY_STEPS = 80
CACHE_BATCHES = 96
BATCH = 2
\`\`\`

The new **attention curriculum** explicitly starts with full attention in
all four passes, then gradually makes the second and fourth passes local,
**without replacing the learned QKV/FFN/CeNN weights**:

| Local optimizer updates | Training attention topology |
|---|---|
| 1–100 (100 updates) | FULL → FULL → FULL → FULL |
| 101–240 (140 updates) | FULL → W129 → FULL → FULL |
| 241–520 (280 updates) | FULL → W129 → FULL → W129 |
| 521–600 (up to 80 updates) | Full-student task recovery with both W129 windows |

Checkpoint selection for *local distillation* considers **only** the final
full/W129/full/W129 phase. Phase 0 and 1 checkpoints can never be deployed,
even if they achieve higher cosine before the window conversions. The
curriculum phase is preserved in the model's state dictionary, so an
accidentally resumed full-window checkpoint cannot silently claim to be
the final two-window architecture.

For each group, cache up to 192 distinct TRAIN decisions and fixed DEV-PROBE
decisions with **one frozen teacher forward plus one current student prefix
forward per batch**, in CPU FP32. The cache is unique to that group and
discarded before the next. **520 distillation optimizer steps run only the
candidate**, alternating mostly *actual changed-student prefix input* and
occasional teacher input. The targets are original teacher outputs after
the full three-layer segment, plus a first-pair auxiliary teacher output.

After local checkpoint selection on DEV-PROBE **in final sliding-window
mode**, install the full candidate. Run **up to 80** full-model recovery
updates using frozen teacher decision distillation, gold CE and final-encoder
hidden alignment. Save/restore the best DEV-GATE checkpoint; early stop
when recovery consistently worsens. **The selected recovery checkpoint must
not reduce teacher agreement from the initial pre-recovery checkpoint.**
No separate CeNN warmup optimizers or uncounted updates are made.

The dataset split is case-disjoint 700 TRAIN cases, 80 PROBE cases, 140
DEV-GATE cases and 200 final sealed HOLDOUT cases. The HOLDOUT is not
materialized or inspected for hyperparameter or candidate selection. All
cases come from the benchmark TRAIN split, and thus cannot establish an
official benchmark TEST result independent of the pretrained teacher.

## Selective deployment gate

For every candidate triple, compare against the entire **currently**
compressed model. Require real-prefix cosine≥0.99, NMSE≤0.025,
cumulative teacher agreement≥0.99, increment agreement loss≤0.01 and
bounded gold-accuracy loss. If failed, restore the 3 original teacher
encoder blocks. Continue checking other groups; accepting zero groups
is a legitimate negative outcome. A GPU speed gate is optional for
selection; final independent benchmark B1/B2 at short/long shapes
always runs.

## Outputs

- \`full_local_x2_progress.json\` — attempted, accepted and rejected groups
  (including measured 600-step budget and direct GPU comparison)
- \`accepted_full_local_x2_XX_YY.pt\` — only accepted candidate states
- \`jev_cenn_global_local_x2_600_report.json\`
- \`jev_cenn_global_local_x2_600_trials.csv\`
- \`jev_cenn_global_local_x2_600_plot.png\`
- \`jev_cenn_global_local_x2_600_EXPERIMENTAL_NOT_DEPLOYABLE.pt\` unless all
  quality, speed and nontrivial-compression gates pass

The official TEST is not opened. The Hugging Face published Jev_CeNN
model is not modified. 0.99 cosine/agreement and speedup are
**unmeasured** before an actual GPU Colab run.
