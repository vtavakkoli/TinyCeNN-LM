# Jev_CeNN — Full encoder progressive 3-to-1 compression

[Open the full-encoder Google Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_AllLayers_3to1_Progressive_Colab.ipynb)

## The prior experimental result is not a success claim

The supplied layers-15–17 report has blind-audit cosine **0.969154**, with
CeNN disabled cosine **0.947775** (a +0.021379 conditional benefit).
But end-to-end teacher decision agreement is only **0.741667** and native
three-block CUDA latency **5.775 ms** versus student one-block **18.359 ms**.
It **failed both quality and speed gates**.

The improvement through step 400 suggests longer training is worth testing,
but does not guarantee a 0.99 solution. Even a perfect single triple may
be insufficient when independently compressing all nine triples due to
compounding error.

## Full architecture: 28 → 10 active complete encoder blocks

Original ModernBERT has nine global+sliding+sliding triples:

- 0/1/2, 3/4/5, 6/7/8,
- 9/10/11, 12/13/14, 15/16/17,
- 18/19/20, 21/22/23, 24/25/26.

Each entire original triple (three attention layers, three norm+residual
paths and three FFNs) becomes one Jev_CeNN pair-window superlayer at the
first slot, with **two full-layer bypasses** in the next two slots. Original
global encoder **layer 27 remains unchanged**, with the original final norm
and task head preserved. This is 9 experimental compression units + 1 native
tail: **10 active encoder blocks**.

The superlayer uses two sequential query-dependent sliding-attention
operations (W45 normal then W45 dilated at stride23), two multiscale CeNN
injections before the second attention's QKV and output, and one FFN.
An optional rank-96 zero-initialized output correction learns residual
errors; it increases compute but starts as exactly the original superlayer.

The two-hop receptive field is global to 512 tokens by exact graph
reachability; **that is not a theorem of exact function approximation or
training convergence**. The current differentiable PyTorch
\`sparse_attention\` gather-based implementation has considerable GPU
overhead compared with fused native SDPA.

## Scientific procedure and controls

- Frozen source teacher loaded from Hugging Face via original pinned
  ModernBERT 4.57.6, independent initial student initialized from the
  same original weights. No published Jev_CeNN model is modified.
- Data is case-split into TRAIN and DEV (from original dataset TRAIN),
  then disjoint development PROBE, blind layer AUDIT and downstream DEV.
  Official TEST is never accessed. The full-encoder DEV cases also
  serve exploratory selection for joint fine-tuning, so **full-model
  quality claims are exploratory**, not an independently held-out TEST.
- Batch-wise frozen teacher hooks capture original inputs at every group
  start and outputs after the first and third complete original layers,
  stored as CPU FP16; one full teacher forward supplies all nine groups.
- For group \(g\), pretrain its CeNN to reconstruct original
  first-layer residual, then jointly train only that group's superlayer
  against the original three-block target, including first-layer
  intermediate and CeNN-only auxiliary losses.
- **Drift-aware exposure:** for groups after index 0, capture the *actual*
  hidden states output by the previously compressed student prefix.
  Alternate training samples between the frozen teacher's input and
  those real student-prefix inputs. Both target original frozen
  teacher group outputs on the same case. Probe checkpoint selection
  combines native/rollout errors.
- After each group, install the replacement and report *blind* group-level
  cosine/NMSE and end-to-end cumulative decision agreement/gold accuracy.
  Continue across all groups even when individual groups fail (diagnostic).
- Optional full-encoder joint training uses teacher logits/actions and
  gold cross entropy. It keeps all outside-group layers and task head
  frozen. Checkpoint selection uses DEV data (exploratory).
- Ablate each CeNN after training in the complete compressed model,
  test numerical effect and benchmark original vs compressed full
  models using CUDA events at B1/B2.

## Tuning and replay

Default starting point is **100 CeNN warmup steps + 600 superlayer steps
per triple**, \`CORRECTION_RANK=96\`, \`ROLL_OUT_WEIGHT=0.5\`,
\`COSINE_WEIGHT=3.0\` and \`JOINT_STEPS=120\`. These are **hypotheses**, not
proven optimized hyperparameters. If a GPU runtime is interrupted,
group checkpoints are saved to \`OUTPUT_DIR\`; set it to a mounted
Google Drive folder to survive Colab restarts and leave
\`REUSE_GROUP_CHECKPOINTS=True\`.

Caution: 9 × (100 + 600) optimizer updates, teacher activation caching,
full-model quality checks and joint fine-tuning can be computationally
substantial on a Colab T4. CPU FP16 cache storage may consume gigabytes
of system RAM.

## Deliverables

- \`jev_cenn_progressive_groups.json\`: per-group progress and quality,
- \`group_XX_YY_trained.pt\`: independent restartable full-superlayer state,
- \`jev_cenn_all_groups_metrics.csv\`, \`jev_cenn_all_groups_report.json\`,
- \`jev_cenn_all_groups_summary.png\`,
- \`jev_cenn_full_encoder_EXPERIMENTAL_NOT_DEPLOYABLE.pt\` unless **all**
  quality and performance gates pass.

Accept only if all group blind cosines ≥0.99 and NMSE ≤0.025, full-model
teacher decision agreement ≥0.99 without more than 1.5-point gold drop,
and paired complete-model CUDA speed is faster than native at every
tested shape. *A gate is never silently overridden to allow publication.*

No automatic Hugging Face upload or official TEST usage is provided.
This is a new experimental architecture, not a modification of the
existing verified Jev_CeNN release.
