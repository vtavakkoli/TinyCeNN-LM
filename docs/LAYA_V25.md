# Laya V2.5: sliding CeNN and multiscale recurrent CeNN

[Run in Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V25_WindowDelta_CeNN_Decision_Colab.ipynb)

The canonical notebook now trains sliding CeNN instead of windowed Delta. Its filename stays unchanged to preserve links. This revision is CPU-tested; full GPU accuracy, convergence and latency remain unmeasured. The notebook clears historical outputs and uses a new output directory, `/content/Laya_V25_SlidingCeNN`.

## Latest saved run and motivation

The [2026-10-06 Delta run](../notebooks/archive/Laya_V25_Delta_recovery_20261006.ipynb) passed FLA preflight and completed all 28 local transfers. Recovery stayed finite, with an AMP overflow correctly skipped. It reached 55.6% teacher agreement at the final step; the selected checkpoint had 55.4% agreement and 0.06993 mean JS. Sliding layer 25 remained a weak local fit (cosine 0.7055, NMSE 0.5320). Gold training was saved only through step 200, so this run has no completed held-out test or speed result. These results support testing a different local mixer; they do not establish that the Delta implementation was broken.

The [earlier T4 run](../notebooks/archive/Laya_V25_T4_failed_recovery_20261005.ipynb) developed NaNs in recovery before the numerical safeguards were added. Both historical notebooks preserve their original outputs.

## Architecture

| Component | Current V2.5 replacement |
|---|---|
| Sliding encoder attention | SlidingCeNN: 384 cell channels, bidirectional templates, two recurrent updates, maximum context offset ±64 tokens |
| Global encoder attention | MultiscaleCeNN: 256 channels, scales 1/4/16, two recurrent updates, global masked-context broadcast |
| Final decision head | Shared option MLP conditioned on CLS, sequence mean and option mean |
| Encoder FFNs, norms and residuals | Preserved from the source checkpoint |

SlidingCeNN projects each token into a trainable cellular state space. Value channels transferred from the teacher are spread across attention heads, and both input/output projections are trainable. The input is normalized and masked before spatial mixing. Starting from zero state, the cellular update is

`x_next = (1 − eta) * x + eta * tanh(A * x + B * u + bias)`.

Here `*` is a depthwise sliding convolution and `eta` is a learned sigmoid leak. Each channel's feedback template A has L1 norm at most 0.9, so its recurrent feedback is contractive for fixed input and parameters. This bound does not guarantee stable optimization or higher accuracy. B is a learned input template; a learned pointwise readout path preserves unsaturated token information before the output projection. Recurrent accumulation and template normalization use FP32 under AMP.

With `window=128` and two updates, A and B each have radius 32 (65 taps). The complete layer has an exact maximum receptive-field radius of 64, or 129 positions including the center. It operates across the sequence without chunks, overlap-add, wrapping or a token-by-token scan. Relative positions are encoded by learned left/right template offsets. Masking after every update prevents padding states from feeding back into valid tokens. Appending padding or shifting a sequence with masked padding leaves its valid outputs unchanged.

Global CeNN uses fixed pooling bins plus global masked pooling to provide distant context. The final `LinearMaskEncoder` bypasses the native quadratic attention-mask builder and skips unused RoPE computation for the all-CeNN path. No QK attention matrix, Delta recurrence, FLA kernel, or Transformer decision head is used by the final model. The original head remains only during intermediate recovery stages. Legacy Delta classes/configurations remain loadable for existing checkpoints.

Sliding convolution work scales linearly with sequence length for fixed cell width, window and recurrence count. Actual speed depends on the GPU and kernels; the notebook measures it rather than assuming an improvement.

## Training schedule

At the user's request, **every normal and gold training phase starts at 0.01 and ends at 0.002**, using exact cosine decay without warmup. This applies to both core and head optimizer groups. Stage counts, batch size, data splits and the order of training are preserved. V2.4's separate configuration is unchanged.

| Stage | Steps | Initial → final LR |
|---|---:|---|
| A: progressive local replacement | 400 per encoder layer | 0.01 → 0.002 |
| B: teacher distillation with original head | 1600 | core and head: 0.01 → 0.002 |
| C: direct gold supervision | 900 | core and head: 0.01 → 0.002 |
| Compact pooled-head distillation | 700 | 0.01 → 0.002 |
| Compact joint distillation | 400 | core and head: 0.01 → 0.002 |
| Compact direct gold supervision | 700 | core and head: 0.01 → 0.002 |

These rates are aggressive experimental settings, not evidence that learning will be faster or better. Shared safeguards reject non-finite losses/parameters, skip AMP gradient overflows with scale backoff, stop persistent overflow and clip gradients before updates. FP16 starts at scale 128. Every recovery/head phase retains the stage-entry dev checkpoint if later training does not improve its selection criterion. Gold selection is lexicographic accuracy, KL, then Brier on the full dev set. Stage B retains marker-representation alignment with weight 0.05.

Stage A saves safetensors with source revision, architecture and training metadata. Old Delta checkpoints are incompatible with this architecture and must not be resumed; the new output directory and metadata checks prevent accidental reuse. Set OUTPUT_DIR to a persistent mount to retain new progress across Colab sessions.

Transformers 4.57.6 is pinned. Neither this notebook nor its all-CeNN export requires FLA. The preflight checks local/global CeNN forward agreement with FP32, padding invariance and finite gradients on the actual GPU/AMP dtype before training.

## Evaluation and conditional Hugging Face export

The unchanged teacher and full/compact students are compared on the official test split after training. The final pooled-head model must meet all default gates:

- Accuracy no more than 0.02 below the teacher, including the paired 95% case-bootstrap lower bound.
- At least 1.10× median speedup on the same five length-stratified dev batches.
- Finite metrics and matching paired evaluation sizes.
- No original encoder attention, Transformer head or Delta layers.
- Local save/reload logits and action outputs agree within mixed-precision tolerance.

A rejected run saves its JSON report and skips upload. A passing run saves weights, tokenizer, architecture/source configuration, metrics, self-contained runtime files and a model card. The export reloads without downloading the teacher before any Hub write.

`PUSH_TO_HUB=True` enables conditional upload. With an empty `HF_REPO_ID`, the default destination is the authenticated account's `Laya-V25-SlidingCeNN` repository. New repositories are private by default; existing visibility is unchanged. Authentication uses the Hugging Face session, a Colab `HF_TOKEN` secret, or notebook login. Credentials are not included in artifacts.

The export provides `load_v25`, not `AutoModel.from_pretrained` or a text-generation pipeline. Install its requirements and run from the downloaded directory:

```python
from tinycenn_lm.decision_v25 import load_v25
model = load_v25(".", device="cuda")  # device="cpu" also supported, without FLA
```

Pack typed decisions with `build_sequence`/`collate_items` from the bundled `standalone_decision` module. CUDA inference should use the validated FP16/BF16 autocast dtype.

## Validation limits

Tests cover exact local support in both directions, no chunk-boundary artifacts, padding/empty-row isolation, finite mixed-precision gradients, a learned local operator at the requested rates, the actual notebook training stages on tiny CPU models, preserved FFNs, stage resume, and self-contained save/reload for both new CeNN and legacy Delta checkpoints. These establish implementation behavior, not benchmark superiority.

This is a bidirectional typed-decision encoder, not a causal text generator. The official test split is held out within the notebook, but it informed earlier research iterations; publication claims require a fresh external benchmark. Teacher data provenance is not established here. Action outputs are distilled from the teacher rather than validated against independent action labels. Latency excludes tokenization; activation-memory measurements exclude resident weights.
