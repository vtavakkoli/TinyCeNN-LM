# CeNN Research Layers: a controlled Transformer-replacement experiment

[Open the new Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/cenn-research-layers-20260914/notebooks/CeNN_Research_Layers_Colab.ipynb)

**Recommended first experiment:** run the three proposed layers on SmolLM2-135M
layer 18, with feature dimension 64 and a 32-token window for the hybrid.
The notebook's balanced profile is a screening experiment. It measures whether
one replacement preserves next-token quality; it cannot promise to match or beat
the Transformer before training and evaluation.

## Why this is a new experiment

The existing
[learned-kernel notebook](notebooks/CeNN_Learned_Kernel_Concepts_Colab.ipynb)
learns positive feature maps while freezing the pretrained projections. Its
default screens reuse six training sequences, evaluate on two sequences, and
optimize dense pairwise kernels. Its reported runtime measures that dense path;
the constant-state memory figures are analytical. Output cosine and the report's
PASS status do not establish downstream language-model parity.

This notebook does not rerun those old variants. It changes **how the recurrent
memory is edited**, implements an actual bounded-state computation, and measures
the language model after replacing an attention layer.

## Proposed layers and research basis

| Name | Mechanism | Local softmax? | Basis |
|---|---|---|---|
| cenn_kda | Channel-wise forgetting; one tied erase/write gate per KV head and token | No | Kimi Delta Attention, 2025 |
| cenn_delta2 | Channel-wise forgetting, independent key-channel erase and value-channel write gates | No | Gated DeltaNet-2, May 2026 |
| cenn_delta2_window | The Delta2 memory plus an exact 32-token causal window and a learned query-dependent mixture | Yes, bounded window | Gated DeltaNet-2 plus attention-transfer/local-window ideas |

My research hypothesis is that **cenn_delta2_window has the best chance of
preserving a pretrained attention layer**, because it retains exact local
retrieval while learning how to use compressed history. If softmax must be
absent from the new layer, prioritize **cenn_delta2**, with cenn_kda as its
tied-gate ablation. This is an inference from the cited work, not a result of
this repository's unrun GPU experiment.

These are independent **adaptations**, not reproductions of the papers' models
or fused CUDA implementations. Their full-model benchmark gains do not transfer
automatically to frozen SmolLM2 projections.

Primary sources, checked 14 September 2026:

1. [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791)
   — Hatamizadeh, Choi and Kautz, May 2026. Source of the separate erase/write
   recurrence, including its tied-gate reduction.
2. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
   — Kimi Team, October 2025. Source of channel-wise forgetting. Kimi Linear's
   reported full architecture is hybrid and retains global attention.
3. [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/abs/2410.10254)
   — Zhang et al., ICLR 2025. Supports first matching attention outputs, then
   correcting downstream language loss. This experiment refines only the new
   layer; it does not implement LoLCATs' LoRA stage.
4. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
   — Yang, Kautz and Hatamizadeh, ICLR 2025. Background for gated delta updates.
5. [Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences](https://arxiv.org/abs/2604.21100)
   — Tumma, Loo and Rus, April 2026. A relevant follow-up if recall still
   saturates. It is not implemented in this focused three-candidate experiment.

## Implemented recurrence

For a KV head, the persistent memory is a grid of cells
$S_t \in \mathbb{R}^{F \times D}$. With normalized learned query/key features:

$$
\bar S_t = \operatorname{diag}(\alpha_t) S_{t-1},
\qquad e_t = b_t \odot k_t,
\qquad z_t = w_t \odot v_t.
$$

$$
S_t = \bar S_t + k_t \left(z_t - \bar S_t^\top e_t\right)^\top,
\qquad o_t = S_t^\top q_t.
$$

For cenn_kda, the vectors $b_t$ and $w_t$ both collapse to the same scalar.
The hybrid adds a query-dependent convex mixture:

$$
o_t^{\mathrm{hybrid}} =
g_t\,\operatorname{SoftmaxAttention}(q_t,K_{t-W+1:t},V_{t-W+1:t})
+(1-g_t)\,o_t^{\mathrm{memory}}.
$$

The bounded window **includes the current token**. Its local expert overlaps
the history in recurrent memory; this is a mixture of two estimators, not an
exact disjoint partition of a softmax denominator.

Here CeNN means the project's recurrent memory-cell interpretation. The update
uses associative reads across the memory grid; it is not a claim to reproduce
a classical nearest-neighbor continuous-time CeNN template or solve its ODE.

### Explicit deviations from the research models

- Pretrained Q/K/V/O and RoPE are frozen. Learned headwise feature adapters,
  forget/erase/write gates, output gain, and the optional mixture gate are new.
- Gates use frozen K/V-derived features instead of fresh projections of the
  full hidden representation. KV groups share memory; the implementation
  counts KV heads, not query heads, when reporting state.
- Queries and keys are L2-normalized after their learned adapters.
- Log-decay is bounded to [-0.25, 0]. Chunk length is at most 32. This keeps
  within-chunk exponential rescaling bounded by exp(8) in the FP32 reference.
  It is an implementation/stability choice, not the papers' unrestricted gate.
- Training uses a triangular solve within fixed-size chunks, with recurrence
  between chunks. Only chunk-by-chunk and T-by-W matrices are constructed.
- The module supports stateful streaming; the Hugging Face replacement wrapper
  intentionally evaluates unpadded complete blocks with use_cache=False.
- No custom CUDA extensions, flash-linear-attention dependency, or paper code
  is copied. The reference may be slower than optimized SDPA at short context.

## Experimental protocol

1. Resolve and record immutable model and dataset revisions.
2. Normalize document whitespace, hash documents, assign whole documents to
   train/validation/test, and reject exact duplicates. Capture one block per
   document. No document is split between adaptation partitions.
3. Cache teacher Q/K/V and exact attention outputs from the frozen model.
4. Check replacement plumbing with an exact-softmax core: per-document NLL
   difference must remain below 0.01, allowing mixed-precision differences.
5. Train each candidate with the same token order, update count, seed,
   optimizer and schedule. Optimize output NMSE plus a cosine term.
6. Reload the best attention-transfer checkpoint selected on validation NMSE.
7. Refine **only the new layer** using next-token cross entropy plus output
   regularization. Keep the checkpoint with the lowest validation NLL,
   including the checkpoint before this refinement.
8. Lock the candidate choice per layer using validation NLL, write
   selection.json, and only then compute test metrics.
9. Report all preregistered candidates, marking the validation-selected one.
   Each layer is replaced independently; multiple selected layers are not
   composed in this experiment.
10. Evaluate the same test documents at training length and a longer context.
    Bootstrap paired document NLL differences for exploratory 95% intervals.

“Held out” here means held out from **this adaptation experiment**. FineWeb-Edu
may overlap the original model's pretraining corpus; no claim about original
pretraining contamination is made.

The balanced profile uses 64 training documents (16,384 unique input tokens at
context 256), 12 validation documents and 24 test documents. It is intentionally
small. The extended profile increases data and training budgets. More seeds,
larger untouched test sets, downstream tasks, and simultaneous multi-layer
replacement are needed before making a full-model claim. Test feedback must not
be used to tune and reevaluate the same held-out documents; use a fresh test
partition for subsequent research decisions.

This is an incremental replacement comparison against a pretrained model,
**not** an equal-parameter or equal-pretraining-compute architecture comparison.
New parameter counts and token presentations are exported.

## How to read the outputs

Lower test NLL and perplexity are better. PPL ratio is candidate PPL / original
Transformer PPL. An exact forward reference already has perfect self-fidelity;
a candidate can only “do better” on a downstream objective, not by being more
identical than the reference.

| Field or label | Meaning |
|---|---|
| output_cosine / output_nmse | Attention-output agreement on test documents |
| grad_q/k/v_cosine, grad_q/k/v_nmse | Input-gradient fidelity on one fixed diagnostic document |
| delta_nll | Paired candidate minus Transformer mean NLL |
| within_declared_nll_margin | Entire 95% paired interval lies inside +/-0.02 nats by default, roughly +/-2% PPL |
| lower_nll_on_this_test | Upper bound of the paired interval is below zero |
| worse_than_declared_margin | Lower bound is above the declared margin |
| inconclusive | Interval does not support one of the above decisions |
| insufficient_test_documents | Fewer than eight test documents; smoke checks do not earn a quality label |
| prefill_speedup / decode_speedup | Actual reference-kernel timing ratios; values below 1 mean slower |
| state_vs_transformer_fp16 | Actual FP32 candidate state vs an analytical FP16 KV cache |
| peak_extra_bytes | CUDA peak allocation above the warmed-up baseline during the measured kernel call |

The “within margin” label is an exploratory practical criterion, not a formal
equivalence trial or evidence across training seeds. Completion status
**completed** never means parity was reached. Raw NLL and uncapped perplexity
are retained.

Delta-rule coefficients can be signed and write gates depend on V. They do not
define a softmax-like probability distribution over tokens, so attention KL and
partition-function loss from the old positive-kernel study are not used.

Prefill and one-token decode timings use equal FP32 inputs, identical shapes,
warm-up, CUDA synchronization and median timing. They include feature/gate
work and GQA expansion, but exclude frozen Q/K/V/O projections, model layers,
tokenization, and data loading. The decode microbenchmark repeats the same
final query against a fixed prefix state. It is not full generation throughput.
One document is used for timing; GPU model and software versions are recorded.

## Run outside Colab

From this branch, after installing the project and compatible dependencies:

~~~bash
python -m pip install -e . "transformers==4.57.6" "datasets>=3,<5" pandas matplotlib
python scripts/benchmark_cenn_research_layers.py \
  --layers 18 \
  --variants cenn_kda,cenn_delta2,cenn_delta2_window \
  --feature-dims 64 --context 256 --test-contexts 256,512 \
  --train-documents 64 --validation-documents 12 --test-documents 24 \
  --steps 400 --lm-steps 40 \
  --output-dir result/cenn-research-layers-run1
~~~

Use a fresh output directory for each run. For a strictly softmax-free candidate
set, set --variants cenn_kda,cenn_delta2. The untouched reference model still
contains Transformer attention. To compare easy/hard/late positions, set
--layers 0,18,29. Existing notebooks and production model classes remain intact.

## Saved artifacts

- manifest.json: revisions, source commit, configuration, document hashes,
  software/hardware and exact-replacement control.
- token_blocks.pt: exact tokenized documents for reproduction.
- checkpoints/*.pt: new-layer weights, complete constructor config and metadata.
- selection.json: validation-selected architecture, written before test evaluation.
- validation_summary.csv and training_history.csv: model selection and loss curves.
- research_layer_summary.csv and research_layer_report.json: complete results.
- test_document_nll.csv: paired per-document observations for audit/reanalysis.
- The Colab additionally saves comparison plots and offers a ZIP download.

Checkpoints are incremental new-layer weights. They are not standalone complete
language models, and training state is not a resumable optimizer checkpoint.
If a run is interrupted, completed checkpoints and progress files remain useful;
use a fresh run directory for a complete rerun.

## Validation

CPU tests check the recurrence against an independent tokenwise oracle,
including derivatives; causal behavior; GQA; bounded streaming state; local
attention against a dense masked reference; checkpoint reload; actual random
Llama integration; restoration of original modules; frozen base weights; both
training phases; and a synthetic transfer-learning sanity check. CI also parses
the notebook and compiles every code cell. These checks validate implementation
behavior, not GPU performance or pretrained-model quality.
