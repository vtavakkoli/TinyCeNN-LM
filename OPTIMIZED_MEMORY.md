# Optimized Memory V2

[Open the Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Optimized_Memory_V2_Colab.ipynb).
Choose a GPU runtime and Run all. `balanced` is the default; `smoke` checks the pipeline first.
This is a new controlled experiment alongside the repository's Memory Fusion and Delta work.
No trained V2 GPU result is included, and no candidate is known to beat the Transformer yet.

## Why this experiment

The supplied V1 layer-18 plots showed approximately 1–2% higher perplexity, output cosine
around 0.62–0.68, and substantial kernel slowdowns despite smaller recurrent state.
A nearly unchanged full-model perplexity after replacing one attention layer does not establish
that a whole Transformer can be replaced. V2 addresses the normalization, optimization and
execution bottlenecks, then tests both individual and jointly composed replacements.

The shortlist is deliberately small:

| Variant | Role | Sequence softmax | Trainable components |
|---|---|---|---|
| `cenn_partition` | Primary quality/efficiency hypothesis | Exact sinks and two terraced blocks | Positive features, memory mass, head readout |
| `cenn_linear` | Pure recurrent-memory ablation | None | Positive features, head readout |
| `sink_window` | Strong no-training control | Exact sinks and two terraced blocks | None |
| `transformer_readout` | Adapted full-attention control | Full causal attention | Head readout |
| Original model | Unchanged reference | Full causal attention | None |

## Mathematical construction

Use row-vector queries and values, head width d, feature width F, block size C, and s sinks.
Positive features retain Q/K input magnitudes:

    phi(x) = softmax([x W^T, -x W^T] / d^(1/4)).

The softmax here is over F features, not over the sequence. Q and K have separate learned
maps, initialized consistently within each grouped-query group. State is shared by KV heads.
These are learned kernel features, not an unbiased estimator of the exponential dot-product kernel.

For a query in block b, exact indices E contain the first s tokens plus blocks b-1 and b,
restricted to positions no later than the query. Compressed indices G contain older non-sink
blocks through b-2. Sink duplicates are masked from local columns. Thus E and G are disjoint
and cover the causal prefix. Store

    S = sum_{i in G} phi(k_i)^T v_i,     z = sum_{i in G} phi(k_i).

The raw hybrid output is

    y = [sum_{i in E} exp(q k_i^T / sqrt(d)) v_i + g(q) phi(q) S]
        / [sum_{i in E} exp(q k_i^T / sqrt(d)) + g(q) phi(q) z^T],

    g(q) = exp(clamp(log_mass + normalize(q) mass_w, -12, 12)).

A shared maximum stabilizes the two positive masses. Unlike separately normalized overlapping
branches, each source position contributes once and the denominator tracks its mass. Before
readout calibration, y is a convex combination of past values in exact arithmetic. If G is
empty it equals exact causal attention. The learned readout need not preserve convexity.

The pure linear variant uses S,z over the entire causal prefix. Within-block triangular
feature products and exclusive block prefix sums produce all outputs in parallel. The hybrid
uses an exclusive prefix delayed by two blocks. There is no sequence-length Python loop or
triangular solve in prefill for either new memory candidate. For fixed F,d,C,s the prefill
work and saved training activations grow linearly with sequence length. This does not imply
lower wall-clock latency than fused SDPA at a particular context length.

Decode compresses a retiring block when entering the next block. State is at most
H_kv * [F*d + F + 2*(2*C+s)*d] FP32 elements for the hybrid (times batch size), or
H_kv * (F*d+F) for pure linear memory. The sink control has just its bounded K/V caches.
Actual tensor bytes are reported; cache slices are cloned to avoid retaining an entire old
allocation. Sink tokens may occupy redundant cache storage temporarily, but their attention
weights are never double-counted.

### Closed-form calibration and readout fusion

For training-only raw attention outputs Y and teacher targets T, solve independently per head

    min_R ||Y R - T||_F^2 + lambda ||R - I||_F^2,
    (Y^T Y + lambda I) R = Y^T T + lambda I.

Lambda is 0.01 times the mean diagonal of the Gram matrix by default. The small d-by-d systems
are solved in CPU float64 with symmetric diagonal preconditioning: for A R = B and
D = diag(A)^(-1/2), solve (D A D) U = D B, then R = D U. This is algebraically equivalent;
conditioning improvements depend on the matrix. It does not guarantee improved validation loss.
Calibration is accepted only when validation output NMSE does not worsen.

Training then uses attention transfer and language loss with validation checkpoint selection.
At inference, fold the learned readout into the original output projection:

    O_new = O blockdiag(R)^T.

The code forms a separate fused buffer and restores the original attention module after each
comparison. Original model weights are never overwritten. Full-model logit equivalence is
unit-tested in FP32; native precision can introduce rounding. Evaluation uses the actual folded
wrapper. The training path stays unfused so gradients reach R and the feature maps.

## Experimental safeguards and limits

- Start from the original pretrained SmolLM2-135M. Freeze all original parameters.
- Same transfer and language-update counts for learned candidates and adapted Transformer;
  parameter counts differ. Ridge adds one training-data pass. Sink attention receives no training.
- Defaults: layers 0,18,29, train context 256, test contexts 256/512/1024, 96/16/32 train/validation/test
  documents, F=64, C=32, s=4, 300 transfer and 40 language steps. This is a small screening budget.
- Select one bounded candidate per layer by validation NLL before any test evaluation. Also evaluate
  all candidates transparently. Jointly compose the selected layers and compare with jointly
  adapted Transformer readouts. There is no joint retraining or full-model replacement claim.
- Hash normalized documents; one block per unique document. Exclude V1 validation/test buckets
  0..19; use 20..29 for V2 test, 30..39 for validation, 40..99 for training. New V2 holdouts may
  overlap V1 training unless `--exclude-manifest` is supplied. It excludes every hash recorded in
  each previous manifest. This cannot establish absence from the original model's pretraining data.
- Record source commit, immutable model/data revisions, seed, GPU, dependencies, hashes and blocks.
  Intervals are paired bootstrap over documents, not over training seeds; repeated model comparisons
  are exploratory, without familywise correction. At least eight test documents are needed for flags.
- A quality win requires the upper 95% NLL-difference bound below zero versus both original and
  adapted Transformer controls. A quality-preserving efficiency win requires upper bounds <=0.02
  nats/token versus both, prefill and decode speedup >1, and state ratio <1. The tolerance corresponds
  to exp(0.02)-1, about 2.02% perplexity; it is not exact equality.
- Measure synchronized warmed kernel medians against native-dtype SDPA and also FP32 SDPA.
  Include feature maps; exclude Q/K/V/O on both sides (readout is folded into O). Decode times average
  a whole block of successive state updates, including block retirement, with cache appends on both
  sides. The last cached Q/K/V is repeated as a synthetic timing workload. State ratio is at the
  starting test context. These are microbenchmarks, not production text-generation throughput.
- State is FP32. Inference matmuls can use the GPU's native FP16/BF16; check validation drift against
  FP32 before selection. Optional torch.compile prefill results have an equivalence gate, record
  failures, and never silently replace eager decision metrics. Compilation cost is separate.
- Full-model joint forward latency includes projections and the LM head, excludes generation caches,
  and is measured on one document. Re-run multiple seeds, documents, devices, and longer contexts
  before claiming a deployment win. A custom fused kernel could help, but is not implemented here.
- The integrated wrapper supports unpadded full causal Llama blocks with use_cache=False. Core-level
  streaming is tested; an HF generate-compatible cache adapter is not part of this experiment.
- These are CeNN-project memory mixers; they are not a claim of classical nearest-neighbor CeNN ODE
  dynamics. No universal dominance over the Transformer follows from bounded recurrent memory.

## Research sources

- [LoLCATs (2024)](https://arxiv.org/abs/2410.10254): learned attention features, transfer/adaptation,
  and hybrid local/linear attention. The block partition here is an independent adaptation.
- [Sliding-window beats linear attention (August 2026)](https://arxiv.org/abs/2608.28444): motivates
  a strong sink-preserving local-attention control. Results in that paper do not validate this code.
- [Preconditioned DeltaNet (April 2026)](https://arxiv.org/abs/2604.21100): motivates examining
  conditioning. V2 uses exact ridge calibration, not that paper's recurrent update algorithm.

## Run and inspect

    python scripts/benchmark_cenn_optimized_memory.py --output-dir result/new-v2-run

Use a fresh output directory. The Colab offers Drive persistence and a ZIP containing checkpoints,
manifest, selection, training history, per-document NLL, optimized_memory_report.json,
optimized_memory_summary.csv, selected_decisions.csv and plots. Interrupted runs retain intermediate
files and logs; automatic resume is not implemented. The notebook rerun starts a new experiment.

Correctness tests:

    OMP_NUM_THREADS=1 python -m pytest -q tests/test_optimized_memory.py tests/test_optimized_memory_benchmark.py

Tests cover independent dense outputs and gradients, causality, stateful equivalence, memory bounds,
ridge calibration, frozen-model preservation, readout fusion, complete offline Llama training/testing,
CLI invocation outside the checkout, and execution of notebook reporting cells on offline results.
