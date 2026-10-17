# Laya FP16 V3: measured layer search

Model: `vtava/Laya-1.2-speedup-standalone-fp16`, pinned to
`2891f7312d8e7c65b43a6e04c29cc226b06b4fac`.

## What the uploaded V2 T4 run actually demonstrated

| Workload | Original eager | Original compiled | Optimized compiled | Improvement beyond compilation |
|---|---:|---:|---:|---:|
| B1, T128 forward | 30.197 ms | 7.964 ms | 7.909 ms | 1.007× |
| B1, T512 forward | 33.025 ms | 23.299 ms | 23.216 ms | 1.004× |
| B4, T128 forward | 32.400 ms | 22.831 ms | 20.573 ms | 1.110× |
| B4, T512 forward | 113.407 ms | 89.256 ms | 86.910 ms | 1.027× |
| Prepared complete prediction API | 32.783 ms | 14.563 ms | 12.550 ms | 1.160× |

Legacy API: 34.429 ms, so the optimized compiled API was 2.743× faster.
Every recorded numerical/decision check passed; maximum probability drift was
0.0020555. These checks used three synthetic probes per forward shape and three
synthetic API cases, not a labelled accuracy evaluation. Most forward speedup
came from compilation. V3 asks whether removing actual encoder computation helps.

## Search space and boundaries

V3 exhaustively tests five alternatives independently at each of all 28 layers
(140 single-layer trials):

- Skip the attention residual branch.
- Skip the FFN residual branch.
- Bypass the entire encoder block.
- Retain approximately 75% of GLU channels.
- Retain approximately 50% of GLU channels.

FFN channels are ranked by mean absolute calibration activation multiplied by
the norm of the corresponding output column. Input and gate rows are kept as a
pair, with matching output columns. Widths are rounded down to 64-channel multiples
for the real checkpoint. This creates genuinely smaller dense matrix multiplications;
it does not zero dense matrices and then claim sparse speedup. No weights are trained.
Retained original tensors remain frozen. Native PyTorch SDPA handles attention;
there are no custom Triton attention kernels. PyTorch Inductor may generate Triton
code internally when compiling.

Four additional trials change every layer together (75%/50% FFNs, attention removal,
FFN removal). The best 12 individual trials, including near misses, generate up to
66 cross-layer pairs. A width-2 beam explores combinations of the best 24 single-layer
edits for up to six steps, recalculating metrics on the accumulated model. Trials
are deduplicated. This is a **bounded brute-force search**, not exhaustive enumeration
of all 6^28 architectures. Compensating combinations outside this pool may be missed.

## Data and selection

The pinned `LocalLLaMA/typed-decisions` **labelled synthetic benchmark** is used with
the source runtime's sequence builder and its exact option order. Choice, score and
boolean (`noul`) questions are included. Defaults are 24 calibration cases, 32 search
cases and 160 validation cases, sampled by workflow from the official train split.
Cases and their questions stay together. Partitions are disjoint; duplicate IDs and
train/test overlaps fail loudly. This isolates V3 selection, but does not establish
that the source checkpoint never saw these cases in earlier training.

Default gates relative to the unchanged original:

- At least 98% decision agreement.
- At most 0.5 percentage points lower gold-label accuracy overall.
- At most 2 percentage points lower accuracy within each workflow and question type.
- Mean Jensen–Shannon divergence at most 0.005.

Up to 12 promising search candidates are checked on validation; up to three are
compiled. Physical parameter count is only a shortlist heuristic, never a speed claim.
Compiled candidates must also preserve every eager candidate decision on validation
and have maximum probability drift <=0.01. The unchanged optimized compiled baseline
is independently checked against the original.

Finalist timings alternate candidate order across seven rounds. A layer change must
beat the optimized compiled baseline by at least 3%, and be faster in at least 80%
of rounds. These are practical filters, not statistical significance tests. Each
timing uses the same real case shapes and includes H2D, forward, softmax and D2H.
Warmup/compilation are reported separately. Lengths are not repadded because Laya's
landmark positions depend on the padded sequence length.

A separate, complete uncached API timing includes state/question tokenization,
collation and Python decoding. It compares original eager, original compiled,
optimized compiled and selected compiled models on validation cases. It does not
drive model selection and differs from V2's prepared-question API workload.

Selection is saved **before** loading the official test split. The selected model
is then audited once over all test cases, with per-type/workflow accuracy, teacher
agreement, probability drift, gold NLL and a paired case-bootstrap accuracy-delta
interval. Failed test results are reported; no alternative is chosen using test
labels. Passing point-estimate gates does not prove statistical non-inferiority.

## Colab use and files

Open `notebooks/Laya_FP16_All28_BruteForce_V3_Colab.ipynb`, select a GPU and run all.
The notebook pins code, model and dataset revisions. It retains Colab's CUDA-enabled
PyTorch. Several hundred model evaluations and variable-shape compilations can take
substantial GPU time. Budgets are exposed in the notebook.

Every completed trial is saved atomically. Restarting with the same output directory
reuses it. A configuration/code/version/hardware mismatch requires a fresh output
directory. Optional Drive storage survives a Colab runtime reset. Interrupting midway
still leaves useful trial reports; rerun the download cell to save partial results.

Key outputs:

- `trials.json`: every attempted architecture, including failures and rejected trials.
- `validation.json`: independent validation quality checks.
- `compiled_validation.json`: compile audits and interleaved finalist timing.
- `selected_before_test.json`: the frozen selection.
- `end_to_end_api.json`: full-request timing, including tokenization and decoding.
- `final_report.json`: final audit, confidence interval, outcome and limitations.
- `selected_recipe.json`: reconstructable architecture recipe with exact channel indices.
- `layer_search.py`, `fast_inference.py`: loader/runtime copied into the result bundle.

The recipe is **not a full standalone model export**. It downloads the immutable
original checkpoint and applies recorded edits without recalibration. From the
extracted results directory:

```python
from layer_search import load_selected
model, tokenizer, runtime = load_selected('selected_recipe.json', device='cuda')
```

Only `recommended: true` means a changed model passed this run's quality, compiler
and speed gates. `no_qualifying_layer_change` keeps the original architecture.
`candidate_failed_final_audit` leaves the changed recipe explicitly experimental.
No model is automatically uploaded or published to Hugging Face.

## Validation before release

Local tests exercise real tiny ModernBERT layers, structural residual semantics,
GLU indexing, immutable originals, label mapping, case splits, calibration hooks,
resumed searches, all three question types, recipe reconstruction and fullgraph
capture using the eager compiler backend. A miniature 28-layer pipeline test uses
mocked Hub data and compilation to exercise the complete orchestration.
These are not T4 measurements or accuracy results for the full checkpoint.
The new full-model GPU search must run in Colab before claiming any further gain.
