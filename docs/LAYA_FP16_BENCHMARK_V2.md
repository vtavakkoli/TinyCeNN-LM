# Laya FP16 V2: compiled control and full prediction API

[Open V2 in Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/laya-fp16-fast-inference/notebooks/Laya_FP16_Compiled_Control_API_V2_Colab.ipynb).

The first T4 report measured 3.53x (batch 1 / 128 tokens), 1.33x (1 / 512),
1.40x (4 / 128), and 1.13x (4 / 512) for compiled optimized inference against
uncompiled original inference. Those comparisons combine compilation and code
changes. The earlier full API comparison only exercised the eager optimized path.

V2 addresses both gaps using the same pinned FP16 checkpoint. No training, weight
conversion, attention-window change, or Hugging Face publishing is involved.

## Four comparable runtimes

| Runtime | Original model computation | Optimizations | Compilation |
|---|---|---|---|
| `original_eager` | Native standalone forward | None | No |
| `original_compiled` | Native standalone forward | None | Yes |
| `optimized_eager` | Existing optimized forward | Device masks, gather-first, marker head | No |
| `optimized_compiled` | Existing optimized forward | Device masks, gather-first, marker head | Yes |

Both compiled paths use identical mode/dynamic-shape settings and require
`fullgraph=True`. Compile failures are logged and excluded instead of being
silently counted as a successful compiled variant. A failed compiled-original
control means V2 cannot attribute the speedup beyond compilation.

For each accepted pair, the report computes:

- `compilation_on_original`: original eager / original compiled.
- `optimization_without_compile`: original eager / optimized eager.
- **`optimization_beyond_compile`: original compiled / optimized compiled.**
- `combined_vs_original_eager`: original eager / optimized compiled.

Ratios above 1 favor the second runtime. A result below 1 means that runtime is
slower. The compiled original can win; V2 does not force selection of our wrapper.

## Complete API benchmark

The four paths use identical `PreparedChoices` preprocessing. Its inference method
calls the supplied model's forward explicitly, including the compiled original.
Calling `.predict()` directly on a compiled wrapper can resolve the original
convenience method and miss compiled execution, so V2 avoids that route.

A fifth `legacy_predict` measurement uses the unmodified original API, letting you
separate preprocessing changes from execution changes. API timing includes state
tokenization, collation, device transfer, model execution, probability calculation,
CPU transfer and decoded answers. Cached question preparation is a one-time cost
outside this timing. A request means one state with all its associated questions.

The script checks prepared input tensors against independently constructed original
input tensors, then compares every candidate's decoded decisions/probabilities
against the original API. It records padded lengths and reference top-two option
margins to help identify whether the supplied examples exercise close decisions.

Three application-like synthetic cases are provided. They have no ground-truth
labels. Supply a UTF-8 JSONL file for representative application states:

```json
{"state":{"road_direction":"left","speed":7.0},"questions":{"steering":{"type":"choice","instructions":"Choose road direction.","criteria":["left","straight","right"]}}}
```

Each line requires `state`; `questions` defaults to the original steering/speed
examples if omitted. Only choice questions are supported by this convenience API.
Custom states and questions are read locally and are not included verbatim in the
result JSON. Use `--dynamic` when a diverse workload generates many shapes; it is
applied equally to both compiled paths. Full-graph dynamic compilation may be
unsupported for some input patterns, in which case failures are reported.

## Fair timing and validation

All candidates are validated and warmed before measurement. Each timing round
randomizes candidate order with a recorded seed. GPU completion is synchronized
before and after every timing block. This reduces ordering bias but cannot remove
thermal/clock variation. The report preserves every round and its order.

Forward results are milliseconds per batch with device-resident inputs. API results
are milliseconds per request averaged over a complete case sweep. Round minima and
maxima are ranges of round means, **not** request-level percentiles. Cold setup,
validation and warmup are reported together separately from steady-state timing.

Every tested decision must match, probabilities must be finite, and maximum
absolute probability drift must be <=0.01. Failed candidates are excluded. This is
an output-agreement gate, not evidence of unchanged held-out task accuracy.

## Run

```bash
pip install 'transformers==4.57.6' 'huggingface_hub>=0.25,<1' safetensors
python scripts/benchmark_laya_fp16_v2.py --output results/laya_fp16_v2.json
# Optional representative workload:
python scripts/benchmark_laya_fp16_v2.py --cases-jsonl my_cases.jsonl --dynamic
```

Keep the CUDA PyTorch supplied by Colab. Start with `reduce-overhead`; optionally
repeat with `--compile-mode max-autotune` using the same workload. Compilation can
take several minutes across shapes. CUDA illegal-memory/device-assert errors stop
the run and preserve the partial report; restart the GPU session before retrying.

The notebook pins the implementation commit and model revision. GPU compilation
and speed must be measured on the target hardware. Local validation: 25 tests
passed, 1 CUDA-only test skipped; includes an end-to-end runner smoke test and
Dynamo fullgraph capture with backend `eager` on tiny models. The latter validates
capture/routing, not Inductor kernel generation or speed.
