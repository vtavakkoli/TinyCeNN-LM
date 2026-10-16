# Laya standalone FP16 fast inference

Target: [`vtava/Laya-1.2-speedup-standalone-fp16`](https://huggingface.co/vtava/Laya-1.2-speedup-standalone-fp16),
revision `2891f7312d8e7c65b43a6e04c29cc226b06b4fac`.

[Open the GPU notebook](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/laya-fp16-fast-inference/notebooks/Laya_FP16_Fast_Inference_Colab.ipynb).

This is an inference implementation for the existing 421M-parameter FP16 model.
It does not train, quantize, remove layers, change windows, or share weights between
encoder layers. It does not create or upload a new Hugging Face checkpoint.

## Changes

| Path | Work avoided | Constraint |
|---|---|---|
| Device masks | CPU construction/copy of the encoder's square window mask; global mask broadcasts over queries | Specialized to Transformers 4.57.6 ModernBERT SDPA; otherwise native encoder |
| Gather first | Type-embedding addition over the entire encoder output | Same token indices, order and masks |
| Marker SDPA head | Unused queries, output projection and FFN positions in the final head layer | All previous head layers and all K/V positions retained |
| Prepared questions | Repeated question tokenization and state tokenization per question | Recreate when tokenizer/questions change |
| Compilation | Python overhead and fusible operations | Optional, shape-specific startup and hardware dependence |
| CUDA graph | Repeated kernel-launch setup | One exact tensor signature per runner; input copies and owned output clone included |
| FlashAttention-2 | Alternative encoder attention backend | Optional compatible GPU/package; separate encoder, numerically gated |

The original runtime remains unmodified. `FastLaya` shares its weights and is
inference-only. Use `torch.inference_mode()` for direct tensor calls; the prepared
prediction API applies it automatically. Floating-point kernels may round
differently even where the mathematical computation is unchanged.

## Run

Use an existing CUDA-enabled PyTorch installation. Do not replace Colab's PyTorch
with a CPU wheel.

```bash
pip install 'transformers==4.57.6' 'huggingface_hub>=0.25,<1' 'safetensors>=0.4'
python scripts/benchmark_laya_fp16.py --compile --cuda-graph
```

The script downloads only the named pinned standalone checkpoint. For an already
downloaded snapshot, add `--model-dir /path/to/snapshot`. Baseline and candidate
use the same actual FP16 weights and the same input tensors, not FP32/autocast as
a substitute baseline.

For a longer context sweep:

```bash
python scripts/benchmark_laya_fp16.py --compile --compile-mode max-autotune \
  --cuda-graph --lengths 128 512 1024 2048 --batch-sizes 1 4
```

FlashAttention-2 is opt-in (`--flash-attention`) after installing a compatible
`flash-attn` package. It is not enabled on T4. Construction errors are recorded
and skipped. CUDA illegal-memory/device-assert errors stop the run; restart the
runtime before retrying. Do not silently keep using a damaged CUDA context.

## Measurement and acceptance

The JSON report contains three distinct synthetic input audits per shape,
identical-decision checks, maximum probability/logit drift, synchronized timing
rounds, setup/validation time, environment and source hashes. CUDA peak memory is
**process-wide**, including resident baseline/candidates, not isolated model memory.

Candidates are eligible only if every tested decision matches and maximum absolute
probability drift is at most 0.01 (configurable). Failed candidates are recorded.
The original is always eligible if its repeatability check passes. The reported
winner is the lowest measured median for each shape, not a universal deployment
recommendation. Small timing differences may be noise; repeat on the target GPU.

Model-forward timing uses device-resident tensors and excludes tokenization and
startup; CUDA graph input copies/output clone are included. A separate API benchmark
includes tokenization, transfer, softmax and decoding. First compile/capture costs
are excluded from steady-state timing and reported with validation costs.

The included CPU result is a smoke benchmark, not evidence of GPU speedup. Local
numerical tests cover FP32/FP16, head depths 1–3, short/long sequences, padded rows,
invalid markers, immutable weights and cached tokenization. The CUDA graph test is
skipped without CUDA. GPU compilation, CUDA replay and FA2 require execution on
supported hardware before deployment. No new held-out accuracy claim is made.

## Recorded CPU smoke result

[Machine-readable report](benchmarks/laya_fp16_cpu.json), using the actual full
FP16 checkpoint, PyTorch 2.14.1+cpu, two CPU threads, one warmup and three single-call
timing rounds per candidate. This short run is primarily a correctness check.

| Workload | Original | Fastest accepted candidate | Ratio |
|---|---:|---:|---:|
| Batch 1, 128 tokens | 2483.1 ms | Device masks + gather: 2364.7 ms | 1.050× |
| Batch 1, 512 tokens | 9387.6 ms | Marker SDPA: 9279.9 ms | 1.012× |
| Two-question prediction API | 1549.2 ms | Prepared marker API: 1661.1 ms | 0.933× |

Every tested decision matched. Maximum probability drift across the model-forward
audits was 0.000118464; API drift was 0.0000125244. The CPU API was slower, and the
small model-forward differences may be noise. These measurements do **not** establish
a substantial speedup or predict the GPU result. The GPU notebook tests launch-overhead
and kernel optimizations that could not be measured in this environment.

## Application API

From the repository root, after downloading the pinned snapshot:

```python
import sys
import torch
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'standalone/laya_1p2_speedup')
from benchmark_laya_fp16 import load_runtime, parity
from fast_inference import optimize_model, PreparedChoices, inference_context

runtime = load_runtime(local_dir)
original, tokenizer = runtime.load_model(local_dir, device='cuda')
fast = optimize_model(original)
questions = {'steering': {'type': 'choice', 'instructions': 'Choose road direction.',
                          'criteria': ['left', 'straight', 'right']}}
prepared = PreparedChoices(runtime, tokenizer, questions)
state = {'road_direction': 'left', 'speed': 7.0}
batch = prepared.batch(state, device='cuda')
with torch.inference_mode(), inference_context(original):
    reference = original(**batch)[0].clone()
    candidate = fast(**batch)[0].clone()
assert parity(reference, candidate, batch['marker_mask'])['passed']
print(prepared.predict(fast, state))
```

Validate representative application inputs, including near-tied decisions, before
choosing a backend. A one-example or synthetic test cannot establish general task
accuracy. `PreparedChoices` supports choice questions; the tensor forward supports
all existing qtype embeddings.

**Do not silently bucket or repad:** the checkpoint chooses head landmarks using
padded sequence length. Different padding/batching can change which real tokens
reach the head. CUDA graph replay therefore rejects shape/dtype/device changes.
Use one runner per worker; runners are not thread-safe. Recreate after changing
weights, dtype or attention backend. Do not nest manual graphs around compiled
models that already manage CUDA graphs.

**No prefix/KV cache:** ModernBERT is bidirectional, so changing the state changes
question-token representations throughout the encoder. Tokenization caching is
valid; reusing prior encoder K/V would change this model's computation.

## References

- [PyTorch torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [ModernBERT documentation](https://huggingface.co/docs/transformers/model_doc/modernbert)
- [Pinned ModernBERT implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py)
