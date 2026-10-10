# Jev_CeNN — CUDA optimized 30-layer hybrid model

[Run Jev_CeNN on Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_CUDA_Fast_AllAttention_Colab.ipynb)

## Architecture and model identity

**Jev_CeNN** is a separately named hybrid replacement derived from a frozen
`convaiinnovations/laya-typed-decisions` teacher. It contains 10 global
FastWindowMemory replacements and 20 SlidingWindowFusion replacements. All
original encoder attention **modules** are replaced, but both replacements
still use attention primitives. **Jev_CeNN is not an attention-free model.**

Default global `W1024` is a full-context quadratic control at 512 tokens,
and sliding `W129` still uses local QK attention. Report real GPU measurements,
rather than extrapolating performance from asymptotic complexity.

## Acceleration

- `jev_cenn_cuda.py` provides a **real Triton CUDA kernel** which computes a
  single query's exact bounded local SDPA directly from strided Q, K and V.
  No T×T score matrix, halo reshaping/unfold, or materialized softmax table.
- **Inference-only:** The optimized kernel is used under `no_grad`; any
  trainable Q/K/V path delegates to the original PyTorch autograd operation.
  When Triton or CUDA is unavailable, the runtime falls back to PyTorch.
- Strict GPU preflight compares length 9/129/290/512, different radii and
  padded sequences to the exact blocked PyTorch reference before enabling
  acceleration. Default thresholds are max absolute error <0.025 in FP16
  (<0.05 BF16), valid-token cosine >0.999.
- The CPU-compressed shared teacher-I/O cache captures all 30 original layers
  from **one group of teacher forwards** for each train/probe/audit batch.
  This saves roughly 29 repeated full teacher passes per batch at the cost
  of several GB system RAM. Toggle `SHARED_TEACHER_CACHE` for small instances.
- Per-layer probe and blind audit still run for **every layer**, while
  expensive end-to-end DEV checkpoints run only every
  `DEV_EVAL_EVERY_LAYERS=5` replaced layers.
- An optional `torch.compile(mode='reduce-overhead')` inference benchmark
  can be enabled, but startup compilation may be slow and it is not exported.

## Recommended reproducible experiment

1. Use an NVIDIA Colab GPU and run the notebook from the top.
2. If your V2.9 global and V3.0 sliding conversion checkpoints exist in
   Colab or Drive, set `USE_V29_CHECKPOINT` and
   `USE_V30_SLIDING_CHECKPOINTS` to `True`, and enter their paths.
   Otherwise the notebook independently trains missing layers.
3. For checkpoint persistence set `OUTPUT_DIR` under mounted Google Drive.
4. Compare per-layer blind cosine / NMSE and DEV teacher agreement, official
   test accuracy, and **paired actual CUDA latency**. The model is not assumed
   to be faster merely because Triton is installed.
5. Review `jev_cenn_cuda_benchmarks.json`:
   `optimized_vs_native > 1` means the **complete model** beats the teacher;
   `triton_vs_pytorch > 1` means it beats the unoptimized Jev_CeNN.
6. The checkpoint is selected on TRAIN/DEV **before** the official TEST
   split is loaded. The original V3.1 ≥0.99 local cosine and model decision
   agreement gates are retained.
7. Export a standalone Hugging Face model with independent, optional Triton
   runtime embedded in the model's `runtime.py`. The model can be loaded
   without the research repository. Add write-enabled `HF_TOKEN` in Colab
   Secrets, set `HF_PRIVATE=True`, and optionally choose `HF_REPO_ID`.
   The default new repository name is `<your-HF-username>/Jev_CeNN`.
   Publication remains gated on held-out DEV quality and CPU strict reload.

## Caveats

**The optimized Triton CUDA kernel has to pass parity tests on your actual
GPU; GitHub's CPU CI cannot run those GPU tests.** Fallback and notebook syntax
are covered in CI. Actual speedup, total runtime, final quality, GPU training
and Hugging Face upload are not implied by successful CI.

The bundled `runtime.py` is executable code; pin trusted revisions before
loading public models.
