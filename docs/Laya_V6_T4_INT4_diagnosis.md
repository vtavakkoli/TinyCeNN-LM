# Laya V6 → V6.1: Why native INT4 did not accelerate on Tesla T4

## Reproduced empirical result

Evidence: user-supplied `Laya_Native_INT4_V6_Results.zip` (operator_trials.json,
model_speed.csv, summary_v6.json, conversion_manifest.json).

Hardware: NVIDIA Tesla T4, CUDA capability 7.5 (Turing).
PyTorch: 2.11.0+cu130; TorchAO: 0.17.0.

- **0 native INT4 projection modules selected**
- **0 INT4 model-level speedups demonstrated**
- All 112 original encoder projections retained FP16
- DEV / GUARD / AUDIT teacher agreement: 100%, because it was an **FP16
  identity self-check**, not a quantized model validation
- End-to-end B1 T256: FP16 34.997 ms, identity-control 30.829 ms (1.135× noise)
- End-to-end B1 T512: FP16 32.341 ms, identity-control 33.732 ms (0.959×)
- End-to-end B4 T512: FP16 117.408 ms, identity-control 117.292 ms (1.001×)

## Actual kernel failures

1. `torchao_tiled_hqq` / `torchao_tiled_tinygemm` for three of the four
   projection shape classes: `Only bfloat16 is supported for
   Int4TilePackedTo4dTensor, got torch.float16`. T4/SM75 has no BF16.
2. `torchao_plain_tinygemm` for those shape classes:
   `ImportError('Requires mslk >= 1.0.0')`.
   Stable MSLK 1.0.0 supports CUDA architectures 8.0, 9.0a, 10.0a,
   and 12.0a, not SM75; installing it does NOT establish T4 support.
3. `mlp.Wo` was executable in all three backends but not faster enough
   at the actual encoder workloads: M=512 gains of 0.996×/1.011×/1.055×
   and M=2048 gains of 1.004×/0.997×/1.004×, below the 1.10× gate.

## Why this is not a contradiction of T4 INT4 tensor cores

Turing Tensor Cores can perform INT4×INT4→INT32 accumulation, but the
experiment used TorchAO **weight-only** INT4 with FP16 activation input.
The matrix operator is a different hardware/software path. Native T4 INT4
acceleration needs both activations and weights quantized to INT4 with a
compatible SM75 CUDA/CUTLASS kernel, and additional model fidelity testing.

## V6.1 changes

- Capability-aware preflight prints GPU, SM and BF16 availability.
- Default SM75 skips unsupported TorchAO weight-only INT4 backends instead
  of wasting an extended trial sweep.
- Explicit `ALLOW_EXPERIMENTAL_T4_KERNELS=True` lets researchers force
  kernel probing, without claiming the result will be usable.
- The baseline-only run is labeled `FP16_identity_self_check_not_INT4_validation`.
- It records `identity_control_FP16` speed = 1.0 rather than pretending
  arbitrary GPU timing noise is native INT4 improvement.
- Real INT4 candidates use paired ABBA measurements, reporting p10
  paired speedup and requiring robust B1/T512 and B4/T512 improvements.
- It exports `hardware_diagnostics.json` and `backend_qualification.json`.

## Next hardware test

Prefer A100 or H100 for current TorchAO BF16+INT4 weight-only kernels.
L4 is a useful experiment, but a 1.10× gain is not guaranteed; test
M=512 and M=2048 rather than LLM decoding M=1.
On SM75, the separate INT4×INT4 activation quantization/CUTLASS path
would be an entirely new runtime and cannot be claimed to be implemented.

Sources:
- https://docs.pytorch.org/ao/stable/workflows/inference.html
- https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int4WeightOnlyConfig.html
- https://pypi.org/project/mslk-cuda/
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/implicit_gemm_convolution.md
