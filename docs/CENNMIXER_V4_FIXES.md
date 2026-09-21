# CeNNMixer-v4 runtime and training corrections

Use the notebook on `fix/cennmixer-v4-runtime-training`. Setup uses an isolated
checkout of that branch and Transformers 5.17.0, the version tested below.
Choose a GPU runtime and run all cells. Upload is disabled by default.

## Corrections

- Pad the streaming convolution history when the initial prompt is shorter than
  `conv_kernel - 1`; one-token decoding now agrees with full-sequence inference.
- Preserve input gradients through the frozen Qwen mixer, needed when earlier
  converted layers feed later progressive mixers.
- Initialize the decay projection bias to zero so `assoc_decay_init` sets the
  intended starting retention.
- Detect native BF16 support rather than accepting emulation on T4. Otherwise
  use FP16 on CUDA, with explicit finite loss/gradient checks before updates.
- Save initial best checkpoints even when a stage needs no updates, validate
  resume architecture/model identity, and persist metrics throughout training.
- Validate schedules and update budgets, fail nonfinite metrics closed, avoid
  NaNs for top-k=1, and bound data sampling attempts.
- Disable thinking for the short generation diagnostic so its token budget can
  be used for answers. Preserve model mode during on-policy distillation.
- Avoid destructive checkout resets, clear obsolete notebook outputs, tolerate
  empty history and older history columns, and make Hub upload explicitly opt-in.

The default output directory is `/content/cennmixer_v4_results`. For an existing
checkpoint, set `OUTPUT_DIR` to its directory and keep the same configuration.
`RESUME_ALPHA=0.4` starts at 0.45 in quick mode. For persistent checkpoints, mount
Google Drive and select a directory there before training.

## Validation and limits

`OMP_NUM_THREADS=1 python -m pytest -q tests/test_qwen35_cennmixer_v4.py`

12 CPU tests pass with PyTorch 2.14.0+cpu and Transformers 5.17.0. Tests cover
short-prefix streaming parity, frozen-weight input gradients, nonfinite gates,
top-k=1, notebook syntax and disabled upload, and a tiny randomly initialized
Qwen3.5 model's backward pass and finalized cached/full decoding parity.

The 0.8B model and datasets were not downloaded or trained; no GPU was available.
These fixes do not establish convergence, generation quality, throughput, or
memory use on Colab. Quick mode still permits diagnostic exploration separately
from acceptance. Existing strict export quality thresholds remain in force.
A quality-gate stop is not a software exception and must not be bypassed to
claim successful replacement. The five generation prompts and validation split
are diagnostics, not an independent final test benchmark.
