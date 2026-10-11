# Jev_CeNN Layer-15 branch ablation

[Open GPU Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_Layer15_Branch_Ablation_Colab.ipynb)

## Scope

The lab changes **only global encoder layer 15**, replacing the original
ModernBERT attention module with the V3.1 Jev_CeNN
`PDeltaV28FastWindowMemory` implementation. Every other original encoder
attention module is kept **native**; there is **no whole-encoder Jev_CeNN
conversion**, task-head training, or Hugging Face publication.

Starting point: one **already trained** layer-15 replacement, either restored
from `/content/Jev_CeNN/global_W1024/layer15_W1024.pt` or distilled on
original teacher layer-15 I/O for `GLOBAL_STEPS=400` using V3.1's unchanged
optimizer. Checkpoint load validates **layer ID, W1024 and strict state keys**.

## Interventions

- `full`: trained unmodified module; mandatory baseline.
- `manual_full`: rewritten forward, numerical parity must match `full`.
- `minus_local`: remove full-window dense SDPA (default W1024).
- `minus_landmarks`: remove compressed landmark SDPA.
- `minus_linear`: remove learned linear-memory branch.
- `minus_value`: remove direct pretrained value shortcut.
- `minus_cenn`: remove learned additive depthwise 5-tap convolution.
- `static_routing`: remove query-conditioned routing while retaining
  learned head-wise mixture logits.
- `only_local`, `only_landmarks`, `only_linear`, `only_value`:
  single branch without residual CeNN.
- `only_cenn`: pretrained value features processed by cellular
  convolution alone, without attention; **not retrained** by default.

Drop-one and only-one arms actually **skip calculating absent paths**, so the
layer latency captures avoidable compute rather than simply multiplying a
precomputed feature by zero. All other parameters start identical; no retrain
is done in the default run. When a branch is dropped, its routing probability
is renormalized among the remaining branches. Effects can interact and are
**not additive feature-importance scores**.

Optional `RECOVERY_STEPS > 0`: separately initialize each recovery arm
from the same trained full checkpoint; apply equal-budget, probe-selected
layer-output distillation; consult blind audit only after checkpoint selection.

## Separate observations

1. Layer-15 valid-token cosine and normalized mean squared error on DEV-AUDIT.
2. Downstream end-to-end typed-decision agreement, gold accuracy and Jensen–
   Shannon divergence on independent DEV cases (reference teacher logits
   cached once).
3. Paired native teacher layer and end-to-end **CUDA median/p95 latency**.
4. Learned mean branch-routing mass on valid input tokens.
5. Optional per-arm recovery curves, checkpoints, audit and task results.

The official held-out **test** split stays unobserved in this study. Multiple
ablation hypotheses are evaluated on DEV; conclusions are exploratory until
repeated on a completely separate held-out dataset.

## Outputs

- `jev_cenn_l15_surgical_ablation.csv`
- `jev_cenn_l15_ablation_report.json`
- `jev_cenn_l15_ablation.png`
- Optional `l15_recovery_<mode>.pt` files

A branch is a promising optimization only if it meaningfully reduces measured
latency while preserving end-to-end decisions and local fidelity. At W1024,
removing full-context SDPA can be fast but may degrade query-dependent token
retrieval. **Do not infer efficiency from cosine alone.**

All numerical results must come from a real Colab GPU run. GitHub CPU CI
checks code/parity contracts; it does not establish benchmark results.
