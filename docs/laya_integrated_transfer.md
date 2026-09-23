# Integrated Memory V2.2: cumulative decision transfer

The previous committed notebook recorded 6/10 full-attention replacements,
93.9% final teacher agreement, and median demo latency of 48.07 ms for the
student versus 40.48 ms for the teacher. These are historical notebook outputs,
not results for this revision. The old final set also overlapped its acceptance
set. Fresh results will therefore not be directly comparable on identical cases.

## Changes

- Opt-in `integrated_functional_training` first fits local attention, then trains
  the candidate through the cumulative student's frozen decision stack. Only
  the current replacement core receives updates; QKV, output projection,
  backbone, decision heads, and prior accepted replacements stay frozen.
- Local loss retains projected-output and pre-output-projection fidelity.
  Decision refinement adds masked decision KL, action KL and centered-logit MSE.
- Checkpoints rank probe gate violations first, then teacher agreement, KL and
  local error. Local fidelity alone no longer triggers early stopping.
- Fit/probe partitions group questions by input state. Gate/final cases use
  disjoint workflow-stratified subsets. The final set never selects checkpoints.
- `target_all_attention` includes full and sliding layers in execution order.
  Full layers retain linear kernel memory. Sliding layers evaluate that kernel
  in bounded query chunks inside the native window. Padding is removed before
  convolution; convolution support cannot exceed the sliding radius.
- Reports distinguish target acceptance, remaining native attention, and final
  decision-quality gates. Export in the notebook requires final quality to pass
  and pins the TinyCeNN source revision in requirements.

## Run

Open `notebooks/Laya_Integrated_Memory_V22_Colab.ipynb` on a GPU runtime.
`smoke` runs one candidate for 60 steps; `extended` targets every attention layer
with up to 2400 steps each. The LR decays from 0.01 to 0.001.

The notebook requests teacher agreement >=0.99, output NMSE <=0.20,
cosine >=0.90, KL <=0.005 and accuracy drop <=0.005. The existing notebook's
NMSE/cosine/KL/accuracy-drop limits are retained; agreement is tightened from
0.90 to 0.99. Rejected layers remain native. Full target acceptance is a goal,
not a guarantee. Teacher agreement is distinct from gold-label accuracy.

The optional first-50-case diagnostic may overlap gate cases; use the report's
independent final evaluation for quality claims. Use warm repeated demo latency
and representative workloads to assess speed. Functional training adds backward
passes through the frozen downstream network, increasing training memory/cost.
The positive feature maps and sliding kernels may remain slower than fused SDPA
on short inputs. No GPU acceptance, latency improvement or full model training
result has been established for this revision.

## Verification

CPU tests cover the dense kernel reference and gradients, padding/window
isolation, actual ModernBERT RoPE/mask integration, state-dict reload, functional
training with a small decision model, frozen parameters, exception cleanup,
runner rejection rollback, disjoint final cases and truthful result reporting.
