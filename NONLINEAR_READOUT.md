# Nonlinear recurrent readout experiment

Two runnable Colabs test PDelta3-GDN2-CLVR temporal memory with a small, shared
nonlinear readout. This is an **active, unvalidated experiment**. There is no
claim that it beats either original model or reaches 2x generation speed.

- [SmolLM2-135M Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_Nonlinear_Recurrent_Readout_Colab.ipynb)
- [Qwen3.5-0.8B Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_Nonlinear_Recurrent_Readout_Colab.ipynb)

Select a GPU runtime and Run all. The balanced profile is the default. Smoke is
for correctness only; extended increases the adaptation and evaluation budget.
Colab mounts Drive and saves checkpoints and optimizer state automatically.
No HF token or automatic model publication is required for these public models.

## Layer

For each KV head, retain the existing input-dependent affine memory recurrence:

\[
\bar S_t=D_tS_{t-1},\qquad
S_t=\bar S_t+k_t(z_t-\bar S_t^Te_t)^T,\qquad r_t=S_t^Tq_t.
\]

Conv4 mixes causal Q/K/V. Independent erase/write gates and channel-wise decay
come from the repository's GDN2 adaptation. CLVR routes current-token values
from the preceding **selected replacement** layer; with nonconsecutive targets,
this is not the immediately preceding decoder layer. The first replacement
uses its own V. Joint-training gradients propagate through the routed value.

Apply a headwise, low-rank nonlinear residual to the recurrent readout:

\[
u_t^{(0)}=r_t,\qquad
u_t^{(j+1)}=u_t^{(j)}+\sigma(\eta)A\,
\operatorname{SiLU}(B\operatorname{RMSNorm}(u_t^{(j)})+
C\operatorname{RMSNorm}(q_t)).
\]

A, B, C, eta are shared across refinement steps. The condition is the current
token's projected, rotary-positioned query. A starts at zero, so adding the
refiner initially preserves the recurrent readout exactly. One and two steps
have identical parameter counts, but two steps cost more computation. Zero
steps has no refiner parameters. The refiner adds no temporal cache.

The result is mixed with exact W-token causal local attention by an
input-dependent sigmoid gate. Qwen retains its original Q/K normalization and
post-attention output gate; the nonlinear readout is not folded through it.
The pretrained projections, FFNs, embeddings, and output head stay frozen.

A recurrent matrix has fixed capacity. Refining its readout cannot recover
information already erased from it. Better retrieval or smaller-state quality
is a hypothesis to test, not a consequence of adding SiLU or recursion.

## Implementation and limitations

- Bounded local attention constructs T-by-W scores, not a dense T-by-T mask.
  GQA queries use a grouped view instead of duplicating local K/V heads.
- Stateful prefill, chunk continuation, and token decoding preserve recurrent
  state, convolution tails, W-1 local K/V entries, and the actual position.
- Single-token delta updates bypass triangular solves and avoid GQA memory
  expansion. Prefill still uses the existing chunkwise PyTorch reference.
- FP32 arithmetic and persistent state are the conservative default. Optional
  FP16 persistent state is rounded at cache-call boundaries and therefore needs
  the cache-equivalence gate. It does not make recurrent arithmetic FP16.
- All candidates and the original use native FP16 on T4 or BF16 on Ampere/newer
  for their base model. The new memory and nonlinear refiner compute in FP32.
- This version does **not** implement fused Triton/CUDA prefill/backward kernels,
  adaptive halting, state nonlinearities, beam search, padding, serving-engine
  integration, or compressed-history cropping. Use the supplied greedy helper.
- The actual two models are SmolLM2-135M (Llama) and Qwen3.5-0.8B (already a
  hybrid of recurrent and full-attention layers). Only selected full-attention
  layers are replaced. Whole-model cache is therefore not context-independent.

## Controlled comparison

| Candidate | Purpose |
|---|---|
| Original | Unmodified pretrained model |
| Attention control | Same selected full-attention layers with trainable head readouts |
| recurrent_r0 | PDelta3-CLVR + local attention without nonlinear refinement |
| recurrent_r1 | Same memory, one nonlinear refinement step |
| recurrent_r2 | Same memory, two shared refinement steps |

Defaults: F=64, W=32, bottleneck rank=16. Replace SmolLM2 layers 0,1,2 or
Qwen full-attention layers 3,7,11. Memory initialization is identical across
R0/R1/R2 per layer. Every learned candidate receives identical token order,
update count, contexts and joint objective; trainable parameter counts differ
and are reported. The attention control helps distinguish adaptation gains
from architectural gains, but is not an equal-parameter comparison.

The balanced run has 96/16/32 training/validation/test documents, training
contexts 256/512, and tests at 256/512/1024. It uses 20 transfer-heavy warmup
updates and 150 joint updates. The objective is next-token CE plus teacher KL
and an attention-output NMSE term. The KL weight decreases during joint
training; all installed replacements train together. The zero-update checkpoint
can win if training makes validation loss worse.

Normalized document hashes define disjoint partitions. There is one block per
document and no synthetic-data fallback. The bundled prior-exclusion manifest
is applied; additional `--exclude-manifest` paths exclude every listed hash.
These documents may overlap older experiment training sets unless their
manifests are supplied, and may overlap the base model's pretraining corpus.
Use new final-test documents after tuning with these results.

Candidates are selected only on validation NLL, before test evaluation. Paired
bootstrap intervals describe document uncertainty for one training seed;
multiple comparisons are exploratory, without familywise correction. A larger
held-out dataset, multiple seeds and downstream/retrieval tasks remain needed
for a general quality claim.

## Measurements and decisions

Full-model prefill and cached decoding include Q/K/V/O, FFNs and the last-token
LM head. Batch size is one. Decode uses identical teacher-forced token inputs,
not divergent generated continuations. Greedy examples are saved separately.
Timings use CUDA synchronization, warmup and medians across documents/repeats.
Weights stay unquantized. GPU timing excludes data loading and model loading.

Only the model being measured is GPU-resident; the distillation teacher is moved
to CPU before candidate timing. Report actual full-model cache bytes and total
peak allocated CUDA bytes (weights + cache + temporaries). CPU runs have no GPU
memory ratio and cannot earn a speed-win label. Peak allocated bytes are not
allocator-reserved bytes or a measurement of another process's GPU use.

- `strict_quality_win`: upper paired 95% NLL-difference bounds below zero versus
  both original and adapted attention control, with at least eight documents.
- `quality_within_margin`: upper bounds <=0.02 nats versus both; this allows
  about 2% worse perplexity and is not a quality win.
- `meets_requested_target`: strict quality win, >=2x measured **decode** speed,
  lower full-model cache and peak allocated memory, on CUDA. Prefill is reported
  separately; this flag does not assert 2x faster prefill or training.
- Qwen target flags are additionally disabled when native FLA/causal-conv
  packages are unavailable or Transformers reports reference-kernel fallbacks.
  Full warning messages are saved in `backend.json` and the Colab log.

Replacing three layers may yield only small whole-model savings. The 2x target
is deliberately demanding; a negative result is still a valid experiment.

## Persistence and rerunning

The Drive directory contains `manifest.json`, `token_blocks.pt`, per-candidate
best adapters and `*-resume.pt` optimizer/scaler state, histories, validation
selection, per-document test NLL, `summary.csv`, `report.json`, raw timing samples,
generation examples and charts. Adapters record the exact base-model revision.

To continue an interrupted run, retain the printed RUN_NAME and all settings,
set RESUME=True, and rerun. Progress resumes at the last evaluation checkpoint;
intervening unsaved updates are repeated. Candidate initialization and the data
order are deterministic. Completed candidates load their best saved state.
Configuration or relevant source-code changes are rejected on resume. Pin
REPO_REF to the manifest's source_commit when reconstructing an old run.

To compare F=96 or W=64, start a separate run. Keep final selection away from the
test documents, and supply prior manifests when you need a fresh final holdout.
The optional custom-prompt cell reconstructs the validation-selected adapter.

## Local commands

```bash
python -m pip install 'transformers==5.17.0' 'datasets==4.8.5' sentencepiece pandas matplotlib
python -m pip install -e . --no-deps
python scripts/benchmark_nonlinear_readout.py --family smollm2 --layers 0,1,2 --output-dir result/smol-nonlinear-new
python scripts/benchmark_nonlinear_readout.py --family qwen35 --layers 3,7,11 --output-dir result/qwen-nonlinear-new
OMP_NUM_THREADS=1 python -m pytest -q tests/test_nonlinear_readout.py tests/test_research_layers.py tests/test_pdelta3_frontier.py
```

The Colabs preserve the runtime's installed CUDA PyTorch rather than replacing
it with a CPU build. Dependencies are installed before launching a fresh Python
process. Public model downloads and actual GPU timings happen in your Colab.

## Research basis

- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791): independent erase/write
  recurrence and chunkwise parallelization.
- [LoLCATs](https://arxiv.org/abs/2410.10254): attention transfer and adaptation
  of pretrained models.

The shared nonlinear readout is this experiment's proposed extension; no
literature novelty claim or pretrained-model performance claim is established.
