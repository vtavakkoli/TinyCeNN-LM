# SmolLM2 integrated memory V3

[Open the new Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb).
Use a GPU, run `smoke` first if desired, then `balanced`.

## Assessment of the supplied V2 run

The uploaded `optimized_memory_report.json` is a completed SmolLM2-135M run on a Tesla T4,
seed 2027, source commit `9ee49ec93d009dc030aed8edbbf39e96837587ae`, with 32 test documents.
It already evaluated the full language model with individual replacements and with layers
0,18,29 replaced together. The joint test did not jointly retrain those replacements.

| Joint partition configuration | Context 256 | Context 512 | Context 1024 |
|---|---:|---:|---:|
| PPL / original | 1.00185 | 1.00976 | 1.02502 |
| PPL / adapted Transformer | 1.00279 | 1.01067 | 1.02587 |
| Complete forward speedup | 0.6667x | 0.9633x | 1.0053x |

This is promising preservation at short context, not a demonstrated overall improvement.
At context 1024 the joint NLL difference versus original has a positive 95% paired interval
[0.01945, 0.03012] nats/token. Quality degradation is distinguishable from zero in that run.
No bounded candidate met the V2 quality-preserving efficiency gate.

At context 1024, the partition candidate's single-layer PPL changes were +0.044% (layer 0),
+2.193% (layer 18) and +0.291% (layer 29). Its layer cache ratio was 0.1963, but its decode
speedup was about 0.22x. The sink control had around 1.88–1.93x reported prefill speedup,
while decode remained slower. Pure linear memory had substantially worse quality, especially
layer 0 (approximately +9.25% PPL at context 256). Consequently it is not in V3's main shortlist.

There is also a precision issue in the old benchmark: `is_bf16_supported()` can include
emulation. The uploaded T4 run selected BF16 and called it the native reference. V3 and the V2
runner now use CUDA compute capability to choose FP16 on T4, and BF16 on Ampere/newer NVIDIA
GPUs. Remeasure speed against this corrected baseline before interpreting a deployment gain.
This correction does not turn the measured V2 quality results into a win.

## Model and configurations

The official small checkpoint is [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
The family has 135M, 360M and 1.7B models; this experiment does not invent a 100M checkpoint
or silently change the parameter budget. Most original parameters remain present, and extra
core parameters are reported explicitly.

- Original pretrained model, untouched.
- Partition conservative: jointly train replacements at 0,29, retain layer 18 as attention.
- Partition expanded: jointly train 0,18,29 to test whether integrated training recovers the loss.
- Separate matched full-attention readout controls for each of those layer sets.
- Untrained sink/local attention at 0,29 as an inexpensive control.

These are partial attention replacements. The partition mixer itself contains exact local/sink
softmax attention. This is neither a fully attention-free model nor a fully converted 30-layer model.
Layer sets can be changed explicitly through the CLI, but the supplied results do not justify
blindly replacing all layers. The architecture choices were informed by the old test results;
V3 therefore uses fresh evaluation documents.

## What V3 changes

1. Ridge initialization and attention-transfer warmup use only training samples and teacher states.
   The raw V2 tables do not contain learned core checkpoint tensors, so this self-contained
   notebook retrains the architectures. It does not claim to reload prior weights from metrics.
2. Every learned configuration then trains all installed cores jointly using the actual student
   hidden-state distribution. All base-model parameters stay frozen. Only learned features,
   memory-mass parameters and/or head readouts receive gradients.
3. The objective is next-token cross entropy plus teacher-to-student KL divergence:

       L = CE(labels, student_logits) + lambda * tau^2 * KL(p_teacher(tau) || p_student(tau)).

   Defaults are lambda=1 and tau=1. The logit loss is chunked across tokens to limit temporary
   vocabulary-sized allocations. Gradient norms are clipped and nonfinite gradients stop the run.
4. Balanced training cycles through contexts 256,512,1024 for 300 joint updates. Every learned
   model receives the same token order, context schedule and update count, with different trainable
   parameter counts. A 100-step per-layer attention warmup precedes joint training. Sink control
   receives zero training. Candidate records report actual joint token presentations.
5. Best checkpoints minimize average validation NLL across training contexts, using the actual
   native-precision fused inference path. The before-joint checkpoint is retained as a fallback,
   so a worse last training step cannot overwrite a better validation checkpoint.
6. Lock the best bounded configuration on validation before any test evaluation. Report test NLL
   versus original, matched adapted attention and the model's own before-joint checkpoint. Test
   results cannot feed back into the saved choice. The best bounded choice can still be worse than
   the original model; the decision flags show whether it actually wins.
7. A mixed cache stores native-dtype KV for unchanged/full-attention layers and bounded FP32
   memory plus local KV for replaced layers. Position accounting works when layer 0 is replaced.
   Readouts are folded into O during inference, with original weights kept intact.
8. Test full-sequence versus cached logits and checkpoint reconstruction. Benchmark complete-model
   prefill and cached decoding, not isolated kernels. Use identical teacher-forced tokens across
   models, batch one, full Q/K/V/O and FFNs, and a last-position LM-head output. Warmup and loading
   are excluded. Multiple documents/repeats are saved, but timing confidence bounds are not claimed.
   Greedy text examples use actual autoregressive predictions in a separate qualitative check.

The underlying disjoint memory equation and ridge/readout derivations remain in
[OPTIMIZED_MEMORY.md](OPTIMIZED_MEMORY.md). V3 changes integration, training and evaluation,
not the claimed mathematical identity of the V2 core.

## Data, saved artifacts and reproducibility

`configs/smollm2_v3_prior_exclusions.json` contains the 144 normalized document hashes and source
revisions from the supplied V2 report; no document text or private Colab paths are committed.
All are excluded from train, validation and test. Hash buckets 0..39 (V1/V2 holdouts) are also
excluded. V3 uses 40..49 for test, 50..59 for validation and 60..99 for training. One sufficiently
long block is taken per unique document. Extra `--exclude-manifest` paths extend the exclusion
set. Other earlier training sets may overlap unless their manifests are also supplied. Original
pretraining contamination cannot be ruled out: SmolLM2 was itself trained on FineWeb-Edu.

The manifest records immutable model/data revisions, Git commit, seed, versions, GPU, compute
capability, precision, document hashes and token budget. Every run uses a new directory. Drive
receives best adapter checkpoints and progress during execution; interrupted optimizer-state
resume is not implemented. Starting the notebook again creates a new experiment.

Artifacts include `integrated_report.json`, `integrated_summary.csv`, `decision_table.csv`,
`validation_summary.csv`, `training_history.csv`, `test_document_nll.csv`, `selection.json`,
`generation_examples.json`, raw timing samples, plots, token blocks, and adapter checkpoints
before and after joint training. The optional notebook prompt cell reconstructs the chosen
model from its exact base revision plus saved adapter tensors.

## Interpretation

A quality-win flag requires the upper paired 95% NLL-difference bound below zero against both
original and matched adapted Transformer controls, with at least eight documents. An efficiency
flag requires both upper bounds <=0.02 nats/token, faster complete-model prefill and decode,
and lower total cache bytes. The tolerance is about 2.02% perplexity, not exact equivalence.
Intervals reflect document sampling, not training-seed variability; multiple comparisons are
exploratory. Confirm a promising model with untouched data, more seeds and your target hardware.

Cache savings must be reported at the model level. If only three of 30 equal-KV layers shrink
to 19.6% of their original cache, total cache is approximately (27+3*0.196)/30=91.96% of the
original: about 8% saved, not 80%. V3 measures the actual tensor bytes across all layers.
This is cache storage only; parameters, temporary activations and allocator overhead are distinct.

Supported inference is unpadded, causal, batch-one greedy decoding through the supplied helper.
Beam search, cropping compressed history, padded batching, offloading and external serving engines
are not supported. No claim of improved GPU quality, accuracy or throughput follows from CPU tests.

## Run

    python scripts/benchmark_smollm2_integrated_memory.py --output-dir result/smollm2-integrated-v3-new

    OMP_NUM_THREADS=1 python -m pytest -q tests/test_integrated_memory.py

The dedicated Colab CI pins Transformers 4.57.6 and validates notebook schema, code cells and
real tiny-Llama integration. Tests cover causal cache equivalence, joint gradients, frozen teacher
weights, adapter reload, native T4 precision selection, fresh document exclusions, full offline
training/selection/evaluation/generation, and the actual notebook reporting cell.

References: [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
[PyTorch BF16 support API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html),
[LoLCATs](https://arxiv.org/abs/2410.10254) for attention-transfer and hybrid-model motivation.
