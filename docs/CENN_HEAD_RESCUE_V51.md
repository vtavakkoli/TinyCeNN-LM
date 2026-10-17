# CeNN V5.1 — checkpoint-guided, quality-first head rescue

## Why a new experiment?

The uploaded `qwen35_cenn_head_rescue_v5.zip` is a **completed V5 quick run**, not an exception traceback. Seven layer-11 variants completed. None met the declared quality and speed acceptance criteria. The chosen `rescue_g0_delta_f64_w64` had validation mixer cosine **0.933948**, teacher next-token agreement **0.96585**, and paired validation speedup **0.98042×**. The held-out WikiText-2 test showed ΔNLL **+0.001727**, agreement **0.95931**, and speedup **0.97415×**, failing both representation and throughput requirements.

CeNN V5.1 is therefore an **accuracy-recovery ablation**, not a speedup claim. It preserves original pretrained Qwen3.5 full-attention KV group 0, and reuses all trained weights of the approximate group-1 V5 delta+Local64 core. The additional components are:

- **Per-query adaptive local/global router** with a headwise bias and a normalized query-dependent linear score. It reduces to V5's fixed per-head local gate when router weights are zero.
- **Zero-initialized, low-rank, query-only correction**, per approximate query head. Its initial output is identically zero; it adds no token-to-token attention outside the existing bounded local window.
- Controlled ablations: frozen V5 baseline, router-only, correction-only, combined router and correction, and joint fine-tuning of the memory basis.
- Optional model-level native-logits KL refinement, evaluated only on the validation split before any final audit.

The reusable V5.1 core resides in `src/tinycenn_lm/cenn_head_rescue_v51.py`; the Qwen wrapper remains `Qwen35HeadRescueV5` unchanged. This prevents duplicating model code in notebooks.

## Reproducible warm-start

The notebook reads `checkpoints/v5_L11_rescue_g0_delta_f64_w64.pt` from the supplied ZIP without unpacking paths. It verifies exact source model revision and checkpoint architecture, loads the old core's 10 state parameters, checks the five expected new parameters, and performs an **initial output parity test**. The original base model, RoPE/QKV, residual paths and output projections remain frozen.

Qwen3.5-0.8B text backbone is loaded from the pinned model SHA in the V5 ZIP. The original layer remains the teacher. The new model supports only **text-only unpadded prefill** with `use_cache=False`. Its standalone state can stream across token chunks but the Qwen wrapper does not yet integrate the hybrid KV cache.

## Scientific protocol

1. Train/local-calibrate on the initial WikiText-2 `train` tokens, rank on WikiText-2 `validation` only, using full-token next-token NLL and agreement.
2. Since the prior V5 run already examined the WikiText-2 **test** split, it cannot be treated as a new blind test for V5.1. Use a predeclared separate **withheld train tail starting at 250,000 Qwen tokens** as a new, disjoint adapter audit; label this honestly rather than claiming a new dataset.
3. Gate candidates at local cosine ≥0.95, native teacher top-1 agreement ≥0.98 and validation ΔNLL ≤0.10. An accepted performance model must then have a positive heldout change under the same thresholds and paired speedup **>1.05×**. If gates fail, keep native attention.
4. Paired runtime medians should be interpreted cautiously at short sequence lengths and small batches. More seeds, hardware configurations, longer context lengths, full-token and inference-latency tests are required before scientific speedup claims.
5. A slower, high-fidelity candidate is a research result, not a faster deployable model. Benchmark memory and cached decoding once those implementations are available.

## Limitations

- No claim of improved accuracy, speed, or novelty until an actual CUDA run has verified it.
- Only Qwen3.5-0.8B **full-attention layer 11**, group 0 exact, group 1 approximated, is included by default. Gated DeltaNet layers are not changed.
- The V5 source ZIP must be supplied by the user at notebook runtime; the repo does not contain model checkpoints.
- The proposed query correction is pointwise (not tokenwise memory retrieval), and the cellular recurrence uses Python chunk loops and a non-fused triangular solve. Native SDPA can be faster.
- No cached `.generate()` or variable-length padding, multimodal, beam search, multi-GPU model parallelism, or production release.

## Repository-first Google Colab (official V5.1 release)

- Code lives at `src/tinycenn_lm/cenn_head_rescue_v51.py` and tests at `tests/test_cenn_head_rescue_v51.py`; notebook uses **only** the checked-out repository source.
- Install dependencies, clone `main`, record the Git commit and SHA-256 of the V5.1 module, run its CPU test suite before loading Qwen.
- Load `checkpoints/v5_L11_rescue_g0_delta_f64_w64.pt` from the V5 result ZIP, verify the model revision and checkpoint shape, and record the checkpoint SHA-256 to distinguish checkpoint runs.
- Warm-start all V5.1 variants; select on validation, then restore **the exact selected architecture flags** for the diagnostic tail audit.
- Audit on a disjoint tail of the *training* partition only. It is not statistically independent of pretraining and must not be called a fresh independent external benchmark.
- Do not claim GPT-like generation, quality preservation or faster inference from CPU tests; cache and fused kernels remain unimplemented.

Open in Colab: https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V51_Warmstart_Colab.ipynb
