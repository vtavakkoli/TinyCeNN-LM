# Laya V5.5 — Brute-force all 28 liquid CeNN FFN output substitutions

[![Open all-28 brute-force Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V55_All28_BruteForce_LiquidCeNN_Colab.ipynb)

## Purpose and truthful scope

The V5.4 experiment replaced **one FFN output projection** with a
liquid CeNN-inspired token-neighbor recurrent cell, recovered the teacher's
gold accuracy on 200 test decisions (76.5%), and yielded meaningful model
parameter reduction. It did **not** replace the remaining 27 FFN
projections or any original attention computations.

V5.5 **FORCES all 28 FFN output projections `encoder.layers[i].mlp.Wo`
for i = 0..27 to be replaced** in one model. This is a 28/28
FFN-projection substitution, **not** a claim that complete Transformer
layers (self-attention, layer norms, residuals, etc.) were all replaced.

## Experimental sequence

1. Rebuild the pretrained ModernBERT 28-layer encoder with the original
   20 selected compressed SVD/delta projections, plus decision head.
2. Create an exact weight-cloned compressed baseline using
   `load_state_dict(strict=True)`. Do **not** refit factors for the baseline.
3. For each of 28 FFN output layers test 3 liquid widths (16,32,64)
   against 2 recurrent-step counts (1,2): up to **168 independent trials**.
   Candidate FFN cells that do not reduce that projection's parameters
   are screened out without training.
4. Fit each local candidate to the real FFN input/output activations on
   TRAIN cases. Screen each on 24 gold-VALIDATION decisions. Save the
   best architecture and local learned factors for each layer.
5. Force all 28 layer substitutions concurrently, compute a cumulative
   28-row validation-fidelity trajectory, and perform a joint 28-cell
   gold + teacher-KL training with checkpoint selection on full
   180-decision VALIDATION.
6. In a **separate ablation** perform greedy rollback/selection on the
   same validation subset to discover a viable smaller set. Never call
   this greedy partial model an all-28 replacement.
7. Evaluate forced 28, partial greedy subset, original teacher and
   compressed baseline on 200 TEST decisions **after frozen selection**.
   Evaluate paired CUDA event and wall latency.
8. Strictly save/reload a complete full FP32 model with all **28**
   CeNN FFN outputs. Verify forward logit and argmax parity.
9. **Publish to Hugging Face only if all strict full-28 quality gates pass**,
   as `vtava/Laya-v55-all28-liquid-cenn`. Existing HF models are not
   overwritten without opt-in.
10. **Automatically download statistics only** at the end:
    `Laya_V55_All28_Statistics.zip` with JSON, CSV, SHA-256 manifest,
    HF status and no weight/tokenizer binaries. The complete model
    remains in Colab and uploads to HF only when validated.

## Cautions

- Brute-forcing all 28 FFN substitutions can seriously hurt accuracy,
  and cumulative errors are precisely what the experiment measures.
- More CeNN layers do not automatically reduce latency; convolutions,
  recurrent steps and launches can outweigh FLOP reductions.
- Teacher agreement is not gold accuracy.
- The TEST split has been used in prior V5.x research, so it is not
  a never-examined cross-study holdout.
- Publication is not guaranteed. If the all-28 model fails any gate,
  it is not uploaded; the research metrics still download.
- Expected to be substantially more computationally expensive than
  a one-layer CeNN scan. Tune `LOCAL_DISTILL_STEPS` or widths if limited
  by Colab credits; report changed settings faithfully.

## Standalone inference

An actually validated, exported full-28 checkpoint loads with:

```python
from modeling_laya_v55 import load_v55_model
model,tokenizer=load_v55_model("/path/to/standalone_model",device="cuda")
```

[Runtime](modeling_laya_v55.py) ·
[Notebook](https://github.com/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V55_All28_BruteForce_LiquidCeNN_Colab.ipynb)

**Status:** The experiment is committed; full-28 GPU results and any
associated Hugging Face publication remain unverified until the Colab runs.
