# Laya V5.4 — Optimized, lower-parameter liquid CeNN experiment

[![Run optimized CeNN in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V54_Optimized_Liquid_CeNN_HF_Colab.ipynb)

## Analysis of the uploaded V5.3 results

The user's provided V5.3 reports show that the 64-channel / 3-step
replacement of layer 24 retained **75.5% gold accuracy on 200 TEST
decisions**, exactly equal to the compressed baseline and 1 pp below the
76.5% original teacher. It reached **83.33% gold on the 180-decision
validation** at its selected training step 140. Relative to the compressed
baseline its measured GPU latency ratio was **0.989x** (about 1.1% slower).

The chosen layer-24 projection originally had 233,472 parameters,
whereas its CeNN replacement had 247,233: that particular replacement
**increased**, rather than decreased, storage. Other candidate layers
(8, 17, 20, 23, 26) removed more parameters in the same V5.3 screen,
though full validation of their combined effects had not been done.

## V5.4 improvements

- Preserve original 28-layer ModernBERT, typed-decision head and 20
  pretrained SVD-compressed projections except one selected FFN output.
- Test narrower 32-channel liquid token-lattice cells with **2** (rather
  than 3) recurrent steps at layers 8, 17, 20, 23 and 26.
- Try 64-channel cells at the two strongest candidate layers.
- Screen real FFN activation imitation on TRAIN cases; select the final
  layer/width using the entire 180-decision validation subset.
- Reject any replacement whose parameters are **not fewer** than the
  original projection at that layer.
- Recreate an exact fully compressed control model with strict
  state_dict cloning (no second expensive SVD fitting), then measure
  paired GPU/wall latency against that control.
- Train only CeNN cell factors with reproducibly shuffled TRAIN examples,
  check full gold-validation score and stop on plateau.
- Perform a one-time 200-decision TEST evaluation after freezing weights.
  Prior experiments have also used the same TEST split, so this is not
  independent *across studies*.
- Export a complete FP32 standalone checkpoint and verify offline reload
  logit and decision parity.
- Only after **all** gates pass, privately upload the complete model to
  `vtava/Laya-v54-optimized-liquid-cenn`. The model can be released
  as fidelity-preserving *compression* even if it does not exceed
  the original teacher's gold accuracy, provided the loss is <=1.01 pp
  on 200 TEST decisions and it does not regress versus the compressed
  baseline. The model card must explicitly disclose that difference.
- **Download statistics only** at the end: `Laya_V54_Statistics.zip`
  contains JSON/CSV metrics, release status, selection results and SHA256
  manifest, but excludes `*.safetensors`, weight binaries and tokenizer.
  Failed release artifacts are retained in the Colab runtime and are
  **not** silently uploaded.

## What qualifies as "working well"

Release gates include unchanged-or-better gold correctness against the
compressed baseline on validation and TEST; <=1.01pp TEST gold loss versus
original teacher; >=90% TEST teacher top-1 agreement; mean JS <=0.008;
>=100MB full FP32 model byte reduction compared to original teacher;
>=0.2MB reduction in the replaced projection compared to the compressed
baseline; >=0.95x GPU and wall speed on sampled TEST batches; exact
standalone reload top-1 decisions with close logits.

A passing gate means the model is **releasable as a compression
experiment**, not that it outperforms the teacher or is universally faster.

## Source

- [Optimized V5.4 Colab](https://github.com/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V54_Optimized_Liquid_CeNN_HF_Colab.ipynb)
- [Liquid-CeNN standalone runtime](../laya_v53_liquid_cenn/modeling_laya_v53.py), embedded verbatim into the Colab.

**Status**: Code committed to GitHub. V5.4 GPU training, HF model
publication, and its measured statistics have **not** yet been run.
