# Laya V5.6 — two separate Liquid CeNN models, FULL then DELTA

[![Run FULL then DELTA on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V56_FiveLayer_FULL_Then_DELTA_LiquidCeNN_Colab.ipynb)

## Evidence from the user's V5.5 statistics

V5.5 all-28 FFN output CeNN substitutions yielded **31% gold on 200 TEST decisions** and **0.845x GPU speed** vs the compressed baseline; full-28 did not meet its release gates. The validation-based greedy model kept five selected FFN outputs at encoder layers **20,24,23,26,8** and achieved **77% gold TEST**, versus **75.5%** of the compressed baseline and **76.5%** of the pretrained teacher; the greedy model was **0.974x** as fast as the compressed baseline. The 1-3 decision differences on 200 cases are not statistically sufficient to establish a generalization gain.

The V5.5 uploaded results ZIP contains summary statistics and architecture specifications but **no five-layer trained CeNN tensors**. Therefore this V5.6 experiment reconstructs the same five liquid architectures *from scratch*, using the pinned pretrained model and real TRAIN FFN activations. It does **not** pretend to resume unavailable V5.5 weights.

## Two real variants, trained sequentially in one Colab

| | FULL first | DELTA second |
|---|---|---|
| Encoder layers | 20,24,23,26,8 | 20,24,23,26,8 |
| FFN output | LiquidCeNN(x) | OriginalCompressedWo(x) + alpha·LiquidCeNN(x) |
| Original FFN Wo remains | **No** | **Yes** |
| Original SVD compressed baseline | same for both | same for both |
| Zero-init exact baseline | No, activation matching required | **Yes**, alpha=0 |
| CeNN parameter saving at replaced FFNs | **Yes**, verify | **No**, additive overhead |
| Fine-tuning | Five CeNNs jointly, gold + teacher KL | Five residual CeNNs jointly, gold + teacher KL |
| HF model repository | `vtava/Laya-v56-FULL-5LiquidCeNN` | `vtava/Laya-v56-DELTA-5LiquidCeNN` |
| Export and acceptance gates | Separate | Separate |

The FULL mode first installs and activation-distills all five liquid FFN outputs *progressively*, records cumulative gold-validation accuracy, trains all five liquid cell matrices jointly, freezes, evaluates, exports and checks strict offline loading.

Only after finishing FULL does the notebook rebuild the untouched 20-SVD compressed baseline, attach five DELTA cells with zero correction scales, verify exact initial decision parity, jointly fine-tune, freeze, evaluate, export, verify and conditionally publish the DELTA model.

Each model is a **28-layer ModernBERT encoder** with five selected Liquid CeNN FFN output locations. No Transformer attention is replaced. A 1D token-neighbor liquid recurrence is *CeNN inspired*, not identical to the canonical Chua CeNN.

## Release and outputs

Each model must pass its own full gold-validation, TEST gold, teacher agreement, JS, actual model storage reduction relative to the original teacher, real GPU/wall latency, and strict saved/reloaded logits/argmax gates. FULL also must save at least 1MB of *theoretical FP32* projection parameters over the 20-SVD compressed baseline. DELTA is not expected to do that because it intentionally keeps the original FFN output projection.

A model failing acceptance is **not uploaded**. Each successful model is uploaded into its own new, private Hugging Face model repository. Existing repositories are not overwritten without explicit `ALLOW_UPDATE_EXISTING=True` in Colab.

The final browser download is **`Laya_V56_FULL_vs_DELTA_Statistics.zip`**, with reports, per-variant checkpoint selection, benchmark results, HF statuses, metadata, CSV and SHA-256 manifest, and **no model weights, adapters or tokenizer**. The full FP32 standalone files are stored separately on Hugging Face only if validated.

These two GPU experiments **have not yet run**. The code, strict loaders and design were prepared for Colab execution; outcomes and HF upload status must be checked in the resulting ZIP.

## Standalone loader

```python
from modeling_laya_v56 import load_v56_model

full_model, tok = load_v56_model("FULL_standalone_model",device="cuda")
delta_model, tok = load_v56_model("DELTA_standalone_model",device="cuda")
```

[Implementation](modeling_laya_v56.py) · [Colab](https://github.com/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V56_FiveLayer_FULL_Then_DELTA_LiquidCeNN_Colab.ipynb)
