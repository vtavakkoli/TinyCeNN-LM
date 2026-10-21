# Laya V5.3 — 1D liquid CeNN FFN replacement / residual experiment

[![Run V5.3 in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V53_Liquid_CeNN_Colab.ipynb)

## Purpose

V5.1 successfully preserved all teacher top-1 decisions on a 120-decision
development PROBE but did not increase gold accuracy. V5.2 supervised gold
training yielded **83.33% student vs 85.00% teacher on a 180-decision
validation** and **75.5% vs 76.5% on a 200-decision test**. Its student
corrected 6 previously wrong teacher answers while introducing 8 errors.

V5.3 tests a different **CeNN-inspired recurrent architecture** in the
original **28-layer ModernBERT encoder**, retaining the 20 selectively
compressed attention and FFN projections of prior experiments.

## Actual architecture

The experiment targets one layer's `encoder.layers[i].mlp.Wo` output
projection and replaces or augments it using `LiquidCeNNProjection`.

Each CeNN unit acts on **a 1D token lattice**, not a 2D image lattice.
It first projects the FFN activation into a configurable 64-channel space.
At each of three recurrent steps the network:

1. Computes **local lateral coupling** with a depthwise 3-tap convolution
   across neighboring tokens.
2. Computes a candidate state from the current input, lateral coupling,
   and recurrent state transform.
3. Uses a **state-dependent, bounded update gate** to relax the state
   toward that candidate.
4. Projects the resulting recurrent state into the FFN output dimension.

An illustrative discretized update:

    u = tanh(Norm(W_in x))
    h_0 = 0
    h_(k+1) = h_k + dt(u,h_k) * (tanh(u + K*h_k + W_h h_k) - h_k)
    y = W_out h_steps

where `K` is a depthwise token-neighbor convolution and `dt` depends on
the current state. This is **CeNN-inspired** and **liquid/gated in its
dynamics**. It is **not** the original continuous-time Chua CeNN, a
canonical Liquid Time-Constant (LTC) differential equation, or a proof
that CeNN can replace Transformer self-attention.

## Two comparison modes

**Mode = replace**: the original dense/SVD output projection is removed
from the model and checkpoint. To avoid comparing a random cell against a
pretrained Transformer, the candidate first learns to imitate real FFN
input/output activations using a relative-MSE loss. The notebook screens
candidate layers on validation only; replaces the FFN projection only when
the resulting whole-model fidelity passes the screen.

**Mode = residual**: retain the original `Wo` and use
`Wo(x) + alpha * LiquidCeNN(x)`. `alpha=0` initially, making the whole
model's outputs **identical** to the existing baseline. If all direct
replacements fail the screen, the notebook uses this residual control.
**Residual mode increases rather than decreases the parameters of that
layer and is not inherently faster.**

## Evaluation

- Compare original frozen teacher, 20-projection compressed baseline and
  liquid intervention on the same examples.
- Experiment screens layers 8,17,20,23,24,26 by default (all 28 optional).
- Pretrains replacement cells on captured FFN activations from TRAIN cases.
- Selects a mode and layer with a subset of gold validation, fine-tunes
  gold correctness, and chooses a validation checkpoint.
- Performs one TEST split audit **in that run** without using TEST examples
  for training/selection. Earlier experiments already used this test split,
  so it is not a pristine cross-study holdout.
- Measures real GPU and wall speed against the unchanged compressed baseline.
- Exports full standalone FP32 `model.safetensors`, tokenizer,
  `config.json`, reports and the exact custom loader below.
- Upload to the **private** Hugging Face model
  `vtava/Laya-v53-liquid-cenn` only if quality, reload and size checks pass.
- Always builds and auto-downloads `Laya_V53_All_Results.zip`, independent
  of Hugging Face publication status, and verifies ZIP contents/manifests.

### Implementation

- [modeling_laya_v53.py](modeling_laya_v53.py): standalone runtime including
  `LiquidCeNNProjection`, `install_liquid_ffn` and `load_v53_model`
- [Colab notebook](https://github.com/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V53_Liquid_CeNN_Colab.ipynb): training, ablations,
  independent evaluation and export

Load a fully exported model without a teacher download:

```python
from modeling_laya_v53 import load_v53_model
model,tokenizer=load_v53_model("standalone_model",device="cuda")
```

**Status:** Implementation and Colab committed; no V5.3 GPU run has been
performed yet. There are **no empirical accuracy, memory or latency
improvements to claim at this time**.
