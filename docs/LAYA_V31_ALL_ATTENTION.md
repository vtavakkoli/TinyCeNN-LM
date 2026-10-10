# Laya V3.1 — Complete ModernBERT Attention Conversion

[Open the training, testing and HF publishing Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V31_AllAttention_FineTune_HF_Colab.ipynb)

## Model architecture

The original `convaiinnovations/laya-typed-decisions` ModernBERT encoder has
30 attention layers (10 global, 20 sliding). V3.1 replaces **every** attention
module by a corresponding learnable research adapter:

- **Global:** V2.9 `PDeltaV28FastWindowMemory` with default W1024; native
  QKV and RoPE preserved; a mix of bidirectional local SDPA, pooled
  landmarks, positive-feature linear memory and direct value residual.
- **Sliding:** V3.0 `SlidingWindowFusion` with default bidirectional radius
  R64 (W129 inclusive); native QKV and local RoPE preserved; bounded/blocked
  sliding SDPA plus 7/33/65-token multiscale convolutions and a learned
  per-head, per-token router.
- **Task head:** The full Laya typed-decision head remains available and can
  be fine-tuned at a much smaller learning rate.

**Not attention-free:** W1024 at 512 tokens is a quadratic full-context
attention control. W129 remains sliding SDPA. Any speedup must be *measured*
against the original encoder; none is presumed.

## Running the Colab

1. Open a CUDA Colab runtime. Run setup + data preparation.
2. If already trained V2.9 global checkpoint is accessible, set
   `USE_V29_CHECKPOINT=True` and point `V29_CHECKPOINT` to the actual
   `v29_all_global_W1024.pt`. Otherwise V3.1 independently trains each
   global adapter (400 updates by default), saving checkpoints.
3. If V3.0's individual sliding-layer files are available, set
   `USE_V30_SLIDING_CHECKPOINTS=True` and point `V30_SLIDING_DIR`
   to `layers_W129` (or matching W). Otherwise V3.1 trains missing
   sliding layers where their pretrained near-native initialization does
   not already meet local fidelity targets.
4. The model is then fine-tuned *jointly*, with teacher consistency and
   supervised labels. Independent DEV evaluation selects the best state;
   the pre-fine-tune checkpoint is retained if optimization regresses.
5. Only then is the **official test split** loaded. The notebook measures
   teacher agreement, gold accuracy, JS divergence, and actual GPU latency.
6. The complete converted model is saved, reconstructed with both adapter
   types, strictly checked against its original checkpoint and independently
   exported to a standalone Hugging Face bundle.
7. With `PUBLISH_TO_HF=True` (default), add a write-enabled `HF_TOKEN`
   to Colab Secrets. Publication requires the quality gate unless
   `PUBLISH_ONLY_IF_PASSES=False` is explicitly selected.
   `HF_PRIVATE=True` is the default; existing public repos are refused
   when private publication is requested. No token is written to the bundle.

Change `OUTPUT_DIR` to a persistent mounted Drive directory if your
per-layer runs span multiple Colab sessions; `/content/` disappears on
runtime reset.

## Metrics and scientific acceptance

Default local criteria: cosine >= **0.99**, NMSE <= **0.025** for each
individually audited attention replacement. Additional DEV model-level
criteria: teacher decision agreement >= **0.99**, and gold accuracy
within **1.5 percentage points** of the teacher. The decision threshold
is a gate for publication, not a claim of measured performance.

Model selection uses TRAIN and DEV only. The official TEST partition is
evaluated once after the final checkpoint is saved.

## Hugging Face deployment

The export contains complete `model.safetensors`, `bundle.json`,
`runtime.py`, `load_model.py`, `encoder/config.json`,
`tokenizer/`, `training_report.json` and `requirements.txt`.
The bundled runtime is independent of `tinycenn_lm` and `laya`, and
uses only standard PyTorch/Transformers/safetensors/HF Hub dependencies.

```bash
pip install torch "transformers==4.57.6" safetensors huggingface_hub
hf download YOUR_USER/YOUR_MODEL --local-dir ./laya-v31
```

```python
import sys
sys.path.insert(0, "./laya-v31")
from load_model import load_model, predict

model, tokenizer = load_model("./laya-v31", device="cpu")
choice = predict(
    "./laya-v31",
    state={"speed": 20},
    question={
        "type": "choice",
        "instructions": "Choose safe action",
        "criteria": {"brake": None, "accelerate": None},
    },
)
print(choice)
```

Loading a model bundle executes its bundled `runtime.py` code.
Review the published code and pin a known HF revision before loading
models from untrusted authors. The model is a **typed-decision** model, not
a generic text generator.

## Validation status

The notebook is a reproducible Colab training program. Its new GPU
training, full test results and Hugging Face upload are **not executed
by GitHub CI**. CI can validate Python syntax, standalone CPU reconstruction
and numerical parity on small ModernBERT fixtures. The user must run
Colab with a GPU, a valid HF token, and valid training data to produce
and publish the actual trained full-size model.
