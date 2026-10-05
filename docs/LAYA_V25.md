# Laya V2.5: windowed Delta and multiscale recurrent CeNN

[Run in Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/laya-v25-window-delta-cenn/notebooks/Laya_Integrated_Memory_V25_WindowDelta_CeNN_Decision_Colab.ipynb)

Status: implementation tested on small CPU models; full CUDA training, kernel execution, quality and speed are **not yet measured**. The notebook contains no inherited outputs. V2.4 is unchanged.

## Architecture

| Component | V2.5 |
|---|---|
| Sliding encoder attention | Bidirectional Gated Delta in independent 128-token windows, stride 64, overlap-add output |
| Global encoder attention | CeNN with trainable input/output projections, fixed scales 1/4/16, two leaky recurrent updates, and global masked-context broadcast |
| Final decision head | Shared option MLP conditioned on CLS, sequence mean and option mean |
| Encoder FFNs / norms / residuals | Preserved from the source checkpoint |

Window states start at zero in each direction. This is an overlapping-chunk approximation, not exact sliding-window attention. Each token receives two window outputs. No state wraps from the sequence end to its start. Padding has neutral Delta updates and is excluded from CeNN/head pooling. CeNN uses fixed pooling bins so adding batch padding does not move cell boundaries. Its bounded local feedback template and sigmoid leak implement recurrent cellular updates, while the global drive carries whole-sequence context; it does not implement exact content-addressable attention.

`LinearMaskEncoder` preserves the pretrained FFN/residual computation but bypasses the native quadratic attention-mask builder. All encoder layers must be converted before it can be installed. The final module audit also rejects a remaining Transformer decision head.

## Preserved training protocol

| Stage | Steps | Initial → final LR |
|---|---:|---|
| A: progressive local replacement | 400 per encoder layer | 0.01 → 0.0005 |
| B: recovery with original decision head | 1600 | core 0.01 → 0.0005; head 0.002 → 0.0001 |
| C: direct gold supervision | 900 | core 0.003 → 0.0003; head 0.0006 → 0.00005 |
| Compact pooled head | 700 | 0.0008 → 0.00008 |
| Compact joint recovery | 400 | core 0.002 → 0.0002; head 0.00015 → 0.00003 |
| Compact direct gold supervision | 700 | core 0.002 → 0.0002; head 0.0004 → 0.00004 |

The V2.4 losses, cosine schedules, split sizes, batch size and checkpoint-selection formulas are retained. Stage A now processes **all** encoder layers, increasing total training work. Core discovery includes both CeNN and Delta parameters even after a blanket parameter freeze. CeNN projections learn; inherited local Delta QKV/output projections remain frozen, matching the V2.4 projection policy. The original-head intermediate is kept for recovery/comparison, but is never eligible for upload.

FP16 gradient scaling and native-BF16 detection are numerical safeguards. Transformers 4.57.6 and FLA 0.5.2 are pinned. The notebook checks FLA forward agreement against an explicit reference recurrence and finite gradients on the actual GPU before expensive training. CPU reference scans are for correctness/debug only, never a silent CUDA performance fallback.

## Evaluation and conditional Hugging Face export

The final model must meet all these defaults:

- Accuracy at most 0.02 below the unchanged teacher.
- The paired 95% case-bootstrap lower bound also at least -0.02.
- At least 1.10× median speedup, measured across the same five length-stratified dev batches for all models.
- Finite quality metrics and matching paired evaluation sizes.
- No original encoder attention or Transformer head modules.
- Local save/reload logits and action outputs agree within the mixed-precision tolerance.

`MAX_ACCURACY_DROP` and `MIN_SPEEDUP` are set before training. This gate is an engineering acceptance criterion, not proof of superiority. A rejected run retains its JSON report and skips upload. A passing run saves weights, tokenizer, architecture/source configuration, metrics, self-contained runtime files and a model card. It reloads without downloading the teacher before any Hub write.

`PUSH_TO_HUB=True` enables conditional upload. Leave `HF_REPO_ID` empty to use the authenticated account's `Laya-V25-WindowDelta-CeNN` model repository, or supply an authorized namespace. `HF_PRIVATE=True` is the default for new repositories; existing repository visibility is not changed. Authentication comes from the current Hugging Face session, a Colab `HF_TOKEN` secret, or interactive notebook login. No token is saved in outputs or artifacts.

The export provides `load_v25`, not `AutoModel.from_pretrained` or a text-generation pipeline. Run from the downloaded directory with its included `tinycenn_lm` runtime, install `requirements.txt`, then:

```python
from tinycenn_lm.decision_v25 import load_v25
model = load_v25(".", device="cuda", backend="fla")
```

Pack typed decisions with `build_sequence`/`collate_items` from the bundled `standalone_decision` module; use CUDA autocast in the validated FP16/BF16 dtype. CPU loading is available with `backend="reference"` for correctness/debug.

## Limits

This is a bidirectional typed-decision encoder, not a causal language model or a pure CeNN. Gated Delta is a recurrent/linear-attention-family mechanism. The original official test split is held out during this notebook run, but it informed earlier V2.4 research; use a fresh external benchmark for publication validation. The source teacher's data provenance is not established here. Action-head outputs are distilled from the teacher, not validated against independent action labels. Latency excludes tokenization; activation peaks exclude resident model weights.
