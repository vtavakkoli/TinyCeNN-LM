# TinyCeNN-LM

**Can shared recurrent CeNN computation add useful depth to a pretrained tiny Transformer without adding Transformer layers?**

TinyCeNN-LM is a small, reproducible research lab built on [`arnir0/Tiny-LLM`](https://huggingface.co/arnir0/Tiny-LLM). The upstream checkpoint is a ~13M-parameter, one-layer Llama-family causal language model pretrained on 32B FineWeb tokens. TinyCeNN-LM keeps that pretrained layer intact and adds a **causal Cellular Neural Network (CeNN) residual state core** whose weights are shared across recurrent iterations.

## Why this first design?

Replacing Tiny-LLM's only pretrained Transformer layer with random weights would throw away the checkpoint's learned computation. Version 0.1 therefore uses a safer experiment:

```text
Tokens
  │
  ▼
Pretrained embedding
  │
  ▼
Pretrained Llama decoder layer ──────────────┐
  │                                         │
  ▼                                         │
Causal CeNN state core                      │
(shared weights; x1/x2/x4/x8 iterations)    │
  │                                         │
  └──────────────── residual ───────────────┘
  │
  ▼
Pretrained norm + LM head
  │
  ▼
Next-token logits
```

The CeNN branch is **exactly zero at initialization**, so injecting it initially reproduces the pretrained model's decoder output. Fine-tuning can then learn the additional recurrent computation instead of first recovering from a destructive layer replacement.

## CeNN core

The adapter treats sequence positions as 1-D cells. Each recurrent update uses:

- strictly **causal depthwise neighborhood mixing**;
- a dilation schedule (default `1,2,4,8`) to expand the receptive field quickly;
- gated **SwiGLU** state updates;
- fp32-accumulated RMS normalization for stability;
- one **shared cell** reused for every recurrent step;
- zero-initialized output projection for safe pretrained-model insertion.

With kernel size 3 and four recurrent steps using dilations `1,2,4,8`, the CeNN branch has a 31-token causal receptive field while keeping the same trainable parameter count as a one-step CeNN.

## Quick start

```bash
git clone https://github.com/vtavakkoli/TinyCeNN-LM.git
cd TinyCeNN-LM
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

### 1. Fast smoke training

Start with 10M tokens and context 256. On an 8 GB CUDA GPU this is intended to be a short architecture test, not full pretraining.

```bash
python scripts/train_adapter.py \
  --max-tokens 10000000 \
  --context-length 256 \
  --batch-size 8 \
  --grad-accum 4 \
  --steps 4 \
  --output-dir checkpoints/tinycenn-base
```

Only the CeNN residual is trainable by default. The pretrained embedding, Transformer layer and LM head remain frozen.

For a longer run after the smoke test:

```bash
python scripts/train_adapter.py \
  --max-tokens 50000000 \
  --context-length 256 \
  --steps 4 \
  --output-dir checkpoints/tinycenn-base-50m
```

### 2. Compare speed with Tiny-LLM

```bash
python scripts/benchmark.py --steps 4 --context-length 256 --batch-size 4
```

To benchmark trained adapter weights:

```bash
python scripts/benchmark.py --adapter checkpoints/tinycenn-base --steps 4
```

### 3. Generate text

```bash
python scripts/generate.py \
  --adapter checkpoints/tinycenn-base \
  --prompt "The future of efficient AI is"
```

## First ablation matrix

Keep everything else identical and vary only recurrent computation:

| Model | Shared CeNN steps | Unique CeNN parameters | Goal |
|---|---:|---:|---|
| Tiny-LLM | 0 | 0 | pretrained baseline |
| TinyCeNN-LM x1 | 1 | fixed | local recurrent adapter |
| TinyCeNN-LM x2 | 2 | fixed | more compute, same parameters |
| TinyCeNN-LM x4 | 4 | fixed | default |
| TinyCeNN-LM x8 | 8 | fixed | test compute-depth scaling |

Measure validation loss/perplexity, tokens/s, peak VRAM, trainable parameters and wall-clock convergence. A useful result is not merely lower loss; it is whether **additional shared-weight CeNN iterations improve quality enough to justify their compute cost**.

## Design goals

1. **Preserve the pretrained model.** CeNN starts as an exact no-op.
2. **No information leakage.** All neighborhood convolutions are left-padded and causal.
3. **Parameter-efficient depth.** Recurrent steps share the same weights.
4. **Fast CUDA path.** The core uses depthwise `conv1d`, linear projections and PyTorch SDPA in the base model.
5. **Robust experiments.** Gradient clipping, mixed precision, finite-loss checks, streaming data, deterministic seeding and lightweight adapter checkpoints are built in.
6. **Easy rollback.** The base Hugging Face checkpoint is never overwritten; TinyCeNN weights are saved separately.

## Roadmap

- **v0.1:** residual CeNN adaptation of the pretrained Tiny-LLM layer.
- **v0.2:** teacher-distilled CeNN-only decoder replacement.
- **v0.3:** continued base-model pretraining and controlled Transformer/CeNN scaling studies.
- **v0.4:** instruction/SFT stage (`TinyCeNN-Chat`) after a successful base model.

## Upstream model

TinyCeNN-LM is an independent research project based on the MIT-licensed `arnir0/Tiny-LLM` checkpoint. See the upstream model card for its original training details and usage conditions.

## License

MIT
