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

## Docker: recommended training path

The repository includes a CUDA Docker image and Docker Compose services for training, benchmarking and generation. The `train` service automatically:

1. verifies that CUDA is visible inside the container;
2. downloads/caches `arnir0/Tiny-LLM` from Hugging Face;
3. streams `HuggingFaceFW/fineweb` (`sample-10BT`);
4. creates a fixed monitoring sample before training;
5. records the initial loss/perplexity;
6. fine-tunes the CeNN adapter;
7. periodically evaluates loss/perplexity and gradient health;
8. saves the best adapter automatically;
9. stops with an error if loss becomes non-finite or strongly diverges;
10. writes `training_report.json` with the complete health history.

The default Docker profile is conservative for an **8 GB GPU**: context 256, batch 4, gradient accumulation 8, CeNN x4 and 1M training tokens.

### Build and train

Docker Compose v2 with NVIDIA GPU support is recommended (Docker Desktop + WSL2 GPU support on Windows works well).

```bash
git clone https://github.com/vtavakkoli/TinyCeNN-LM.git
cd TinyCeNN-LM

docker compose build train
docker compose run --rm train
```

The first run downloads the base model and streams the dataset. Hugging Face files are kept in the persistent `hf-cache` volume for later runs.

Outputs are mounted back to the host:

```text
checkpoints/
├── tinycenn-base/
│   ├── cenn_adapter.pt
│   ├── cenn_config.json
│   ├── tokenizer files...
│   └── training_report.json
└── tinycenn-base-best/
    ├── cenn_adapter.pt
    └── cenn_config.json
```

At the end of the run you will see one of:

- `HEALTHY` - the best monitoring loss improved by at least the configured threshold;
- `WARNING_NO_IMPROVEMENT` - training stayed numerically stable but did not improve enough yet;
- `DIVERGED` - non-finite gradients/loss or excessive validation-loss growth; the container exits non-zero.

`training_report.json` includes initial/final/best loss, perplexity, best update, relative improvement, tokens processed, elapsed time, peak VRAM and the full evaluation history. FineWeb `sample-10BT` has no official validation split, so this fixed separately shuffled sample is a **training-health monitor**, not a publication-grade held-out benchmark.

### Configure a longer run

Copy the provided environment template:

```bash
cp .env.example .env
```

For a 10M-token run, change:

```dotenv
MAX_TOKENS=10000000
```

For 50M:

```dotenv
MAX_TOKENS=50000000
```

Then run the same command:

```bash
docker compose run --rm train
```

Useful 8 GB tuning variables are available in `.env`: `CONTEXT_LENGTH`, `BATCH_SIZE`, `GRAD_ACCUM`, `CENN_STEPS`, `LEARNING_RATE`, `EVAL_EVERY`, `EVAL_BATCHES`, `HEALTH_MIN_IMPROVEMENT`, and `NO_COMPILE`.

Set `FAIL_ON_NO_IMPROVEMENT=1` if you want CI/automation to return a non-zero exit code when the model stays stable but fails to improve by the requested threshold.

### Benchmark the best model

```bash
docker compose run --rm benchmark
```

By default this loads `checkpoints/tinycenn-base-best`. Override it with, for example:

```bash
ADAPTER=/workspace/checkpoints/tinycenn-base docker compose run --rm benchmark
```

### Generate text

```bash
PROMPT="The future of efficient AI is" docker compose run --rm generate
```

Generation intentionally uses the complete prefix (`use_cache=False`) in v0.1, because a normal Transformer KV cache does not contain the CeNN recurrent neighborhood state.

## Native Python quick start

```bash
git clone https://github.com/vtavakkoli/TinyCeNN-LM.git
cd TinyCeNN-LM
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

### Fast smoke training

```bash
python scripts/train_adapter.py \
  --max-tokens 1000000 \
  --context-length 256 \
  --batch-size 4 \
  --grad-accum 8 \
  --steps 4 \
  --output-dir checkpoints/tinycenn-base
```

Only the CeNN residual is trainable by default. The pretrained embedding, Transformer layer and LM head remain frozen.

For a longer run:

```bash
python scripts/train_adapter.py \
  --max-tokens 50000000 \
  --context-length 256 \
  --steps 4 \
  --output-dir checkpoints/tinycenn-base-50m
```

### Compare speed with Tiny-LLM

```bash
python scripts/benchmark.py --steps 4 --context-length 256 --batch-size 4
```

To benchmark trained adapter weights:

```bash
python scripts/benchmark.py --adapter checkpoints/tinycenn-base-best --steps 4
```

### Generate text

```bash
python scripts/generate.py \
  --adapter checkpoints/tinycenn-base-best \
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

Measure monitoring/held-out loss and perplexity, tokens/s, peak VRAM, trainable parameters and wall-clock convergence. A useful result is not merely lower loss; it is whether **additional shared-weight CeNN iterations improve quality enough to justify their compute cost**.

## Design goals

1. **Preserve the pretrained model.** CeNN starts as an exact no-op.
2. **No information leakage.** All neighborhood convolutions are left-padded and causal.
3. **Parameter-efficient depth.** Recurrent steps share the same weights.
4. **Fast CUDA training path.** The core uses depthwise `conv1d`, linear projections, optional `torch.compile`, and PyTorch SDPA in the base model.
5. **Correct autoregressive semantics.** v0.1 intentionally uses `use_cache=False` during generation. A normal Transformer KV cache does not preserve the recurrent CeNN neighborhood states, so disabling it avoids a silent train/inference mismatch.
6. **Robust experiments.** Gradient clipping, mixed precision, finite-loss/gradient checks, streaming data, deterministic seeding, periodic evaluation, best-checkpoint saving and JSON health reports are built in.
7. **Easy rollback.** The base Hugging Face checkpoint is never overwritten; TinyCeNN weights are saved separately.
8. **Reproducible container path.** The default CUDA/PyTorch image is pinned and can be overridden with `PYTORCH_IMAGE`.

## Roadmap

- **v0.1:** residual CeNN adaptation of the pretrained Tiny-LLM layer.
- **v0.2:** teacher-distilled CeNN-only decoder replacement plus a dedicated streaming CeNN state cache for fast token-by-token generation.
- **v0.3:** continued base-model pretraining and controlled Transformer/CeNN scaling studies.
- **v0.4:** instruction/SFT stage (`TinyCeNN-Chat`) after a successful base model.

## Upstream model

TinyCeNN-LM is an independent research project based on the MIT-licensed `arnir0/Tiny-LLM` checkpoint. See the upstream model card for its original training details and usage conditions.

## License

MIT
