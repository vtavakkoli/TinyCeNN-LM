# Transformer → CeNN Teacher Distillation

This experiment removes the only Transformer decoder layer from `arnir0/Tiny-LLM` and replaces it with a recurrent causal CeNN core. The pretrained tokenizer, token embeddings, final RMSNorm and LM head are retained as the language interface; the Transformer attention and FFN parameters are not present in the student.

## Architecture

```text
                         frozen teacher
input ids ─────────────► Tiny-LLM Transformer ─────► teacher logits
    │                            │
    │                            └───────────────► teacher hidden state
    │
    └─► pretrained embedding
              │
              ▼
       CeNN-only core ×7
       shared recurrent cell
       dilations 1,2,4,8,16,32,64
       receptive field = 255 tokens
              │
              ▼
       pretrained final norm
              │
              ▼
        pretrained LM head ───────────────────────► student logits
```

Default loss:

```text
L = 1.0 * CE(next token)
  + 1.0 * KL(student logits || teacher logits, T=2)
  + 0.25 * (1 - cosine(student hidden, teacher hidden))
```

The student is trained only through the CeNN replacement core in v0.2. The copied embedding, final norm and LM head remain frozen so the experiment isolates whether CeNN dynamics can recover the removed Transformer computation.

The original 10M-token experiment remains in `scripts/train_distill.py` unchanged for reproducibility. The stronger continuation protocol is versioned separately in `scripts/train_distill_rigorous.py`.

## Rigorous-v2 benchmark protocol

The continuation experiment strengthens the original 10M-token proof-of-concept without changing the CeNN architecture.

### Deterministic held-out split

FineWeb documents are assigned by a stable BLAKE2 hash:

- buckets 0–989 / 1000: training
- buckets 990–999 / 1000: validation

The validation side is never shuffled and the exact token batches are fingerprinted with SHA-256. The default rigorous benchmark uses:

```text
64 batches × 4 sequences × 256 tokens = 65,536 held-out tokens
```

The SHA-256 token fingerprint is stored in `distillation_report.json` and must match again after the model is uploaded to and downloaded from Hugging Face.

### Shuffled streaming training

The 99% training stream is passed through a deterministic bounded-memory shuffle before token packing:

```text
shuffle buffer = 4096 documents
seed           = 42
```

This removes the previous dependence on raw FineWeb streaming order while preserving bounded memory usage.

### Continuation instead of restart

`train_distill_rigorous.py` accepts `--resume-student-dir`. The rigorous Colab notebook downloads the published 10M-token `TinyCeNN-LM-Distilled` best checkpoint and continues from it with:

```text
additional tokens = 30,000,000
learning rate     = 3e-4
CeNN steps        = 7
context           = 256
```

The report records tokens from the previous checkpoint, tokens in the current run, and cumulative distillation tokens separately. The resumed model is snapshotted as **best-at-update-0**, so the reported best metric and published best weights remain identical even if continuation does not improve.

### Teacher-gap target

The rigorous run recomputes three points on the same 65,536-token validation benchmark:

1. **cold CeNN**: freshly replaced, untrained CeNN core;
2. **run start**: resumed published checkpoint;
3. **best continued student**.

Teacher-gap recovery is calculated against the cold CeNN and frozen Tiny-LLM teacher on this exact same benchmark. The default target is **90% recovery**.

Statuses are:

- `target_reached`: teacher-gap recovery ≥ target;
- `healthy_progress`: continued CE improves meaningfully but target is not reached;
- `warning_no_improvement`: no meaningful continued improvement;
- `diverged`: non-finite optimization/evaluation metrics.

## Colab notebooks

Initial Transformer → CeNN distillation:

`notebooks/TinyCeNN_Distill_Colab.ipynb`

Rigorous continuation from the published 10M checkpoint:

`notebooks/TinyCeNN_Rigorous_Continue_Colab.ipynb`

The rigorous notebook publishes a versioned model as `<HF-user>/TinyCeNN-LM-Distilled-v2`, preserving the original 10M checkpoint for reproducibility.

## CLI — rigorous continuation

```bash
python scripts/train_distill_rigorous.py \
  --resume-student-dir /path/to/TinyCeNN-LM-Distilled \
  --max-tokens 30000000 \
  --context-length 256 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 0.0003 \
  --steps 7 \
  --dilations 1,2,4,8,16,32,64 \
  --shuffle-buffer 4096 \
  --eval-batches 64 \
  --eval-batch-size 4 \
  --eval-every 250 \
  --target-gap-recovery 0.90 \
  --temperature 2.0 \
  --ce-weight 1.0 \
  --kl-weight 1.0 \
  --hidden-weight 0.25 \
  --output-dir checkpoints/cenn-student-rigorous-v2
```

## Reproduce a published checkpoint

The evaluator can use a local checkpoint or download directly from Hugging Face. It reconstructs the same deterministic held-out batches from the saved dataset/split/text-field settings, verifies their SHA-256 fingerprint, evaluates teacher and student, and fails if the reloaded student CE differs from the saved best CE by more than the configured tolerance.

```bash
python scripts/eval_distilled.py \
  --hf-repo YOUR_USER/TinyCeNN-LM-Distilled-v2 \
  --ce-tolerance 0.02
```

Successful output ends with:

```text
RIGOROUS REMOTE BENCHMARK: PASS
```

## Key report fields

`distillation_report.json` now records:

- benchmark protocol version;
- dataset, dataset config, split and text field;
- deterministic validation split;
- exact held-out batch/token count;
- SHA-256 benchmark fingerprint;
- training shuffle method, buffer and seed;
- cold CeNN, resumed run-start, best and final metrics;
- teacher/student CE and perplexity;
- KL divergence and hidden-state cosine loss;
- previous, current-run and cumulative training tokens;
- CeNN receptive field and parameter counts;
- teacher-gap recovery and target status;
- peak VRAM, elapsed time and complete evaluation history.

## Research interpretation

A successful result is not merely that training loss decreases. The stronger result is that a causal shared-weight CeNN core can recover a large fraction of the behavior of the pretrained Transformer core on a reproducible held-out benchmark while using a different computational primitive. Only after the plain CeNN continuation is measured under this protocol should additional architectural capacity such as routed experts/MoE be introduced.
