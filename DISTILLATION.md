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

## Clean held-out evaluation

The distillation trainer uses a deterministic document-level text hash split:

- buckets 0–989 / 1000: training
- buckets 990–999 / 1000: validation

This prevents the same FineWeb document from appearing in both the training and monitoring sets.

## Colab

Open:

`notebooks/TinyCeNN_Distill_Colab.ipynb`

The notebook trains the student, checks distillation health, publishes the best checkpoint to `<HF-user>/TinyCeNN-LM-Distilled`, downloads it again from Hugging Face, reconstructs the Transformer-free model and runs generation/sanity tests.

## CLI

```bash
python scripts/train_distill.py \
  --max-tokens 10000000 \
  --context-length 256 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 0.001 \
  --steps 7 \
  --dilations 1,2,4,8,16,32,64 \
  --temperature 2.0 \
  --ce-weight 1.0 \
  --kl-weight 1.0 \
  --hidden-weight 0.25 \
  --output-dir checkpoints/cenn-student-distill
```

For a first compatibility run, add `--no-compile`. Once the eager path is confirmed on a GPU, remove it to test `torch.compile` throughput.

## Key report fields

`distillation_report.json` records:

- teacher and student CE/perplexity;
- KL divergence;
- hidden-state cosine loss;
- CeNN receptive field;
- student parameter counts;
- best optimizer update;
- relative student CE improvement;
- teacher-gap recovery fraction;
- peak VRAM and elapsed time;
- complete held-out evaluation history.

The most useful metric for this stage is **teacher-gap recovery**: how much of the loss gap caused by removing the Transformer is recovered by the CeNN student.

## Research interpretation

A successful result is not merely that training loss decreases. The stronger claim is that a causal shared-weight CeNN core can recover a meaningful fraction of the behavior of the pretrained Transformer core while using a different computational primitive. Follow-up experiments should compare x1/x2/x4/x7/x8 recurrent steps at matched unique parameters and report both parameter efficiency and compute/FLOPs.
