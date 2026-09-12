# Continued learning after the first 10M tokens

Use [the optimized Colab notebook](notebooks/TinyCeNN_Optimized_Continue_Colab.ipynb)
to continue an existing dense CeNN checkpoint. It keeps the Transformer-free
architecture and the deterministic held-out benchmark.

## What changed

- **FP32 student parameters, mixed-precision computation.** All four trainers
  previously loaded trainable parameters in BF16/FP16 on CUDA. AdamW updates at
  a low learning rate can round away in that storage format; FP16 parameters also
  conflict with ordinary GradScaler unscaling. The student now loads in FP32,
  while CUDA autocast still accelerates computation and the teacher stays frozen.
- **Optional interface adaptation.** `--train-interfaces all` trains embeddings,
  final normalization and the output head alongside the core. For the default
  model this expands the trainable set from about 0.48M to 12.77M parameters;
  it does not add parameters or Transformer layers. Interfaces use 5% of the core
  LR by default. `norm` adapts only final normalization; `none` retains the
  original core-only experiment. Checkpoints include every adapted interface,
  including interfaces frozen again during a later continuation.
- **A schedule for longer runs.** Optional `wsd` warms up, holds the peak LR,
  then decays during the final 20% of the token budget to a 10% floor. The original
  cosine schedule remains available. Norms and biases receive no weight decay.
- **Less teacher constraint late in training.** The optimized recipe gradually
  reduces KL weight from 1 to 0.25 and hidden-state alignment from 0.25 to 0.
  Next-token CE remains active throughout. This is an experimental recipe;
  compare it with constant weights on the same held-out set.
- **Continuation advances the data stream.** New checkpoints save a token offset
  and a signature covering dataset/revision, tokenizer, split and shuffle settings.
  Resume reconstructs that stream and skips the saved prefix before optimization.
  This reads/tokenizes the prefix again, but avoids teacher/student computation
  on those tokens. Changing the seed alone would still reuse source documents.
- **Recent progress is visible.** Reports include training CE/KL/hidden loss,
  gradient norms, scheduled weights and LR multipliers, plus a recent held-out CE
  plateau flag. A run can improve overall and still plateau near its end.

## Run locally or in an existing GPU container

Install the current checkout (`pip install -e .`) and use a new output directory:

```bash
python scripts/train_distill_rigorous.py \
  --resume-student-dir /path/to/your/dense-cenn-checkpoint \
  --output-dir checkpoints/cenn-optimized-01 \
  --max-tokens 30000000 \
  --train-interfaces all --interface-lr-scale 0.05 \
  --learning-rate 3e-4 --lr-schedule wsd --decay-ratio 0.2 \
  --kl-weight 1.0 --kl-final-weight 0.25 \
  --hidden-weight 0.25 --hidden-final-weight 0.0
```

Match the source checkpoint's CeNN steps/dilations/kernel/expansion, dataset and
evaluation settings. The defaults match the seven-step, 256-token FineWeb
experiment. The Colab notebook reads these settings from the source metadata.
It defaults to the existing `TinyCeNN-LM-Distilled-v2` Hub checkpoint and publishes
to a separate `TinyCeNN-LM-Distilled-v3` repository when its upload cell is run.

For a smaller intervention, start with `--train-interfaces none` and keep the
original constant distillation weights. This isolates the numerical and stream
fixes. Use `all` if the small shared core still stops improving. Training and
checkpoint memory increase when interfaces become trainable.

## Checkpoints and evaluation

Version-1 core-only checkpoints remain loadable. Version-2 checkpoints have a
manifest of the saved core/interface tensors; missing interface weights fail
strict loading. MoE warm-start also preserves adapted dense interfaces.

The **best** and **final** directories each contain their own `checkpoint` entry
in `distillation_report.json`, with the weights' metrics and token count. The
run-wide `cumulative_training_tokens` still describes the whole run. Use
`checkpoint.cumulative_training_tokens` when describing the particular model
being published. The evaluator selects that model's metrics and preserves FP32
storage under autocast:

```bash
python scripts/eval_distilled.py --student-dir checkpoints/cenn-optimized-01-best
python scripts/eval_distilled.py --student-dir checkpoints/cenn-optimized-01
```

The benchmark remains `rigorous-v2`: identical document split and token batches.
Evaluation always uses the initial fixed loss weights so its combined loss stays
comparable throughout a run, even when the training weights anneal. Student CE
determines the best checkpoint. A changed held-out fingerprint rejects resume.

Continuation starts **new AdamW moments and a new schedule** for the additional
token budget; it is not a bitwise restoration of an interrupted optimizer. The
saved stream offset is exact only when source data and tokenization are unchanged.
Pin `--dataset-revision` for repeatable future runs. Old checkpoints lack an exact
stream cursor and use an explicitly reported `legacy_estimate`; their earlier
prefix coverage cannot be guaranteed. `--train-skip-tokens N` provides an explicit
offset when migrating to another stream, and `0` intentionally replays it.

The trainer refuses to mix a new run with existing output checkpoints. If the
dataset ends early, it reports `data_exhausted`, saves completed updates, and
does not pretend the requested budget was consumed. Budgets round up to a whole
gradient-accumulation update; token counts include packed input tokens and EOS.

## What the validation establishes

Offline regression tests reproduce BF16 update rounding, check LR groups and
schedules, verify exact stream suffixes, exercise legacy/new checkpoint reload,
and run the actual trainer against a tiny local Llama teacher and synthetic corpus.
These checks establish numerical behavior and training/reload correctness. They
do **not** establish a better FineWeb result after 10M or 40M tokens. Run the
optimized recipe on the same held-out fingerprint and compare CE/perplexity to
your source model before claiming that the plateau is resolved. No training
schedule can guarantee indefinite improvement from a fixed-capacity model.
