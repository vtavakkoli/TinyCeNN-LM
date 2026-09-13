# Run notebook experiments from Python

The Colab notebooks now have a matching command-line workflow. After cloning the repository, install it once and run any experiment with `scripts/run_experiment.py`.

```bash
git clone https://github.com/vtavakkoli/TinyCeNN-LM.git
cd TinyCeNN-LM
python -m pip install -e .
```

Each run trains the model, reloads the local checkpoint, runs a forward/generation smoke test, and stores the artifacts under:

```text
result/<model_name>/
├── model/                  # final checkpoint and trainer report
├── model-best/             # best checkpoint when the trainer creates one
├── run_manifest.json       # exact command and warm-start source
├── train.log               # complete training stdout/stderr
├── eval.log                # local reload + generation test
├── evaluation.json         # machine-readable local test result
├── rigorous_eval.log       # only for rigorous benchmark models
└── rigorous_evaluation.json# only when the rigorous evaluator is available
```

## Live status in Colab

All repository `train_*.py` commands now use immediate line-buffered output. Existing notebook cells can keep using `subprocess.run(cmd, check=True)`; TinyCeNN automatically makes the child Python process unbuffered and, when the notebook has imported `tinycenn_lm`, streams stdout and stderr through the Colab cell with explicit status markers.

A typical training cell now displays output continuously like this:

```text
[TinyCeNN][START] train_smollm2_amcenn_v2
[TinyCeNN][COMMAND] python .../train_smollm2_amcenn_v2.py ...
[TinyCeNN][OUTPUT] /content/TinyCeNN-LM/checkpoints/smollm2-amcenn-top2-v2
[TinyCeNN][LIVE] streaming training output...
[TinyCeNN][PROCESS START] train_smollm2_amcenn_v2
device=cuda dtype=torch.float16
...
calibration ...
...
[TinyCeNN][PROCESS DONE] train_smollm2_amcenn_v2 completed in 42m 18s
[TinyCeNN][DONE] train_smollm2_amcenn_v2 completed in 42m 20s
```

If training raises an uncaught exception, the cell prints `PROCESS FAILED` / `FAILED` before propagating the error. A local live log is also written below `.colab_live_backup/<run_id>/train.log`. If a Hugging Face token is configured, the existing private live backup continues to run; missing Hub credentials no longer disable live console logging.

## One command per notebook

| Notebook | Python command |
|---|---|
| `TinyCeNN_LM_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-lm` |
| `TinyCeNN_Distill_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-distill` |
| `TinyCeNN_Rigorous_Continue_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-rigorous-continue` |
| `TinyCeNN_Optimized_Continue_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-optimized-continue` |
| `TinyCeNN_MoE_Top2_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-moe-top2` |
| `TinyCeNN_SharedFFN_Top2_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-sharedffn-top2` |
| `TinyCeNN_Story_AntiRepeat_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-story-antirepeat` |
| `TinyCeNN_Story_v2_Colab.ipynb` | `python scripts/run_experiment.py --model tinycenn-story-v2` |
| `SmolLM2_AMCeNN_Top2_Colab.ipynb` | `python scripts/run_experiment.py --model smollm2-amcenn-top2` |
| `SmolLM2_AMCeNN_Top2_v2_Colab.ipynb` | `python scripts/run_experiment.py --model smollm2-amcenn-top2-v2` |

The continuation/MoE/story experiments automatically download the same public Hugging Face warm-start checkpoints used by the notebooks. Override the source with either a local directory or another Hugging Face repo:

```bash
python scripts/run_experiment.py \
  --model tinycenn-rigorous-continue \
  --source /path/to/TinyCeNN-LM-Distilled
```

or

```bash
python scripts/run_experiment.py \
  --model tinycenn-rigorous-continue \
  --source vtava/TinyCeNN-LM-Distilled
```

If a source repository requires authentication, export `HF_TOKEN` before running.

## Quick end-to-end test

Use `--smoke` to reduce the token/runtime budget while still exercising training, checkpoint saving, local reload, and evaluation:

```bash
python scripts/run_experiment.py --model tinycenn-lm --smoke
```

## Override trainer arguments

Unknown arguments are passed to the underlying trainer after the notebook-equivalent defaults. For example:

```bash
python scripts/run_experiment.py \
  --model smollm2-amcenn-top2 \
  --max-tokens 5000000 \
  --max-runtime-minutes 90
```

The final occurrence of an argparse option wins, so these values override the defaults assembled by the runner.

## Re-run evaluation only

The trained checkpoint can be tested again without retraining:

```bash
python scripts/evaluate_local.py \
  --model tinycenn-lm \
  --model-dir result/tinycenn-lm/model-best \
  --output result/tinycenn-lm/evaluation.json
```

For `tinycenn-rigorous-continue`, `tinycenn-optimized-continue`, and `tinycenn-sharedffn-top2`, the runner additionally executes the repository's deterministic held-out benchmark evaluator and records the result in `rigorous_evaluation.json`.

SmolLM2 and Story notebooks intentionally do not define the same long held-out benchmark; their `evaluation.json` therefore records a local checkpoint reload, finite-loss sanity test, architecture information, and generation test instead of inventing a benchmark that the original notebook did not run.
