# TinyCeNN-LM on Google Colab

The ready-to-run notebook is:

`notebooks/TinyCeNN_LM_Colab.ipynb`

Open it directly in Colab:

https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/TinyCeNN_LM_Colab.ipynb

## What the notebook does

1. checks the assigned Colab GPU with `nvidia-smi`;
2. clones and installs TinyCeNN-LM;
3. logs in to Hugging Face using the Colab Secret `HF_TOKEN` or interactive login;
4. trains the CeNN adapter on FineWeb;
5. inspects `training_report.json` and stops publication if training diverged;
6. selects the best adapter checkpoint;
7. adds tokenizer files, the health report and a generated model card;
8. creates `<your-hf-user>/TinyCeNN-LM-Base` automatically and uploads the checkpoint;
9. downloads that uploaded model back from Hugging Face;
10. reconstructs Tiny-LLM + CeNN and runs deterministic text-generation and numerical sanity tests.

## Hugging Face token

Create a Hugging Face token with write permission and add it in Colab under **Secrets** with the name:

`HF_TOKEN`

Do not put a real `hf_...` token directly into the notebook or commit it to Git.

The notebook also shows the direct API form with a placeholder:

```python
from huggingface_hub import login
login("hf_REPLACE_WITH_YOUR_NEW_TOKEN")
```

## Default training run

The notebook starts conservatively with 1M training tokens, context 256, batch 8, gradient accumulation 4 and CeNN x4. If the health report is good, increase the token budget to 10M and then 50M.

## Test the published Hugging Face checkpoint with Docker

Set the repository produced by the notebook:

```bash
cp .env.example .env
# edit .env:
# HF_MODEL_REPO=your-hf-user/TinyCeNN-LM-Base
```

Then run:

```bash
docker compose -f docker-compose.test.yml run --rm test-hf
```

Or without editing `.env`:

```bash
HF_MODEL_REPO=your-hf-user/TinyCeNN-LM-Base \
  docker compose -f docker-compose.test.yml run --rm test-hf
```

The test container downloads the model from Hugging Face, rejects a published checkpoint marked as diverged, computes a finite sanity loss/perplexity, and runs several deterministic generation prompts. It exits successfully only after the full remote reload test passes.

For a private Hugging Face repository, also provide `HF_TOKEN` in `.env` or the shell environment.
