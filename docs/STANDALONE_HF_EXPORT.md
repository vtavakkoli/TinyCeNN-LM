# Self-contained Hugging Face model export

The V2.9.1 Colab exports a complete converted Laya model, **including the
fine-tuned decision head**. It can optionally upload the bundle to the
authenticated user's Hugging Face model repository.

## Publish V2.9.1 from Colab

1. Open the [V2.9.1 Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V29_AllGlobal_FastWindowMemory_Colab.ipynb).
2. Choose a GPU and execute all training and validation cells.
3. Add a Hugging Face **write** access token in Colab Secrets as `HF_TOKEN`
   (or sign in with `huggingface_hub.login()`). Never paste tokens into a notebook.
4. In the configuration cell set `HF_REPO_ID` to `username/model-name` or
   leave empty for an automatic repo name; `UPLOAD_TO_HF=True`,
   `HF_PRIVATE=True` by default. Existing public repos are refused when private
   publication is requested.
5. The final cell exports the bundle, independently **loads and checks output
   parity** via its embedded loader, then calls the official Hugging Face upload API.

If the GPU VM disconnects before publishing, saved files under
`/content/Laya_V29_AllGlobal_FastWindowMemory/` are lost unless copied to
persistent storage. Saving `*.pt` checkpoints locally is separate from publishing.

## Bundle files

Each published repository contains:

- `model.safetensors`: **complete** model state, not just delta weights.
- `bundle.json`: immutable runtime protocol metadata and architecture settings.
- `encoder/config.json`: original ModernBERT encoder settings.
- `tokenizer/*`: vocabulary and special-token metadata.
- `runtime.py`: self-contained model and FastWindowMemory implementation.
- `load_model.py`: standalone loader for either a HF repo ID or local folder.
- `README.md`, `requirements.txt` and `training_report.json`.

The loader has **zero imports from `tinycenn_lm` or `laya`**. It still
requires the ordinary deep-learning environment: PyTorch,
Transformers **4.57.6**, safetensors and huggingface_hub (for remote downloads).

## Load into any new environment

```bash
pip install torch "transformers==4.57.6" safetensors huggingface_hub
hf download USERNAME/MODEL --local-dir ./converted-model
```

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("./converted-model").resolve()))
from load_model import load_model, predict

model, tokenizer = load_model("./converted-model", device="cpu")
result = predict(
    "./converted-model",
    state={"speed": 20},
    question={
        "type": "choice",
        "instructions": "Select the preferred action.",
        "criteria": {"accelerate": None, "brake": None},
    },
)
print(result["choice"])
```

For direct repo loading (without downloading the whole directory in advance),
use `from load_model import load_model` from the model repository's bundled
loader and call `load_model("USERNAME/MODEL", device="cuda")`. It downloads
a cached snapshot through the standard Hugging Face Hub client.

For untrusted remote models, review the bundled `runtime.py` and pin a
specific model revision. The bundled runtime executes Python code.

## Reuse for other standalone models

Use the generic model-independent exporter:

```python
from tinycenn_lm.hf_standalone import (
    export_standalone_bundle, upload_standalone_to_hub,
)
folder = export_standalone_bundle(
    model=your_model,
    output_dir="./exported",
    runtime_source="./your_runtime.py",
    manifest={"runtime_format": 1, "architecture": "your-architecture"},
    tokenizer=your_tokenizer,          # optional
    encoder_config=your_encoder_config, # optional
    report=your_metrics,              # optional
)
upload_standalone_to_hub(
    folder, "username/your-model", token=None, private=True,
)
```

The custom `your_runtime.py` must define:

```python
def build_model(bundle_directory, manifest):
    # Reconstruct the exact model class from JSON config/manifest.
    # The exported loader then loads all state_dict keys strictly.
    return model
```

This protocol works with other architecture implementations; their runtime
code must be included in the model repo. `export_v29_fastwindow()` is a
convenience adapter specializing the generic exporter to the V2.9 architecture.

## Accuracy and performance

The saved **original V2.9** W1024 run reported 10/10 converted global layers,
official-test teacher agreement **0.9975**, teacher gold accuracy **0.7655**,
converted gold accuracy **0.7660**, and batch-1 speedup **0.594×** (slower).

Those results are not evidence of increased accuracy or speed due to joint
fine-tuning: the original 60-step joint run chose **step 0** after validation
regressed. V2.9.1 adds supervised task loss, gentler rates and validation
rollback. Its result must be measured on a new run.

At sequence length ≤512, W1024's local attention path is essentially full
attention and retains quadratic complexity. Do not claim a linear-attention
or fast-attention replacement from its high cosine score.
