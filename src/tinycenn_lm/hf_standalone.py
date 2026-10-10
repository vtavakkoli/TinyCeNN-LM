"""Generic reusable exporter for Hugging Face standalone models.

The uploaded repository contains model weights, tokenizer, encoder config,
a standalone runtime module and a loader. No TinyCeNN-LM project dependency.
The runtime_source contract is build_model(bundle_dir, manifest).
"""
from __future__ import annotations
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Mapping
import torch


def export_standalone_bundle(*, model, tokenizer, output_dir,
                             runtime_source, manifest, encoder_config,
                             report=None, loader_source=None):
    """Export a self-contained model with safetensors and executable runtime."""
    from safetensors.torch import save_file
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Expected torch.nn.Module")
    if manifest.get("runtime_format") != 1 or not manifest.get("architecture"):
        raise ValueError("A runtime_format=1 and architecture are required")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["runtime_file"] = "runtime.py"
    (root / "bundle.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    (root / "encoder").mkdir(exist_ok=True)
    encoder_config.save_pretrained(str(root / "encoder"))
    (root / "tokenizer").mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(root / "tokenizer"))
    source = Path(runtime_source)
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, root / "runtime.py")
    loader = (Path(loader_source) if loader_source is not None
              else Path(__file__).with_name("standalone_hf_loader.py"))
    if not loader.is_file():
        raise FileNotFoundError(loader)
    shutil.copyfile(loader, root / "load_model.py")

    # Clone all tensors to CPU to remove shared-storage aliases and GPU context.
    weights = {k: v.detach().to("cpu").contiguous().clone()
               for k, v in model.state_dict().items()}
    if not weights:
        raise ValueError("Refusing to publish empty weights")
    save_file(weights, str(root / "model.safetensors"),
              metadata={"format": "pt", "architecture": str(manifest["architecture"])})
    (root / "requirements.txt").write_text(
        "torch>=2.2\ntransformers==4.57.6\nsafetensors>=0.4\n"
        "huggingface_hub>=0.25\n", encoding="utf-8"
    )
    card = """---
library_name: pytorch
tags: [pytorch, standalone, modernbert, decision-model]
pipeline_tag: text-classification
---

# Self-contained trained model

The model weights, tokenizer, config and all custom Python runtime code are
included in this repository. The original research package is NOT required.

Install: pip install torch transformers==4.57.6 safetensors huggingface_hub

Download the load_model.py file or the entire repository, then run:

~~~python
from load_model import load_model, predict
model, tokenizer = load_model("USERNAME/MODEL", device="cpu")
output = predict("USERNAME/MODEL", state={"speed": 20},
                 question={"type": "choice", "instructions": "Select action",
                           "criteria": {"accelerate": None, "brake": None}})
print(output)
~~~

The bundled loader executes bundled runtime.py. Review/pin trusted code before use.
The model is a typed-decision encoder, NOT a text-generation model.

W1024 at 512-token input is a quadratic full-context control and
does NOT demonstrate a faster or attention-free replacement.
Consult training_report.json for actual held-out metrics.
"""
    (root / "README.md").write_text(card, encoding="utf-8")
    if report is not None:
        (root / "training_report.json").write_text(
            json.dumps(dict(report), indent=2, allow_nan=False,
                       default=lambda x: x.item() if hasattr(x, "item") else str(x)),
            encoding="utf-8"
        )
    return root


def export_v29_fastwindow(model, tokenizer, *, output_dir, global_layers,
                          window, max_len=512, feature_dim=96,
                          landmark_stride=16, max_landmarks=32,
                          source_model=None, report=None):
    """V2.9 adapter for the reusable exporter (no notebook-specific code)."""
    for i in global_layers:
        layer = model.encoder.layers[i].attn
        if getattr(layer, "architecture", None) != "pdelta_v28_fast_window_memory":
            raise TypeError(f"Layer {i} is not FastWindowMemory")
    head_layers = len(model.head.layers) if model.head is not None else 0
    manifest = {
        "architecture": "laya-v29-fastwindow",
        "runtime_format": 1,
        "source_model": source_model,
        "head_layers": int(head_layers),
        "n_act": int(model.act_head[-1].out_features),
        "global_layers": list(map(int, global_layers)),
        "window": int(window), "max_len": int(max_len),
        "feature_dim": int(feature_dim),
        "landmark_stride": int(landmark_stride),
        "max_landmarks": int(max_landmarks),
        "quadratic_full_context_control": int(window) >= 2*int(max_len)-1,
    }
    return export_standalone_bundle(
        model=model, tokenizer=tokenizer, output_dir=output_dir,
        runtime_source=Path(__file__).with_name("standalone_fastwindow_runtime.py"),
        manifest=manifest, encoder_config=model.encoder.config, report=report
    )


def verify_standalone_bundle(folder, reference_model, sample_batch,
                             atol=0.005, rtol=0.005):
    """Strict weight/shape loading and output parity check on CPU."""
    root = Path(folder)
    spec = importlib.util.spec_from_file_location(
        "_verified_export_loader", str(root / "load_model.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    restored, tokenizer = module.load_model(root, device="cpu")
    orig_training = reference_model.training
    reference_model.eval()
    reference_device = next(reference_model.parameters()).device
    with torch.inference_mode():
        lhs = reference_model(**{
            k: v.to(reference_device) for k, v in sample_batch.items()
        })
        rhs = restored(**{k: v.cpu() for k, v in sample_batch.items()})
    result = {}
    for i, (expected, actual) in enumerate(zip(lhs, rhs)):
        torch.testing.assert_close(
            actual.float().cpu(), expected.float().cpu(), rtol=rtol, atol=atol
        )
        result[f"logit_{i}_max_abs"] = float(
            (actual.float().cpu()-expected.float().cpu()).abs().max()
        )
    reference_model.train(orig_training)
    return result


def upload_standalone_to_hub(folder, repo_id, *, token=None, private=True,
                             commit_message="Publish self-contained trained model"):
    """Explicit synchronous Hub upload with authentication; never store tokens."""
    from huggingface_hub import HfApi
    folder = Path(folder)
    for name in ("bundle.json", "runtime.py", "load_model.py",
                 "model.safetensors", "README.md", "encoder/config.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Incomplete model bundle: {name}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        commit_message=commit_message,
        ignore_patterns=["*.pt", "*.bin", "*.ckpt", "__pycache__/*", "*.pyc"]
    )
    return str(commit.commit_url)


# Reusable for other architectures that provide their own runtime_source.
save_standalone_model = export_standalone_bundle
