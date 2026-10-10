"""Generic, dependency-free-of-the-project Hugging Face model bundle loader.

This file is distributed *inside* each exported model repository.
Install: pip install torch transformers==4.57.6 safetensors huggingface_hub
It does not import tinycenn_lm or the original research repository.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path


def _download_or_local(path_or_repo, *, revision=None, token=None):
    path = Path(str(path_or_repo)).expanduser()
    if path.is_dir() and (path / "bundle.json").exists():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=str(path_or_repo), repo_type="model",
        revision=revision, token=token,
    ))


def load_model(path_or_repo, *, device="cpu", revision=None, token=None):
    """Load an entire standalone model directly from a HF repo ID or local folder.

    Return (nn.Module, tokenizer). No TinyCeNN-LM/Laya imports needed.
    Always review runtime.py from a model repository before executing it.
    """
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    root = _download_or_local(path_or_repo, revision=revision, token=token)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    runtime_name = manifest.get("runtime_file", "runtime.py")
    if runtime_name != "runtime.py":
        raise ValueError("Unrecognized runtime path in bundle manifest")
    module_path = root / runtime_name
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location(
        "_standalone_bundle_runtime", str(module_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load model runtime file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model(root, manifest)
    # Load tensors on CPU; move only after strict key/shape validation.
    weights = load_file(str(root / "model.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    return model, tokenizer


def predict(path_or_model, tokenizer=None, state=None, question=None, *,
            device="cpu", max_len=None, revision=None, token=None):
    """Task inference convenience for standalone typed-decision bundles."""
    if isinstance(path_or_model, (str, Path)):
        model, loaded_tokenizer = load_model(
            path_or_model, device=device, revision=revision, token=token
        )
        tokenizer = loaded_tokenizer
        root = _download_or_local(path_or_model, revision=revision, token=token)
    else:
        model = path_or_model
        root = None
    if tokenizer is None:
        raise ValueError("tokenizer is required when passing a model object")
    if question is None:
        raise ValueError("question is required")
    if root is not None:
        manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
        if max_len is None:
            max_len = int(manifest.get("max_len", 512))
    if max_len is None:
        max_len = 512
    # Import the bundled runtime from the model's directory, not from project.
    # For model objects without a root, use the module supplied by the caller.
    if root is None:
        raise ValueError("For a model object, import predict from the bundled runtime.py")
    spec = importlib.util.spec_from_file_location(
        "_standalone_predict_runtime", str(root / "runtime.py")
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime.predict(
        model, tokenizer, state, question, max_len=max_len, device=device
    )
