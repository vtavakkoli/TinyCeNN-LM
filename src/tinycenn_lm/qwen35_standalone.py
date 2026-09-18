from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from .qwen35_flyffn_v2 import (
        FlyFFNV2Config,
        assert_qwen35_flyffn_v2,
        replace_ffns_with_fly_v2,
    )
except ImportError:
    from qwen35_flyffn_v2 import (
        FlyFFNV2Config,
        assert_qwen35_flyffn_v2,
        replace_ffns_with_fly_v2,
    )


STATE_FILE = "standalone_state.pt"


def _fly_keys(state: dict[str, torch.Tensor]) -> list[str]:
    markers = (
        ".mlp.gate_weight",
        ".mlp.up_weight",
        ".mlp.down_weight",
        ".mlp.output_scale",
        ".mlp.active_k_state",
        ".mlp.route_mix_state",
        ".mlp.router.",
        "flyffn_shared_graph.adjacency",
    )
    return [k for k in state if any(m in k for m in markers)]


def _write_loader(export_dir: Path) -> None:
    loader = """from pathlib import Path
import torch
from qwen35_standalone import load_standalone

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    model, tokenizer = load_standalone(HERE)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain sparse neural networks in one paragraph."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
"""
    (export_dir / "load_model.py").write_text(loader, encoding="utf-8")


def _write_readme(export_dir: Path, metadata: dict[str, Any] | None = None) -> None:
    metadata = metadata or {}
    ce_gap = metadata.get("fly_ce_gap_vs_qwen", "N/A")
    ppl_ratio = metadata.get("fly_ppl_ratio_vs_qwen", "N/A")
    anchors = metadata.get("dense_anchor_layers", [])
    readme = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-0.8B
tags:
- qwen3.5
- flyffn
- sparse-ffn
- experimental
---

# Qwen3.5-0.8B FlyFFN-v2 Standalone

Experimental FFN-only FlyFFN-v2 conversion of Qwen3.5-0.8B.

- Qwen token mixers remain unchanged.
- Dense FFN anchors: {anchors}
- Training/evaluation CE gap versus teacher: {ce_gap}
- Perplexity ratio versus teacher: {ppl_ratio}
- Full custom checkpoint: standalone_state.pt

Important: this model contains custom FlyFFN-v2 modules. The authoritative
standalone checkpoint is standalone_state.pt. Use qwen35_standalone.load_standalone
or load_model.py to reconstruct the custom architecture.

The current research implementation still computes all shards during dense/sparse
blending, so it should not yet be interpreted as a fused sparse-compute speed
implementation.

Source: https://github.com/vtavakkoli/TinyCeNN-LM
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def export_standalone(
    model,
    tokenizer,
    fly_cfg: FlyFFNV2Config,
    export_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model.config.save_pretrained(export_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    cfg_payload = {"base_model": "Qwen/Qwen3.5-0.8B", **fly_cfg.to_dict()}
    (export_dir / "flyffn_config.json").write_text(
        json.dumps(cfg_payload, indent=2), encoding="utf-8"
    )

    package_dir = Path(__file__).resolve().parent
    for name in ("smollm2_flyffn_v2.py", "qwen35_flyffn_v2.py", "qwen35_standalone.py"):
        src = package_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, export_dir / name)

    qwen_path = export_dir / "qwen35_flyffn_v2.py"
    qwen_text = qwen_path.read_text(encoding="utf-8").replace(
        "from .smollm2_flyffn_v2 import (",
        "from smollm2_flyffn_v2 import (",
    )
    qwen_path.write_text(qwen_text, encoding="utf-8")

    raw = model.state_dict()
    expected_fly_keys = _fly_keys(raw)
    if not expected_fly_keys:
        raise RuntimeError("No FlyFFN-v2 keys found in the trained model state_dict")

    state = {k: v.detach().cpu() for k, v in raw.items()}
    state_path = export_dir / STATE_FILE
    torch.save(state, state_path)
    del state
    gc.collect()

    verify = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    missing = [k for k in expected_fly_keys if k not in verify]
    if missing:
        raise RuntimeError(f"Standalone export missing FlyFFN keys: {missing[:20]}")
    if "flyffn_shared_graph.adjacency" not in verify:
        raise RuntimeError("Standalone export is missing FlyWire adjacency")

    custom_count = len(_fly_keys(verify))
    total_count = len(verify)
    del verify
    gc.collect()

    if metadata is not None:
        (export_dir / "report.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        for name in (
            "summary.csv",
            "chat_samples.json",
            "bio_progressive_calibration.csv",
            "bio_training_history.csv",
            "fast_eval_50_qwen35.csv",
        ):
            src = artifacts_dir / name
            if src.exists():
                shutil.copy2(src, export_dir / name)

    _write_loader(export_dir)
    _write_readme(export_dir, metadata)

    manifest = {
        "format": "TinyCeNN-LM Qwen3.5 FlyFFN-v2 standalone v1",
        "state_file": STATE_FILE,
        "state_keys": total_count,
        "flyffn_state_keys": custom_count,
        "state_bytes": state_path.stat().st_size,
        "verified": True,
    }
    (export_dir / "standalone_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_standalone(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    model_dir = Path(model_dir).resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    fc = json.loads((model_dir / "flyffn_config.json").read_text(encoding="utf-8"))
    fly_cfg = FlyFFNV2Config(
        fly_nodes=int(fc["fly_nodes"]),
        router_rank=int(fc["router_rank"]),
        num_shards=int(fc["num_shards"]),
        graph_steps=int(fc["graph_steps"]),
        graph_mix_init=float(fc["graph_mix_init"]),
        router_noise_std=float(fc.get("router_noise_std", 5e-3)),
        anchor_every=int(fc["anchor_every"]),
    )

    state_path = model_dir / STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)

    adj = state.get("flyffn_shared_graph.adjacency")
    if adj is None:
        raise RuntimeError("FlyWire adjacency missing from standalone checkpoint")

    hf_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_config(hf_config)
    replace_ffns_with_fly_v2(model, fly_cfg, adj.float())

    incompatible = model.load_state_dict(state, strict=False, assign=True)
    important_missing = [
        k for k in incompatible.missing_keys
        if k.startswith("flyffn_shared_graph.") or ".mlp." in k
    ]
    if important_missing:
        raise RuntimeError(f"Important FlyFFN weights missing: {important_missing[:20]}")

    assert_qwen35_flyffn_v2(model, fly_cfg.anchor_every)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def upload_standalone(
    export_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
) -> str:
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN to Colab Secrets or environment."
        )

    export_dir = Path(export_dir)
    manifest_path = export_dir / "standalone_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Standalone package has not been verified")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("verified"):
        raise RuntimeError("Standalone package verification failed")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        commit_message="Upload verified standalone Qwen3.5-0.8B FlyFFN-v2",
    )
    return f"https://huggingface.co/{repo_id}"
