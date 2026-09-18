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
    from .qwen35_flycore_v1 import (
        FlyFFNV3Config,
        FlyVocabConfig,
        assert_qwen35_flycore,
        replace_qwen_with_flycore,
    )
except ImportError:
    from qwen35_flycore_v1 import (
        FlyFFNV3Config,
        FlyVocabConfig,
        assert_qwen35_flycore,
        replace_qwen_with_flycore,
    )


STATE_FILE = "standalone_state.pt"


def _flycore_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return [
        k for k in state
        if (
            ".mlp.gate_weight" in k
            or ".mlp.up_weight" in k
            or ".mlp.down_weight" in k
            or ".mlp.router." in k
            or k.startswith("flyffn_shared_graph.")
            or k.startswith("fly_vocab_core.")
        )
    ]


def _write_loader(export_dir: Path) -> None:
    loader = """from pathlib import Path
import torch
from qwen35_flycore_standalone import load_flycore_standalone

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    model, tokenizer = load_flycore_standalone(HERE)
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
    stats = metadata.get("vocab_parameter_stats", {})
    factor = metadata.get("factorization", {})
    ce_gap = metadata.get("fly_ce_gap_vs_qwen", "N/A")
    p_ratio = metadata.get("parameter_ratio_fly_over_qwen", "N/A")
    rank = metadata.get("config", {}).get("vocab_latent_dim", "N/A")

    readme = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-0.8B
tags:
- qwen3.5
- flycore
- flyembedding
- flylmhead
- flyffn
- sparse-ffn
- experimental
---

# Qwen3.5-0.8B FlyCore-v1 Standalone

Experimental Qwen3.5-0.8B architecture with FlyEmbedding, FlyLMHead,
one shared latent Fly vocabulary core, and FlyFFN-v3 in all 24 FFN layers.
Qwen3.5 Gated DeltaNet / full-attention token mixers remain unchanged.

Vocabulary latent rank: {rank}
Original tied vocabulary parameters: {stats.get("original_tied_vocab_params", "N/A")}
Fly vocabulary parameters: {stats.get("fly_vocab_trainable_params", "N/A")}
Vocabulary parameter reduction: {stats.get("vocab_param_reduction_pct", "N/A")} percent
SVD retained energy: {factor.get("retained_energy", "N/A")}
Initial relative reconstruction MSE: {factor.get("relative_reconstruction_mse", "N/A")}
Final CE gap versus Qwen: {ce_gap}
Whole-model parameter ratio versus Qwen: {p_ratio}

Architecture:
token id -> latent codebook -> Fly graph residual -> 1024 hidden ->
24 FlyFFN-v3 layers -> 384 latent -> Fly graph residual -> shared codebook -> logits.

Loading:
from qwen35_flycore_standalone import load_flycore_standalone
model, tokenizer = load_flycore_standalone(".", device="cuda")

The authoritative checkpoint is standalone_state.pt.

Important limitations:
- This is an experimental research model.
- Rank reduction of the vocabulary matrix is lossy.
- FastEval is diagnostic rather than an official leaderboard evaluation.
- Current FlyFFN still computes all shards during dense/sparse blending.
- Optimized Qwen3.5 kernels should be installed before speed conclusions.

Source:
https://github.com/vtavakkoli/TinyCeNN-LM
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def _rewrite_standalone_imports(export_dir: Path) -> None:
    p = export_dir / "qwen35_flyffn_v3.py"
    text = p.read_text(encoding="utf-8").replace(
        "from .smollm2_flyffn_v2 import (",
        "from smollm2_flyffn_v2 import (",
    )
    p.write_text(text, encoding="utf-8")

    p = export_dir / "qwen35_flycore_v1.py"
    text = p.read_text(encoding="utf-8").replace(
        "from .qwen35_flyffn_v3 import (",
        "from qwen35_flyffn_v3 import (",
    )
    p.write_text(text, encoding="utf-8")


def export_flycore_standalone(
    model,
    tokenizer,
    ffn_cfg: FlyFFNV3Config,
    vocab_cfg: FlyVocabConfig,
    export_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    assert_qwen35_flycore(model)

    model.config.save_pretrained(export_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    config_payload = {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "ffn": ffn_cfg.to_dict(),
        "vocab": vocab_cfg.to_dict(),
    }
    (export_dir / "flycore_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    package_dir = Path(__file__).resolve().parent
    for name in (
        "smollm2_flyffn_v2.py",
        "qwen35_flyffn_v3.py",
        "qwen35_flycore_v1.py",
        "qwen35_flycore_standalone.py",
    ):
        src = package_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, export_dir / name)
    _rewrite_standalone_imports(export_dir)

    raw = model.state_dict()
    expected = _flycore_keys(raw)
    if not expected:
        raise RuntimeError("No FlyCore keys found in model state_dict")
    if not any(k.startswith("fly_vocab_core.") for k in expected):
        raise RuntimeError("Fly vocabulary core parameters are missing before export")

    state = {k: v.detach().cpu() for k, v in raw.items()}
    state_path = export_dir / STATE_FILE
    torch.save(state, state_path)
    del state
    gc.collect()

    verify = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    missing = [k for k in expected if k not in verify]
    if missing:
        raise RuntimeError(f"Standalone FlyCore export missing keys: {missing[:20]}")

    required = (
        "fly_vocab_core.codebook.weight",
        "fly_vocab_core.basis",
        "fly_vocab_core.fly_down.weight",
        "fly_vocab_core.fly_up.weight",
        "fly_vocab_core.adjacency",
        "flyffn_shared_graph.adjacency",
    )
    absent = [k for k in required if k not in verify]
    if absent:
        raise RuntimeError(f"Standalone FlyCore export missing required keys: {absent}")

    total_count = len(verify)
    flycore_count = len(_flycore_keys(verify))
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
            "bio_vocab_training_history.csv",
            "rewired_progressive_calibration.csv",
            "rewired_training_history.csv",
            "rewired_vocab_training_history.csv",
            "fast_eval_50_qwen35_flycore.csv",
        ):
            src = artifacts_dir / name
            if src.exists():
                shutil.copy2(src, export_dir / name)

    _write_loader(export_dir)
    _write_readme(export_dir, metadata)

    manifest = {
        "format": "TinyCeNN-LM Qwen3.5 FlyCore-v1 standalone",
        "state_file": STATE_FILE,
        "state_keys": total_count,
        "flycore_state_keys": flycore_count,
        "state_bytes": state_path.stat().st_size,
        "verified": True,
    }
    (export_dir / "standalone_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_flycore_standalone(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    model_dir = Path(model_dir).resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    payload = json.loads((model_dir / "flycore_config.json").read_text(encoding="utf-8"))
    f = payload["ffn"]
    v = payload["vocab"]

    ffn_cfg = FlyFFNV3Config(
        fly_nodes=int(f["fly_nodes"]),
        router_rank=int(f["router_rank"]),
        num_shards=int(f["num_shards"]),
        graph_steps=int(f["graph_steps"]),
        graph_mix_init=float(f["graph_mix_init"]),
        router_noise_std=float(f.get("router_noise_std", 5e-3)),
    )
    vocab_cfg = FlyVocabConfig(
        latent_dim=int(v["latent_dim"]),
        fly_nodes=int(v["fly_nodes"]),
        graph_steps=int(v["graph_steps"]),
        graph_mix_init=float(v["graph_mix_init"]),
    )

    state_path = model_dir / STATE_FILE
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)

    adjacency = state.get("flyffn_shared_graph.adjacency")
    if adjacency is None:
        raise RuntimeError("FlyFFN adjacency missing from standalone checkpoint")

    hf_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if hasattr(hf_config, "tie_word_embeddings"):
        hf_config.tie_word_embeddings = False

    model = AutoModelForCausalLM.from_config(hf_config)
    replace_qwen_with_flycore(
        model,
        ffn_cfg,
        vocab_cfg,
        adjacency.float(),
        factorization=None,
    )

    incompatible = model.load_state_dict(state, strict=False, assign=True)
    important_missing = [
        k for k in incompatible.missing_keys
        if (
            ".mlp." in k
            or k.startswith("flyffn_shared_graph.")
            or k.startswith("fly_vocab_core.")
        )
    ]
    if important_missing:
        raise RuntimeError(f"Important FlyCore weights missing: {important_missing[:20]}")

    assert_qwen35_flycore(model)

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
    model.fly_vocab_core.fly_down.float()
    model.fly_vocab_core.fly_up.float()
    model.fly_vocab_core.graph_mix_logit.data = model.fly_vocab_core.graph_mix_logit.data.float()
    model.fly_vocab_core.adjacency.data = model.fly_vocab_core.adjacency.data.float()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def upload_flycore_standalone(
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
    manifest = json.loads(
        (export_dir / "standalone_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("verified"):
        raise RuntimeError("Standalone FlyCore package is not verified")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        commit_message="Upload verified Qwen3.5 FlyCore-v1 standalone model",
    )
    return f"https://huggingface.co/{repo_id}"
