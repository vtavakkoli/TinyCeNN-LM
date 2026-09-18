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
    from .qwen35_flycore_v2 import (
        FlyVocabV2Config,
        FlyVocabCoreV2,
        FlyEmbeddingV2,
        FlyLMHeadV2,
        assert_qwen35_flycore_v2,
    )
    from .qwen35_flyffn_v3 import FlyFFNV3Config
except ImportError:
    from qwen35_flycore_v2 import (
        FlyVocabV2Config,
        FlyVocabCoreV2,
        FlyEmbeddingV2,
        FlyLMHeadV2,
        assert_qwen35_flycore_v2,
    )
    from qwen35_flyffn_v3 import FlyFFNV3Config

try:
    from .smollm2_flyffn_v2 import replace_ffns_with_fly_v2
except ImportError:
    from smollm2_flyffn_v2 import replace_ffns_with_fly_v2


STATE_FILE = "standalone_state.pt"


def _keys(state):
    return [
        k for k in state
        if (
            ".mlp." in k
            or k.startswith("flyffn_shared_graph.")
            or k.startswith("fly_vocab_core_v2.")
        )
    ]


def _write_loader(export_dir: Path):
    text = """from pathlib import Path
import torch
from qwen35_flycore_v2_standalone import load_flycore_v2_standalone

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    model, tokenizer = load_flycore_v2_standalone(HERE)
    prompt = tokenizer.apply_chat_template(
        [{"role":"user","content":"Explain neural networks in one paragraph."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
"""
    (export_dir / "load_model.py").write_text(text, encoding="utf-8")


def _write_readme(export_dir: Path, metadata: dict[str, Any] | None):
    metadata = metadata or {}
    stats = metadata.get("vocab_parameter_stats", {})
    cfg = metadata.get("config", {})
    readme = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: vtava/Qwen35-0.8B-FlyFFN-v3-AllFFN
tags:
- qwen3.5
- flycore
- flyembedding
- flylmhead
- flyffn
- experimental
---

# Qwen3.5-0.8B FlyCore-v2

FlyCore-v2 starts from the trained 24/24 FlyFFN-v3 model and replaces only the
tied vocabulary embedding/head with a compressed graph-aware vocabulary core.

Key differences from FlyCore-v1:

- latent rank: {cfg.get("vocab_latent_dim", "N/A")}
- hot-token residual rows: {cfg.get("hot_token_count", "N/A")}
- adjoint-consistent Fly transform: T on input, T^T on output
- trained FlyFFN-v3 layers remain frozen
- explicit embedding + head + logit distillation losses
- automatic generation-degeneration checks before upload

Vocabulary parameter reduction:
{stats.get("vocab_param_reduction_pct", "N/A")} percent

Load with:

from qwen35_flycore_v2_standalone import load_flycore_v2_standalone
model, tokenizer = load_flycore_v2_standalone(".", device="cuda")

Source:
https://github.com/vtavakkoli/TinyCeNN-LM
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")


def _rewrite_imports(export_dir: Path):
    p = export_dir / "qwen35_flyffn_v3.py"
    t = p.read_text(encoding="utf-8").replace(
        "from .smollm2_flyffn_v2 import (",
        "from smollm2_flyffn_v2 import (",
    )
    p.write_text(t, encoding="utf-8")

    p = export_dir / "qwen35_flycore_v2.py"
    t = p.read_text(encoding="utf-8").replace(
        "from .qwen35_flyffn_v3 import ",
        "from qwen35_flyffn_v3 import ",
    )
    p.write_text(t, encoding="utf-8")


def export_flycore_v2_standalone(
    model,
    tokenizer,
    ffn_cfg: FlyFFNV3Config,
    vocab_cfg: FlyVocabV2Config,
    export_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
):
    assert_qwen35_flycore_v2(model)
    export_dir = Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model.config.save_pretrained(export_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    payload = {
        "base_model": "vtava/Qwen35-0.8B-FlyFFN-v3-AllFFN",
        "ffn": ffn_cfg.to_dict(),
        "vocab": vocab_cfg.to_dict(),
    }
    (export_dir / "flycore_v2_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    package = Path(__file__).resolve().parent
    for name in (
        "smollm2_flyffn_v2.py",
        "qwen35_flyffn_v3.py",
        "qwen35_flycore_v2.py",
        "qwen35_flycore_v2_standalone.py",
    ):
        shutil.copy2(package / name, export_dir / name)
    _rewrite_imports(export_dir)

    raw = model.state_dict()
    expected = _keys(raw)
    state = {k: v.detach().cpu() for k, v in raw.items()}
    state_path = export_dir / STATE_FILE
    torch.save(state, state_path)
    del state
    gc.collect()

    verify = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    missing = [k for k in expected if k not in verify]
    if missing:
        raise RuntimeError(f"standalone v2 export missing keys: {missing[:20]}")
    required = [
        "fly_vocab_core_v2.codebook.weight",
        "fly_vocab_core_v2.basis",
        "fly_vocab_core_v2.hot_token_ids",
        "fly_vocab_core_v2.hot_residual",
        "fly_vocab_core_v2.fly_down.weight",
        "fly_vocab_core_v2.fly_up.weight",
        "fly_vocab_core_v2.fly_scale_raw",
        "fly_vocab_core_v2.adjacency",
        "flyffn_shared_graph.adjacency",
    ]
    absent = [k for k in required if k not in verify]
    if absent:
        raise RuntimeError(f"standalone v2 missing required keys: {absent}")

    manifest = {
        "format": "TinyCeNN-LM Qwen3.5 FlyCore-v2 standalone",
        "state_file": STATE_FILE,
        "state_keys": len(verify),
        "flycore_state_keys": len(_keys(verify)),
        "state_bytes": state_path.stat().st_size,
        "verified": True,
    }
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
            "vocab_training_history.csv",
            "chat_samples.json",
            "fast_eval_50_flycore_v2_from_v3.csv",
            "fast_eval_details_flycore_v2_from_v3.csv",
        ):
            src = artifacts_dir / name
            if src.exists():
                shutil.copy2(src, export_dir / name)

    _write_loader(export_dir)
    _write_readme(export_dir, metadata)
    (export_dir / "standalone_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_flycore_v2_standalone(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    model_dir = Path(model_dir).resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    payload = json.loads(
        (model_dir / "flycore_v2_config.json").read_text(encoding="utf-8")
    )
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
    vocab_cfg = FlyVocabV2Config(
        latent_dim=int(v["latent_dim"]),
        fly_nodes=int(v["fly_nodes"]),
        graph_steps=int(v["graph_steps"]),
        graph_mix_init=float(v["graph_mix_init"]),
        max_fly_scale=float(v["max_fly_scale"]),
        hot_token_count=int(v["hot_token_count"]),
    )

    state = torch.load(
        model_dir / STATE_FILE, map_location="cpu", weights_only=True, mmap=True
    )
    adjacency = state["flyffn_shared_graph.adjacency"].float()
    hot_ids = state["fly_vocab_core_v2.hot_token_ids"].long()

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_config(config)

    # Reconstruct the 24 FlyFFNs with no anchors.
    internal = ffn_cfg.to_v2(len(model.model.layers))
    replace_ffns_with_fly_v2(model, internal, adjacency)

    reference = next(model.parameters())
    core = FlyVocabCoreV2(
        int(model.config.vocab_size),
        int(model.config.hidden_size),
        vocab_cfg,
        adjacency,
        hot_ids,
        device=reference.device,
        dtype=reference.dtype,
    )
    model.fly_vocab_core_v2 = core
    model.model.embed_tokens = FlyEmbeddingV2(core)
    model.lm_head = FlyLMHeadV2(core)
    model.config.tie_word_embeddings = False
    try:
        model._tied_weights_keys = {}
    except Exception:
        pass

    incompatible = model.load_state_dict(state, strict=False, assign=True)
    important = [
        k for k in incompatible.missing_keys
        if (
            ".mlp." in k
            or k.startswith("flyffn_shared_graph.")
            or k.startswith("fly_vocab_core_v2.")
        )
    ]
    if important:
        raise RuntimeError(f"important FlyCore-v2 weights missing: {important[:20]}")

    assert_qwen35_flycore_v2(model)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if device.type == "cuda" else torch.float32)
        )

    model = model.to(device=device, dtype=dtype).eval()
    core = model.fly_vocab_core_v2
    core.fly_down.float()
    core.fly_up.float()
    core.fly_scale_raw.data = core.fly_scale_raw.data.float()
    core.graph_mix_logit.data = core.graph_mix_logit.data.float()
    core.adjacency.data = core.adjacency.data.float()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def upload_flycore_v2_standalone(
    export_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
):
    from huggingface_hub import HfApi
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face token found")
    export_dir = Path(export_dir)
    manifest = json.loads(
        (export_dir / "standalone_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("verified"):
        raise RuntimeError("FlyCore-v2 standalone package is not verified")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        commit_message="Upload verified Qwen3.5 FlyCore-v2 standalone model",
    )
    return f"https://huggingface.co/{repo_id}"
