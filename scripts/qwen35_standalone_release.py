#!/usr/bin/env python3
"""Create validated standalone releases of the three published TinyCeNN/Qwen3.5 models.

Goals:
- preserve the custom TinyCeNN architecture and weights;
- save a complete single-file ``model.safetensors`` checkpoint;
- bundle only the runtime source files required by each model;
- run a very small deterministic sanity suite (not an official benchmark);
- reload in a fresh Python process with no TinyCeNN-LM checkout on PYTHONPATH;
- upload only after strict reload validation passes.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qwen35_gguf_export import load_memory_fusion, load_packaged_loader
from tinycenn_lm.qwen35_pdelta3_runtime import (
    QwenPDelta3CLVRConfig,
    replace_full_attention_layers,
)

MODELS = (
    {
        "source": "vtava/Qwen3.5-0.8B-MemoryFusion",
        "target": "vtava/Qwen3.5-0.8B-MemoryFusion-Standalone",
        "variant": "memory_fusion",
        "class": "MemoryFusionQwen35Attention",
    },
    {
        "source": "vtava/Qwen3.5-0.8B-CeNN-Integrated-V1",
        "target": "vtava/Qwen3.5-0.8B-CeNN-Integrated-V1-Standalone",
        "variant": "cenn_integrated",
        "class": "Qwen35IntegratedAttention",
    },
    {
        "source": "vtava/Qwen3.5-0.8B-PDelta3-CLVR-Local32",
        "target": "vtava/Qwen3.5-0.8B-PDelta3-CLVR-Local32-Standalone",
        "variant": "pdelta3_clvr",
        "class": "QwenPDelta3CLVRAttention",
    },
)

RUNTIME_FILES = {
    "memory_fusion": (
        "qwen3_5_memory_fusion.py",
        "memory_attention.py",
        "cellular_attention.py",
    ),
    "cenn_integrated": (
        "qwen35_integrated_memory.py",
        "optimized_memory.py",
    ),
    "pdelta3_clvr": (
        "qwen35_pdelta3_runtime.py",
        "pdelta3_frontier.py",
        "pdelta2_er.py",
        "pdelta2_features.py",
        "research_layers.py",
    ),
}

QUICK_TESTS = (
    ("Knowledge", "What is the capital of France?", ("Berlin", "Paris", "Rome", "Madrid"), "B"),
    ("STEM", "What is 12 multiplied by 8?", ("86", "92", "96", "108"), "C"),
    ("Reasoning", "Complete the sequence: 2, 4, 8, 16, ?", ("18", "24", "30", "32"), "D"),
    ("Multilingual", "Was ist die Hauptstadt von Österreich?", ("Graz", "Salzburg", "Wien", "Linz"), "C"),
    (
        "Context",
        "Read carefully. We discuss rivers and books. The secret word is MARBLE. "
        "Then we discuss gardens and clouds. What was the secret word?",
        ("RIVER", "MARBLE", "GARDEN", "MUSIC"),
        "B",
    ),
)

REFERENCE = {
    "MMLU-Pro (non-thinking)": 29.7,
    "MMLU-Redux (non-thinking)": 48.5,
    "C-Eval (non-thinking)": 46.4,
    "IFEval (non-thinking)": 52.1,
    "MMMLU (non-thinking)": 34.1,
    "MMLU-Pro (thinking)": 42.3,
    "GPQA (thinking)": 11.9,
    "LongBench v2 (thinking)": 26.1,
    "MMMLU (thinking)": 44.3,
    "Global PIQA (thinking)": 59.4,
    "WMT24++ (thinking)": 27.2,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="/content/tinycenn_standalone")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--only",
        default="all",
        help="all or comma-separated variants: memory_fusion,cenn_integrated,pdelta3_clvr",
    )
    return p.parse_args()


def device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def backbone(model):
    base = getattr(model, "model", model)
    return getattr(base, "language_model", base)


def class_modules(model, class_name):
    return [(name, m) for name, m in model.named_modules() if m.__class__.__name__ == class_name]


def state_keys_in_file(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return list(f.keys())


def infer_pdelta_config(local: Path, source: str):
    weight = local / "model.safetensors"
    if not weight.exists():
        raise FileNotFoundError(f"PDelta3 standalone source requires {weight}")
    keys = state_keys_in_file(weight)
    layers = sorted({
        int(m.group(1))
        for key in keys
        if (m := re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.core\.", key))
    })
    if not layers:
        raise RuntimeError("Could not infer PDelta3 replacement layers from checkpoint tensor names")

    def tensor_shape(marker):
        with safe_open(str(weight), framework="pt", device="cpu") as f:
            name = next((k for k in f.keys() if marker in k), None)
            return tuple(f.get_tensor(name).shape) if name else None

    a_shape = tensor_shape(".self_attn.core.A_log")
    conv_shape = tensor_shape(".self_attn.core.v_conv_weight")
    if not a_shape or not conv_shape:
        raise RuntimeError("PDelta3 checkpoint is missing A_log or v_conv_weight")
    feature_dim = int(a_shape[-1])
    conv_kernel = int(conv_shape[-1])
    local = re.search(r"Local(\d+)", source)
    local_window = int(local.group(1)) if local else 32
    cfg = QwenPDelta3CLVRConfig(
        feature_dim=feature_dim,
        local_window=local_window,
        chunk_size=min(32, local_window),
        conv_kernel=conv_kernel,
        state_dtype="fp16",
        variant="conv4_gdn2_clvr_f96",
        local_gate_init=0.72,
        warm_start_previous_core=True,
    )
    return layers, cfg


def load_pdelta3_strict(local: Path, source: str, device: torch.device):
    layers, cfg = infer_pdelta_config(local, source)
    config = AutoConfig.from_pretrained(local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    replace_full_attention_layers(model, cfg, layers)
    missing, unexpected = load_safetensors_model(
        model, str(local / "model.safetensors"), strict=True, device="cpu"
    )
    if missing or unexpected:
        raise RuntimeError(
            f"PDelta3 strict load mismatch: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True, use_fast=True)
    model.config.use_cache = False
    return model.to(device).eval(), tokenizer, "strict full-checkpoint reconstruction"


def load_source(spec, local: Path, device: torch.device):
    variant = spec["variant"]
    if variant == "pdelta3_clvr":
        return load_pdelta3_strict(local, spec["source"], device)
    if variant == "memory_fusion":
        value = load_memory_fusion(local)
    else:
        value = load_packaged_loader(local)
    if not value:
        raise RuntimeError(f"Could not reconstruct {spec['source']}")
    model, tokenizer, mode = value
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", use_fast=True)
    return model.eval(), tokenizer, mode


@torch.inference_mode()
def next_token(model, tokenizer, prompt="The capital of Austria is"):
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    if not torch.isfinite(out.logits).all():
        raise RuntimeError("Non-finite logits")
    token = int(out.logits[0, -1].argmax())
    return token, tokenizer.decode([token])


@torch.inference_mode()
def quickcheck(model, tokenizer):
    device = next(model.parameters()).device
    rows = []
    for category, question, options, answer in QUICK_TESTS:
        prompt = (
            f"Question: {question}\nA. {options[0]}\nB. {options[1]}\n"
            f"C. {options[2]}\nD. {options[3]}\nAnswer:"
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[0, -1].float()
        scores = {}
        for letter in "ABCD":
            candidates = tokenizer(" " + letter, add_special_tokens=False).input_ids
            if not candidates:
                candidates = tokenizer(letter, add_special_tokens=False).input_ids
            scores[letter] = float(logits[candidates[0]])
        pred = max(scores, key=scores.get)
        rows.append({
            "category": category,
            "prediction": pred,
            "expected": answer,
            "correct": pred == answer,
        })
        print(f"  {category:13s} {'PASS' if pred == answer else 'FAIL'} {pred}/{answer}")
    correct = sum(int(r["correct"]) for r in rows)
    return {
        "name": "TinyCeNN QuickCheck v1",
        "official_benchmark": False,
        "score": round(100.0 * correct / len(rows), 1),
        "correct": correct,
        "total": len(rows),
        "items": rows,
    }


def release_metadata(model, spec):
    bb = backbone(model)
    layers = [
        i for i, layer in enumerate(bb.layers)
        if getattr(layer, "self_attn", None).__class__.__name__ == spec["class"]
    ]
    if not layers:
        raise RuntimeError(f"{spec['class']} is not active; refusing to publish plain Qwen3.5")
    meta = {
        "format": "tinycenn-qwen35-standalone-v2",
        "source": spec["source"],
        "variant": spec["variant"],
        "expected_class": spec["class"],
        "layers": layers,
    }
    if spec["variant"] == "memory_fusion":
        attn = bb.layers[layers[0]].self_attn
        meta["variant_config"] = {
            "feature_dim": int(attn.core.feature_dim),
            "memory_rank": int(attn.core.memory_rank),
            "dilations": list(attn.core.dilations),
            "shifted_window": int(attn.core.shifted_window),
            "train_output_projection": True,
        }
    elif spec["variant"] == "cenn_integrated":
        meta["core_configs"] = {
            str(i): dict(bb.layers[i].self_attn.core.config) for i in layers
        }
    else:
        attn = bb.layers[layers[0]].self_attn
        meta["variant_config"] = {
            "feature_dim": int(attn.core.feature_dim),
            "local_window": int(attn.local_window),
            "chunk_size": int(attn.core.chunk_size),
            "conv_kernel": int(attn.core.conv_kernel),
            "state_dtype": str(attn.core.state_dtype),
            "variant": str(attn.core.variant),
            "local_gate_init": 0.72,
            "warm_start_previous_core": True,
        }
    return meta


def custom_weight_keys(model):
    markers = (".self_attn.core.", ".self_attn.local_gate_")
    return [k for k in model.state_dict().keys() if any(m in k for m in markers)]


def bundle_runtime(export: Path, variant: str):
    package = export / "tinycenn_lm"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        '"""Minimal TinyCeNN runtime bundled with this model release."""\n',
        encoding="utf-8",
    )
    for name in RUNTIME_FILES[variant]:
        shutil.copy2(ROOT / "src" / "tinycenn_lm" / name, package / name)


LOADER = r'''from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_model as load_safetensors_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _backbone(model):
    base = getattr(model, "model", model)
    return getattr(base, "language_model", base)


def load_model(repo_dir=".", device=None):
    root = Path(repo_dir).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    meta = json.loads((root / "standalone_config.json").read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    variant = meta["variant"]

    if variant == "memory_fusion":
        from tinycenn_lm.qwen3_5_memory_fusion import Qwen35MemoryFusionConfig, replace_attention_layers
        replace_attention_layers(
            model,
            Qwen35MemoryFusionConfig.from_dict(meta["variant_config"]),
            meta["layers"],
        )
        model.config.use_cache = False
    elif variant == "cenn_integrated":
        from tinycenn_lm.optimized_memory import OptimizedMemory
        from tinycenn_lm.qwen35_integrated_memory import Qwen35IntegratedAttention
        bb = _backbone(model)
        for raw in meta["layers"]:
            i = int(raw)
            original = bb.layers[i].self_attn
            core = OptimizedMemory(**meta["core_configs"][str(i)])
            bb.layers[i].self_attn = Qwen35IntegratedAttention(original, core, i)
    elif variant == "pdelta3_clvr":
        from tinycenn_lm.qwen35_pdelta3_runtime import QwenPDelta3CLVRConfig, replace_full_attention_layers
        replace_full_attention_layers(
            model,
            QwenPDelta3CLVRConfig.from_dict(meta["variant_config"]),
            meta["layers"],
        )
        model.config.use_cache = False
    else:
        raise ValueError(f"Unknown TinyCeNN variant: {variant}")

    missing, unexpected = load_safetensors_model(
        model,
        str(root / "model.safetensors"),
        strict=True,
        device="cpu",
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Strict weight load failed: missing={list(missing)[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)
    return model.to(device).eval(), tokenizer
'''


LOAD_TEST = r'''from pathlib import Path
import json
import torch
from load_model import load_model

root = Path(__file__).resolve().parent
meta = json.loads((root / "standalone_config.json").read_text())
probe = json.loads((root / "validation_probe.json").read_text())
model, tokenizer = load_model(root)
classes = [m.__class__.__name__ for m in model.modules()]
assert meta["expected_class"] in classes, f"Missing custom class: {meta['expected_class']}"
import tinycenn_lm
runtime = Path(tinycenn_lm.__file__).resolve()
assert str(runtime).startswith(str(root)), f"External TinyCeNN runtime used: {runtime}"
ids = tokenizer(probe["prompt"], return_tensors="pt").input_ids.to(next(model.parameters()).device)
with torch.inference_mode():
    out = model(input_ids=ids, use_cache=False, return_dict=True)
assert torch.isfinite(out.logits).all()
actual = int(out.logits[0, -1].argmax())
assert actual == int(probe["next_token_id"]), (actual, probe["next_token_id"])
print("STANDALONE_RELOAD_PASS")
print("architecture=", meta["expected_class"])
print("layers=", meta["layers"])
print("runtime=", runtime)
print("next_token=", tokenizer.decode([actual]))
'''


def model_card(spec, meta, quick, params, custom_tensors, probe):
    qrows = "\n".join(
        f"| {r['category']} | {'PASS' if r['correct'] else 'FAIL'} | {r['prediction']} | {r['expected']} |"
        for r in quick["items"]
    )
    refs = "\n".join(f"| {k} | {v} |" for k, v in REFERENCE.items())
    return f'''---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- qwen3.5
- tinycenn
- recurrent-attention
- efficient-attention
- standalone
- safetensors
---

# {spec["target"].split("/")[-1]}

Standalone release of `{spec["source"]}` based on **Qwen3.5-0.8B**.

## Why this release is different

This repository contains the **complete model weights**, tokenizer, Qwen3.5 config,
and the minimal TinyCeNN inference runtime required for this architecture. You do
not need to clone the TinyCeNN-LM repository.

The release was uploaded only after a fresh Python process reconstructed the
custom architecture, loaded `model.safetensors` with strict key checking, and
produced finite logits with the same validation next token as the source model.

## Architecture

- Variant: `{meta["variant"]}`
- Custom class: `{meta["expected_class"]}`
- Replaced full-attention layers: `{meta["layers"]}`
- Parameters: `{params:,}`
- Custom TinyCeNN tensors: `{custom_tensors}`
- Standalone reload: **PASS**
- Probe next token: `{probe["next_token_text"]}`

## Install

Until Qwen3.5 support is available in a stable Transformers release used by your
environment:

```bash
pip install torch safetensors huggingface_hub
pip install git+https://github.com/huggingface/transformers.git@main
```

## Load

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

path = Path(snapshot_download("{spec['target']}"))
sys.path.insert(0, str(path))
from load_model import load_model
model, tokenizer = load_model(path, device="cuda")
```

## TinyCeNN QuickCheck v1

This is a small deterministic regression/sanity suite designed to finish quickly.
It is **not** an official MMLU, GPQA, IFEval, MMMLU, or LongBench run and its
percentage must not be compared directly with those benchmark scores.

**Score: {quick['score']}% ({quick['correct']}/{quick['total']})**

| Category | Result | Prediction | Expected |
|---|---:|---:|---:|
{qrows}

Full machine-readable results are in `quickcheck.json`.

## Qwen3.5-0.8B reference context

The values below are reference values supplied for context; this release script
does **not** rerun those benchmark suites.

| Benchmark | Qwen3.5-0.8B |
|---|---:|
{refs}

## Files

- `model.safetensors` — complete reconstructed model weights
- `config.json` — Qwen3.5 model configuration
- tokenizer files
- `tinycenn_lm/` — minimal inference-only TinyCeNN runtime
- `load_model.py` — standalone loader with strict weight checking
- `standalone_config.json` — custom architecture metadata
- `quickcheck.json` — fast sanity-suite results
- `validation_probe.json` — reload-equivalence probe
- `release_report.json` — release summary

## Limitations

- Research model; not a production inference kernel.
- The bundled runtime is reference PyTorch code, not a fused CUDA kernel.
- QuickCheck is only a regression test, not a publication-grade benchmark.
- GGUF still requires native TinyCeNN operator support in llama.cpp.
'''


def build_release(spec, work: Path, token: str | None, upload: bool, private: bool):
    source = spec["source"]
    target = spec["target"]
    source_dir = work / "sources" / source.replace("/", "__")
    export = work / "exports" / target.replace("/", "__")
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    export.parent.mkdir(parents=True, exist_ok=True)
    local = Path(snapshot_download(source, token=token, local_dir=source_dir))
    device, _ = device_dtype()

    print(f"\n{'=' * 96}\n{source}\n{'=' * 96}", flush=True)
    model, tokenizer, mode = load_source(spec, local, device)
    modules = class_modules(model, spec["class"])
    if not modules:
        raise RuntimeError(f"{spec['class']} is not active")
    ckeys = custom_weight_keys(model)
    if not ckeys:
        raise RuntimeError("No TinyCeNN custom tensors found in reconstructed model")
    if spec["variant"] == "pdelta3_clvr":
        for marker in ("A_log", "decay_w", "route_proj"):
            if not any(marker in k for k in ckeys):
                raise RuntimeError(f"PDelta3 custom tensor {marker} is missing")

    print(f"load_mode={mode}", flush=True)
    print(f"custom_layers={len(modules)} custom_tensors={len(ckeys)}", flush=True)
    probe_id, probe_text = next_token(model, tokenizer)
    probe = {
        "prompt": "The capital of Austria is",
        "next_token_id": probe_id,
        "next_token_text": probe_text,
    }
    print(f"probe next token: {probe_text!r}", flush=True)
    print("QuickCheck:", flush=True)
    quick = quickcheck(model, tokenizer)
    print(f"QuickCheck score={quick['score']}%", flush=True)

    meta = release_metadata(model, spec)
    params = sum(p.numel() for p in model.parameters())
    shutil.rmtree(export, ignore_errors=True)
    export.mkdir(parents=True)
    tokenizer.save_pretrained(export)
    model.config.save_pretrained(export)
    if getattr(model, "generation_config", None) is not None:
        try:
            model.generation_config.save_pretrained(export)
        except Exception:
            pass
    bundle_runtime(export, spec["variant"])
    (export / "standalone_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (export / "quickcheck.json").write_text(json.dumps(quick, indent=2), encoding="utf-8")
    (export / "validation_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (export / "load_model.py").write_text(LOADER, encoding="utf-8")
    (export / "load_test.py").write_text(LOAD_TEST, encoding="utf-8")
    (export / "requirements.txt").write_text(
        "torch\nsafetensors>=0.4\nhuggingface_hub>=0.30\n"
        "transformers @ git+https://github.com/huggingface/transformers.git@main\n",
        encoding="utf-8",
    )
    license_path = ROOT / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, export / "TINY_CENN_LICENSE")

    print("Saving complete model.safetensors ...", flush=True)
    model = model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_safetensors_model(
        model,
        str(export / "model.safetensors"),
        metadata={"format": "pt", "source": source, "variant": spec["variant"]},
    )
    size = (export / "model.safetensors").stat().st_size
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Fresh-process standalone reload test ...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(export)
    env["PYTHONUNBUFFERED"] = "1"
    test = subprocess.run(
        [sys.executable, str(export / "load_test.py")],
        cwd="/tmp",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(test.stdout, flush=True)
    if test.returncode:
        raise RuntimeError("Standalone reload failed:\n" + test.stdout[-8000:])

    report = {
        "source": source,
        "target": target,
        "variant": spec["variant"],
        "source_load_mode": mode,
        "custom_class": spec["class"],
        "custom_layers": meta["layers"],
        "custom_tensor_count": len(ckeys),
        "parameters": params,
        "model_bytes": size,
        "quickcheck": quick,
        "probe": probe,
        "standalone_reload": "PASS",
        "uploaded": False,
    }
    (export / "README.md").write_text(
        model_card(spec, meta, quick, params, len(ckeys), probe), encoding="utf-8"
    )
    (export / "release_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if upload:
        if not token:
            raise RuntimeError("HF_TOKEN with write permission is required for upload")
        api = HfApi(token=token)
        api.create_repo(target, repo_type="model", private=private, exist_ok=True)
        api.upload_large_folder(repo_id=target, repo_type="model", folder_path=str(export))
        report["uploaded"] = True
        report["url"] = f"https://huggingface.co/{target}"
        (export / "release_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("UPLOADED:", report["url"], flush=True)
    return report


def main():
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    selected = {x.strip() for x in args.only.split(",") if x.strip()}
    specs = list(MODELS) if "all" in selected else [m for m in MODELS if m["variant"] in selected]
    if not specs:
        raise SystemExit(f"No models selected by --only={args.only!r}")

    results = []
    for spec in specs:
        try:
            results.append(build_release(spec, work, args.token, args.upload, args.private))
        except Exception as exc:
            traceback.print_exc()
            results.append({
                "source": spec["source"],
                "target": spec["target"],
                "variant": spec["variant"],
                "standalone_reload": "FAIL",
                "uploaded": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        (work / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nFINAL SUMMARY", flush=True)
    for r in results:
        print(
            r["source"],
            "reload=", r.get("standalone_reload"),
            "quickcheck=", r.get("quickcheck", {}).get("score"),
            "uploaded=", r.get("uploaded"),
            r.get("url", ""),
            flush=True,
        )
        if r.get("error"):
            print("  ERROR:", r["error"], flush=True)
    print("Report:", work / "summary.json", flush=True)


if __name__ == "__main__":
    main()
