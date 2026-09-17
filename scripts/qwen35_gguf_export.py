#!/usr/bin/env python3
"""Test three published TinyCeNN/Qwen3.5 models, convert compatible checkpoints to GGUF, validate with llama.cpp, and publish.

The exporter is deliberately strict: it never drops TinyCeNN custom tensors or substitutes base-Qwen weights. A target HF repo is uploaded only after llama-cli successfully loads and runs the generated Q4_K_M GGUF.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

MODELS = [
    ("vtava/Qwen3.5-0.8B-MemoryFusion", "vtava/Qwen3.5-0.8B-MemoryFusion-GGUF"),
    ("vtava/Qwen3.5-0.8B-CeNN-Integrated-V1", "vtava/Qwen3.5-0.8B-CeNN-Integrated-V1-GGUF"),
    ("vtava/Qwen3.5-0.8B-PDelta3-CLVR-Local32", "vtava/Qwen3.5-0.8B-PDelta3-CLVR-Local32-GGUF"),
]
PROMPT = "The capital of Austria is"


def sh(cmd, *, cwd=None, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-12000:], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
    return p


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def snapshot(repo_id: str, work: Path, token: str) -> Path:
    out = work / "sources" / repo_id.replace("/", "__")
    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(repo_id=repo_id, repo_type="model", token=token, local_dir=str(out)))


def load_packaged_loader(local: Path):
    loader = local / "load_model.py"
    if not loader.exists():
        return None
    if str(local) not in sys.path:
        sys.path.insert(0, str(local))
    name = "tinycenn_loader_" + re.sub(r"\W+", "_", local.name)
    spec = importlib.util.spec_from_file_location(name, str(loader))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    value = mod.load_model(str(local), device=("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(value, tuple) and len(value) >= 2:
        return value[0], value[1], "load_model.py"
    return value, None, "load_model.py"


def load_memory_fusion(local: Path):
    cfg_path = local / "qwen35_memory_fusion_config.json"
    state_path = local / "qwen35_memory_fusion.pt"
    full_path = local / "qwen35_memory_fusion_full_state.pt"
    if not cfg_path.exists() and not full_path.exists():
        return None
    from tinycenn_lm.qwen3_5_memory_fusion import Qwen35MemoryFusionConfig, replace_attention_layers, load_selected_attention_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device.type == "cuda" else torch.float32)
    if cfg_path.exists():
        meta = read_json(cfg_path)
        base = meta.get("base_model", "Qwen/Qwen3.5-0.8B")
        layers = [int(x) for x in meta.get("accepted_layers", [])]
        config = Qwen35MemoryFusionConfig.from_dict(meta.get("memory_fusion", {}))
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(full_path, map_location="cpu", weights_only=False)
        base = payload.get("base_model", "Qwen/Qwen3.5-0.8B")
        layers = [int(x) for x in payload.get("target_layers", [])]
        config = Qwen35MemoryFusionConfig.from_dict(payload.get("config", {}))
        state = payload["attention_state"]

    # Do not import Qwen3_5ForCausalLM directly. Some Transformers builds register
    # Qwen3.5 with AutoModelForCausalLM but do not export the concrete class at
    # transformers.Qwen3_5ForCausalLM. AutoModel keeps the exporter compatible
    # with both native registrations and repositories that provide remote code.
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    replace_attention_layers(model, config, layers)
    load_selected_attention_state(model, state, layers)
    return model, AutoTokenizer.from_pretrained(base, trust_remote_code=True, use_fast=True), "memory-fusion reconstruction"


def load_source(repo_id: str, local: Path, token: str):
    errors = []
    for fn in (load_packaged_loader, load_memory_fusion):
        try:
            value = fn(local)
            if value:
                model, tok, mode = value
                if tok is None:
                    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, token=token)
                return model, tok, mode
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, token=token, use_fast=True)
        for cls in (AutoModelForCausalLM, AutoModel):
            try:
                model = cls.from_pretrained(repo_id, trust_remote_code=True, token=token, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
                return model, tok, cls.__name__
            except Exception as e:
                errors.append(f"{cls.__name__}: {e}")
    except Exception as e:
        errors.append(f"tokenizer: {e}")
    raise RuntimeError("; ".join(errors))


@torch.inference_mode()
def source_test(repo_id: str, local: Path, token: str, new_tokens=10):
    model = tok = None
    try:
        model, tok, mode = load_source(repo_id, local, token)
        ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
        start = ids.shape[1]
        for _ in range(new_tokens):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError("model output has no logits")
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if tok.eos_token_id is not None and int(nxt[0]) == int(tok.eos_token_id):
                break
        if ids.shape[1] <= start:
            raise RuntimeError("no generated tokens")
        return True, mode, tok.decode(ids[0], skip_special_tokens=True), None
    except Exception as e:
        return False, None, None, f"{type(e).__name__}: {e}"
    finally:
        try:
            del model, tok
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def model_card(source: str, target: str, result: dict, llama_commit: str):
    return f"""---
library_name: llama.cpp
base_model: {source}
tags: [gguf, qwen3.5, tinycenn, llama.cpp]
---
# {target.split('/')[-1]}

Validated GGUF export of `{source}`.

- Source smoke test: **{result.get('source_test_ok')}**
- llama.cpp conversion: **{result.get('conversion_ok')}**
- llama-cli test: **{result.get('gguf_test_ok')}**
- llama.cpp commit: `{llama_commit}`
- Q4_K_M: `{result.get('q4_file')}`
- F16: `{result.get('f16_file')}`

This repository is uploaded only after `llama-cli` executes the converted model. The exporter never intentionally drops TinyCeNN custom tensors or substitutes the base Qwen model.
"""


def process(source: str, target: str, *, work: Path, llama: Path, token: str, keep_f16: bool, upload: bool):
    converter = llama / "convert_hf_to_gguf.py"
    cli = llama / "build/bin/llama-cli"
    quantize = llama / "build/bin/llama-quantize"
    result = {"source": source, "target": target, "source_test_ok": False, "conversion_ok": False, "gguf_test_ok": False, "uploaded": False}
    print("\n" + "=" * 100 + f"\n{source}\n" + "=" * 100)
    local = snapshot(source, work, token)
    files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
    cfg = read_json(local / "config.json")
    result["source_files"] = files
    result["source_architectures"] = cfg.get("architectures")
    result["source_model_type"] = cfg.get("model_type")

    ok, mode, text, err = source_test(source, local, token)
    result.update(source_test_ok=ok, source_load_mode=mode, source_text=text, source_test_error=err)
    print("SOURCE TEST:", "PASS" if ok else f"FAIL: {err}")
    if text:
        print(text)

    weights = list(local.glob("*.safetensors")) + list(local.glob("pytorch_model*.bin"))
    if not weights:
        result["conversion_error"] = "adapter/custom-loader package has no standalone HF weight file; faithful stock llama.cpp conversion is unavailable"
        print("CONVERSION SKIPPED:", result["conversion_error"])
        return result

    export = work / "exports" / target.replace("/", "__")
    shutil.rmtree(export, ignore_errors=True)
    export.mkdir(parents=True, exist_ok=True)
    stem = target.split("/")[-1]
    f16, q4 = export / f"{stem}-F16.gguf", export / f"{stem}-Q4_K_M.gguf"
    conv = sh([sys.executable, converter, local, "--outfile", f16, "--outtype", "f16"], check=False)
    result["converter_log"] = conv.stdout[-16000:]
    if conv.returncode or not f16.exists():
        result["conversion_error"] = f"convert_hf_to_gguf.py failed ({conv.returncode}); custom TinyCeNN architecture is not representable by stock llama.cpp"
        return result
    result.update(conversion_ok=True, f16_file=f16.name, f16_bytes=f16.stat().st_size)

    q = sh([quantize, f16, q4, "Q4_K_M"], check=False)
    result["quantize_log"] = q.stdout[-12000:]
    if q.returncode or not q4.exists():
        result.update(conversion_ok=False, conversion_error=f"Q4_K_M quantization failed ({q.returncode})")
        return result
    result.update(q4_file=q4.name, q4_bytes=q4.stat().st_size)

    test = sh([cli, "-m", q4, "-p", PROMPT, "-n", "10", "--temp", "0", "-c", "1024"], check=False)
    result["gguf_test_output"] = test.stdout[-8000:]
    result["gguf_test_ok"] = test.returncode == 0 and bool(test.stdout.strip())
    if not result["gguf_test_ok"]:
        result["upload_error"] = "strict gate: llama-cli could not execute the generated GGUF"
        return result

    llama_commit = sh(["git", "rev-parse", "HEAD"], cwd=llama).stdout.strip()
    if not keep_f16:
        f16.unlink(missing_ok=True)
        result["f16_file"] = None
    (export / "conversion_report.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (export / "README.md").write_text(model_card(source, target, result, llama_commit), encoding="utf-8")
    if upload:
        api = HfApi(token=token)
        api.create_repo(target, repo_type="model", private=False, exist_ok=True)
        api.upload_large_folder(repo_id=target, repo_type="model", folder_path=str(export))
        result.update(uploaded=True, url=f"https://huggingface.co/{target}")
        print("UPLOADED:", result["url"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-dir", default="/content/llama.cpp")
    p.add_argument("--work-dir", default="/content/tinycenn_gguf_export")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--keep-f16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if not args.token:
        raise SystemExit("HF_TOKEN with write permission is required")
    llama, work = Path(args.llama_dir), Path(args.work_dir)
    if not (llama / "convert_hf_to_gguf.py").exists():
        raise SystemExit(f"llama.cpp not found at {llama}")
    results = []
    for source, target in MODELS:
        try:
            result = process(source, target, work=work, llama=llama, token=args.token, keep_f16=args.keep_f16, upload=args.upload)
        except Exception as e:
            traceback.print_exc()
            result = {"source": source, "target": target, "source_test_ok": False, "conversion_ok": False, "gguf_test_ok": False, "uploaded": False, "fatal_error": f"{type(e).__name__}: {e}"}
        results.append(result)
        work.mkdir(parents=True, exist_ok=True)
        (work / "all_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("\nFINAL SUMMARY")
    for r in results:
        print(r["source"], "source=", r.get("source_test_ok"), "convert=", r.get("conversion_ok"), "llama=", r.get("gguf_test_ok"), "uploaded=", r.get("uploaded"), r.get("url", ""))
        error = r.get("conversion_error") or r.get("source_test_error") or r.get("upload_error") or r.get("fatal_error")
        if error:
            print("  ", error)


if __name__ == "__main__":
    main()
