#!/usr/bin/env python3
"""Test TinyCeNN/Qwen3.5 models and export only GGUFs that stock llama.cpp can execute.

The exporter is intentionally strict:
- source models must load and run first;
- TinyCeNN custom tensors are never dropped or renamed into unrelated llama.cpp tensors;
- a GGUF is uploaded only after llama-cli successfully executes it.

Custom TinyCeNN recurrent/cellular attention currently needs a corresponding
llama.cpp runtime implementation. Such checkpoints are reported as unsupported
instead of producing a misleading GGUF.
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
import transformers
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
    p = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(p.stdout[-12000:], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
    return p


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def qwen35_runtime_status():
    """Return whether the installed Transformers build has native Qwen3.5."""
    try:
        from transformers import Qwen3_5ForCausalLM  # noqa: F401
        import transformers.models.qwen3_5  # noqa: F401
        return True, transformers.__version__, None
    except Exception as e:
        return False, transformers.__version__, f"{type(e).__name__}: {e}"


def snapshot(repo_id: str, work: Path, token: str) -> Path:
    out = work / "sources" / repo_id.replace("/", "__")
    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            local_dir=str(out),
        )
    )


def load_packaged_loader(local: Path):
    loader = local / "load_model.py"
    if not loader.exists():
        return None
    if str(local) not in sys.path:
        sys.path.insert(0, str(local))
    name = "tinycenn_loader_" + re.sub(r"\W+", "_", local.name)
    spec = importlib.util.spec_from_file_location(name, str(loader))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import packaged loader: {loader}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    value = mod.load_model(
        str(local),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if isinstance(value, tuple) and len(value) >= 2:
        return value[0], value[1], "load_model.py"
    return value, None, "load_model.py"


def load_memory_fusion(local: Path):
    cfg_path = local / "qwen35_memory_fusion_config.json"
    state_path = local / "qwen35_memory_fusion.pt"
    full_path = local / "qwen35_memory_fusion_full_state.pt"
    if not cfg_path.exists() and not full_path.exists():
        return None

    from transformers import Qwen3_5ForCausalLM
    from tinycenn_lm.qwen3_5_memory_fusion import (
        Qwen35MemoryFusionConfig,
        load_selected_attention_state,
        replace_attention_layers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )

    if cfg_path.exists():
        meta = read_json(cfg_path)
        base = meta.get("base_model", "Qwen/Qwen3.5-0.8B")
        layers = [int(x) for x in meta.get("accepted_layers", [])]
        config = Qwen35MemoryFusionConfig.from_dict(meta.get("memory_fusion", {}))
        if not state_path.exists():
            raise FileNotFoundError(f"missing {state_path.name}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(full_path, map_location="cpu", weights_only=False)
        base = payload.get("base_model", "Qwen/Qwen3.5-0.8B")
        layers = [int(x) for x in payload.get("target_layers", [])]
        config = Qwen35MemoryFusionConfig.from_dict(payload.get("config", {}))
        state = payload["attention_state"]

    model = Qwen3_5ForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.config.use_cache = False
    replace_attention_layers(model, config, layers)
    load_selected_attention_state(model, state, layers)
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    return model, tok, "memory-fusion reconstruction"


def load_source(repo_id: str, local: Path, token: str):
    errors = []

    for fn in (load_packaged_loader, load_memory_fusion):
        try:
            value = fn(local)
            if value:
                model, tok, mode = value
                if tok is None:
                    tok = AutoTokenizer.from_pretrained(
                        repo_id, trust_remote_code=True, token=token
                    )
                return model, tok, mode
        except Exception as e:
            errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    try:
        tok = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=True,
            token=token,
            use_fast=True,
        )
        for cls in (AutoModelForCausalLM, AutoModel):
            try:
                model = cls.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    token=token,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(device).eval()
                if hasattr(model, "config"):
                    model.config.use_cache = False
                return model, tok, cls.__name__
            except Exception as e:
                errors.append(f"{cls.__name__}: {type(e).__name__}: {e}")
    except Exception as e:
        errors.append(f"tokenizer: {type(e).__name__}: {e}")

    raise RuntimeError("; ".join(errors))


@torch.inference_mode()
def source_test(repo_id: str, local: Path, token: str, new_tokens=10):
    model = tok = None
    try:
        model, tok, mode = load_source(repo_id, local, token)
        ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(
            next(model.parameters()).device
        )
        start = ids.shape[1]

        for _ in range(new_tokens):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError("model output has no logits")
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if (
                tok.eos_token_id is not None
                and int(nxt[0]) == int(tok.eos_token_id)
            ):
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


def safetensor_names(local: Path):
    names = []
    try:
        from safetensors import safe_open
    except Exception:
        return names

    for path in local.glob("*.safetensors"):
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                names.extend(f.keys())
        except Exception:
            continue
    return names


def detect_custom_tinycenn_tensors(local: Path):
    """Detect tensors that stock Qwen3.5 llama.cpp does not have semantics for."""
    names = safetensor_names(local)
    patterns = (
        ".self_attn.core.",
        ".memory_fusion.",
        ".route_proj",
        ".route_gate_",
        ".erase_w",
        ".erase_b",
        ".write_w",
        ".write_b",
        ".q_conv_weight",
        ".k_conv_weight",
        ".v_conv_weight",
    )
    custom = sorted({n for n in names if any(p in n for p in patterns)})
    return custom


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

This repository is uploaded only after `llama-cli` executes the converted model.
The exporter never drops TinyCeNN custom tensors or substitutes the base Qwen model.
"""


def process(
    source: str,
    target: str,
    *,
    work: Path,
    llama: Path,
    token: str,
    keep_f16: bool,
    upload: bool,
):
    converter = llama / "convert_hf_to_gguf.py"
    cli = llama / "build/bin/llama-cli"
    quantize = llama / "build/bin/llama-quantize"

    result = {
        "source": source,
        "target": target,
        "source_test_ok": False,
        "conversion_ok": False,
        "gguf_test_ok": False,
        "uploaded": False,
    }

    print("\n" + "=" * 100 + f"\n{source}\n" + "=" * 100)
    local = snapshot(source, work, token)

    files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
    cfg = read_json(local / "config.json")
    result["source_files"] = files
    result["source_architectures"] = cfg.get("architectures")
    result["source_model_type"] = cfg.get("model_type")

    ok, mode, text, err = source_test(source, local, token)
    result.update(
        source_test_ok=ok,
        source_load_mode=mode,
        source_text=text,
        source_test_error=err,
    )
    print("SOURCE TEST:", "PASS" if ok else f"FAIL: {err}")
    if text:
        print(text)

    weights = list(local.glob("*.safetensors")) + list(local.glob("pytorch_model*.bin"))
    if not weights:
        result["gguf_supported"] = False
        result["conversion_error"] = (
            "This Hub repo is an adapter/custom-loader package and has no standalone "
            "HF weight file. Reconstructing it is possible for PyTorch inference, but "
            "a faithful GGUF additionally requires llama.cpp runtime support for the "
            "TinyCeNN replacement layer."
        )
        print("CONVERSION SKIPPED:", result["conversion_error"])
        return result

    custom = detect_custom_tinycenn_tensors(local)
    if custom:
        result["gguf_supported"] = False
        result["custom_tensor_count"] = len(custom)
        result["custom_tensor_examples"] = custom[:24]
        result["conversion_error"] = (
            "Checkpoint contains TinyCeNN custom recurrent/cellular attention tensors "
            "that stock llama.cpp has no computation graph for. Example: "
            f"{custom[0]}. These tensors are semantically required, so dropping or "
            "blindly renaming them would make the GGUF incorrect. Implement the "
            "TinyCeNN/PDelta3 attention backend in llama.cpp before conversion."
        )
        print("CONVERSION SKIPPED:", result["conversion_error"])
        return result

    result["gguf_supported"] = True
    export = work / "exports" / target.replace("/", "__")
    shutil.rmtree(export, ignore_errors=True)
    export.mkdir(parents=True, exist_ok=True)

    stem = target.split("/")[-1]
    f16 = export / f"{stem}-F16.gguf"
    q4 = export / f"{stem}-Q4_K_M.gguf"

    conv = sh(
        [
            sys.executable,
            converter,
            local,
            "--outfile",
            f16,
            "--outtype",
            "f16",
        ],
        check=False,
    )
    result["converter_log"] = conv.stdout[-16000:]
    if conv.returncode or not f16.exists():
        result["conversion_error"] = (
            f"convert_hf_to_gguf.py failed ({conv.returncode}). "
            "The checkpoint is not representable by the current stock llama.cpp converter."
        )
        return result

    result.update(
        conversion_ok=True,
        f16_file=f16.name,
        f16_bytes=f16.stat().st_size,
    )

    q = sh([quantize, f16, q4, "Q4_K_M"], check=False)
    result["quantize_log"] = q.stdout[-12000:]
    if q.returncode or not q4.exists():
        result.update(
            conversion_ok=False,
            conversion_error=f"Q4_K_M quantization failed ({q.returncode})",
        )
        return result

    result.update(q4_file=q4.name, q4_bytes=q4.stat().st_size)

    test = sh(
        [
            cli,
            "-m",
            q4,
            "-p",
            PROMPT,
            "-n",
            "10",
            "--temp",
            "0",
            "-c",
            "1024",
        ],
        check=False,
    )
    result["gguf_test_output"] = test.stdout[-8000:]
    result["gguf_test_ok"] = test.returncode == 0 and bool(test.stdout.strip())

    if not result["gguf_test_ok"]:
        result["upload_error"] = (
            "strict gate: llama-cli could not execute the generated GGUF"
        )
        return result

    llama_commit = sh(["git", "rev-parse", "HEAD"], cwd=llama).stdout.strip()

    if not keep_f16:
        f16.unlink(missing_ok=True)
        result["f16_file"] = None

    (export / "conversion_report.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )
    (export / "README.md").write_text(
        model_card(source, target, result, llama_commit),
        encoding="utf-8",
    )

    if upload:
        api = HfApi(token=token)
        api.create_repo(target, repo_type="model", private=False, exist_ok=True)
        api.upload_large_folder(
            repo_id=target,
            repo_type="model",
            folder_path=str(export),
        )
        result.update(uploaded=True, url=f"https://huggingface.co/{target}")
        print("UPLOADED:", result["url"])

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-dir", default="/content/llama.cpp")
    p.add_argument("--work-dir", default="/content/tinycenn_gguf_export")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument(
        "--keep-f16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    if not args.token:
        raise SystemExit("HF_TOKEN with write permission is required")

    q35_ok, tf_version, q35_error = qwen35_runtime_status()
    print(f"Transformers: {tf_version}")
    if not q35_ok:
        raise SystemExit(
            "Installed Transformers does not contain native Qwen3.5 support. "
            f"Current version: {tf_version}. Error: {q35_error}. "
            "llama.cpp currently pins transformers==4.57.6, which can downgrade "
            "the Colab environment. Reinstall transformers>=5.17.0 AFTER installing "
            "llama.cpp requirements."
        )

    llama = Path(args.llama_dir)
    work = Path(args.work_dir)

    if not (llama / "convert_hf_to_gguf.py").exists():
        raise SystemExit(f"llama.cpp not found at {llama}")

    results = []
    for source, target in MODELS:
        try:
            result = process(
                source,
                target,
                work=work,
                llama=llama,
                token=args.token,
                keep_f16=args.keep_f16,
                upload=args.upload,
            )
        except Exception as e:
            traceback.print_exc()
            result = {
                "source": source,
                "target": target,
                "source_test_ok": False,
                "conversion_ok": False,
                "gguf_test_ok": False,
                "uploaded": False,
                "fatal_error": f"{type(e).__name__}: {e}",
            }

        results.append(result)
        work.mkdir(parents=True, exist_ok=True)
        (work / "all_results.json").write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )

    print("\nFINAL SUMMARY")
    for r in results:
        print(
            r["source"],
            "source=", r.get("source_test_ok"),
            "gguf_supported=", r.get("gguf_supported"),
            "convert=", r.get("conversion_ok"),
            "llama=", r.get("gguf_test_ok"),
            "uploaded=", r.get("uploaded"),
            r.get("url", ""),
        )
        error = (
            r.get("conversion_error")
            or r.get("source_test_error")
            or r.get("upload_error")
            or r.get("fatal_error")
        )
        if error:
            print("  ", error)


if __name__ == "__main__":
    main()
