#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_smollm2_memory_fusion_sequential as seq
from train_qwen35_pdelta3_clvr_sequential import (
    QwenPDelta3CLVRConfig,
    replace_full_attention_layers,
)


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    v.add_argument("--output-dir", required=True)
    v.add_argument("--probe-context", type=int, default=128)
    v.add_argument("--probe-blocks", type=int, default=6)
    v.add_argument("--accept-nmse", type=float, default=0.15)
    v.add_argument("--accept-cosine", type=float, default=0.94)
    v.add_argument("--accept-incremental-delta-nll", type=float, default=0.015)
    v.add_argument("--accept-cumulative-delta-nll", type=float, default=0.05)

    u = sub.add_parser("upload")
    u.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    u.add_argument("--output-dir", required=True)
    u.add_argument("--repo-id", default="vtava/Qwen3.5-0.8B-PDelta3-CLVR-Local32")
    u.add_argument("--export-dir", default="/content/qwen35_pdelta3_clvr_hf_export")

    return p.parse_args()


def _device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _load_progress(out: Path):
    path = out / "qwen35_progress.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing accepted checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    accepted = [int(x) for x in payload.get("accepted_full_attention_layers", [])]
    if not accepted:
        raise RuntimeError("No accepted Qwen3.5 full-attention layers are saved yet.")
    cfg = QwenPDelta3CLVRConfig.from_dict(payload["config"])
    return payload, accepted, cfg


def _apply_state(model, payload, accepted):
    replace_full_attention_layers(model, QwenPDelta3CLVRConfig.from_dict(payload["config"]), accepted)
    incompatible = model.load_state_dict(payload["attention_state"], strict=False)
    prefixes = tuple(f"model.layers.{i}.self_attn." for i in accepted)
    missing = [k for k in incompatible.missing_keys if prefixes and k.startswith(prefixes)]
    if missing:
        raise RuntimeError(f"Accepted checkpoint missing replacement keys: {missing[:12]}")


def verify(args):
    out = Path(args.output_dir)
    payload, accepted, cfg = _load_progress(out)
    device, dtype = _device_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe_blocks = seq.build_probe_blocks(tokenizer, context=args.probe_context, count=args.probe_blocks)
    amp = seq.amp_factory(device, dtype)

    prompts = [
        "The future of small language models is",
        "Artificial intelligence can help scientists by",
        "A good software architecture should",
        "The capital of Austria is",
        "Once upon a time, a small robot",
    ]

    @torch.no_grad()
    def generate_all(model):
        model.eval()
        result = {}
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            result[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
        return result

    print("Loading baseline...")
    baseline = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        token=False,
        attn_implementation="eager",
    ).to(device)
    baseline.config.use_cache = False
    baseline_nll = seq.probe_nll(baseline, probe_blocks, device, amp)
    baseline_text = generate_all(baseline)
    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading accepted PDelta3-CLVR checkpoint...")
    candidate = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        token=False,
        attn_implementation="eager",
    ).to(device)
    candidate.config.use_cache = False
    _apply_state(candidate, payload, accepted)
    candidate_nll = seq.probe_nll(candidate, probe_blocks, device, amp)
    candidate_text = generate_all(candidate)
    delta_nll = candidate_nll - baseline_nll

    accepted_reports = {}
    for report in payload.get("reports", []):
        if bool(report.get("accepted", False)):
            accepted_reports[int(report["layer"])] = report

    checks = []
    for layer in accepted:
        r = accepted_reports.get(layer)
        if r is None:
            checks.append({"layer": layer, "pass": False, "reason": "no accepted report"})
            continue
        ok = (
            float(r["nmse"]) <= args.accept_nmse
            and float(r["cosine"]) >= args.accept_cosine
            and float(r["incremental_delta_nll"]) <= args.accept_incremental_delta_nll
            and float(r["cumulative_delta_nll"]) <= args.accept_cumulative_delta_nll
        )
        checks.append({
            "layer": layer,
            "pass": bool(ok),
            "nmse": float(r["nmse"]),
            "cosine": float(r["cosine"]),
            "incremental_delta_nll": float(r["incremental_delta_nll"]),
            "cumulative_delta_nll": float(r["cumulative_delta_nll"]),
        })

    quality_pass = bool(delta_nll <= args.accept_cumulative_delta_nll)
    gates_pass = all(x["pass"] for x in checks)
    verified = bool(quality_pass and gates_pass)

    examples = []
    for prompt in prompts:
        print("\n" + "=" * 100)
        print("PROMPT:", prompt)
        print("\nBASELINE:\n" + baseline_text[prompt])
        print("\nPDELTA3-CLVR:\n" + candidate_text[prompt])
        examples.append({
            "prompt": prompt,
            "baseline": baseline_text[prompt],
            "pdelta3_clvr": candidate_text[prompt],
        })

    report = {
        "verified": verified,
        "base_model": args.base_model,
        "accepted_full_attention_layers": accepted,
        "config": cfg.to_dict(),
        "probe_context": args.probe_context,
        "probe_blocks": args.probe_blocks,
        "baseline_probe_nll": baseline_nll,
        "candidate_probe_nll": candidate_nll,
        "delta_nll": delta_nll,
        "release_cumulative_delta_nll_limit": args.accept_cumulative_delta_nll,
        "quality_pass": quality_pass,
        "all_saved_layer_gates_pass": gates_pass,
        "layer_checks": checks,
        "prompt_examples": examples,
    }
    path = out / "qwen35_verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nVERIFICATION SUMMARY")
    print(json.dumps({
        "verified": verified,
        "accepted_full_attention_layers": accepted,
        "baseline_probe_nll": baseline_nll,
        "candidate_probe_nll": candidate_nll,
        "delta_nll": delta_nll,
        "all_saved_layer_gates_pass": gates_pass,
    }, indent=2))
    print("Saved:", path)


def upload(args):
    out = Path(args.output_dir)
    verification_path = out / "qwen35_verification.json"
    if not verification_path.exists():
        raise FileNotFoundError("Run verification first.")
    verification = json.loads(verification_path.read_text())
    if not verification.get("verified", False):
        raise RuntimeError("Upload blocked: verification did not pass.")

    payload, accepted, cfg = _load_progress(out)
    if accepted != [int(x) for x in verification["accepted_full_attention_layers"]]:
        raise RuntimeError("Verification and progress checkpoint refer to different accepted layers.")

    export = Path(args.export_dir)
    if export.exists():
        shutil.rmtree(export)
    export.mkdir(parents=True)

    print("Building verified export model...")
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        token=False,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    _apply_state(model, payload, accepted)
    model.save_pretrained(export, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=False, use_fast=True)
    tokenizer.save_pretrained(export)

    meta = {
        "format_version": 1,
        "architecture": "qwen3.5-pdelta3-gdn2-clvr-localw",
        "base_model": args.base_model,
        "accepted_full_attention_layers": accepted,
        "replacement_config": cfg.to_dict(),
        "verification_file": "qwen35_verification.json",
        "loader": "load_model.py",
    }
    (export / "tinycenn_qwen35.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name in ("qwen35_verification.json", "qwen35_progress.json", "qwen35_run_status.json"):
        src = out / name
        if src.exists():
            shutil.copy2(src, export / name)

    shutil.copytree(REPO_ROOT / "src" / "tinycenn_lm", export / "src" / "tinycenn_lm")
    (export / "scripts").mkdir(parents=True, exist_ok=True)
    for name in (
        "train_qwen35_pdelta3_clvr_sequential.py",
        "train_smollm2_memory_fusion_sequential.py",
    ):
        shutil.copy2(REPO_ROOT / "scripts" / name, export / "scripts" / name)

    loader = '''from __future__ import annotations\n\nimport json\nimport sys\nfrom pathlib import Path\nimport torch\nfrom safetensors.torch import load_file\nfrom transformers import AutoTokenizer, Qwen3_5ForCausalLM\n\ndef load_model(path=None, device="cpu", dtype=None):\n    root = Path(path or Path(__file__).resolve().parent)\n    for p in (root / "src", root, root / "scripts"):\n        if str(p) not in sys.path:\n            sys.path.insert(0, str(p))\n    from train_qwen35_pdelta3_clvr_sequential import QwenPDelta3CLVRConfig, replace_full_attention_layers\n    meta = json.loads((root / "tinycenn_qwen35.json").read_text())\n    accepted = [int(x) for x in meta["accepted_full_attention_layers"]]\n    cfg = QwenPDelta3CLVRConfig.from_dict(meta["replacement_config"])\n    if dtype is None:\n        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (torch.float16 if device.startswith("cuda") else torch.float32)\n    model = Qwen3_5ForCausalLM.from_pretrained(root, dtype=dtype, local_files_only=True, attn_implementation="eager")\n    replace_full_attention_layers(model, cfg, accepted)\n    single = root / "model.safetensors"\n    if single.exists():\n        model.load_state_dict(load_file(str(single), device="cpu"), strict=False)\n    else:\n        index = json.loads((root / "model.safetensors.index.json").read_text())\n        for shard in sorted(set(index["weight_map"].values())):\n            model.load_state_dict(load_file(str(root / shard), device="cpu"), strict=False)\n    model.config.use_cache = False\n    model.to(device).eval()\n    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)\n    return model, tokenizer\n'''
    (export / "load_model.py").write_text(loader, encoding="utf-8")
    (export / "requirements.txt").write_text(
        "torch\ntransformers==4.57.6\ndatasets>=3,<5\nsafetensors\n",
        encoding="utf-8",
    )

    readme = f'''---\nlicense: apache-2.0\nbase_model: {args.base_model}\ntags:\n- qwen3.5\n- tinycenn\n- pdelta3\n- gdn2\n- clvr\n- recurrent-attention\n- research\n---\n\n# Qwen3.5-0.8B PDelta3-CLVR Local32\n\n- Base model: `{args.base_model}`\n- Verified replacement layers: `{accepted}`\n- Verification delta NLL: `{verification["delta_nll"]:.6f}`\n- Verification status: **PASS**\n\nThis is a custom adapted architecture. Load with:\n\n```python\nfrom load_model import load_model\nmodel, tokenizer = load_model(device="cuda")\n```\n'''
    (export / "README.md").write_text(readme, encoding="utf-8")

    del model
    gc.collect()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(export),
        commit_message=f"Upload verified PDelta3-CLVR checkpoint for layers {accepted}",
    )
    print("UPLOADED: https://huggingface.co/" + args.repo_id)


def main():
    args = parse_args()
    if args.command == "verify":
        verify(args)
    else:
        upload(args)


if __name__ == "__main__":
    main()
