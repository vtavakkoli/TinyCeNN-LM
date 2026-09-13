#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load and smoke-test a locally trained TinyCeNN-LM experiment")
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=24)
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_model(model_key: str, root: Path, device: torch.device, dtype: torch.dtype):
    if model_key == "tinycenn-lm":
        from tinycenn_lm import build_from_adapter
        return build_from_adapter(root, device=device, dtype=dtype), {"builder": "build_from_adapter"}
    if model_key in {"tinycenn-distill", "tinycenn-rigorous-continue", "tinycenn-optimized-continue"}:
        from tinycenn_lm import CeNNReplacementLayer, build_cenn_student
        model = build_cenn_student(root, device=device, dtype=dtype)
        count = sum(isinstance(m, CeNNReplacementLayer) for m in model.modules())
        return model, {"builder": "build_cenn_student", "cenn_replacement_layers": count}
    if model_key == "tinycenn-moe-top2":
        from tinycenn_lm import MoECeNNReplacementLayer, build_moe_cenn_student
        model = build_moe_cenn_student(root, device=device, dtype=dtype)
        count = sum(isinstance(m, MoECeNNReplacementLayer) for m in model.modules())
        return model, {"builder": "build_moe_cenn_student", "moe_cenn_layers": count}
    if model_key in {"tinycenn-sharedffn-top2", "tinycenn-story-antirepeat"}:
        from tinycenn_lm import ShardedMoECeNNReplacementLayer, build_sharded_moe_student
        model = build_sharded_moe_student(root, device=device, dtype=dtype)
        count = sum(isinstance(m, ShardedMoECeNNReplacementLayer) for m in model.modules())
        return model, {"builder": "build_sharded_moe_student", "sharded_cenn_layers": count}
    if model_key == "tinycenn-story-v2":
        from tinycenn_lm.story_v2 import LowRankLMHeadAdapter, StoryV2ReplacementLayer, build_story_v2_student
        model = build_story_v2_student(root, device=device, dtype=dtype)
        return model, {
            "builder": "build_story_v2_student",
            "story_v2_layers": sum(isinstance(m, StoryV2ReplacementLayer) for m in model.modules()),
            "low_rank_heads": sum(isinstance(m, LowRankLMHeadAdapter) for m in model.modules()),
        }
    if model_key == "smollm2-amcenn-top2":
        from tinycenn_lm.smollm2_amcenn import AMCeNNAttention, ShardedTop2LlamaMLP, build_smollm2_amcenn
        model = build_smollm2_amcenn(root, device=device, dtype=dtype)
        return model, {
            "builder": "build_smollm2_amcenn",
            "amcenn_layers": sum(isinstance(m, AMCeNNAttention) for m in model.modules()),
            "sharded_ffn_layers": sum(isinstance(m, ShardedTop2LlamaMLP) for m in model.modules()),
        }
    if model_key == "smollm2-amcenn-top2-v2":
        from tinycenn_lm.smollm2_amcenn import ShardedTop2LlamaMLP
        from tinycenn_lm.smollm2_amcenn_v2 import AMCeNNAttentionV2, build_smollm2_amcenn_v2
        model = build_smollm2_amcenn_v2(root, device=device, dtype=dtype)
        return model, {
            "builder": "build_smollm2_amcenn_v2",
            "amcenn_v2_layers": sum(isinstance(m, AMCeNNAttentionV2) for m in model.modules()),
            "sharded_ffn_layers": sum(isinstance(m, ShardedTop2LlamaMLP) for m in model.modules()),
        }
    raise ValueError(f"unknown model key: {model_key}")


def load_training_report(root: Path) -> tuple[str | None, dict | None]:
    candidates = [
        "training_report.json",
        "distillation_report.json",
        "moe_distillation_report.json",
        "sharded_moe_distillation_report.json",
        "smollm2_amcenn_training_report.json",
        "smollm2_amcenn_v2_training_report.json",
        "story_training_report.json",
        "story_v2_training_report.json",
    ]
    for name in candidates:
        path = root / name
        if path.exists():
            return name, json.loads(path.read_text(encoding="utf-8"))
    return None, None


def main() -> None:
    args = parse_args()
    root = Path(args.model_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    model, structure = build_model(args.model, root, device, dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sanity_text = (
        "Story beginning:\nMia found a small robot in the park.\nContinue the story:\n"
        if "story" in args.model
        else "TinyCeNN is a compact recurrent language model."
    )
    sample = tokenizer(sanity_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**sample, labels=sample["input_ids"], use_cache=False)
    loss = float(outputs.loss.detach().float().cpu())
    if not math.isfinite(loss):
        raise RuntimeError(f"non-finite local sanity loss: {loss}")

    prompt = (
        "Story beginning:\nMia found a small robot under a tree.\nContinue the story:\n"
        if "story" in args.model
        else "The capital of Austria is"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = int(inputs["input_ids"].shape[1])
    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    if int(generated.shape[1]) <= input_len:
        raise RuntimeError("generation produced no new tokens")
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    report_name, report = load_training_report(root)
    evaluation = {
        "status": "PASS",
        "model": args.model,
        "model_dir": str(root),
        "device": str(device),
        "dtype": str(dtype),
        "sanity_loss": loss,
        "sanity_perplexity": math.exp(min(loss, 20.0)),
        "generated_tokens": int(generated.shape[1]) - input_len,
        "prompt": prompt,
        "generation": generated_text,
        "structure": structure,
        "training_report": report_name,
    }
    if report is not None:
        evaluation["training_status"] = report.get("status")
        evaluation["architecture"] = report.get("architecture")
        evaluation["evaluation_performed_during_training"] = report.get("evaluation_performed")
        for key in ("best_eval_loss", "best_perplexity", "teacher_gap_recovery_fraction", "stop_reason"):
            if key in report:
                evaluation[key] = report[key]
        if isinstance(report.get("best"), dict):
            evaluation["best"] = report["best"]
        if isinstance(report.get("evaluation"), dict):
            evaluation["training_evaluation"] = report["evaluation"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
