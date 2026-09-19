#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_flyffn_v3 import (
    FlyFFNV3Config,
    assert_qwen35_flyffn_v3,
    flyffn_v2_modules,
    replace_all_ffns_with_fly_v3,
    routing_schedule,
    set_route_state,
)
from tinycenn_lm.qwen35_standalone import export_standalone, upload_standalone

# Reuse the already-tested v2 training utilities; this runner only adapts the backbone.
_V2_PATH = Path(__file__).with_name("run_smollm2_flyffn_v2.py")
_spec = importlib.util.spec_from_file_location("flyffn_v2_base_runner", _V2_PATH)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5-0.8B FlyFFN-v3 all-FFN experiment")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--router-rank", type=int, default=96)
    p.add_argument("--max-edges", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix-init", type=float, default=0.50)
    p.add_argument("--max-ce-gap", type=float, default=None)
    p.add_argument("--quality-rescue", action=argparse.BooleanOptionalAction, default=True,
                   help="After global distillation, selectively relax sparse routing on sensitive layers until the CE gate is recovered.")
    p.add_argument("--rescue-probe-batches", type=int, default=4,
                   help="Number of held-out probe batches used by adaptive quality rescue.")
    p.add_argument("--rewired", action="store_true")
    p.add_argument("--output-dir", default="results/flyffn_v3_qwen35_08b")
    p.add_argument("--standalone-dir", default=None)
    p.add_argument("--upload-hf", action="store_true")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--seed", type=int, default=5321)
    return p.parse_args()


def load_model(dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)


def build_student(dtype, device, cfg, adjacency, seed):
    base.base.set_seed(seed)
    model = load_model(dtype, device)
    replace_all_ffns_with_fly_v3(model, cfg, adjacency)
    assert_qwen35_flyffn_v3(model)
    return model


@torch.no_grad()
def generate_chat_samples(model, tokenizer, device, max_new_tokens=128):
    prompts = [
        "Explain why the sky is blue in a way a 10-year-old can understand.",
        "Write a Python function that returns the two largest unique numbers in a list.",
        "A train travels 180 km in 2 hours and then 120 km in 1.5 hours. What is its average speed for the whole trip? Explain briefly.",
        "Give me three practical ideas for reducing energy use in a data center without reducing reliability.",
    ]
    rows = []
    model.eval()
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text_prompt, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"prompt": prompt, "reply": reply})
    return rows


@torch.no_grad()
def adaptive_quality_rescue(student, teacher, probe_batches, args, device, dtype, max_ce_gap):
    """Recover the CE quality gate with the smallest useful loss of sparsity.

    Global distillation can move a model away from the progressive calibration
    gate.  We therefore test one extra shard at a time on the most aggressively
    sparse layers, keep only changes that improve held-out CE, and stop as soon
    as the requested quality gap is recovered.
    """
    batches = probe_batches[:max(1, min(len(probe_batches), int(args.rescue_probe_batches)))]

    def score():
        m = base.evaluate(student, teacher, batches, device, dtype, args.seq_len, args.batch_size)
        return m, float(m["ce"] - m["teacher_ce"])

    metrics, gap = score()
    history = [{
        "step": 0, "layer": None, "old_k": None, "new_k": None,
        "ce_gap": gap, "accepted": True, "reason": "initial",
    }]
    if gap <= max_ce_gap:
        print(f"RESCUE not needed: held-out CE gap={gap:.4f} <= {max_ce_gap:.4f}", flush=True)
        return metrics, history

    print(f"RESCUE start: held-out CE gap={gap:.4f} > {max_ce_gap:.4f}", flush=True)
    step = 0
    while gap > max_ce_gap:
        candidates = [
            m for m in flyffn_v2_modules(student)
            if m.route_mix > 0.0 and m.active_k < m.num_shards
        ]
        candidates.sort(
            key=lambda m: (m.route_mix * (m.num_shards - m.active_k), -m.layer_idx),
            reverse=True,
        )
        if not candidates:
            break

        best = None
        for m in candidates:
            old_k = m.active_k
            m.set_route_state(old_k + 1, m.route_mix)
            cand_metrics, cand_gap = score()
            m.set_route_state(old_k, m.route_mix)
            improvement = gap - cand_gap
            if best is None or improvement > best["improvement"]:
                best = {
                    "module": m, "old_k": old_k, "new_k": old_k + 1,
                    "metrics": cand_metrics, "gap": cand_gap, "improvement": improvement,
                }

        if best is None or best["improvement"] <= 1e-4:
            print("RESCUE stopped: no single-layer k increase improved held-out CE.", flush=True)
            break

        m = best["module"]
        m.set_route_state(best["new_k"], m.route_mix)
        metrics, gap = best["metrics"], best["gap"]
        step += 1
        history.append({
            "step": step, "layer": int(m.layer_idx),
            "old_k": int(best["old_k"]), "new_k": int(best["new_k"]),
            "ce_gap": gap, "accepted": True, "reason": "best_single_layer_backoff",
        })
        print(
            f"RESCUE step={step} layer={m.layer_idx} k={best['old_k']}→{best['new_k']} "
            f"improvement={best['improvement']:.4f} held-out-gap={gap:.4f}",
            flush=True,
        )

    print(
        f"RESCUE final: held-out CE gap={gap:.4f} | "
        f"{'PASS' if gap <= max_ce_gap else 'BEST_EFFORT'}",
        flush=True,
    )
    return metrics, history


def train_variant(name, adjacency, teacher, tokenizer, cfg, calib_batches, probe_batches,
                  train_batches, eval_batches, args, device, dtype, steps_per_stage,
                  group_size, max_ce_gap, train_updates, grad_accum):
    print(f"STAGE building {name} Qwen3.5 FlyFFN-v3 student", flush=True)
    student = build_student(dtype, device, cfg, adjacency, args.seed)
    eq = base.dense_equivalence(student, teacher, eval_batches[0][:, :-1].to(device), device, dtype)
    print(f"DENSE_EQ {name} max_abs_logit_diff={eq:.8g}", flush=True)
    hist = base.calibrate_progressively(
        name, student, teacher, calib_batches, probe_batches, device, dtype,
        steps_per_stage, group_size, max_ce_gap, args.num_shards,
    )
    print(f"STAGE global distillation: {name}", flush=True)
    metrics, train_hist = base.global_train(
        name, student, teacher, train_batches, eval_batches, args, device, dtype,
        train_updates, grad_accum,
    )
    return student, metrics, hist, train_hist, base.v2_state(student), eq


def main():
    args = parse_args()
    base.base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.base.choose_dtype(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"DEVICE {device} | dtype={dtype} | gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )

    if args.run_mode == "quick":
        steps_per_stage, group_size = 4, 2
        train_updates, grad_accum, eval_count, probe_count = 600, 1, 12, 4
        max_ce_gap = 0.25 if args.max_ce_gap is None else args.max_ce_gap
    else:
        # Layer-wise quality gating prevents one difficult layer from forcing
        # two neighboring layers to roll back with it.
        steps_per_stage, group_size = 8, 1
        train_updates, grad_accum, eval_count, probe_count = 4000, 2, 24, 8
        max_ce_gap = 0.20 if args.max_ce_gap is None else args.max_ce_gap

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bio_adj, rew_adj, edge_count = base.base.extract_graph(
        args.fly_nodes, args.max_edges, args.seed, Path("/content/qwen35_flyffn_v3_cache")
    )

    print("STAGE loading Qwen3.5-0.8B teacher/tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = load_model(dtype, device)
    teacher.eval()
    teacher.config.use_cache = True
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = FlyFFNV3Config(
        fly_nodes=args.fly_nodes,
        router_rank=args.router_rank,
        num_shards=args.num_shards,
        graph_steps=args.graph_steps,
        graph_mix_init=args.graph_mix_init,
    )

    temp = build_student(dtype, device, cfg, bio_adj, args.seed)
    fly_layers = [m.layer_idx for m in flyffn_v2_modules(temp)]
    anchors = []
    n_groups = math.ceil(len(fly_layers) / group_size)
    layer_types = list(getattr(temp.config, "layer_types", []))
    del temp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"Architecture: Qwen token mixers untouched; ALL {len(fly_layers)} FFNs are FlyFFN-v3; "
        f"dense anchors=0", flush=True,
    )
    if layer_types:
        print(
            f"Qwen token mixers: linear_attention={layer_types.count('linear_attention')} | "
            f"full_attention={layer_types.count('full_attention')}", flush=True,
        )
    print(f"Progressive schedule: {base.stage_schedule(args.num_shards)} | CE gate={max_ce_gap:.3f}", flush=True)

    print("STAGE preparing FineWeb-Edu calibration/probe/train/eval blocks", flush=True)
    calib_needed = n_groups * len(base.stage_schedule(args.num_shards)) * steps_per_stage * args.batch_size
    train_needed = train_updates * grad_accum * args.batch_size
    calib_batches = base.base.make_batches(
        list(base.base.token_blocks(tokenizer, args.seed + 5, calib_needed, args.seq_len)), args.batch_size
    )
    probe_batches = base.base.make_batches(
        list(base.base.token_blocks(tokenizer, args.seed + 777, probe_count * args.batch_size, args.seq_len)), args.batch_size
    )
    train_batches = base.base.make_batches(
        list(base.base.token_blocks(tokenizer, args.seed + 10, train_needed, args.seq_len)), args.batch_size
    )
    eval_batches = base.base.make_batches(
        list(base.base.token_blocks(tokenizer, args.seed + 999, eval_count * args.batch_size, args.seq_len)), args.batch_size
    )

    bio, bio_metrics, bio_calib, bio_hist, bio_state, bio_eq = train_variant(
        "biological", bio_adj, teacher, tokenizer, cfg, calib_batches, probe_batches,
        train_batches, eval_batches, args, device, dtype, steps_per_stage, group_size,
        max_ce_gap, train_updates, grad_accum,
    )
    pd.DataFrame(bio_calib).to_csv(out_dir / "bio_progressive_calibration.csv", index=False)
    pd.DataFrame(bio_hist).to_csv(out_dir / "bio_training_history.csv", index=False)
    bio_schedule = routing_schedule(bio)

    rewired_metrics = rew_schedule = rew_eq = None
    if args.rewired:
        del bio
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rew, rewired_metrics, rew_calib, rew_hist, _, rew_eq = train_variant(
            "rewired", rew_adj, teacher, tokenizer, cfg, calib_batches, probe_batches,
            train_batches, eval_batches, args, device, dtype, steps_per_stage, group_size,
            max_ce_gap, train_updates, grad_accum,
        )
        pd.DataFrame(rew_calib).to_csv(out_dir / "rewired_progressive_calibration.csv", index=False)
        pd.DataFrame(rew_hist).to_csv(out_dir / "rewired_training_history.csv", index=False)
        rew_schedule = routing_schedule(rew)
        del rew
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        bio = build_student(dtype, device, cfg, bio_adj, args.seed)
        base.load_v2_state(bio, bio_state)

    assert_qwen35_flyffn_v3(bio)

    rescue_history = []
    rescue_probe_metrics = None
    if args.quality_rescue:
        print("STAGE post-training adaptive quality rescue", flush=True)
        rescue_probe_metrics, rescue_history = adaptive_quality_rescue(
            bio, teacher, probe_batches, args, device, dtype, max_ce_gap
        )
        pd.DataFrame(rescue_history).to_csv(out_dir / "bio_quality_rescue.csv", index=False)
        # Re-evaluate on the untouched eval split after the rescue decision.
        bio_metrics = base.evaluate(
            bio, teacher, eval_batches, device, dtype, args.seq_len, args.batch_size
        )
        bio_state = base.v2_state(bio)

    bio_schedule = routing_schedule(bio)
    test_ids = eval_batches[0][:, :-1].to(device)
    print("STAGE benchmarking Qwen3.5-0.8B vs FlyFFN-v3", flush=True)
    tbench = base.base.benchmark(teacher, test_ids, device)
    fbench = base.base.benchmark(bio, test_ids, device)
    print("STAGE generating chat samples", flush=True)
    chat_samples = generate_chat_samples(bio, tokenizer, device)
    (out_dir / "chat_samples.json").write_text(json.dumps(chat_samples, indent=2), encoding="utf-8")

    report = {
        "architecture": "Qwen3.5-0.8B + FlyFFN-v3 on all 24 FFNs (no dense anchors)",
        "base_model": BASE_MODEL,
        "token_mixers_unchanged": True,
        "flyffn_layers": len(fly_layers),
        "dense_anchor_layers": anchors,
        "qwen_layer_types": layer_types,
        "dense_equivalence_biological_max_abs_logit_diff": bio_eq,
        "dense_equivalence_rewired_max_abs_logit_diff": rew_eq,
        "implementation_note": "v3.2 adds post-training adaptive quality rescue on top of exact dense reconstruction, confidence-tempered routing, and selected-shard sparse execution",
        "device": str(device),
        "dtype": str(dtype),
        "config": {
            "run_mode": args.run_mode,
            "seq_len": args.seq_len,
            "fly_nodes": args.fly_nodes,
            "fly_edges": edge_count,
            "router_rank": args.router_rank,
            "num_shards": args.num_shards,
            "graph_steps": args.graph_steps,
            "graph_mix_init": args.graph_mix_init,
            "replace_all_ffns": True,
            "dense_anchors": 0,
            "quality_gate_max_ce_gap": max_ce_gap,
            "post_training_quality_rescue": bool(args.quality_rescue),
            "rescue_probe_batches": int(args.rescue_probe_batches),
            "stage_schedule": base.stage_schedule(args.num_shards),
            "steps_per_stage": steps_per_stage,
            "global_train_updates": train_updates,
            "grad_accum": grad_accum,
        },
        "biological": bio_metrics,
        "post_training_rescue_probe": rescue_probe_metrics,
        "quality_rescue_history": rescue_history,
        "rewired": rewired_metrics,
        "biological_routing_schedule": bio_schedule,
        "rewired_routing_schedule": rew_schedule,
        "benchmark_teacher": tbench,
        "benchmark_biological": fbench,
        "fly_ce_gap_vs_qwen": bio_metrics["ce"] - bio_metrics["teacher_ce"],
        "fly_ppl_ratio_vs_qwen": bio_metrics["perplexity"] / bio_metrics["teacher_perplexity"],
        "decode_speed_ratio_fly_over_qwen": fbench["decode_tokens_s"] / tbench["decode_tokens_s"],
        "parameter_ratio_fly_over_qwen": sum(p.numel() for p in bio.parameters()) / sum(p.numel() for p in teacher.parameters()),
    }
    if rewired_metrics:
        report["biological_topology_ce_gain"] = rewired_metrics["ce"] - bio_metrics["ce"]
        report["biological_topology_ppl_gain_pct"] = 100 * (
            rewired_metrics["perplexity"] - bio_metrics["perplexity"]
        ) / rewired_metrics["perplexity"]

    rows = {
        "Qwen3.5-0.8B": {"ce": bio_metrics["teacher_ce"], "perplexity": bio_metrics["teacher_perplexity"], **tbench},
        "Qwen3.5 FlyFFN-v3 biological": {"ce": bio_metrics["ce"], "perplexity": bio_metrics["perplexity"], "teacher_kl": bio_metrics["teacher_kl"], **fbench},
    }
    if rewired_metrics:
        rows["Qwen3.5 FlyFFN-v3 rewired"] = {
            "ce": rewired_metrics["ce"], "perplexity": rewired_metrics["perplexity"], "teacher_kl": rewired_metrics["teacher_kl"]
        }
    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(bio_state, out_dir / "biological_qwen35_flyffn_v3.pt")

    standalone_dir = Path(args.standalone_dir) if args.standalone_dir else out_dir / "standalone"
    print("\nSTAGE exporting verified standalone model", flush=True)
    manifest = export_standalone(
        bio,
        tokenizer,
        cfg.to_v2(len(bio.model.layers)),
        standalone_dir,
        metadata=report,
        artifacts_dir=out_dir,
    )
    print(
        f"STANDALONE verified=True state_keys={manifest['state_keys']} "
        f"flyffn_keys={manifest['flyffn_state_keys']} "
        f"size_gb={manifest['state_bytes'] / 1024**3:.3f}",
        flush=True,
    )

    hf_url = None
    if args.upload_hf:
        if not args.hf_repo_id:
            raise RuntimeError("--upload-hf requires --hf-repo-id")
        print(f"STAGE uploading verified standalone model to {args.hf_repo_id}", flush=True)
        hf_url = upload_standalone(
            standalone_dir,
            args.hf_repo_id,
            private=args.hf_private,
        )
        print(f"HUGGINGFACE {hf_url}", flush=True)

    print("\nQWEN3.5 FLYFFN-V3 ALL-FFN CHECK", flush=True)
    print("Token mixers unchanged: True", flush=True)
    print(f"FlyFFN-v3 layers: {len(fly_layers)}/{len(bio.model.layers)} | dense anchors: 0", flush=True)
    print(f"Dense-equivalence max logit diff: {bio_eq}", flush=True)
    print("\nSUMMARY", flush=True)
    print(pd.DataFrame(rows).T, flush=True)
    print("\nKEY REPORT", flush=True)
    keys = (
        "fly_ce_gap_vs_qwen", "fly_ppl_ratio_vs_qwen", "parameter_ratio_fly_over_qwen",
        "decode_speed_ratio_fly_over_qwen", "biological_topology_ce_gain", "biological_topology_ppl_gain_pct",
    )
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=2), flush=True)
    print("\nFINAL BIOLOGICAL ROUTING SCHEDULE", flush=True)
    print(json.dumps(bio_schedule, indent=2), flush=True)
    print("\nCHAT SAMPLES", flush=True)
    for item in chat_samples:
        print("=" * 88, flush=True)
        print("USER:", item["prompt"], flush=True)
        print("FLY:", item["reply"], flush=True)
    print("\nSaved:", out_dir, flush=True)
    print("Standalone:", standalone_dir, flush=True)
    if hf_url:
        print("Hugging Face:", hf_url, flush=True)


if __name__ == "__main__":
    main()
