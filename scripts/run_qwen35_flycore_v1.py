#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_flycore_v1 import (
    FlyFFNV3Config,
    FlyVocabConfig,
    assert_qwen35_flycore,
    factorize_embedding_weight,
    fly_vocab_parameter_groups,
    replace_qwen_with_flycore,
)
from tinycenn_lm.qwen35_flyffn_v3 import flyffn_v2_modules, routing_schedule
from tinycenn_lm.qwen35_flycore_standalone import export_flycore_standalone, upload_flycore_standalone

_V3_PATH = Path(__file__).with_name("run_qwen35_flyffn_v3.py")
_spec = importlib.util.spec_from_file_location("flyffn_v3_base_runner", _V3_PATH)
v3 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(v3)
base = v3.base

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen3.5-0.8B FlyCore: FlyEmbedding + FlyLMHead + FlyFFN-v3 on all FFNs"
    )
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--router-rank", type=int, default=96)
    p.add_argument("--max-edges", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix-init", type=float, default=0.50)
    p.add_argument("--vocab-latent-dim", type=int, default=384)
    p.add_argument("--vocab-graph-mix-init", type=float, default=0.10)
    p.add_argument("--factor-chunk-rows", type=int, default=4096)
    p.add_argument("--max-ce-gap", type=float, default=None)
    p.add_argument("--rewired", action="store_true")
    p.add_argument("--output-dir", default="results/flycore_v1_qwen35_08b")
    p.add_argument("--standalone-dir", default=None)
    p.add_argument("--upload-hf", action="store_true")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--seed", type=int, default=6421)
    return p.parse_args()


def load_model(dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)


def build_student(dtype, device, ffn_cfg, vocab_cfg, adjacency, factorization, seed):
    base.base.set_seed(seed)
    model = load_model(dtype, device)
    replace_qwen_with_flycore(
        model,
        ffn_cfg,
        vocab_cfg,
        adjacency,
        factorization=factorization,
    )
    assert_qwen35_flycore(model)
    return model


@torch.no_grad()
def probe_model(student, teacher, batches, device, dtype):
    tce, sce, kl, emb = [], [], [], []
    student.eval()
    teacher.eval()
    for cpu_ids in batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        with base.base.amp_ctx(device, dtype):
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            s = student(input_ids=x, use_cache=False, return_dict=True)
            te = teacher.model.embed_tokens(x)
            se = student.model.embed_tokens(x)
        tce.append(float(base.base.causal_ce(t.logits, ids)))
        sce.append(float(base.base.causal_ce(s.logits, ids)))
        kl.append(float(base.base.distill_kl(s.logits, t.logits)))
        denom = te.float().square().mean().clamp_min(1e-8)
        emb.append(float(((se.float() - te.float()).square().mean() / denom).item()))
    return {
        "teacher_ce": sum(tce) / len(tce),
        "student_ce": sum(sce) / len(sce),
        "ce_gap": sum(sce) / len(sce) - sum(tce) / len(tce),
        "teacher_kl": sum(kl) / len(kl),
        "embedding_relative_mse": sum(emb) / len(emb),
    }


def calibrate_vocab(
    student,
    teacher,
    train_batches,
    probe_batches,
    updates,
    device,
    dtype,
):
    groups, trainable = fly_vocab_parameter_groups(
        student,
        code_lr=1.5e-4,
        basis_lr=4e-5,
        fly_lr=3e-4,
        weight_decay=0.01,
    )
    opt = base.base.make_optimizer(groups, device)
    scaler = base.base.make_scaler(device, dtype)
    history = []
    student.train()
    teacher.eval()
    t0 = time.perf_counter()

    for update in range(1, updates + 1):
        ids = train_batches[(update - 1) % len(train_batches)].to(device)
        x = ids[:, :-1]
        opt.zero_grad(set_to_none=True)

        with torch.no_grad(), base.base.amp_ctx(device, dtype):
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            teacher_emb = teacher.model.embed_tokens(x)

        with base.base.amp_ctx(device, dtype):
            s = student(input_ids=x, use_cache=False, return_dict=True)
            student_emb = student.model.embed_tokens(x)
            ce = base.base.causal_ce(s.logits, ids)
            kl = base.base.distill_kl(s.logits, t.logits)
            denom = teacher_emb.float().square().mean().clamp_min(1e-8)
            emb = (student_emb.float() - teacher_emb.float()).square().mean() / denom
            loss = 0.35 * ce + 0.45 * kl + 0.20 * emb

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()

        row = {
            "update": update,
            "loss": float(loss.detach()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "embedding_relative_mse": float(emb.detach()),
        }
        history.append(row)

        if update == 1 or update % 20 == 0 or update == updates:
            elapsed = time.perf_counter() - t0
            print(
                f"VOCAB {update}/{updates} loss={row['loss']:.4f} "
                f"ce={row['ce']:.4f} kl={row['kl']:.4f} emb={row['embedding_relative_mse']:.4f} "
                f"ups={update/max(elapsed,1e-9):.3f}",
                flush=True,
            )

    metrics = probe_model(student, teacher, probe_batches, device, dtype)
    return metrics, history


def global_train_flycore(
    name,
    student,
    teacher,
    train_batches,
    eval_batches,
    args,
    device,
    dtype,
    train_updates,
    grad_accum,
):
    layers = [m.layer_idx for m in flyffn_v2_modules(student) if m.route_mix > 0.0]

    # Freeze the backbone. If no FFN group passed the gate, continue training
    # the Fly vocabulary core instead of constructing empty optimizer groups.
    if layers:
        ffn_groups, ffn_trainable = base.flyffn_v2_parameter_groups(
            student,
            layers,
            router_lr=4e-4,
            shard_lr=6e-6,
            weight_decay=0.01,
        )
    else:
        for p in student.parameters():
            p.requires_grad = False
        ffn_groups, ffn_trainable = [], []

    core = student.fly_vocab_core
    core.codebook.weight.requires_grad = True
    core.basis.requires_grad = True
    for p in core.fly_down.parameters():
        p.requires_grad = True
    for p in core.fly_up.parameters():
        p.requires_grad = True
    core.graph_mix_logit.requires_grad = True

    vocab_fly_params = [
        *core.fly_down.parameters(),
        *core.fly_up.parameters(),
        core.graph_mix_logit,
    ]
    groups = [
        *ffn_groups,
        {"params": [core.codebook.weight], "lr": 8e-5, "weight_decay": 0.01},
        {"params": [core.basis], "lr": 2e-5, "weight_decay": 0.01},
        {"params": vocab_fly_params, "lr": 1.5e-4, "weight_decay": 0.01},
    ]
    trainable = [
        *ffn_trainable,
        core.codebook.weight,
        core.basis,
        *vocab_fly_params,
    ]

    opt = base.base.make_optimizer(groups, device)
    scaler = base.base.make_scaler(device, dtype)
    initial_lrs = [float(g["lr"]) for g in groups]
    warmup = max(20, train_updates // 20)
    history = []
    micro = 0
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    student.train()

    for update in range(1, train_updates + 1):
        ce_a = kl_a = hid_a = emb_a = reg_a = loss_a = 0.0

        for _ in range(grad_accum):
            ids = train_batches[micro % len(train_batches)].to(device)
            micro += 1
            x = ids[:, :-1]

            with torch.no_grad(), base.base.amp_ctx(device, dtype):
                t = teacher(
                    input_ids=x,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                teacher_emb = teacher.model.embed_tokens(x)

            with base.base.amp_ctx(device, dtype):
                s = student(
                    input_ids=x,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                student_emb = student.model.embed_tokens(x)
                ce = base.base.causal_ce(s.logits, ids)
                kl = base.base.distill_kl(s.logits, t.logits)
                hid = base.base.hidden_alignment(s.hidden_states, t.hidden_states)
                denom = teacher_emb.float().square().mean().clamp_min(1e-8)
                emb = (student_emb.float() - teacher_emb.float()).square().mean() / denom
                reg = base.flyffn_v2_router_regularizer(student, layers)
                raw = 0.30 * ce + 0.40 * kl + 0.15 * hid + 0.15 * emb + 0.005 * reg
                loss = raw / grad_accum

            scaler.scale(loss).backward()
            ce_a += float(ce.detach()) / grad_accum
            kl_a += float(kl.detach()) / grad_accum
            hid_a += float(hid.detach()) / grad_accum
            emb_a += float(emb.detach()) / grad_accum
            reg_a += float(reg.detach()) / grad_accum
            loss_a += float(raw.detach()) / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)

        if update <= warmup:
            mult = max(update / warmup, 1e-3)
        else:
            p = (update - warmup) / max(train_updates - warmup, 1)
            mult = 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g, lr0 in zip(opt.param_groups, initial_lrs):
            g["lr"] = lr0 * mult

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        history.append({
            "update": update,
            "loss": loss_a,
            "ce": ce_a,
            "kl": kl_a,
            "hidden": hid_a,
            "embedding_relative_mse": emb_a,
            "router_reg": reg_a,
        })

        if update == 1 or update % 25 == 0 or update == train_updates:
            elapsed = time.perf_counter() - t0
            ups = update / max(elapsed, 1e-9)
            print(
                f"GLOBAL {name} {update}/{train_updates} loss={loss_a:.4f} "
                f"ce={ce_a:.4f} kl={kl_a:.4f} hid={hid_a:.4f} emb={emb_a:.4f} "
                f"reg={reg_a:.4f} ups={ups:.3f}",
                flush=True,
            )

    metrics = base.evaluate(
        student,
        teacher,
        eval_batches,
        device,
        dtype,
        args.seq_len,
        args.batch_size,
    )
    metrics.update(student.fly_vocab_core.parameter_stats())
    metrics["train_tokens_s"] = (
        train_updates
        * grad_accum
        * args.batch_size
        * args.seq_len
        / max(time.perf_counter() - t0, 1e-9)
    )
    return metrics, history


def state_for_flycore(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if (
            ".mlp." in k
            or k.startswith("flyffn_shared_graph.")
            or k.startswith("fly_vocab_core.")
        )
    }


def load_flycore_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    important = [
        k for k in incompatible.missing_keys
        if (
            ".mlp." in k
            or k.startswith("flyffn_shared_graph.")
            or k.startswith("fly_vocab_core.")
        )
    ]
    if important:
        raise RuntimeError(f"missing FlyCore keys: {important[:20]}")


def train_variant(
    name,
    adjacency,
    teacher,
    tokenizer,
    ffn_cfg,
    vocab_cfg,
    factorization,
    calib_batches,
    probe_batches,
    train_batches,
    eval_batches,
    args,
    device,
    dtype,
    vocab_updates,
    steps_per_stage,
    group_size,
    max_ce_gap,
    train_updates,
    grad_accum,
):
    print(f"STAGE building {name} Qwen3.5 FlyCore student", flush=True)
    student = build_student(
        dtype,
        device,
        ffn_cfg,
        vocab_cfg,
        adjacency,
        factorization,
        args.seed,
    )

    initial = probe_model(student, teacher, probe_batches, device, dtype)
    print("INITIAL FLY VOCAB PROBE", json.dumps(initial, indent=2), flush=True)

    print(f"STAGE FlyEmbedding + FlyLMHead calibration: {name}", flush=True)
    vocab_metrics, vocab_hist = calibrate_vocab(
        student,
        teacher,
        train_batches,
        probe_batches,
        vocab_updates,
        device,
        dtype,
    )
    print("VOCAB CALIBRATION RESULT", json.dumps(vocab_metrics, indent=2), flush=True)

    if vocab_metrics["ce_gap"] > max_ce_gap:
        print(
            f"WARNING vocab CE gap={vocab_metrics['ce_gap']:.4f} exceeds "
            f"quality gate={max_ce_gap:.4f}; FFN stages will remain conservative",
            flush=True,
        )

    hist = base.calibrate_progressively(
        name,
        student,
        teacher,
        calib_batches,
        probe_batches,
        device,
        dtype,
        steps_per_stage,
        group_size,
        max_ce_gap,
        args.num_shards,
    )

    print(f"STAGE global FlyCore distillation: {name}", flush=True)
    metrics, train_hist = global_train_flycore(
        name,
        student,
        teacher,
        train_batches,
        eval_batches,
        args,
        device,
        dtype,
        train_updates,
        grad_accum,
    )

    return (
        student,
        metrics,
        hist,
        train_hist,
        vocab_metrics,
        vocab_hist,
        state_for_flycore(student),
        initial,
    )


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
        text_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(text_prompt, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(
            out[0, enc.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        rows.append({"prompt": prompt, "reply": reply})
    return rows


def main():
    args = parse_args()
    base.base.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.base.choose_dtype(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"DEVICE {device} | dtype={dtype} | "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )

    if args.run_mode == "quick":
        vocab_updates = 60
        steps_per_stage, group_size = 3, 3
        train_updates, grad_accum, eval_count, probe_count = 400, 1, 10, 3
        max_ce_gap = 0.30 if args.max_ce_gap is None else args.max_ce_gap
    else:
        vocab_updates = 300
        steps_per_stage, group_size = 12, 3
        train_updates, grad_accum, eval_count, probe_count = 3000, 2, 20, 6
        max_ce_gap = 0.22 if args.max_ce_gap is None else args.max_ce_gap

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bio_adj, rew_adj, edge_count = base.base.extract_graph(
        args.fly_nodes,
        args.max_edges,
        args.seed,
        Path("/content/qwen35_flycore_cache"),
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

    ffn_cfg = FlyFFNV3Config(
        fly_nodes=args.fly_nodes,
        router_rank=args.router_rank,
        num_shards=args.num_shards,
        graph_steps=args.graph_steps,
        graph_mix_init=args.graph_mix_init,
    )
    vocab_cfg = FlyVocabConfig(
        latent_dim=args.vocab_latent_dim,
        fly_nodes=args.fly_nodes,
        graph_steps=args.graph_steps,
        graph_mix_init=args.vocab_graph_mix_init,
    )

    print(
        f"STAGE factorizing tied vocabulary matrix: vocab={teacher.config.vocab_size} "
        f"hidden={teacher.config.hidden_size} rank={args.vocab_latent_dim}",
        flush=True,
    )
    factorization = factorize_embedding_weight(
        teacher.model.embed_tokens.weight,
        args.vocab_latent_dim,
        args.factor_chunk_rows,
    )
    factor_stats = {
        k: v for k, v in factorization.items()
        if not isinstance(v, torch.Tensor)
    }
    print("VOCAB FACTORIZATION", json.dumps(factor_stats, indent=2), flush=True)

    fly_layers = list(range(len(teacher.model.layers)))
    n_groups = math.ceil(len(fly_layers) / group_size)
    layer_types = list(getattr(teacher.config, "layer_types", []))

    print(
        f"Architecture: FlyEmbedding + FlyLMHead + ALL {len(fly_layers)} FlyFFN-v3 layers; "
        f"dense anchors=0; token mixers unchanged",
        flush=True,
    )
    print(
        f"Progressive FFN schedule: {base.stage_schedule(args.num_shards)} | "
        f"CE gate={max_ce_gap:.3f}",
        flush=True,
    )

    print("STAGE preparing FineWeb-Edu calibration/probe/train/eval blocks", flush=True)
    calib_needed = (
        n_groups
        * len(base.stage_schedule(args.num_shards))
        * steps_per_stage
        * args.batch_size
    )
    train_needed = max(
        train_updates * grad_accum * args.batch_size,
        vocab_updates * args.batch_size,
    )
    calib_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 5,
                calib_needed,
                args.seq_len,
            )
        ),
        args.batch_size,
    )
    probe_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 777,
                probe_count * args.batch_size,
                args.seq_len,
            )
        ),
        args.batch_size,
    )
    train_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 10,
                train_needed,
                args.seq_len,
            )
        ),
        args.batch_size,
    )
    eval_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 999,
                eval_count * args.batch_size,
                args.seq_len,
            )
        ),
        args.batch_size,
    )

    (
        bio,
        bio_metrics,
        bio_calib,
        bio_hist,
        bio_vocab_metrics,
        bio_vocab_hist,
        bio_state,
        bio_initial,
    ) = train_variant(
        "biological",
        bio_adj,
        teacher,
        tokenizer,
        ffn_cfg,
        vocab_cfg,
        factorization,
        calib_batches,
        probe_batches,
        train_batches,
        eval_batches,
        args,
        device,
        dtype,
        vocab_updates,
        steps_per_stage,
        group_size,
        max_ce_gap,
        train_updates,
        grad_accum,
    )

    pd.DataFrame(bio_calib).to_csv(
        out_dir / "bio_progressive_calibration.csv",
        index=False,
    )
    pd.DataFrame(bio_hist).to_csv(
        out_dir / "bio_training_history.csv",
        index=False,
    )
    pd.DataFrame(bio_vocab_hist).to_csv(
        out_dir / "bio_vocab_training_history.csv",
        index=False,
    )
    bio_schedule = routing_schedule(bio)

    rewired_metrics = rew_schedule = rew_vocab_metrics = None
    if args.rewired:
        del bio
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        (
            rew,
            rewired_metrics,
            rew_calib,
            rew_hist,
            rew_vocab_metrics,
            rew_vocab_hist,
            _,
            _,
        ) = train_variant(
            "rewired",
            rew_adj,
            teacher,
            tokenizer,
            ffn_cfg,
            vocab_cfg,
            factorization,
            calib_batches,
            probe_batches,
            train_batches,
            eval_batches,
            args,
            device,
            dtype,
            vocab_updates,
            steps_per_stage,
            group_size,
            max_ce_gap,
            train_updates,
            grad_accum,
        )
        pd.DataFrame(rew_calib).to_csv(
            out_dir / "rewired_progressive_calibration.csv",
            index=False,
        )
        pd.DataFrame(rew_hist).to_csv(
            out_dir / "rewired_training_history.csv",
            index=False,
        )
        pd.DataFrame(rew_vocab_hist).to_csv(
            out_dir / "rewired_vocab_training_history.csv",
            index=False,
        )
        rew_schedule = routing_schedule(rew)
        del rew
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        bio = build_student(
            dtype,
            device,
            ffn_cfg,
            vocab_cfg,
            bio_adj,
            factorization,
            args.seed,
        )
        load_flycore_state(bio, bio_state)

    assert_qwen35_flycore(bio)

    test_ids = eval_batches[0][:, :-1].to(device)
    print("STAGE benchmarking Qwen3.5-0.8B vs FlyCore", flush=True)
    tbench = base.base.benchmark(teacher, test_ids, device)
    fbench = base.base.benchmark(bio, test_ids, device)

    print("STAGE generating chat samples", flush=True)
    chat_samples = generate_chat_samples(bio, tokenizer, device)
    (out_dir / "chat_samples.json").write_text(
        json.dumps(chat_samples, indent=2),
        encoding="utf-8",
    )

    vocab_stats = bio.fly_vocab_core.parameter_stats()
    report = {
        "architecture": (
            "Qwen3.5-0.8B FlyCore-v1: FlyEmbedding + FlyLMHead + "
            "FlyFFN-v3 on all 24 FFNs"
        ),
        "base_model": BASE_MODEL,
        "token_mixers_unchanged": True,
        "flyffn_layers": len(fly_layers),
        "dense_anchor_layers": [],
        "fly_embedding": True,
        "fly_lm_head": True,
        "shared_fly_vocab_core": True,
        "qwen_layer_types": layer_types,
        "factorization": factor_stats,
        "initial_fly_vocab_probe": bio_initial,
        "vocab_calibration_biological": bio_vocab_metrics,
        "vocab_calibration_rewired": rew_vocab_metrics,
        "vocab_parameter_stats": vocab_stats,
        "implementation_note": (
            "FlyEmbedding and FlyLMHead share one rank-reduced vocabulary core. "
            "FlyFFN remains a quality prototype that computes all shards during blending."
        ),
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
            "vocab_latent_dim": args.vocab_latent_dim,
            "vocab_graph_mix_init": args.vocab_graph_mix_init,
            "replace_all_ffns": True,
            "dense_anchors": 0,
            "quality_gate_max_ce_gap": max_ce_gap,
            "stage_schedule": base.stage_schedule(args.num_shards),
            "vocab_calibration_updates": vocab_updates,
            "steps_per_stage": steps_per_stage,
            "global_train_updates": train_updates,
            "grad_accum": grad_accum,
        },
        "biological": bio_metrics,
        "rewired": rewired_metrics,
        "biological_routing_schedule": bio_schedule,
        "rewired_routing_schedule": rew_schedule,
        "benchmark_teacher": tbench,
        "benchmark_biological": fbench,
        "fly_ce_gap_vs_qwen": bio_metrics["ce"] - bio_metrics["teacher_ce"],
        "fly_ppl_ratio_vs_qwen": (
            bio_metrics["perplexity"] / bio_metrics["teacher_perplexity"]
        ),
        "decode_speed_ratio_fly_over_qwen": (
            fbench["decode_tokens_s"] / tbench["decode_tokens_s"]
        ),
        "parameter_ratio_fly_over_qwen": (
            sum(p.numel() for p in bio.parameters())
            / sum(p.numel() for p in teacher.parameters())
        ),
    }

    if rewired_metrics:
        report["biological_topology_ce_gain"] = (
            rewired_metrics["ce"] - bio_metrics["ce"]
        )

    rows = {
        "Qwen3.5-0.8B": {
            "ce": bio_metrics["teacher_ce"],
            "perplexity": bio_metrics["teacher_perplexity"],
            **tbench,
        },
        "FlyCore-v1 biological": {
            "ce": bio_metrics["ce"],
            "perplexity": bio_metrics["perplexity"],
            "teacher_kl": bio_metrics["teacher_kl"],
            **fbench,
        },
    }
    if rewired_metrics:
        rows["FlyCore-v1 rewired"] = {
            "ce": rewired_metrics["ce"],
            "perplexity": rewired_metrics["perplexity"],
            "teacher_kl": rewired_metrics["teacher_kl"],
        }

    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    torch.save(
        bio_state,
        out_dir / "biological_qwen35_flycore_v1.pt",
    )

    standalone_dir = (
        Path(args.standalone_dir)
        if args.standalone_dir
        else out_dir / "standalone"
    )
    print("STAGE exporting verified standalone FlyCore model", flush=True)
    manifest = export_flycore_standalone(
        bio,
        tokenizer,
        ffn_cfg,
        vocab_cfg,
        standalone_dir,
        metadata=report,
        artifacts_dir=out_dir,
    )
    print(
        f"STANDALONE verified=True state_keys={manifest['state_keys']} "
        f"flycore_keys={manifest['flycore_state_keys']} "
        f"size_gb={manifest['state_bytes']/1024**3:.3f}",
        flush=True,
    )

    hf_url = None
    if args.upload_hf:
        if not args.hf_repo_id:
            raise RuntimeError("--upload-hf requires --hf-repo-id")
        print(f"STAGE uploading FlyCore to {args.hf_repo_id}", flush=True)
        hf_url = upload_flycore_standalone(
            standalone_dir,
            args.hf_repo_id,
            private=args.hf_private,
        )
        print(f"HUGGINGFACE {hf_url}", flush=True)

    print("\nQWEN3.5 FLYCORE-V1 CHECK", flush=True)
    print("Token mixers unchanged: True", flush=True)
    print(f"FlyEmbedding: True | FlyLMHead: True | shared core: True", flush=True)
    print(f"FlyFFN-v3 layers: {len(fly_layers)}/{len(fly_layers)} | dense anchors: 0", flush=True)
    print("VOCAB PARAMETER STATS", json.dumps(vocab_stats, indent=2), flush=True)
    print("\nSUMMARY", flush=True)
    print(pd.DataFrame(rows).T, flush=True)
    print("\nKEY REPORT", flush=True)
    keys = (
        "fly_ce_gap_vs_qwen",
        "fly_ppl_ratio_vs_qwen",
        "parameter_ratio_fly_over_qwen",
        "decode_speed_ratio_fly_over_qwen",
        "biological_topology_ce_gain",
    )
    print(
        json.dumps({k: report[k] for k in keys if k in report}, indent=2),
        flush=True,
    )
    print("\nFINAL BIOLOGICAL ROUTING SCHEDULE", flush=True)
    print(json.dumps(bio_schedule, indent=2), flush=True)
    print("\nCHAT SAMPLES", flush=True)
    for item in chat_samples:
        print("=" * 88, flush=True)
        print("USER:", item["prompt"], flush=True)
        print("FLYCORE:", item["reply"], flush=True)

    print("\nSaved:", out_dir, flush=True)
    print("Standalone:", standalone_dir, flush=True)
    if hf_url:
        print("Hugging Face:", hf_url, flush=True)


if __name__ == "__main__":
    main()
