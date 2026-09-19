#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_flycore_v2 import (
    FlyVocabV2Config,
    assert_qwen35_fly_embedding_v2,
    choose_hot_tokens,
    factorize_embedding_weight_v2,
    freeze_source_ffn_train_vocab,
    install_fly_embedding_v2,
)

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5-0.8B + input-only FlyEmbedding-v2")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--vocab-latent-dim", type=int, default=768)
    p.add_argument("--hot-token-count", type=int, default=8192)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--vocab-graph-mix-init", type=float, default=0.05)
    p.add_argument("--max-fly-scale", type=float, default=0.05)
    p.add_argument("--factor-chunk-rows", type=int, default=4096)
    p.add_argument("--max-final-ce-gap", type=float, default=0.08)
    p.add_argument("--output-dir", default="results/flycore_v2_from_qwen35_08b")
    p.add_argument("--seed", type=int, default=8621)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_adjacency(n: int) -> torch.Tensor:
    # Deterministic sparse small-world graph. Row-normalized.
    a = torch.zeros(n, n, dtype=torch.float32)
    strides = (1, 3, 7, 17)
    for i in range(n):
        a[i, i] = 1.0
        for s in strides:
            a[i, (i + s) % n] = 1.0
            a[i, (i - s) % n] = 1.0
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1.0)


def token_blocks(tokenizer, seed: int, count: int, seq_len: int):
    # Lightweight deterministic text mixture; avoids dependence on the old v3 runner.
    texts = [
        "Artificial intelligence systems learn patterns from data and use those patterns to make predictions.",
        "Vienna is the capital of Austria and is known for music, architecture, science, and public transport.",
        "A neural network transforms an input through a sequence of learned linear and nonlinear operations.",
        "Efficient language models try to reduce memory and computation while preserving useful knowledge.",
        "The sky appears blue because shorter wavelengths of sunlight are scattered more strongly by the atmosphere.",
        "Seventeen plus twenty five equals forty two.",
        "Software architecture describes components, interfaces, constraints, data flows, and operational qualities.",
        "Machine learning evaluation should separate training data from held out validation and test data.",
        "A sparse model activates only a subset of parameters for each token, which can reduce computation.",
        "Scientific experiments should report methods, baselines, uncertainty, and reproducible measurements.",
    ]
    rng = random.Random(seed)
    eos = tokenizer.eos_token or ""
    for _ in range(count):
        s = " ".join(rng.choice(texts) for _ in range(12)) + eos
        ids = tokenizer(s, return_tensors="pt", truncation=True, max_length=seq_len + 1).input_ids[0]
        if ids.numel() < seq_len + 1:
            reps = math.ceil((seq_len + 1) / max(ids.numel(), 1))
            ids = ids.repeat(reps)
        yield ids[: seq_len + 1].unsqueeze(0)


def causal_ce(logits, ids):
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
    )


def distill_kl(student_logits, teacher_logits):
    s = torch.log_softmax(student_logits.float(), dim=-1)
    t = torch.softmax(teacher_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(s, t, reduction="batchmean") / max(student_logits.shape[1], 1)


@torch.no_grad()
def probe(student, teacher, batches, device):
    student.eval()
    teacher.eval()
    tce = sce = kl = emb = 0.0
    top1 = total = 0
    for cpu_ids in batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        t = teacher(input_ids=x, use_cache=False, return_dict=True)
        s = student(input_ids=x, use_cache=False, return_dict=True)
        te = teacher.model.embed_tokens(x)
        se = student.model.embed_tokens(x)
        tce += float(causal_ce(t.logits, ids))
        sce += float(causal_ce(s.logits, ids))
        kl += float(distill_kl(s.logits, t.logits))
        den = te.float().square().mean().clamp_min(1e-8)
        emb += float(((se.float() - te.float()).square().mean() / den).item())
        top1 += int((t.logits.argmax(-1) == s.logits.argmax(-1)).sum().item())
        total += int(t.logits.shape[0] * t.logits.shape[1])
    n = max(len(batches), 1)
    return {
        "teacher_ce": tce / n,
        "student_ce": sce / n,
        "ce_gap": (sce - tce) / n,
        "teacher_kl": kl / n,
        "embedding_relative_mse": emb / n,
        "top1_logit_agreement": top1 / max(total, 1),
    }


def train_embedding(student, teacher, train_batches, probe_batches, updates, device):
    trainable = freeze_source_ffn_train_vocab(student)
    core = student.fly_vocab_core_v2
    # Input-only training: Qwen body + original lm_head stay frozen.
    groups = [
        {"params": [core.codebook.weight], "lr": 1.0e-5},
        {"params": [core.basis], "lr": 3.0e-6},
        {"params": [core.hot_residual], "lr": 2.0e-5},
        {"params": [core.fly_down.weight, core.fly_up.weight], "lr": 2.0e-5},
        {"params": [core.fly_scale_raw, core.graph_mix_logit], "lr": 5.0e-5},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    start = probe(student, teacher, probe_batches, device)
    best = dict(start)
    best_score = max(start["ce_gap"], 0.0) + 0.20 * start["teacher_kl"] + 0.30 * start["embedding_relative_mse"]
    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    hist = []
    teacher.eval()
    t0 = time.perf_counter()

    for step in range(1, updates + 1):
        student.train()
        ids = train_batches[(step - 1) % len(train_batches)].to(device)
        x = ids[:, :-1]
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            te = teacher.model.embed_tokens(x)
        s = student(input_ids=x, use_cache=False, return_dict=True)
        se = student.model.embed_tokens(x)
        ce = causal_ce(s.logits, ids)
        kl = distill_kl(s.logits, t.logits)
        den = te.float().square().mean().clamp_min(1e-8)
        emb = (se.float() - te.float()).square().mean() / den
        loss = 0.35 * ce + 0.30 * kl + 0.35 * emb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()

        row = {
            "update": step,
            "loss": float(loss.detach()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "embedding_relative_mse": float(emb.detach()),
            "fly_scale": float(core.fly_scale.detach()),
        }
        hist.append(row)
        if step == 1 or step % 25 == 0 or step == updates:
            print(
                f"EMBED {step}/{updates} loss={row['loss']:.4f} ce={row['ce']:.4f} "
                f"kl={row['kl']:.4f} emb={row['embedding_relative_mse']:.6f} "
                f"scale={row['fly_scale']:.4f} ups={step/max(time.perf_counter()-t0,1e-9):.2f}",
                flush=True,
            )
        if step % 50 == 0 or step == updates:
            m = probe(student, teacher, probe_batches, device)
            score = max(m["ce_gap"], 0.0) + 0.20 * m["teacher_kl"] + 0.30 * m["embedding_relative_mse"]
            print("PROBE", step, json.dumps(m), "score=", score, flush=True)
            if score < best_score:
                best_score = score
                best = dict(m)
                best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}

    core.load_state_dict(best_state, strict=True)
    student.eval()
    return probe(student, teacher, probe_batches, device), hist, start, best


@torch.no_grad()
def generation_sanity(model, tokenizer, device):
    prompts = [
        "Explain in two sentences why the sky is blue.",
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
    ]
    rows = []
    ok = True
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=80,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        ids = out[0, enc.input_ids.shape[1]:]
        reply = tokenizer.decode(ids, skip_special_tokens=True).strip()
        passed = bool(reply) and len(ids) > 1
        ok = ok and passed
        rows.append({"prompt": prompt, "reply": reply, "tokens": int(len(ids)), "passed": passed})
    return ok, rows


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print("DEVICE", device, "| dtype", dtype, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = dtype if device.type == "cuda" else torch.float32
    print("Loading original Qwen teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=load_dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print("Loading Qwen student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=load_dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    updates = 400 if args.run_mode == "quick" else 1600
    train_batches = list(token_blocks(tokenizer, args.seed + 10, updates * args.batch_size, args.seq_len))
    probe_batches = list(token_blocks(tokenizer, args.seed + 777, 6 if args.run_mode == "quick" else 12, args.seq_len))

    # Preserve every tokenizer-declared special token plus all control tokens
    # emitted by Qwen's chat template. These tokens are disproportionately
    # important: approximating role/end markers can cause immediate EOS.
    special_ids = set(int(x) for x in tokenizer.all_special_ids if x is not None)
    chat_probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello!"}],
        tokenize=True,
        add_generation_prompt=False,
    )
    # Transformers versions differ here: this may be a plain list/tensor of
    # token ids or a BatchEncoding/dict containing "input_ids".
    if hasattr(chat_probe, "input_ids"):
        chat_probe = chat_probe.input_ids
    elif isinstance(chat_probe, dict):
        chat_probe = chat_probe["input_ids"]
    if isinstance(chat_probe, torch.Tensor):
        chat_probe = chat_probe.detach().cpu().reshape(-1).tolist()
    elif chat_probe and isinstance(chat_probe[0], (list, tuple)):
        chat_probe = chat_probe[0]
    chat_probe = [int(x) for x in chat_probe]
    special_ids.update(
        x for x in chat_probe
        if x >= int(teacher.config.vocab_size) - 512
    )
    special_ids = sorted(special_ids)
    print("Protected special/chat-control token ids:", special_ids, flush=True)
    hot_ids = choose_hot_tokens(
        train_batches[: min(256, len(train_batches))] + probe_batches,
        int(teacher.config.vocab_size),
        args.hot_token_count,
        special_ids=special_ids,
    )
    adjacency = make_adjacency(args.fly_nodes).to(device)

    print(f"Factorizing original Qwen embedding at rank {args.vocab_latent_dim}...", flush=True)
    fact = factorize_embedding_weight_v2(
        teacher.model.embed_tokens.weight,
        args.vocab_latent_dim,
        args.factor_chunk_rows,
    )
    fact_stats = {k: v for k, v in fact.items() if not isinstance(v, torch.Tensor)}
    print("FACTORIZATION", json.dumps(fact_stats, indent=2), flush=True)

    cfg = FlyVocabV2Config(
        latent_dim=args.vocab_latent_dim,
        fly_nodes=args.fly_nodes,
        graph_steps=args.graph_steps,
        graph_mix_init=args.vocab_graph_mix_init,
        max_fly_scale=args.max_fly_scale,
        hot_token_count=len(hot_ids),
    )

    original_head = student.lm_head
    install_fly_embedding_v2(
        student,
        cfg,
        adjacency,
        hot_ids,
        factorization=fact,
        teacher_weight=teacher.model.embed_tokens.weight.detach(),
    )
    assert_qwen35_fly_embedding_v2(student)
    if student.lm_head is not original_head:
        raise RuntimeError("Qwen lm_head changed in input-only mode")
    print("✓ Qwen body unchanged; only input embedding replaced; original lm_head preserved", flush=True)

    final_probe, hist, initial_probe, best_probe = train_embedding(
        student, teacher, train_batches, probe_batches, updates, device
    )
    sanity_ok, sanity_rows = generation_sanity(student, tokenizer, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / "embedding_training_history.csv", index=False)
    (out_dir / "chat_samples.json").write_text(json.dumps(sanity_rows, indent=2), encoding="utf-8")
    torch.save(
        {
            "fly_vocab_core_v2": {k: v.detach().cpu() for k, v in student.fly_vocab_core_v2.state_dict().items()},
            "config": cfg.to_dict(),
            "base_model": args.base_model,
        },
        out_dir / "fly_embedding_v2.pt",
    )
    report = {
        "architecture": "Qwen3.5-0.8B + input-only FlyEmbedding-v2",
        "base_model": args.base_model,
        "qwen_body_unchanged": True,
        "qwen_lm_head_unchanged": True,
        "flyffn_v3_used": False,
        "factorization": fact_stats,
        "initial_probe": initial_probe,
        "best_probe": best_probe,
        "final_probe": final_probe,
        "generation_sanity_passed": sanity_ok,
        "generation_sanity": sanity_rows,
        "embedding_parameter_stats": student.fly_vocab_core_v2.parameter_stats(),
        "quality_gate_passed": bool(sanity_ok and final_probe["ce_gap"] <= args.max_final_ce_gap),
        "config": vars(args),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(report, indent=2), flush=True)
    print("Saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
