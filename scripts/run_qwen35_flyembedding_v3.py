#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from tinycenn_lm.qwen35_flyembedding_v3 import (
    FlyEmbeddingV3Config,
    assert_fly_embedding_v3_identity,
    assert_qwen35_fly_embedding_v3,
    freeze_qwen_train_fly_v3,
    install_fly_embedding_v3,
)

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5-0.8B + identity-preserving FlyEmbedding-v3")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix-init", type=float, default=0.05)
    p.add_argument("--max-residual-scale", type=float, default=0.05)
    p.add_argument("--lr-core", type=float, default=2e-4)
    p.add_argument("--lr-gate", type=float, default=5e-4)
    p.add_argument("--output-dir", default="results/flyembedding_v3_qwen35_08b")
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
    a = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        a[i, i] = 1.0
        for s in (1, 3, 7, 17):
            a[i, (i + s) % n] = 1.0
            a[i, (i - s) % n] = 1.0
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1.0)


def token_blocks(tokenizer, seed: int, count: int, seq_len: int):
    texts = [
        "Artificial intelligence systems learn patterns from data and use those patterns to make predictions.",
        "Vienna is the capital of Austria and is known for music, architecture, science, and public transport.",
        "A neural network transforms an input through a sequence of learned linear and nonlinear operations.",
        "Efficient language models try to reduce computation while preserving useful knowledge and stable generation.",
        "The sky appears blue because shorter wavelengths of sunlight are scattered more strongly by the atmosphere.",
        "Seventeen plus twenty five equals forty two.",
        "Software architecture describes components, interfaces, constraints, data flows, and operational qualities.",
        "Machine learning evaluation should separate training data from held out validation and test data.",
        "A sparse model activates only a subset of parameters for each token, which can reduce computation.",
        "Scientific experiments should report methods, baselines, uncertainty, and reproducible measurements.",
        "Graph neural computation propagates local state through sparse connections and nonlinear transformations.",
        "Residual adapters preserve a pretrained model while learning a small task-specific correction.",
    ]
    rng = random.Random(seed)
    eos = tokenizer.eos_token or ""
    for _ in range(count):
        s = " ".join(rng.choice(texts) for _ in range(14)) + eos
        ids = tokenizer(
            s, return_tensors="pt", truncation=True, max_length=seq_len + 1
        ).input_ids[0]
        if ids.numel() < seq_len + 1:
            reps = math.ceil((seq_len + 1) / max(ids.numel(), 1))
            ids = ids.repeat(reps)
        yield ids[: seq_len + 1].unsqueeze(0)


def causal_ce(logits, targets):
    if logits.shape[1] != targets.shape[1]:
        raise ValueError(f"logit/target length mismatch {logits.shape[1]} vs {targets.shape[1]}")
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
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
    max_logit_abs = 0.0
    for cpu_ids in batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        target = ids[:, 1:]
        t = teacher(input_ids=x, use_cache=False, return_dict=True)
        s = student(input_ids=x, use_cache=False, return_dict=True)
        te = teacher.model.embed_tokens(x)
        se = student.model.embed_tokens(x)
        tce += float(causal_ce(t.logits, target))
        sce += float(causal_ce(s.logits, target))
        kl += float(distill_kl(s.logits, t.logits))
        den = te.float().square().mean().clamp_min(1e-12)
        emb += float(((se.float() - te.float()).square().mean() / den).item())
        top1 += int((t.logits.argmax(-1) == s.logits.argmax(-1)).sum().item())
        total += int(t.logits.shape[0] * t.logits.shape[1])
        max_logit_abs = max(max_logit_abs, float((s.logits.float() - t.logits.float()).abs().max().item()))
    n = max(len(batches), 1)
    return {
        "teacher_ce": tce / n,
        "student_ce": sce / n,
        "ce_gap": (sce - tce) / n,
        "teacher_kl": kl / n,
        "embedding_relative_mse": emb / n,
        "top1_logit_agreement": top1 / max(total, 1),
        "max_abs_logit_error": max_logit_abs,
    }


@torch.no_grad()
def generation_compare(student, teacher, tokenizer, device):
    prompts = [
        "Explain in two sentences why the sky is blue.",
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "In one sentence, explain what a residual neural connection does.",
    ]
    rows = []
    all_good = True
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(text, return_tensors="pt").to(device)

        def gen(m):
            out = m.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            ids = out[0, enc.input_ids.shape[1]:]
            return ids.tolist(), tokenizer.decode(ids, skip_special_tokens=True).strip()

        tids, ttxt = gen(teacher)
        sids, stxt = gen(student)
        prefix = 0
        for a, b in zip(tids, sids):
            if a != b:
                break
            prefix += 1
        exact = tids == sids
        a, b = set(tids), set(sids)
        jaccard = len(a & b) / max(len(a | b), 1)
        valid = bool(stxt) and "�" not in stxt and jaccard >= 0.50
        all_good = all_good and valid
        rows.append({
            "prompt": prompt,
            "qwen_reply": ttxt,
            "fly_reply": stxt,
            "exact_token_match": exact,
            "matching_prefix_tokens": prefix,
            "token_jaccard": jaccard,
            "passed": valid,
        })
    return all_good, rows


def quality_score(m):
    # Lower is better. Improvements in CE are allowed, but drift from Qwen is
    # explicitly expensive because this experiment is preservation-first.
    return (
        max(float(m["ce_gap"]), 0.0)
        + 0.60 * float(m["teacher_kl"])
        + 0.50 * float(m["embedding_relative_mse"])
        + 0.50 * (1.0 - float(m["top1_logit_agreement"]))
    )


def train_adapter(student, teacher, train_batches, probe_batches, updates, device, args):
    trainable = freeze_qwen_train_fly_v3(student)
    core = student.fly_embedding_v3_core
    opt = torch.optim.AdamW([
        {"params": [core.down.weight, core.up.weight, core.feature_gain_raw], "lr": args.lr_core},
        {"params": [core.residual_scale_raw, core.graph_mix_logit], "lr": args.lr_gate},
    ], weight_decay=0.0)

    start = probe(student, teacher, probe_batches, device)
    best = dict(start)
    best_score = quality_score(start)
    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    hist = []
    teacher.eval()
    t0 = time.perf_counter()

    for step in range(1, updates + 1):
        student.train()
        ids = train_batches[(step - 1) % len(train_batches)].to(device)
        x, target = ids[:, :-1], ids[:, 1:]
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            te = teacher.model.embed_tokens(x)

        s = student(input_ids=x, use_cache=False, return_dict=True)
        se = student.model.embed_tokens(x)
        ce = causal_ce(s.logits, target)
        kl = distill_kl(s.logits, t.logits)
        den = te.float().square().mean().clamp_min(1e-12)
        emb = (se.float() - te.float()).square().mean() / den

        # CE supplies an adaptation signal; KL + embedding anchoring keep the
        # residual from destroying pretrained behavior.
        loss = 0.20 * ce + 0.60 * kl + 0.20 * emb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()

        row = {
            "update": step,
            "loss": float(loss.detach()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "embedding_relative_mse": float(emb.detach()),
            "residual_scale": float(core.residual_scale.detach()),
            "graph_mix": float(core.graph_mix.detach()),
        }
        hist.append(row)

        if step == 1 or step % 25 == 0 or step == updates:
            print(
                f"V3 {step}/{updates} loss={row['loss']:.5f} ce={row['ce']:.5f} "
                f"kl={row['kl']:.6f} emb={row['embedding_relative_mse']:.8f} "
                f"scale={row['residual_scale']:.6f} ups={step/max(time.perf_counter()-t0,1e-9):.2f}",
                flush=True,
            )

        if step % 50 == 0 or step == updates:
            m = probe(student, teacher, probe_batches, device)
            score = quality_score(m)
            print("PROBE", step, json.dumps(m), "score=", score, flush=True)
            if score < best_score:
                best_score = score
                best = dict(m)
                best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}

    core.load_state_dict(best_state, strict=True)
    student.eval()
    return probe(student, teacher, probe_batches, device), hist, start, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print("DEVICE", device, "| dtype", dtype, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print("Loading Qwen + FlyEmbedding-v3 student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    original_embedding = student.model.embed_tokens
    original_lm_head = student.lm_head
    cfg = FlyEmbeddingV3Config(
        fly_nodes=args.fly_nodes,
        graph_steps=args.graph_steps,
        graph_mix_init=args.graph_mix_init,
        max_residual_scale=args.max_residual_scale,
    )
    adjacency = make_adjacency(args.fly_nodes).to(device)
    install_fly_embedding_v3(student, cfg, adjacency)
    assert_qwen35_fly_embedding_v3(student)

    if student.lm_head is not original_lm_head:
        raise RuntimeError("FlyEmbedding-v3 changed Qwen lm_head")
    if student.model.embed_tokens.weight.data_ptr() != original_embedding.weight.data_ptr():
        raise RuntimeError("FlyEmbedding-v3 did not preserve original Qwen embedding weight")

    test_ids = torch.tensor([[1, 2, 3, 100, 1000]], device=device)
    identity = assert_fly_embedding_v3_identity(student, original_embedding, test_ids)
    print("IDENTITY EMBEDDING CHECK", json.dumps(identity), flush=True)

    updates = 300 if args.run_mode == "quick" else 1200
    train_batches = list(token_blocks(tokenizer, args.seed + 11, updates, args.seq_len))
    probe_batches = list(token_blocks(tokenizer, args.seed + 777, 8 if args.run_mode == "quick" else 16, args.seq_len))

    initial_probe = probe(student, teacher, probe_batches, device)
    print("INITIAL PROBE", json.dumps(initial_probe, indent=2), flush=True)
    if initial_probe["top1_logit_agreement"] != 1.0 or initial_probe["max_abs_logit_error"] != 0.0:
        raise RuntimeError("Identity contract failed at model-logit level")

    initial_generation_ok, initial_generation = generation_compare(
        student, teacher, tokenizer, device
    )
    print("INITIAL GENERATION", json.dumps(initial_generation, indent=2), flush=True)
    if not all(row["exact_token_match"] for row in initial_generation):
        raise RuntimeError("Identity contract failed during deterministic generation")

    final_probe, hist, train_start, best_probe = train_adapter(
        student, teacher, train_batches, probe_batches, updates, device, args
    )
    generation_ok, generation_rows = generation_compare(
        student, teacher, tokenizer, device
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False)
    (out_dir / "generation_samples.json").write_text(
        json.dumps(generation_rows, indent=2), encoding="utf-8"
    )
    torch.save({
        "fly_embedding_v3_core": {
            k: v.detach().cpu() for k, v in student.fly_embedding_v3_core.state_dict().items()
        },
        "config": cfg.to_dict(),
        "base_model": args.base_model,
    }, out_dir / "fly_embedding_v3_adapter.pt")

    stats = student.fly_embedding_v3_core.parameter_stats(original_embedding.weight.numel())
    report = {
        "architecture": "Qwen3.5-0.8B + identity-preserving FlyEmbedding-v3 residual adapter",
        "base_model": args.base_model,
        "identity_embedding_check": identity,
        "initial_probe": initial_probe,
        "best_probe": best_probe,
        "final_probe": final_probe,
        "initial_generation_exact": bool(all(r["exact_token_match"] for r in initial_generation)),
        "generation_sanity_passed": generation_ok,
        "generation_samples": generation_rows,
        "qwen_embedding_preserved": True,
        "qwen_lm_head_preserved": True,
        "parameter_stats": stats,
        "quality_gate_passed": bool(
            generation_ok
            and final_probe["top1_logit_agreement"] >= 0.98
            and final_probe["teacher_kl"] <= 0.02
            and final_probe["ce_gap"] <= 0.05
        ),
        "config": vars(args),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(report, indent=2), flush=True)
    print("Saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
