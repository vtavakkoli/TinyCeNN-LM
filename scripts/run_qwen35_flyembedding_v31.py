#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random, sys, time
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
    p = argparse.ArgumentParser(description="FlyEmbedding-v3.1 preservation-first experiment")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix-init", type=float, default=0.05)
    p.add_argument("--max-residual-scale", type=float, default=0.05)
    p.add_argument("--lr-core", type=float, default=1.5e-4)
    p.add_argument("--lr-gate", type=float, default=3.0e-4)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--min-top1", type=float, default=0.99)
    p.add_argument("--max-kl", type=float, default=0.005)
    p.add_argument("--max-embedding-mse", type=float, default=0.001)
    p.add_argument("--early-stop-top1", type=float, default=0.985)
    p.add_argument("--early-stop-kl", type=float, default=0.0075)
    p.add_argument("--output-dir", default="results/flyembedding_v31_qwen35_08b")
    p.add_argument("--seed", type=int, default=8621)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_adjacency(n):
    a = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        a[i, i] = 1.0
        for s in (1, 3, 7, 17):
            a[i, (i+s) % n] = 1.0
            a[i, (i-s) % n] = 1.0
    return a / a.sum(-1, keepdim=True).clamp_min(1.0)


def synthetic_texts():
    return [
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


def blocks_from_texts(tokenizer, texts, count, seq_len, seed):
    rng = random.Random(seed)
    clean = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) > 30]
    if not clean:
        clean = synthetic_texts()
    eos = tokenizer.eos_token or ""
    out = []
    while len(out) < count:
        s = " ".join(rng.choice(clean) for _ in range(10)) + eos
        ids = tokenizer(s, return_tensors="pt", truncation=False).input_ids[0]
        if ids.numel() < seq_len + 1:
            continue
        max_start = max(0, ids.numel() - (seq_len + 1))
        start = rng.randint(0, max_start) if max_start else 0
        out.append(ids[start:start+seq_len+1].unsqueeze(0))
    return out


def load_real_blocks(tokenizer, run_mode, seq_len, seed):
    train_n = 320 if run_mode == "quick" else 1200
    val_n = 24 if run_mode == "quick" else 64
    source = "synthetic-fallback"
    try:
        from datasets import load_dataset
        ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        train_text = [x["text"] for x in ds_train if x.get("text", "").strip()]
        val_text = [x["text"] for x in ds_val if x.get("text", "").strip()]
        train = blocks_from_texts(tokenizer, train_text, train_n, seq_len, seed + 10)
        val = blocks_from_texts(tokenizer, val_text, val_n, seq_len, seed + 777)
        source = "WikiText-2 raw train/validation"
    except Exception as e:
        print("WARNING: real dataset load failed, using deterministic synthetic fallback:", repr(e), flush=True)
        base = synthetic_texts()
        train = blocks_from_texts(tokenizer, base, train_n, seq_len, seed + 10)
        val = blocks_from_texts(tokenizer, base, val_n, seq_len, seed + 777)
    return train, val, source


def causal_ce(logits, targets):
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
    student.eval(); teacher.eval()
    tce = sce = kl = emb = 0.0
    top1 = total = 0
    max_logit_abs = 0.0
    for cpu_ids in batches:
        ids = cpu_ids.to(device)
        x, target = ids[:, :-1], ids[:, 1:]
        t = teacher(input_ids=x, use_cache=False, return_dict=True)
        s = student(input_ids=x, use_cache=False, return_dict=True)
        te = teacher.model.embed_tokens(x)
        se = student.model.embed_tokens(x)
        tce += float(causal_ce(t.logits, target))
        sce += float(causal_ce(s.logits, target))
        kl += float(distill_kl(s.logits, t.logits))
        den = te.float().square().mean().clamp_min(1e-12)
        emb += float(((se.float()-te.float()).square().mean()/den).item())
        top1 += int((t.logits.argmax(-1) == s.logits.argmax(-1)).sum())
        total += int(t.logits.shape[0] * t.logits.shape[1])
        max_logit_abs = max(max_logit_abs, float((s.logits.float()-t.logits.float()).abs().max()))
    n = max(len(batches), 1)
    return {
        "teacher_ce": tce/n,
        "student_ce": sce/n,
        "ce_gap": (sce-tce)/n,
        "teacher_kl": kl/n,
        "embedding_relative_mse": emb/n,
        "top1_logit_agreement": top1/max(total,1),
        "max_abs_logit_error": max_logit_abs,
    }


def eligible(m, args):
    return (
        m["top1_logit_agreement"] >= args.min_top1
        and m["teacher_kl"] <= args.max_kl
        and m["embedding_relative_mse"] <= args.max_embedding_mse
    )


def must_stop(m, args):
    return (
        m["top1_logit_agreement"] < args.early_stop_top1
        or m["teacher_kl"] > args.early_stop_kl
    )


@torch.no_grad()
def generation_suite(student, teacher, tokenizer, device):
    prompts = [
        "Explain in two sentences why the sky is blue.",
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "In one sentence, explain what a residual neural connection does.",
        "What is the capital of France? Give only the city.",
        "If a train travels 60 km in one hour, how far in 3 hours? Give only the answer.",
        "Explain in one sentence what an API is.",
        "Write a polite one-sentence thank-you message.",
        "Name the largest planet in the Solar System.",
        "Complete the pattern: 2, 4, 8, 16, ?",
        "In one sentence, explain why validation data should be separate from training data.",
        "Translate 'Good morning' into German. Give only the translation.",
    ]
    rows = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt").to(device)
        def gen(m):
            out = m.generate(
                **enc, max_new_tokens=64, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            ids = out[0, enc.input_ids.shape[1]:]
            return ids.tolist(), tokenizer.decode(ids, skip_special_tokens=True).strip()
        tids, ttxt = gen(teacher)
        sids, stxt = gen(student)
        prefix = 0
        for a,b in zip(tids,sids):
            if a != b: break
            prefix += 1
        a,b = set(tids), set(sids)
        jac = len(a & b) / max(len(a | b), 1)
        passed = bool(stxt) and "�" not in stxt and jac >= 0.50
        rows.append({
            "prompt":prompt, "qwen_reply":ttxt, "fly_reply":stxt,
            "exact_token_match":tids==sids,
            "matching_prefix_tokens":prefix,
            "token_jaccard":jac, "passed":passed,
        })
    exact_rate = sum(r["exact_token_match"] for r in rows)/len(rows)
    pass_rate = sum(r["passed"] for r in rows)/len(rows)
    mean_jaccard = sum(r["token_jaccard"] for r in rows)/len(rows)
    return {
        "exact_rate": exact_rate,
        "pass_rate": pass_rate,
        "mean_token_jaccard": mean_jaccard,
        "rows": rows,
    }


def train(student, teacher, train_batches, val_batches, updates, device, args):
    trainable = freeze_qwen_train_fly_v3(student)
    core = student.fly_embedding_v3_core
    opt = torch.optim.AdamW([
        {"params":[core.down.weight, core.up.weight, core.feature_gain_raw], "lr":args.lr_core},
        {"params":[core.residual_scale_raw, core.graph_mix_logit], "lr":args.lr_gate},
    ], weight_decay=0.0)

    initial = probe(student, teacher, val_batches, device)
    best = dict(initial)
    best_step = 0
    best_state = {k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
    best_ce = initial["student_ce"]
    hist, probes = [], [{"step":0, **initial, "eligible":True}]
    stop_reason = "completed"
    teacher.eval()
    t0 = time.perf_counter()

    for step in range(1, updates+1):
        student.train()
        ids = train_batches[(step-1) % len(train_batches)].to(device)
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
        emb = (se.float()-te.float()).square().mean()/den

        loss = 0.20*ce + 0.65*kl + 0.15*emb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()

        row = {
            "update":step, "loss":float(loss.detach()), "ce":float(ce.detach()),
            "kl":float(kl.detach()), "embedding_relative_mse":float(emb.detach()),
            "residual_scale":float(core.residual_scale.detach()),
            "graph_mix":float(core.graph_mix.detach()),
        }
        hist.append(row)

        if step == 1 or step % 25 == 0 or step == updates:
            print(
                f"V3.1 {step}/{updates} loss={row['loss']:.5f} ce={row['ce']:.5f} "
                f"kl={row['kl']:.6f} emb={row['embedding_relative_mse']:.8f} "
                f"scale={row['residual_scale']:.6f} ups={step/max(time.perf_counter()-t0,1e-9):.2f}",
                flush=True
            )

        if step % args.probe_every == 0 or step == updates:
            m = probe(student, teacher, val_batches, device)
            ok = eligible(m, args)
            probes.append({"step":step, **m, "eligible":ok})
            print("VAL", step, json.dumps(m), "eligible=", ok, flush=True)

            if ok and m["student_ce"] < best_ce:
                best_ce = m["student_ce"]
                best = dict(m)
                best_step = step
                best_state = {k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
                print(f"✓ NEW SAFE BEST at step {step}: CE {best_ce:.6f}", flush=True)

            if must_stop(m, args):
                stop_reason = (
                    f"preservation boundary crossed at step {step}: "
                    f"top1={m['top1_logit_agreement']:.6f}, kl={m['teacher_kl']:.6f}"
                )
                print("EARLY STOP:", stop_reason, flush=True)
                break

    core.load_state_dict(best_state, strict=True)
    student.eval()
    final = probe(student, teacher, val_batches, device)
    return final, hist, probes, initial, best, best_step, stop_reason


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print("DEVICE", device, "| dtype", dtype, flush=True)
    if device.type != "cuda":
        print("WARNING: GPU runtime strongly recommended.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print("Loading Qwen + FlyEmbedding-v3.1 student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    original_embedding = student.model.embed_tokens
    original_lm_head = student.lm_head
    cfg = FlyEmbeddingV3Config(
        fly_nodes=args.fly_nodes, graph_steps=args.graph_steps,
        graph_mix_init=args.graph_mix_init,
        max_residual_scale=args.max_residual_scale,
    )
    install_fly_embedding_v3(student, cfg, make_adjacency(args.fly_nodes).to(device))
    assert_qwen35_fly_embedding_v3(student)
    if student.lm_head is not original_lm_head:
        raise RuntimeError("Qwen lm_head changed")
    if student.model.embed_tokens.weight.data_ptr() != original_embedding.weight.data_ptr():
        raise RuntimeError("Original Qwen embedding weight not preserved")

    identity = assert_fly_embedding_v3_identity(
        student, original_embedding,
        torch.tensor([[1,2,3,100,1000]], device=device)
    )
    print("IDENTITY CHECK", json.dumps(identity), flush=True)

    train_batches, val_batches, data_source = load_real_blocks(
        tokenizer, args.run_mode, args.seq_len, args.seed
    )
    print("DATA SOURCE:", data_source, "| train blocks", len(train_batches), "| val blocks", len(val_batches), flush=True)

    initial_probe = probe(student, teacher, val_batches, device)
    print("INITIAL VALIDATION", json.dumps(initial_probe, indent=2), flush=True)
    if initial_probe["top1_logit_agreement"] != 1.0 or initial_probe["max_abs_logit_error"] > 1e-5:
        raise RuntimeError("Identity contract failed at logits")

    initial_generation = generation_suite(student, teacher, tokenizer, device)
    if initial_generation["exact_rate"] != 1.0:
        raise RuntimeError("Identity contract failed in deterministic generation")
    print("INITIAL GENERATION exact_rate=1.0", flush=True)

    updates = 250 if args.run_mode == "quick" else 800
    final_probe, hist, probes, train_initial, best_probe, best_step, stop_reason = train(
        student, teacher, train_batches, val_batches, updates, device, args
    )
    gen = generation_suite(student, teacher, tokenizer, device)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir/"training_history.csv", index=False)
    pd.DataFrame(probes).to_csv(out_dir/"validation_probes.csv", index=False)
    (out_dir/"generation_samples.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")
    torch.save({
        "fly_embedding_v3_core":{k:v.detach().cpu() for k,v in student.fly_embedding_v3_core.state_dict().items()},
        "config":cfg.to_dict(), "base_model":args.base_model,
        "selected_step":best_step,
    }, out_dir/"fly_embedding_v31_adapter.pt")

    stats = student.fly_embedding_v3_core.parameter_stats(original_embedding.weight.numel())
    quality = bool(
        eligible(final_probe, args)
        and gen["pass_rate"] >= 0.90
        and gen["mean_token_jaccard"] >= 0.70
        and final_probe["student_ce"] <= initial_probe["student_ce"]
    )
    report = {
        "architecture":"Qwen3.5-0.8B + FlyEmbedding-v3.1 preservation-first residual adapter",
        "base_model":args.base_model,
        "data_source":data_source,
        "identity_embedding_check":identity,
        "initial_probe":initial_probe,
        "selected_step":best_step,
        "selected_probe":best_probe,
        "final_probe":final_probe,
        "stop_reason":stop_reason,
        "generation_summary":{
            "exact_rate":gen["exact_rate"],
            "pass_rate":gen["pass_rate"],
            "mean_token_jaccard":gen["mean_token_jaccard"],
        },
        "generation_samples":gen["rows"],
        "constraints":{
            "min_top1":args.min_top1,
            "max_kl":args.max_kl,
            "max_embedding_mse":args.max_embedding_mse,
            "early_stop_top1":args.early_stop_top1,
            "early_stop_kl":args.early_stop_kl,
        },
        "parameter_stats":stats,
        "quality_gate_passed":quality,
        "config":vars(args),
    }
    (out_dir/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(report, indent=2), flush=True)
    print("Saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
