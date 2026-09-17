#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.smollm2_fly_graph_attention import (
    FlyGraphAttentionConfig,
    assert_pure_fly_attention,
    fly_attention_modules,
    fly_attention_stats,
    fly_parameter_groups,
    replace_all_attention_with_fly,
    set_fly_streaming,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
GRAPH_URL = "https://zenodo.org/records/21549559/files/connections_biological.csv.gz?download=1"


def parse_args():
    p = argparse.ArgumentParser(description="FlyWire graph linear-attention replacement experiment")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--max-edges", type=int, default=2048)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--rewired", action="store_true")
    p.add_argument("--output-dir", default="results/flygraph_attention_smollm2_135m")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_ctx(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()


def extract_graph(feature_dim, max_edges, seed, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_file = cache_dir / "connections_biological.csv.gz"
    if not graph_file.exists():
        with requests.get(GRAPH_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(graph_file, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="FlyWire graph"
            ) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    cols = ["pre_root_id", "post_root_id", "syn_count"]
    degree = pd.Series(dtype=np.float64)
    for ch in tqdm(
        pd.read_csv(graph_file, usecols=cols, compression="gzip", chunksize=1_000_000),
        desc="degree pass",
    ):
        w = np.log1p(ch.syn_count.to_numpy(np.float64))
        a = pd.Series(w, index=ch.pre_root_id.to_numpy()).groupby(level=0).sum()
        b = pd.Series(w, index=ch.post_root_id.to_numpy()).groupby(level=0).sum()
        degree = degree.add(a, fill_value=0).add(b, fill_value=0)

    candidates = set(int(x) for x in degree.nlargest(max(feature_dim * 6, 2048)).index)
    parts = []
    for ch in tqdm(
        pd.read_csv(graph_file, usecols=cols, compression="gzip", chunksize=1_000_000),
        desc="edge pass",
    ):
        q = ch[ch.pre_root_id.isin(candidates) & ch.post_root_id.isin(candidates)]
        if len(q):
            parts.append(q)

    edges = (
        pd.concat(parts, ignore_index=True)
        .groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
        .sort_values("syn_count", ascending=False)
    )
    selected, seen = [], set()
    for row in edges.head(max_edges * 16).itertuples(index=False):
        for rid in (int(row.pre_root_id), int(row.post_root_id)):
            if rid not in seen:
                seen.add(rid)
                selected.append(rid)
            if len(selected) >= feature_dim:
                break
        if len(selected) >= feature_dim:
            break
    if len(selected) < feature_dim:
        raise RuntimeError(f"only found {len(selected)} graph nodes; need {feature_dim}")

    selected = selected[:feature_dim]
    selected_set = set(selected)
    edges = edges[
        edges.pre_root_id.isin(selected_set) & edges.post_root_id.isin(selected_set)
    ].head(max_edges).copy()
    id_to_idx = {rid: i for i, rid in enumerate(selected)}
    src = edges.pre_root_id.astype(int).map(id_to_idx).to_numpy(np.int64)
    dst = edges.post_root_id.astype(int).map(id_to_idx).to_numpy(np.int64)
    syn = edges.syn_count.to_numpy(np.float32)

    raw = np.log1p(syn).astype(np.float32)
    incoming = np.zeros(feature_dim, dtype=np.float32)
    np.add.at(incoming, dst, raw)
    base = raw / np.maximum(incoming[dst], 1e-6)

    rng = np.random.default_rng(seed + 1)
    dst_rewired = dst.copy()
    rng.shuffle(dst_rewired)
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(dst_rewired, dtype=torch.long),
        torch.tensor(base, dtype=torch.float32),
        int(len(src)),
    )


def token_blocks(tokenizer, seed, needed, seq_len):
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    ).shuffle(seed=seed, buffer_size=2048)
    eos = tokenizer.eos_token_id
    buf, produced = [], 0
    for row in ds:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        if not ids:
            continue
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= seq_len + 1 and produced < needed:
            yield torch.tensor(buf[: seq_len + 1], dtype=torch.long)
            del buf[: seq_len + 1]
            produced += 1
        if produced >= needed:
            return


def make_batches(blocks, batch_size):
    return [
        torch.stack(blocks[i:i + batch_size])
        for i in range(0, len(blocks) - batch_size + 1, batch_size)
    ]


def causal_ce(logits, ids):
    # Model input is ids[:, :-1]; every returned logit predicts ids[:, 1:].
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    )


def distill_kl(student_logits, teacher_logits, temperature):
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    return (
        F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction="batchmean")
        * temperature**2 / max(s.size(1), 1)
    )


def hidden_alignment(student_hidden, teacher_hidden):
    last = min(len(student_hidden), len(teacher_hidden)) - 1
    indices = sorted({max(1, round(last * x)) for x in (0.25, 0.5, 0.75, 1.0)})
    terms = []
    for i in indices:
        s, t = student_hidden[i].float(), teacher_hidden[i].float()
        cosine = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        nmse = (s - t).square().mean() / t.square().mean().clamp_min(1e-5)
        terms.append(cosine + 0.15 * nmse)
    return torch.stack(terms).mean()


def build_student(base_model, dtype, device, cfg, src, dst, base, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    student = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype if device.type == "cuda" else torch.float32
    ).to(device)
    replace_all_attention_with_fly(student, cfg, src, dst, base)
    assert_pure_fly_attention(student)
    return student


@torch.no_grad()
def evaluate(student, teacher, eval_batches, device, dtype, temperature, seq_len, batch_size):
    student.eval()
    set_fly_streaming(student, False, reset=True)
    rows = []
    for cpu_ids in eval_batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        with amp_ctx(device, dtype):
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            s = student(input_ids=x, use_cache=False, return_dict=True)
        rows.append((
            float(causal_ce(t.logits, ids)),
            float(causal_ce(s.logits, ids)),
            float(distill_kl(s.logits, t.logits, temperature)),
        ))
    a = np.asarray(rows)
    tce, sce = float(a[:, 0].mean()), float(a[:, 1].mean())
    return {
        "teacher_ce": tce,
        "teacher_perplexity": math.exp(min(tce, 20)),
        "ce": sce,
        "perplexity": math.exp(min(sce, 20)),
        "teacher_kl": float(a[:, 2].mean()),
        "eval_tokens": len(eval_batches) * batch_size * seq_len,
        **fly_attention_stats(student),
    }


def train_one(name, dst, teacher, train_batches, eval_batches, cfg, src, base, args,
              device, dtype, train_updates, grad_accum):
    student = build_student(
        BASE_MODEL, dtype, device, cfg, src, dst, base, args.seed
    )
    groups, trainable = fly_parameter_groups(
        student, main_lr=8e-4, qkvo_lr=2e-5, weight_decay=0.01
    )
    try:
        opt = torch.optim.AdamW(groups, fused=(device.type == "cuda"))
    except Exception:
        opt = torch.optim.AdamW(groups)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and dtype == torch.float16)
    )
    warmup = max(20, train_updates // 20)
    student.train()
    set_fly_streaming(student, False, reset=True)
    opt.zero_grad(set_to_none=True)
    micro = 0
    history, t0 = [], time.perf_counter()

    for update in range(train_updates):
        ce_acc = kl_acc = hid_acc = loss_acc = 0.0
        for _ in range(grad_accum):
            ids = train_batches[micro].to(device)
            micro += 1
            x = ids[:, :-1]
            with torch.no_grad(), amp_ctx(device, dtype):
                t = teacher(
                    input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True
                )
            with amp_ctx(device, dtype):
                s = student(
                    input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True
                )
                ce = causal_ce(s.logits, ids)
                kl = distill_kl(s.logits, t.logits, 2.0)
                hid = hidden_alignment(s.hidden_states, t.hidden_states)
                raw = 0.35 * ce + 0.50 * kl + 0.15 * hid
                loss = raw / grad_accum
            scaler.scale(loss).backward()
            ce_acc += float(ce.detach()) / grad_accum
            kl_acc += float(kl.detach()) / grad_accum
            hid_acc += float(hid.detach()) / grad_accum
            loss_acc += float(raw.detach()) / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if update < warmup:
            mult = max((update + 1) / warmup, 1e-3)
        else:
            p = (update - warmup) / max(train_updates - warmup, 1)
            mult = 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        opt.param_groups[0]["lr"] = 8e-4 * mult
        opt.param_groups[1]["lr"] = 2e-5 * mult
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        history.append({"update": update + 1, "loss": loss_acc, "ce": ce_acc,
                        "kl": kl_acc, "hidden": hid_acc})
        if update == 0 or (update + 1) % 25 == 0 or update + 1 == train_updates:
            print(f"{name:>10s} {update+1:4d}/{train_updates} "
                  f"loss={loss_acc:.4f} ce={ce_acc:.4f} kl={kl_acc:.4f} hid={hid_acc:.4f}")

    metrics = evaluate(
        student, teacher, eval_batches, device, dtype, 2.0, args.seq_len, args.batch_size
    )
    metrics["train_tokens_s"] = (
        train_updates * grad_accum * args.batch_size * args.seq_len
        / (time.perf_counter() - t0)
    )
    metrics["params_m"] = sum(p.numel() for p in student.parameters()) / 1e6
    state = {
        k: v.detach().cpu().clone()
        for k, v in student.state_dict().items()
        if ".self_attn." in k
    }
    return student, metrics, history, state


def weight_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20


def peak_extra(fn, device):
    if device.type != "cuda":
        t = time.perf_counter()
        value = fn()
        return value, time.perf_counter() - t, float("nan")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    extra = max(0, torch.cuda.max_memory_allocated() - baseline) / 2**20
    return value, dt, extra


@torch.no_grad()
def benchmark_teacher(model, ids, device, steps=32):
    model.eval()
    out, dt, pre_mem = peak_extra(
        lambda: model(input_ids=ids, use_cache=True, return_dict=True), device
    )
    past = out.past_key_values
    token = out.logits[:, -1].argmax(-1, keepdim=True)

    def decode():
        nonlocal past, token
        for _ in range(steps):
            o = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
            past = o.past_key_values
            token = o.logits[:, -1].argmax(-1, keepdim=True)

    _, dt2, dec_mem = peak_extra(decode, device)
    return {
        "weight_mb": weight_mb(model),
        "prefill_tokens_s": ids.numel() / dt,
        "prefill_peak_extra_mb": pre_mem,
        "decode_tokens_s": steps / dt2,
        "decode_peak_extra_mb": dec_mem,
    }


@torch.no_grad()
def benchmark_fly(model, ids, device, steps=32):
    model.eval()

    def prefill():
        set_fly_streaming(model, True, reset=True)
        pos = torch.arange(ids.size(1), device=device)[None, :]
        return model(input_ids=ids, position_ids=pos, use_cache=False, return_dict=True)

    out, dt, pre_mem = peak_extra(prefill, device)
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    pos_num = ids.size(1)

    def decode():
        nonlocal token, pos_num
        for _ in range(steps):
            o = model(
                input_ids=token,
                position_ids=torch.tensor([[pos_num]], device=device),
                use_cache=False,
                return_dict=True,
            )
            token = o.logits[:, -1].argmax(-1, keepdim=True)
            pos_num += 1

    _, dt2, dec_mem = peak_extra(decode, device)
    set_fly_streaming(model, False, reset=True)
    return {
        "weight_mb": weight_mb(model),
        "prefill_tokens_s": ids.numel() / dt,
        "prefill_peak_extra_mb": pre_mem,
        "decode_tokens_s": steps / dt2,
        "decode_peak_extra_mb": dec_mem,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_updates, grad_accum, eval_count = (
        (300, 2, 12) if args.run_mode == "quick" else (2500, 4, 24)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src, dst_bio, dst_rewired, base, edge_count = extract_graph(
        args.feature_dim, args.max_edges, args.seed, Path("/content/flygraph_attention_cache")
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype if device.type == "cuda" else torch.float32
    ).to(device)
    teacher.eval()
    teacher.config.use_cache = True
    for p in teacher.parameters():
        p.requires_grad_(False)

    needed = train_updates * grad_accum * args.batch_size
    train = make_batches(
        list(token_blocks(tokenizer, args.seed + 10, needed, args.seq_len)),
        args.batch_size,
    )
    ev = make_batches(
        list(token_blocks(tokenizer, args.seed + 999, eval_count * args.batch_size, args.seq_len)),
        args.batch_size,
    )

    cfg = FlyGraphAttentionConfig(
        feature_dim=args.feature_dim,
        graph_steps=args.graph_steps,
        graph_gate_init=0.10,
        content_gate_init=0.25,
        feature_seed=7331,
    )

    bio, bio_metrics, bio_hist, bio_state = train_one(
        "biological", dst_bio, teacher, train, ev, cfg, src, base, args,
        device, dtype, train_updates, grad_accum,
    )
    pd.DataFrame(bio_hist).to_csv(out_dir / "bio_training_history.csv", index=False)

    rewired_metrics = None
    if args.rewired:
        del bio
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rew, rewired_metrics, rew_hist, _ = train_one(
            "rewired", dst_rewired, teacher, train, ev, cfg, src, base, args,
            device, dtype, train_updates, grad_accum,
        )
        pd.DataFrame(rew_hist).to_csv(out_dir / "rewired_training_history.csv", index=False)
        del rew
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        bio = build_student(BASE_MODEL, dtype, device, cfg, src, dst_bio, base, args.seed)
        inc = bio.load_state_dict(bio_state, strict=False)
        missing_attn = [k for k in inc.missing_keys if ".self_attn." in k]
        if missing_attn:
            raise RuntimeError(f"missing attention keys on reload: {missing_attn[:5]}")
    assert_pure_fly_attention(bio)

    test_ids = ev[0][:, :-1].to(device)[:, : min(32, args.seq_len)]
    with torch.no_grad(), amp_ctx(device, dtype):
        set_fly_streaming(bio, False, reset=True)
        full = bio(
            input_ids=test_ids,
            position_ids=torch.arange(test_ids.size(1), device=device)[None, :],
            use_cache=False,
            return_dict=True,
        ).logits[:, -1]
        split = test_ids.size(1) - 1
        set_fly_streaming(bio, True, reset=True)
        bio(
            input_ids=test_ids[:, :split],
            position_ids=torch.arange(split, device=device)[None, :],
            use_cache=False,
            return_dict=True,
        )
        streamed = bio(
            input_ids=test_ids[:, split:],
            position_ids=torch.arange(split, test_ids.size(1), device=device)[None, :],
            use_cache=False,
            return_dict=True,
        ).logits[:, -1]
    stream_diff = float((full.float() - streamed.float()).abs().max().cpu())
    set_fly_streaming(bio, False, reset=True)

    bench_ids = ev[0][:, :-1].to(device)
    tbench = benchmark_teacher(teacher, bench_ids, device)
    fbench = benchmark_fly(bio, bench_ids, device)

    report = {
        "architecture": "SmolLM2-135M with every self-attention layer replaced by FlyGraph causal linear attention",
        "pure_attention_replacement": True,
        "standard_llama_attention_modules": 0,
        "streaming_full_max_abs_logit_diff": stream_diff,
        "config": {
            "run_mode": args.run_mode,
            "seq_len": args.seq_len,
            "feature_dim": args.feature_dim,
            "fly_edges": edge_count,
            "graph_steps": args.graph_steps,
            "train_updates": train_updates,
            "grad_accum": grad_accum,
        },
        "biological": bio_metrics,
        "rewired": rewired_metrics,
        "benchmark_teacher": tbench,
        "benchmark_biological": fbench,
        "fly_ce_gap_vs_smollm2": bio_metrics["ce"] - bio_metrics["teacher_ce"],
        "fly_ppl_ratio_vs_smollm2": bio_metrics["perplexity"] / bio_metrics["teacher_perplexity"],
        "decode_speed_ratio_fly_over_smollm2": fbench["decode_tokens_s"] / tbench["decode_tokens_s"],
    }
    if rewired_metrics:
        report["biological_topology_ce_gain"] = rewired_metrics["ce"] - bio_metrics["ce"]
        report["biological_topology_ppl_gain_pct"] = 100 * (
            rewired_metrics["perplexity"] - bio_metrics["perplexity"]
        ) / rewired_metrics["perplexity"]

    rows = {
        "SmolLM2-135M": {
            "ce": bio_metrics["teacher_ce"],
            "perplexity": bio_metrics["teacher_perplexity"],
            **tbench,
        },
        "FlyGraph biological": {
            "ce": bio_metrics["ce"],
            "perplexity": bio_metrics["perplexity"],
            "teacher_kl": bio_metrics["teacher_kl"],
            **fbench,
        },
    }
    if rewired_metrics:
        rows["FlyGraph rewired"] = {
            "ce": rewired_metrics["ce"],
            "perplexity": rewired_metrics["perplexity"],
            "teacher_kl": rewired_metrics["teacher_kl"],
        }
    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(bio_state, out_dir / "biological_fly_attention.pt")

    print("\nPURE REPLACEMENT CHECK")
    print("LlamaAttention remaining: 0")
    print("FlyGraph layers:", len(fly_attention_modules(bio)))
    print("stream/full max logit diff:", stream_diff)
    print("\nSUMMARY")
    print(pd.DataFrame(rows).T)
    print("\nKEY REPORT")
    print(json.dumps({
        k: report[k] for k in report
        if k in (
            "fly_ce_gap_vs_smollm2",
            "fly_ppl_ratio_vs_smollm2",
            "decode_speed_ratio_fly_over_smollm2",
            "biological_topology_ce_gain",
            "biological_topology_ppl_gain_pct",
        )
    }, indent=2))
    print("\nSaved:", out_dir)


if __name__ == "__main__":
    main()
