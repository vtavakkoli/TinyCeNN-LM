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

from tinycenn_lm.smollm2_flyffn import (
    FlyFFNConfig,
    assert_flyffn_replacement,
    flyffn_modules,
    flyffn_parameter_groups,
    flyffn_router_regularizer,
    flyffn_stats,
    replace_ffns_with_fly,
    set_sparse_layers,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
GRAPH_URL = "https://zenodo.org/records/21549559/files/connections_biological.csv.gz?download=1"


def parse_args():
    p = argparse.ArgumentParser(description="FlyFFN SmolLM2-135M isolated FFN experiment")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--router-rank", type=int, default=64)
    p.add_argument("--max-edges", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix-init", type=float, default=0.50)
    p.add_argument("--rewired", action="store_true")
    p.add_argument("--output-dir", default="results/flyffn_smollm2_135m")
    p.add_argument("--seed", type=int, default=321)
    return p.parse_args()


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_ctx(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_graph(fly_nodes, max_edges, seed, cache_dir):
    print("STAGE graph: preparing FlyWire topology", flush=True)
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

    candidates = set(int(x) for x in degree.nlargest(max(fly_nodes * 6, 2048)).index)
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
            if len(selected) >= fly_nodes:
                break
        if len(selected) >= fly_nodes:
            break
    if len(selected) < fly_nodes:
        raise RuntimeError(f"only found {len(selected)} graph nodes; need {fly_nodes}")

    selected = selected[:fly_nodes]
    selected_set = set(selected)
    edges = edges[
        edges.pre_root_id.isin(selected_set) & edges.post_root_id.isin(selected_set)
    ].head(max_edges).copy()
    id_to_idx = {rid: i for i, rid in enumerate(selected)}
    src = edges.pre_root_id.astype(int).map(id_to_idx).to_numpy(np.int64)
    dst = edges.post_root_id.astype(int).map(id_to_idx).to_numpy(np.int64)
    syn = edges.syn_count.to_numpy(np.float32)

    raw = np.log1p(syn).astype(np.float32)
    incoming = np.zeros(fly_nodes, dtype=np.float32)
    np.add.at(incoming, dst, raw)
    weight = raw / np.maximum(incoming[dst], 1e-6)

    def dense_adjacency(dst_array):
        a = torch.zeros(fly_nodes, fly_nodes, dtype=torch.float32)
        for s, d, w in zip(src, dst_array, weight):
            a[int(d), int(s)] += float(w)
        row_sum = a.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
        return a / row_sum

    bio = dense_adjacency(dst)
    rng = np.random.default_rng(seed + 1)
    dst_rewired = dst.copy()
    rng.shuffle(dst_rewired)
    rewired = dense_adjacency(dst_rewired)
    print(f"Graph ready: {fly_nodes} nodes, {len(src)} edges", flush=True)
    return bio, rewired, int(len(src))


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
    return [torch.stack(blocks[i:i + batch_size]) for i in range(0, len(blocks), batch_size) if len(blocks[i:i + batch_size]) == batch_size]


def causal_ce(logits, ids):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))


def distill_kl(student_logits, teacher_logits, temperature=2.0):
    t = float(temperature)
    s = student_logits.float() / t
    q = teacher_logits.float() / t
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(q, dim=-1), reduction="batchmean") * (t * t) / max(student_logits.shape[1], 1)


def hidden_alignment(student_h, teacher_h):
    n = min(len(student_h), len(teacher_h)) - 1
    idxs = sorted(set(max(1, round(n * f)) for f in (0.25, 0.50, 0.75, 1.0)))
    losses = []
    for idx in idxs:
        s = student_h[idx].float()
        t = teacher_h[idx].float()
        cos = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        nmse = (s - t).pow(2).mean() / t.pow(2).mean().clamp_min(1e-6)
        losses.append(cos + 0.15 * nmse)
    return torch.stack(losses).mean()


def mlp_alignment_loss(pred, target):
    p, t = pred.float(), target.float()
    cos = 1.0 - F.cosine_similarity(p, t, dim=-1).mean()
    nmse = (p - t).pow(2).mean() / t.pow(2).mean().clamp_min(1e-6)
    return cos + 0.20 * nmse


@torch.no_grad()
def capture_teacher_mlp_io(teacher, x, layer_indices, amp):
    captures, handles = {}, []
    for idx in layer_indices:
        module = teacher.model.layers[idx].mlp

        def pre_hook(_module, args, idx=idx):
            captures[idx] = {"hidden": args[0].detach()}

        def post_hook(_module, _args, output, idx=idx):
            captures[idx]["target"] = output.detach()

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    try:
        with amp():
            teacher(input_ids=x, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captures


def build_student(dtype, device, cfg, adjacency, seed):
    set_seed(seed)
    student = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype if device.type == "cuda" else torch.float32
    ).to(device)
    replace_ffns_with_fly(student, cfg, adjacency)
    assert_flyffn_replacement(student)
    return student


def make_optimizer(groups, device):
    try:
        return torch.optim.AdamW(groups, fused=(device.type == "cuda"))
    except Exception:
        return torch.optim.AdamW(groups)


def make_scaler(device, dtype):
    enabled = device.type == "cuda" and dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def calibrate_groups(name, student, teacher, calib_batches, device, dtype, steps_per_group, group_size):
    all_layers = [m.layer_idx for m in flyffn_modules(student)]
    groups_idx = [all_layers[i:i + group_size] for i in range(0, len(all_layers), group_size)]
    amp = lambda: amp_ctx(device, dtype)
    cursor, history = 0, []

    # Exact dense reconstruction before a layer is converted.
    set_sparse_layers(student, enabled=False)

    for gi, layer_group in enumerate(groups_idx, 1):
        set_sparse_layers(student, layer_group, enabled=True)
        params, trainable = flyffn_parameter_groups(
            student, layer_group, router_lr=1e-3, shard_lr=2e-5, weight_decay=0.0
        )
        opt = make_optimizer(params, device)
        scaler = make_scaler(device, dtype)
        student.train()

        for step in range(1, steps_per_group + 1):
            ids = calib_batches[cursor % len(calib_batches)].to(device)
            cursor += 1
            x = ids[:, :-1]
            captures = capture_teacher_mlp_io(teacher, x, layer_group, amp)

            opt.zero_grad(set_to_none=True)
            losses = []
            with amp():
                for idx in layer_group:
                    pred = student.model.layers[idx].mlp(captures[idx]["hidden"])
                    losses.append(mlp_alignment_loss(pred, captures[idx]["target"]))
                align = torch.stack(losses).mean()
                reg = flyffn_router_regularizer(student, layer_group)
                loss = align + 0.02 * reg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()

            lv = float(loss.detach())
            history.append({"group": gi, "step": step, "layers": str(layer_group), "loss": lv})
            if step == 1 or step == steps_per_group or step % max(1, steps_per_group // 3) == 0:
                print(
                    f"CALIB {name} group={gi}/{len(groups_idx)} layers={layer_group} "
                    f"step={step}/{steps_per_group} loss={lv:.5f}", flush=True
                )
    return history


@torch.no_grad()
def evaluate(student, teacher, eval_batches, device, dtype, seq_len, batch_size):
    student.eval(); teacher.eval()
    set_sparse_layers(student, enabled=True)
    rows = []
    for cpu_ids in eval_batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        with amp_ctx(device, dtype):
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            s = student(input_ids=x, use_cache=False, return_dict=True)
        rows.append((float(causal_ce(t.logits, ids)), float(causal_ce(s.logits, ids)), float(distill_kl(s.logits, t.logits))))
    a = np.asarray(rows)
    tce, sce = float(a[:, 0].mean()), float(a[:, 1].mean())
    return {
        "teacher_ce": tce,
        "teacher_perplexity": math.exp(min(tce, 20)),
        "ce": sce,
        "perplexity": math.exp(min(sce, 20)),
        "teacher_kl": float(a[:, 2].mean()),
        "eval_tokens": len(eval_batches) * batch_size * seq_len,
        **flyffn_stats(student),
    }


def global_train(name, student, teacher, train_batches, eval_batches, args, device, dtype, train_updates, grad_accum):
    layers = [m.layer_idx for m in flyffn_modules(student)]
    set_sparse_layers(student, enabled=True)
    groups, trainable = flyffn_parameter_groups(
        student, layers, router_lr=7e-4, shard_lr=1.5e-5, weight_decay=0.01
    )
    opt = make_optimizer(groups, device)
    scaler = make_scaler(device, dtype)
    warmup = max(20, train_updates // 20)
    opt.zero_grad(set_to_none=True)
    student.train()
    history, micro = [], 0
    t0 = time.perf_counter()

    for update in range(1, train_updates + 1):
        ce_acc = kl_acc = hid_acc=reg_acc=loss_acc = 0.0
        for _ in range(grad_accum):
            ids = train_batches[micro % len(train_batches)].to(device)
            micro += 1
            x = ids[:, :-1]
            with torch.no_grad(), amp_ctx(device, dtype):
                t = teacher(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
            with amp_ctx(device, dtype):
                s = student(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
                ce = causal_ce(s.logits, ids)
                kl = distill_kl(s.logits, t.logits)
                hid = hidden_alignment(s.hidden_states, t.hidden_states)
                reg = flyffn_router_regularizer(student)
                raw = 0.35 * ce + 0.45 * kl + 0.20 * hid + 0.01 * reg
                loss = raw / grad_accum
            scaler.scale(loss).backward()
            ce_acc += float(ce.detach()) / grad_accum
            kl_acc += float(kl.detach()) / grad_accum
            hid_acc += float(hid.detach()) / grad_accum
            reg_acc += float(reg.detach()) / grad_accum
            loss_acc += float(raw.detach()) / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if update <= warmup:
            mult = max(update / warmup, 1e-3)
        else:
            p = (update - warmup) / max(train_updates - warmup, 1)
            mult = 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        opt.param_groups[0]["lr"] = 7e-4 * mult
        opt.param_groups[1]["lr"] = 1.5e-5 * mult
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        row = {"update": update, "loss": loss_acc, "ce": ce_acc, "kl": kl_acc, "hidden": hid_acc, "router_reg": reg_acc}
        history.append(row)
        if update == 1 or update % 25 == 0 or update == train_updates:
            elapsed = time.perf_counter() - t0
            ups = update / max(elapsed, 1e-6)
            eta = (train_updates - update) / max(ups, 1e-9)
            print(
                f"GLOBAL {name} {update}/{train_updates} loss={loss_acc:.4f} ce={ce_acc:.4f} "
                f"kl={kl_acc:.4f} hid={hid_acc:.4f} reg={reg_acc:.4f} ups={ups:.3f} eta_s={eta:.1f}",
                flush=True,
            )

    metrics = evaluate(student, teacher, eval_batches, device, dtype, args.seq_len, args.batch_size)
    metrics["train_tokens_s"] = train_updates * grad_accum * args.batch_size * args.seq_len / max(time.perf_counter() - t0, 1e-9)
    metrics["params_m"] = sum(p.numel() for p in student.parameters()) / 1e6
    return metrics, history


def flyffn_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".mlp." in k}


def load_flyffn_state(model, state):
    inc = model.load_state_dict(state, strict=False)
    missing = [k for k in inc.missing_keys if ".mlp." in k]
    if missing:
        raise RuntimeError(f"missing FlyFFN keys: {missing[:8]}")


def weight_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20


def buffer_mb(model):
    return sum(b.numel() * b.element_size() for b in model.buffers()) / 2**20


def peak_extra(fn, device):
    if device.type != "cuda":
        t = time.perf_counter(); value = fn(); return value, time.perf_counter() - t, float("nan")
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); value = fn(); torch.cuda.synchronize(); dt = time.perf_counter() - t
    return value, dt, max(0, torch.cuda.max_memory_allocated() - baseline) / 2**20


@torch.no_grad()
def benchmark(model, ids, device, steps=32):
    model.eval()
    out, dt, pre_mem = peak_extra(lambda: model(input_ids=ids, use_cache=True, return_dict=True), device)
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
        "buffer_mb": buffer_mb(model),
        "prefill_tokens_s": ids.numel() / dt,
        "prefill_peak_extra_mb": pre_mem,
        "decode_tokens_s": steps / dt2,
        "decode_peak_extra_mb": dec_mem,
    }


@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=48):
    prompts = [
        "Artificial intelligence can",
        "The city of Vienna is",
        "A neural network learns",
        "The future of efficient computing",
    ]
    model.eval(); outputs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        outputs.append({"prompt": prompt, "text": tokenizer.decode(out[0], skip_special_tokens=True)})
    return outputs


def train_variant(name, adjacency, teacher, tokenizer, cfg, calib_batches, train_batches, eval_batches, args, device, dtype, calib_steps, calib_group_size, train_updates, grad_accum):
    print(f"STAGE building {name} FlyFFN student (attention unchanged)", flush=True)
    student = build_student(dtype, device, cfg, adjacency, args.seed)
    calib_hist = calibrate_groups(name, student, teacher, calib_batches, device, dtype, calib_steps, calib_group_size)
    print(f"STAGE global distillation: {name}", flush=True)
    metrics, train_hist = global_train(name, student, teacher, train_batches, eval_batches, args, device, dtype, train_updates, grad_accum)
    state = flyffn_state(student)
    return student, metrics, calib_hist, train_hist, state


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"DEVICE {device} | dtype={dtype} | gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)

    if args.run_mode == "quick":
        calib_steps, calib_group_size = 6, 3
        train_updates, grad_accum, eval_count = 300, 2, 12
    else:
        calib_steps, calib_group_size = 30, 3
        train_updates, grad_accum, eval_count = 2500, 4, 24

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bio_adj, rew_adj, edge_count = extract_graph(args.fly_nodes, args.max_edges, args.seed, Path("/content/flyffn_cache"))

    print("STAGE loading standard SmolLM2-135M teacher/tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype if device.type == "cuda" else torch.float32).to(device)
    teacher.eval(); teacher.config.use_cache = True
    for p in teacher.parameters(): p.requires_grad_(False)

    cfg = FlyFFNConfig(
        fly_nodes=args.fly_nodes, router_rank=args.router_rank, num_shards=args.num_shards,
        top_k=args.top_k, graph_steps=args.graph_steps, graph_mix_init=args.graph_mix_init,
    )
    temp = build_student(dtype, device, cfg, bio_adj, args.seed)
    layers = [m.layer_idx for m in flyffn_modules(temp)]
    n_groups = math.ceil(len(layers9 / calib_group_size)
    del temp; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    print(
        f"Architecture: standard SmolLM2 attention untouched; {len(layers)} FFNs -> "
        f"FlyWire-routed {args.top_k}-of-{args.num_shards} SwiGLU shards",
        flush=True,
    )
    print(f"Theoretical active FFN shard fraction: {args.top_k/args.num_shards:.1%}", flush=True)

    print("STAGE preparing FineWeb-Edu calibration/train/eval blocks", flush=True)
    calib_needed = n_groups * calib_steps * args.batch_size
    train_needed = train_updates * grad_accum * args.batch_size
    calib_batches = make_batches(list(token_blocks(tokenizer, args.seed + 5, calib_needed, args.seq_len)), args.batch_size)
    train_batches = make_batches(list(token_blocks(tokenizer, args.seed + 10, train_needed, args.seq_len)), args.batch_size)
    eval_batches = make_batches(list(token_blocks(tokenizer, args.seed + 999, eval_count * args.batch_size, args.seq_len)), args.batch_size)

    bio, bio_metrics, bio_calib, bio_hist, bio_state = train_variant(
        "biological", bio_adj, teacher, tokenizer, cfg, calib_batches, train_batches, eval_batches,
        args, device, dtype, calib_steps, calib_group_size, train_updates, grad_accum
    )
    pd.DataFrame(bio_calib).to_csv(out_dir / "bio_calibration_history.csv", index=False)
    pd.DataFrame(bio_hist).to_csv(out_dir / "bio_training_history.csv", index=False)

    rewired_metrics = None
    if args.rewired:
        del bio; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        rew, rewired_metrics, rew_calib, rew_hist, _ = train_variant(
            "rewired", rew_adj, teacher, tokenizer, cfg, calib_batches, train_batches, eval_batches,
            args, device, dtype, calib_steps, calib_group_size, train_updates, grad_accum
        )
        pd.DataFrame(rew_calib).to_csv(out_dir / "rewired_calibration_history.csv", index=False)
        pd.DataFrame(rew_hist).to_csv(out_dir / "rewired_training_history.csv", index=False)
        del rew; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        bio = build_student(dtype, device, cfg, bio_adj, args.seed)
        set_sparse_layers(bio, enabled=True)
        load_flyffn_state(bio, bio_state)
    else:
        set_sparse_layers(bio, enabled=True)

    assert_flyffn_replacement(bio)
    test_ids = eval_batches[0][:, :-1].to(device)
    print("STAGE benchmarking standard SmolLM2 vs FlyFFN", flush=True)
    tbench = benchmark(teacher, test_ids, device)
    fbench = benchmark(bio, test_ids, device)
    print("STAGE generating qualitative samples", flush=True)
    samples = generate_samples(bio, tokenizer, device)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    report = {
        "architecture": "FlyFFN v1: standard SmolLM2 attention + FlyWire-routed sparse SwiGLU FFN",
        "base_model": BASE_MODEL,
        "attention_unchanged": True,
        "flyffn_layers": len(layers),
        "theoretical_active_ffn_fraction": args.top_k / args.num_shards,
        "implementation_note": "quality prototype computes all shards before Top-k gather; theoretical sparse compute is top_k/num_shards until a fused dispatch kernel is added",
        "device": str(device),
        "dtype": str(dtype),
        "config": {
            "run_mode": args.run_mode, "seq_len": args.seq_len, "fly_nodes": args.fly_nodes,
            "fly_edges": edge_count, "router_rank": args.router_rank, "num_shards": args.num_shards,
            "top_k": args.top_k, "graph_steps": args.graph_steps, "graph_mix_init": args.graph_mix_init,
            "calibration_steps_per_group": calib_steps, "global_train_updates": train_updates,
            "grad_accum": grad_accum,
        },
        "biological": bio_metrics,
        "rewired": rewired_metrics,
        "benchmark_teacher": tbench,
        "benchmark_biological": fbench,
        "fly_ce_gap_vs_smollm2": bio_metrics["ce"] - bio_metrics["teacher_ce"],
        "fly_ppl_ratio_vs_smollm2": bio_metrics["perplexity"] / bio_metrics["teacher_perplexity"],
        "decode_speed_ratio_fly_over_smollm2": fbench["decode_tokens_s"] / tbench["decode_tokens_s"],
        "parameter_ratio_fly_over_smollm2": sum(p.numel() for p in bio.parameters()) / sum(p.numel() for p in teacher.parameters()),
    }
    if rewired_metrics:
        report["biological_topology_ce_gain"] = rewired_metrics["ce"] - bio_metrics["ce"]
        report["biological_topology_ppl_gain_pct"] = 100 * (rewired_metrics["perplexity"] - bio_metrics["perplexity"]) / rewired_metrics["perplexity"]

    rows = {
        "SmolLM2-135M": {"ce": bio_metrics["teacher_ce"], "perplexity": bio_metrics["teacher_perplexity"], **tbench},
        "FlyFFN biological": {"ce": bio_metrics["ce"], "perplexity": bio_metrics["perplexity"], "teacher_kl": bio_metrics["teacher_kl"], **fbench},
    }
    if rewired_metrics:
        rows["FlyFFN rewired"] = {"ce": rewired_metrics["ce"], "perplexity": rewired_metrics["perplexity"], "teacher_kl": rewired_metrics["teacher_kl"]}
    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(bio_state, out_dir / "biological_flyffn.pt")

    print("\nFLYFFN CHECK", flush=True)
    print("Attention unchanged: True", flush=True)
    print(f"FlyFFN layers: {len(layers)} | sparse Top-k: {args.top_k}/{args.num_shards}", flush=True)
    print("\nSUMMARY", flush=True); print(pd.DataFrame(rows).T, flush=True)
    print("\nKEY REPORT", flush=True)
    keys = ("fly_ce_gap_vs_smollm2", "fly_ppl_ratio_vs_smollm2", "parameter_ratio_fly_over_smollm2", "decode_speed_ratio_fly_over_smollm2", "biological_topology_ce_gain", "biological_topology_ppl_gain_pct")
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=2), flush=True)
    print("\nSAMPLES", flush=True)
    for item in samples:
        print("=" * 88, flush=True); print("PROMPT:", item["prompt"], flush=True); print(item["text"], flush=True)
    print("\nSaved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
