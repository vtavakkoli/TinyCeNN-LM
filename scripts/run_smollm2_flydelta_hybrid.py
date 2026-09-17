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

from tinycenn_lm.smollm2_flydelta_hybrid import (
    FlyDeltaHybridConfig,
    anchor_layer_indices,
    assert_flydelta_replacement,
    calibration_parameter_groups,
    flydelta_stats,
    global_parameter_groups,
    hybrid_layer_indices,
    replace_all_attention_with_flydelta,
    set_flydelta_streaming,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
GRAPH_URL = "https://zenodo.org/records/21549559/files/connections_biological.csv.gz?download=1"


def parse_args():
    p = argparse.ArgumentParser(description="FlyDelta-Hybrid v2 SmolLM2 experiment")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--max-edges", type=int, default=2048)
    p.add_argument("--local-window", type=int, default=64)
    p.add_argument("--anchor-window", type=int, default=256)
    p.add_argument("--anchor-every", type=int, default=4)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--rewired", action="store_true")
    p.add_argument("--output-dir", default="results/flydelta_hybrid_smollm2_135m")
    p.add_argument("--seed", type=int, default=123)
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
    print("STAGE graph: preparing FlyWire biological topology", flush=True)
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
    return [
        torch.stack(blocks[i : i + batch_size])
        for i in range(0, len(blocks) - batch_size + 1, batch_size)
    ]


def causal_ce(logits, ids):
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )


def distill_kl(student_logits, teacher_logits, temperature=2.0):
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature
    per_token = F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean() * (temperature ** 2)


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


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(v) for v in value)
    if isinstance(value, list):
        return [_detach_tree(v) for v in value]
    return value


def capture_teacher_attention_io(teacher, ids, layer_indices, amp):
    captures = {idx: {} for idx in layer_indices}
    handles = []
    for idx in layer_indices:
        module = teacher.model.layers[idx].self_attn

        def pre_hook(mod, args, kwargs, layer_idx=idx):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError("teacher attention hidden_states unavailable")
            captures[layer_idx]["hidden"] = hidden.detach()
            for key in ("position_embeddings", "position_ids", "attention_mask", "cache_position"):
                if key in kwargs and kwargs[key] is not None:
                    captures[layer_idx][key] = _detach_tree(kwargs[key])

        def post_hook(mod, args, kwargs, output, layer_idx=idx):
            target = output[0] if isinstance(output, (tuple, list)) else output
            captures[layer_idx]["target"] = target.detach()

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    try:
        with torch.no_grad(), amp():
            teacher(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    for idx in layer_indices:
        if "hidden" not in captures[idx] or "target" not in captures[idx]:
            raise RuntimeError(f"failed to capture teacher layer {idx}")
    return captures


def attention_alignment_loss(pred, target):
    p, t = pred.float(), target.float()
    nmse = (p - t).square().mean() / t.square().mean().clamp_min(1e-5)
    cosine = 1.0 - F.cosine_similarity(p, t, dim=-1).mean()
    return nmse + 0.25 * cosine


def build_student(dtype, device, cfg, adjacency, seed):
    set_seed(seed)
    student = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype if device.type == "cuda" else torch.float32
    ).to(device)
    replace_all_attention_with_flydelta(student, cfg, adjacency)
    assert_flydelta_replacement(student)
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


def calibrate_groups(
    name, student, teacher, calib_batches, device, dtype, steps_per_group, group_size
):
    hybrid = hybrid_layer_indices(student)
    groups_idx = [hybrid[i : i + group_size] for i in range(0, len(hybrid), group_size)]
    amp = lambda: amp_ctx(device, dtype)
    cursor = 0
    history = []

    for gi, layer_group in enumerate(groups_idx, 1):
        params, trainable = calibration_parameter_groups(
            student,
            layer_group,
            main_lr=1e-3,
            qkvo_lr=1e-5,
            weight_decay=0.0,
        )
        opt = make_optimizer(params, device)
        scaler = make_scaler(device, dtype)
        student.train()
        set_flydelta_streaming(student, False, reset=True)

        for step in range(1, steps_per_group + 1):
            ids = calib_batches[cursor % len(calib_batches)].to(device)
            cursor += 1
            x = ids[:, :-1]
            captures = capture_teacher_attention_io(teacher, x, layer_group, amp)

            opt.zero_grad(set_to_none=True)
            losses = []
            with amp():
                for idx in layer_group:
                    c = captures[idx]
                    kwargs = {"use_cache": False}
                    for key in ("position_embeddings", "position_ids", "attention_mask", "cache_position"):
                        if key in c:
                            kwargs[key] = c[key]
                    pred = student.model.layers[idx].self_attn(c["hidden"], **kwargs)[0]
                    losses.append(attention_alignment_loss(pred, c["target"]))
                loss = torch.stack(losses).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()

            lv = float(loss.detach())
            history.append(
                {"group": gi, "step": step, "layers": str(layer_group), "alignment_loss": lv}
            )
            if step == 1 or step == steps_per_group or step % max(1, steps_per_group // 4) == 0:
                print(
                    f"CALIB {name} group={gi}/{len(groups_idx)} "
                    f"layers={layer_group} step={step}/{steps_per_group} loss={lv:.5f}",
                    flush=True,
                )
    return history


@torch.no_grad()
def evaluate(student, teacher, eval_batches, device, dtype, seq_len, batch_size):
    student.eval()
    set_flydelta_streaming(student, False, reset=True)
    rows = []
    for cpu_ids in eval_batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        with amp_ctx(device, dtype):
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            s = student(input_ids=x, use_cache=False, return_dict=True)
        rows.append(
            (
                float(causal_ce(t.logits, ids)),
                float(causal_ce(s.logits, ids)),
                float(distill_kl(s.logits, t.logits)),
            )
        )
    a = np.asarray(rows)
    tce, sce = float(a[:, 0].mean()), float(a[:, 1].mean())
    return {
        "teacher_ce": tce,
        "teacher_perplexity": math.exp(min(tce, 20)),
        "ce": sce,
        "perplexity": math.exp(min(sce, 20)),
        "teacher_kl": float(a[:, 2].mean()),
        "eval_tokens": len(eval_batches) * batch_size * seq_len,
        **flydelta_stats(student),
    }


def global_train(
    name, student, teacher, train_batches, eval_batches, args, device, dtype,
    train_updates, grad_accum
):
    groups, trainable = global_parameter_groups(
        student, main_lr=6e-4, qkvo_lr=1e-5, weight_decay=0.01
    )
    opt = make_optimizer(groups, device)
    scaler = make_scaler(device, dtype)
    warmup = max(20, train_updates // 20)
    opt.zero_grad(set_to_none=True)
    set_flydelta_streaming(student, False, reset=True)
    student.train()
    history = []
    micro = 0
    t0 = time.perf_counter()

    for update in range(1, train_updates + 1):
        ce_acc = kl_acc = hid_acc = loss_acc = 0.0
        for _ in range(grad_accum):
            ids = train_batches[micro % len(train_batches)].to(device)
            micro += 1
            x = ids[:, :-1]
            with torch.no_grad(), amp_ctx(device, dtype):
                t = teacher(
                    input_ids=x,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            with amp_ctx(device, dtype):
                s = student(
                    input_ids=x,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                ce = causal_ce(s.logits, ids)
                kl = distill_kl(s.logits, t.logits)
                hid = hidden_alignment(s.hidden_states, t.hidden_states)
                raw = 0.30 * ce + 0.50 * kl + 0.20 * hid
                loss = raw / grad_accum

            scaler.scale(loss).backward()
            ce_acc += float(ce.detach()) / grad_accum
            kl_acc += float(kl.detach()) / grad_accum
            hid_acc += float(hid.detach()) / grad_accum
            loss_acc += float(raw.detach()) / grad_accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)

        if update <= warmup:
            mult = max(update / warmup, 1e-3)
        else:
            p = (update - warmup) / max(train_updates - warmup, 1)
            mult = 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        opt.param_groups[0]["lr"] = 6e-4 * mult
        opt.param_groups[1]["lr"] = 1e-5 * mult

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        row = {
            "update": update,
            "loss": loss_acc,
            "ce": ce_acc,
            "kl": kl_acc,
            "hidden": hid_acc,
        }
        history.append(row)

        if update == 1 or update % 25 == 0 or update == train_updates:
            elapsed = time.perf_counter() - t0
            ups = update / max(elapsed, 1e-6)
            eta = (train_updates - update) / max(ups, 1e-9)
            print(
                f"GLOBAL {name} {update}/{train_updates} "
                f"loss={loss_acc:.4f} ce={ce_acc:.4f} kl={kl_acc:.4f} "
                f"hid={hid_acc:.4f} ups={ups:.3f} eta_s={eta:.1f}",
                flush=True,
            )

    metrics = evaluate(
        student, teacher, eval_batches, device, dtype, args.seq_len, args.batch_size
    )
    metrics["train_tokens_s"] = (
        train_updates * grad_accum * args.batch_size * args.seq_len
        / max(time.perf_counter() - t0, 1e-9)
    )
    metrics["params_m"] = sum(p.numel() for p in student.parameters()) / 1e6
    return metrics, history


def attention_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if ".self_attn." in k
    }


def load_attention_state(model, state):
    inc = model.load_state_dict(state, strict=False)
    missing = [k for k in inc.missing_keys if ".self_attn." in k]
    if missing:
        raise RuntimeError(f"missing FlyDelta attention keys: {missing[:8]}")


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
def benchmark_flydelta(model, ids, device, steps=32):
    model.eval()

    def prefill():
        set_flydelta_streaming(model, True, reset=True)
        pos = torch.arange(ids.size(1), device=device)[None, :]
        return model(
            input_ids=ids,
            position_ids=pos,
            use_cache=False,
            return_dict=True,
        )

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
    set_flydelta_streaming(model, False, reset=True)
    return {
        "weight_mb": weight_mb(model),
        "prefill_tokens_s": ids.numel() / dt,
        "prefill_peak_extra_mb": pre_mem,
        "decode_tokens_s": steps / dt2,
        "decode_peak_extra_mb": dec_mem,
    }


@torch.no_grad()
def streaming_equivalence(model, ids, device, dtype):
    test_ids = ids[:, : min(48, ids.size(1))]
    pos = torch.arange(test_ids.size(1), device=device)[None, :]
    model.eval()

    with amp_ctx(device, dtype):
        set_flydelta_streaming(model, False, reset=True)
        full = model(
            input_ids=test_ids,
            position_ids=pos,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1]

        split = test_ids.size(1) - 1
        set_flydelta_streaming(model, True, reset=True)
        model(
            input_ids=test_ids[:, :split],
            position_ids=pos[:, :split],
            use_cache=False,
            return_dict=True,
        )
        streamed = model(
            input_ids=test_ids[:, split:],
            position_ids=pos[:, split:],
            use_cache=False,
            return_dict=True,
        ).logits[:, -1]

    diff = float((full.float() - streamed.float()).abs().max().cpu())
    set_flydelta_streaming(model, False, reset=True)
    return diff


@torch.no_grad()
def generate_samples(model, tokenizer, device, dtype, max_new_tokens=48):
    prompts = [
        "Artificial intelligence can",
        "The city of Vienna is",
        "A neural network learns",
        "The future of efficient computing",
    ]
    outputs = []
    model.eval()
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        set_flydelta_streaming(model, True, reset=True)
        pos = torch.arange(ids.size(1), device=device)[None, :]
        with amp_ctx(device, dtype):
            out = model(input_ids=ids, position_ids=pos, use_cache=False, return_dict=True)
        generated = ids.clone()
        logits = out.logits[:, -1]
        next_pos = ids.size(1)
        for _ in range(max_new_tokens):
            token = logits.argmax(-1, keepdim=True)
            generated = torch.cat([generated, token], dim=1)
            if tokenizer.eos_token_id is not None and int(token.item()) == tokenizer.eos_token_id:
                break
            with amp_ctx(device, dtype):
                out = model(
                    input_ids=token,
                    position_ids=torch.tensor([[next_pos]], device=device),
                    use_cache=False,
                    return_dict=True,
                )
            logits = out.logits[:, -1]
            next_pos += 1
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append({"prompt": prompt, "text": text})
        set_flydelta_streaming(model, False, reset=True)
    return outputs


def train_variant(
    name, adjacency, teacher, tokenizer, cfg, calib_batches, train_batches, eval_batches,
    args, device, dtype, calib_steps, calib_group_size, train_updates, grad_accum
):
    print(f"STAGE building {name} FlyDelta-Hybrid student", flush=True)
    student = build_student(dtype, device, cfg, adjacency, args.seed)

    calib_hist = calibrate_groups(
        name, student, teacher, calib_batches, device, dtype,
        steps_per_group=calib_steps, group_size=calib_group_size
    )

    print(f"STAGE global distillation: {name}", flush=True)
    metrics, train_hist = global_train(
        name, student, teacher, train_batches, eval_batches, args,
        device, dtype, train_updates, grad_accum
    )
    state = attention_state(student)
    return student, metrics, calib_hist, train_hist, state


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.run_mode == "quick":
        calib_steps, calib_group_size = 8, 3
        train_updates, grad_accum, eval_count = 300, 2, 12
    else:
        calib_steps, calib_group_size = 40, 3
        train_updates, grad_accum, eval_count = 2500, 4, 24

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bio_adj, rew_adj, edge_count = extract_graph(
        args.fly_nodes, args.max_edges, args.seed, Path("/content/flydelta_cache")
    )

    print("STAGE loading SmolLM2-135M teacher/tokenizer", flush=True)
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

    cfg = FlyDeltaHybridConfig(
        fly_nodes=args.fly_nodes,
        local_window=args.local_window,
        anchor_window=args.anchor_window,
        anchor_every=args.anchor_every,
        graph_steps=args.graph_steps,
        delta_gate_init=0.01,
        graph_gain_init=0.10,
    )
    temp = build_student(dtype, device, cfg, bio_adj, args.seed)
    n_groups = math.ceil(len(hybrid_layer_indices(temp)) / calib_group_size)
    anchors = anchor_layer_indices(temp)
    hybrids = hybrid_layer_indices(temp)
    del temp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"Architecture: {len(hybrids)} FlyDelta hybrid layers + "
        f"{len(anchors)} exact/local anchor layers; anchors={anchors}",
        flush=True,
    )

    print("STAGE preparing FineWeb-Edu calibration/train/eval blocks", flush=True)
    calib_needed = n_groups * calib_steps * args.batch_size
    train_needed = train_updates * grad_accum * args.batch_size
    calib_batches = make_batches(
        list(token_blocks(tokenizer, args.seed + 5, calib_needed, args.seq_len)),
        args.batch_size,
    )
    train_batches = make_batches(
        list(token_blocks(tokenizer, args.seed + 10, train_needed, args.seq_len)),
        args.batch_size,
    )
    eval_batches = make_batches(
        list(token_blocks(tokenizer, args.seed + 999, eval_count * args.batch_size, args.seq_len)),
        args.batch_size,
    )

    bio, bio_metrics, bio_calib, bio_hist, bio_state = train_variant(
        "biological", bio_adj, teacher, tokenizer, cfg, calib_batches, train_batches,
        eval_batches, args, device, dtype, calib_steps, calib_group_size,
        train_updates, grad_accum
    )
    pd.DataFrame(bio_calib).to_csv(out_dir / "bio_calibration_history.csv", index=False)
    pd.DataFrame(bio_hist).to_csv(out_dir / "bio_training_history.csv", index=False)

    rewired_metrics = None
    if args.rewired:
        del bio
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        rew, rewired_metrics, rew_calib, rew_hist, _ = train_variant(
            "rewired", rew_adj, teacher, tokenizer, cfg, calib_batches, train_batches,
            eval_batches, args, device, dtype, calib_steps, calib_group_size,
            train_updates, grad_accum
        )
        pd.DataFrame(rew_calib).to_csv(out_dir / "rewired_calibration_history.csv", index=False)
        pd.DataFrame(rew_hist).to_csv(out_dir / "rewired_training_history.csv", index=False)
        del rew
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        bio = build_student(dtype, device, cfg, bio_adj, args.seed)
        load_attention_state(bio, bio_state)

    assert_flydelta_replacement(bio)
    test_ids = eval_batches[0][:, :-1].to(device)
    stream_diff = streaming_equivalence(bio, test_ids, device, dtype)

    print("STAGE benchmarking teacher and FlyDelta-Hybrid", flush=True)
    tbench = benchmark_teacher(teacher, test_ids, device)
    fbench = benchmark_flydelta(bio, test_ids, device)

    print("STAGE generating qualitative samples", flush=True)
    samples = generate_samples(bio, tokenizer, device, dtype)
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    report = {
        "architecture": "FlyDelta-Hybrid v2: exact/local anchors + FlyWire-gated delta memory",
        "base_model": BASE_MODEL,
        "standard_llama_attention_modules": 0,
        "streaming_full_max_abs_logit_diff": stream_diff,
        "config": {
            "run_mode": args.run_mode,
            "seq_len": args.seq_len,
            "fly_nodes": args.fly_nodes,
            "fly_edges": edge_count,
            "local_window": args.local_window,
            "anchor_window": args.anchor_window,
            "anchor_every": args.anchor_every,
            "graph_steps": args.graph_steps,
            "hybrid_layers": hybrids,
            "anchor_layers": anchors,
            "calibration_steps_per_group": calib_steps,
            "global_train_updates": train_updates,
            "grad_accum": grad_accum,
        },
        "biological": bio_metrics,
        "rewired": rewired_metrics,
        "benchmark_teacher": tbench,
        "benchmark_biological": fbench,
        "fly_ce_gap_vs_smollm2": bio_metrics["ce"] - bio_metrics["teacher_ce"],
        "fly_ppl_ratio_vs_smollm2": bio_metrics["perplexity"] / bio_metrics["teacher_perplexity"],
        "decode_speed_ratio_fly_over_smollm2": fbench["decode_tokens_s"] / tbench["decode_tokens_s"],
        "prefill_memory_ratio_fly_over_smollm2": (
            fbench["prefill_peak_extra_mb"] / max(tbench["prefill_peak_extra_mb"], 1e-9)
        ),
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
        "FlyDelta biological": {
            "ce": bio_metrics["ce"],
            "perplexity": bio_metrics["perplexity"],
            "teacher_kl": bio_metrics["teacher_kl"],
            **fbench,
        },
    }
    if rewired_metrics:
        rows["FlyDelta rewired"] = {
            "ce": rewired_metrics["ce"],
            "perplexity": rewired_metrics["perplexity"],
            "teacher_kl": rewired_metrics["teacher_kl"],
        }

    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(bio_state, out_dir / "biological_flydelta_attention.pt")

    print("\nFLYDELTA-HYBRID CHECK", flush=True)
    print("LlamaAttention remaining: 0", flush=True)
    print(
        f"Hybrid layers: {len(hybrids)} | exact/local anchor layers: {len(anchors)}",
        flush=True,
    )
    print("stream/full max logit diff:", stream_diff, flush=True)
    print("\nSUMMARY", flush=True)
    print(pd.DataFrame(rows).T, flush=True)
    print("\nKEY REPORT", flush=True)
    keys = (
        "fly_ce_gap_vs_smollm2",
        "fly_ppl_ratio_vs_smollm2",
        "decode_speed_ratio_fly_over_smollm2",
        "prefill_memory_ratio_fly_over_smollm2",
        "biological_topology_ce_gain",
        "biological_topology_ppl_gain_pct",
    )
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=2), flush=True)

    print("\nSAMPLES", flush=True)
    for item in samples:
        print("=" * 88, flush=True)
        print("PROMPT:", item["prompt"], flush=True)
        print(item["text"], flush=True)

    print("\nSaved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
