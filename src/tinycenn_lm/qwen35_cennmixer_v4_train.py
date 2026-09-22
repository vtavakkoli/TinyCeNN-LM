from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from .qwen35_cennmixer_v4 import direct_mixer_losses_v4, reset_stream_state_v4


def _token_stream(tokenizer, texts, batch_size=128):
    """Tokenize documents independently, then join with EOS boundaries."""
    eos = tokenizer.eos_token_id
    pieces = []
    clean = [x.strip() for x in texts if isinstance(x, str) and x.strip()]
    for start in range(0, len(clean), batch_size):
        enc = tokenizer(
            clean[start:start + batch_size],
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        for ids in enc:
            if ids:
                pieces.extend(ids)
                if eos is not None:
                    pieces.append(eos)
    if not pieces:
        raise ValueError("No usable tokens in corpus")
    return torch.tensor(pieces, dtype=torch.long)


def _windows(stream, n, seq_len, seed):
    if n < 1 or seq_len < 32:
        raise ValueError("n >= 1 and seq_len >= 32 required")
    if stream.numel() < seq_len + 1:
        raise ValueError(f"Corpus has {stream.numel()} tokens, needs {seq_len + 1}")
    rng = random.Random(seed)
    hi = stream.numel() - seq_len - 1
    return [
        stream[rng.randint(0, hi):][:seq_len + 1].unsqueeze(0)
        for _ in range(n)
    ]


def context_lengths(a):
    vals = sorted(set(int(x) for x in str(a.context_lengths).split(",") if x.strip()))
    if not vals:
        vals = [a.seq_len]
    if vals[0] < 32 or vals[-1] > a.seq_len:
        raise ValueError("context lengths must be >= 32 and <= seq-len")
    if a.seq_len not in vals:
        vals.append(a.seq_len)
    return sorted(set(vals))


def load_data(tokenizer, a):
    from datasets import load_dataset

    tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    va = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

    tr_stream = _token_stream(tokenizer, [x["text"] for x in tr])
    va_stream = _token_stream(tokenizer, [x["text"] for x in va])

    train = _windows(tr_stream, a.train_blocks, a.seq_len, a.seed + 11)
    validation = {
        length: _windows(va_stream, a.val_blocks, length, a.seed + 777 + length)
        for length in context_lengths(a)
    }
    return train, validation


def _flat_chunks(student, teacher, fn, chunk_size=16):
    s = student.reshape(-1, student.shape[-1])
    t = teacher.reshape(-1, teacher.shape[-1])
    total = s.new_zeros((), dtype=torch.float32)
    count = max(s.shape[0], 1)
    for st in range(0, s.shape[0], chunk_size):
        ss = s[st:st + chunk_size]
        tt = t[st:st + chunk_size]
        total = total + fn(ss, tt) * (ss.shape[0] / count)
    return total


def causal_ce(logits, targets, chunk_size=16):
    s = logits.reshape(-1, logits.shape[-1])
    y = targets.reshape(-1)
    total = s.new_zeros((), dtype=torch.float32)
    count = max(s.shape[0], 1)
    for st in range(0, s.shape[0], chunk_size):
        ss = s[st:st + chunk_size].float()
        yy = y[st:st + chunk_size]
        total = total + F.cross_entropy(ss, yy) * (ss.shape[0] / count)
    return total


def forward_kl(student, teacher, temperature=1.5):
    T = float(temperature)

    def loss(s, t):
        ls = F.log_softmax(s.float() / T, -1)
        lt = F.log_softmax(t.float() / T, -1)
        return (lt.exp() * (lt - ls)).sum(-1).mean() * (T * T)

    return _flat_chunks(student, teacher, loss)


def reverse_kl(student, teacher, temperature=1.5):
    T = float(temperature)

    def loss(s, t):
        ls = F.log_softmax(s.float() / T, -1)
        lt = F.log_softmax(t.float() / T, -1)
        return (ls.exp() * (ls - lt)).sum(-1).mean() * (T * T)

    return _flat_chunks(student, teacher, loss)


def rel_mse(student, teacher):
    t = teacher.float()
    s = student.float()
    return (s - t).square().mean() / t.square().mean().clamp_min(1e-12)


def hidden_losses(student_hidden, teacher_hidden, layer):
    # One replacement layer: compare the decoder state immediately after it and
    # its temporal derivative.  Later layers are already constrained by logits.
    s = student_hidden[layer + 1]
    t = teacher_hidden[layer + 1]
    hm = rel_mse(s, t)
    if s.shape[1] > 1:
        dm = rel_mse(s[:, 1:] - s[:, :-1], t[:, 1:] - t[:, :-1])
    else:
        dm = hm.new_zeros(())
    return hm, dm


def topk_rank_loss(student_logits, teacher_logits, k):
    k = min(int(k), int(teacher_logits.shape[-1]))
    with torch.no_grad():
        idx = teacher_logits.topk(k, dim=-1).indices
        t = teacher_logits.gather(-1, idx).float()
        t = t - t.mean(-1, keepdim=True)
        scale = t.std(-1, keepdim=True, correction=0).clamp_min(0.25)
        t = t / scale
    s = student_logits.gather(-1, idx).float()
    s = (s - s.mean(-1, keepdim=True)) / scale
    return F.smooth_l1_loss(s, t)


def top1_margin_loss(student_logits, teacher_logits):
    s = student_logits.float()
    t = teacher_logits.float()
    with torch.no_grad():
        tv, ti = t.topk(2, dim=-1)
        target = ti[..., 0]
        teacher_gap = (tv[..., 0] - tv[..., 1]).clamp(0.05, 1.5)
    target_logit = s.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    sv, si = s.topk(2, dim=-1)
    other = torch.where(si[..., 0].eq(target), sv[..., 1], sv[..., 0])
    return F.relu(teacher_gap + other - target_logit).mean()


def select_positions(logits, targets=None, max_tokens=128):
    n = logits.shape[1]
    keep = min(int(max_tokens), n)
    if keep >= n:
        return logits, targets
    idx = torch.linspace(0, n - 1, steps=keep, device=logits.device).round().long().unique()
    out = logits.index_select(1, idx)
    if targets is None:
        return out, None
    return out, targets.index_select(1, idx)


def quality_targets(a):
    return {
        "min_top1": a.min_top1,
        "max_kl": a.max_kl,
        "max_hidden_mse": a.max_hidden_mse,
        "max_mixer_mse": a.max_mixer_mse,
        "max_mixer_cosine": a.max_mixer_cosine,
        "max_mixer_delta": a.max_mixer_delta,
        "max_mixer_multiscale": a.max_mixer_multiscale,
        "max_ce_gap": a.max_ce_gap,
    }


def quality_ok(m, a):
    t = quality_targets(a)
    needed = (
        "top1", "kl", "hidden_mse", "mixer_mse", "mixer_cosine",
        "mixer_delta", "mixer_multiscale", "ce_gap", "generation_ok",
    )
    if any(k not in m or not math.isfinite(float(m[k])) for k in needed):
        return False
    return (
        m["generation_ok"] == 1
        and abs(m["ce_gap"]) <= t["max_ce_gap"]
        and m["top1"] >= t["min_top1"]
        and m["kl"] <= t["max_kl"]
        and m["hidden_mse"] <= t["max_hidden_mse"]
        and m["mixer_mse"] <= t["max_mixer_mse"]
        and m["mixer_cosine"] <= t["max_mixer_cosine"]
        and m["mixer_delta"] <= t["max_mixer_delta"]
        and m["mixer_multiscale"] <= t["max_mixer_multiscale"]
    )


def quality_violation(m, a):
    t = quality_targets(a)
    if any(not math.isfinite(float(v)) for v in m.values() if isinstance(v, (int, float))):
        return float("inf")
    return (
        (1.0 - m.get("generation_ok", 1.0))
        + max(abs(m.get("ce_gap", 0.0)) - t["max_ce_gap"], 0.0) / max(t["max_ce_gap"], 1e-6)
        + max(t["min_top1"] - m.get("top1", 0.0), 0.0) / max(1.0 - t["min_top1"], 1e-5)
        + max(m.get("kl", 1e9) - t["max_kl"], 0.0) / max(t["max_kl"], 1e-6)
        + max(m.get("hidden_mse", 1e9) - t["max_hidden_mse"], 0.0) / max(t["max_hidden_mse"], 1e-6)
        + max(m.get("mixer_mse", 1e9) - t["max_mixer_mse"], 0.0) / max(t["max_mixer_mse"], 1e-6)
        + max(m.get("mixer_cosine", 1e9) - t["max_mixer_cosine"], 0.0) / max(t["max_mixer_cosine"], 1e-6)
        + max(m.get("mixer_delta", 1e9) - t["max_mixer_delta"], 0.0) / max(t["max_mixer_delta"], 1e-6)
        + max(m.get("mixer_multiscale", 1e9) - t["max_mixer_multiscale"], 0.0)
        / max(t["max_mixer_multiscale"], 1e-6)
    )


def quality_key(m, a):
    return (
        0 if quality_ok(m, a) else 1,
        quality_violation(m, a),
        m.get("mixer_mse", 1e9),
        m.get("kl", 1e9),
        1.0 - m.get("top1", 0.0),
        m.get("hidden_mse", 1e9),
    )


@torch.no_grad()
def probe(student, teacher, batches, layer, device, local_teacher=True, logit_tokens=128):
    student.eval()
    teacher.eval()
    totals = {
        "teacher_ce": 0.0,
        "student_ce": 0.0,
        "kl": 0.0,
        "reverse_kl": 0.0,
        "hidden_mse": 0.0,
        "delta_mse": 0.0,
        "top1_margin_loss": 0.0,
    }
    if local_teacher:
        totals.update({
            "mixer_mse": 0.0,
            "mixer_cosine": 0.0,
            "mixer_delta": 0.0,
            "mixer_multiscale": 0.0,
            "mixer_rms": 0.0,
            "mixer_tail": 0.0,
        })

    top = 0
    count = 0
    for cpu in batches:
        ids = cpu.to(device)
        x, y = ids[:, :-1], ids[:, 1:]

        to = teacher(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
        so = student(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)

        hm, dm = hidden_losses(so.hidden_states, to.hidden_states, layer)
        sl, sy = select_positions(so.logits, y, logit_tokens)
        tl, _ = select_positions(to.logits, None, logit_tokens)

        totals["teacher_ce"] += float(causal_ce(tl, sy))
        totals["student_ce"] += float(causal_ce(sl, sy))
        totals["kl"] += float(forward_kl(sl, tl))
        totals["reverse_kl"] += float(reverse_kl(sl, tl))
        totals["hidden_mse"] += float(hm)
        totals["delta_mse"] += float(dm)
        totals["top1_margin_loss"] += float(top1_margin_loss(sl, tl))

        if local_teacher:
            loc = direct_mixer_losses_v4(student)
            totals["mixer_mse"] += float(loc["mse"])
            totals["mixer_cosine"] += float(loc["cosine"])
            totals["mixer_delta"] += float(loc["delta"])
            totals["mixer_multiscale"] += float(loc["multiscale"])
            totals["mixer_rms"] += float(loc["rms"])
            totals["mixer_tail"] += float(loc["tail"])

        top += int((so.logits.argmax(-1) == to.logits.argmax(-1)).sum())
        count += so.logits.shape[0] * so.logits.shape[1]

    n = max(len(batches), 1)
    for k in totals:
        totals[k] /= n
    totals["ce_gap"] = totals["student_ce"] - totals["teacher_ce"]
    totals["top1"] = top / max(count, 1)
    return totals


def probe_contexts(student, teacher, validation, layer, device, local_teacher=True, logit_tokens=128):
    per = {
        str(n): probe(student, teacher, b, layer, device, local_teacher, logit_tokens)
        for n, b in validation.items()
    }
    first = next(iter(per.values()))
    worst = {}
    for k in first:
        vals = [m[k] for m in per.values()]
        if not all(math.isfinite(float(v)) for v in vals):
            worst[k] = float("nan")
        elif k == "top1":
            worst[k] = min(vals)
        elif k == "ce_gap":
            worst[k] = max(vals, key=lambda v: abs(v))
        elif k == "teacher_ce":
            worst[k] = max(vals)
        else:
            worst[k] = max(vals)
    return worst, per


def retrieval_prompt(tokenizer, length, seed):
    rng = random.Random(seed)
    key = f"{rng.randrange(10000, 100000)}"
    fillers = [
        f"Archive entry {i}: the parcel is stored on shelf {rng.randrange(100)}."
        for i in range(max(12, length // 8))
    ]
    pos = rng.randrange(max(1, len(fillers) // 2))
    fillers.insert(pos, f"The access code is {key}.")
    prefix = "\n".join(fillers)
    ids = tokenizer(prefix, add_special_tokens=False).input_ids
    prefix = tokenizer.decode(ids[:max(16, length - 96)], skip_special_tokens=True)
    if key not in prefix:
        prefix = f"The access code is {key}.\n" + prefix
    return prefix + "\nWhat is the access code? Reply with the code only.", key


def chat_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )


def repetition_rate(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n + 1))]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def generation_suite(student, teacher, tok, device, context_sizes=(1024,), seed=9001):
    prompts = [
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "Explain in one sentence what an API is.",
        "Translate 'Good morning' into German. Give only the translation.",
        "The secret number is 81427. Remember it. What is the secret number? Give only the number.",
    ]
    cases = [(p, None) for p in prompts]
    cases[0] = (prompts[0], "42")
    cases[-1] = (prompts[-1], "81427")
    cases.extend(retrieval_prompt(tok, n, seed + i) for i, n in enumerate(context_sizes))

    rows = []
    for p, expected in cases:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tok(chat, return_tensors="pt").to(device)

        def run(m):
            reset_stream_state_v4(m)
            y = m.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
            z = y[0, enc.input_ids.shape[1]:]
            return z.tolist(), tok.decode(z, skip_special_tokens=True).strip()

        ti, tt = run(teacher)
        si, st = run(student)
        a, b = set(ti), set(si)
        rows.append({
            "prompt": p,
            "qwen": tt,
            "cenn": st,
            "exact": ti == si,
            "jaccard": len(a & b) / max(len(a | b), 1),
            "expected": expected,
            "teacher_correct": expected is None or tt.strip() == expected,
            "student_correct": expected is None or st.strip() == expected,
            "repetition": repetition_rate(si),
            "prompt_tokens": int(enc.input_ids.shape[1]),
        })
    return rows


def generation_health(rows):
    scored = [r for r in rows if r.get("expected") is not None and r.get("teacher_correct")]
    return (
        bool(scored)
        and all(r["student_correct"] for r in scored)
        and all(bool(r["cenn"].strip()) and r["repetition"] < 0.5 for r in rows)
    )


def on_policy_distill_loss(student, teacher, prefix, device, max_new_tokens=16, logit_tokens=64):
    """Teacher feedback on a prefix actually visited by the active compact model."""
    was_training = student.training
    student.eval()
    reset_stream_state_v4(student)
    with torch.no_grad():
        gen = student.generate(
            input_ids=prefix.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(student.config, "eos_token_id", None),
        )
    reset_stream_state_v4(student)

    x = gen[:, :-1]
    with torch.no_grad():
        to = teacher(input_ids=x, use_cache=False, return_dict=True)
    student.train(was_training)
    so = student(input_ids=x, use_cache=False, return_dict=True)

    start = max(prefix.shape[1] - 1, 0)
    sl = so.logits[:, start:]
    tl = to.logits[:, start:]
    sl, _ = select_positions(sl, None, logit_tokens)
    tl, _ = select_positions(tl, None, logit_tokens)
    return (
        0.45 * reverse_kl(sl, tl)
        + 0.35 * forward_kl(sl, tl)
        + 0.20 * top1_margin_loss(sl, tl)
    )
