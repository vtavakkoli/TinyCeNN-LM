#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_cennmixer_v2 import (
    CeNNMixerV2,
    CeNNMixerV2Config,
    clone_cenn_state_v2,
    direct_mixer_loss_v2,
    finalize_cenn_only_v2,
    freeze_all_except_cenn_v2,
    install_cenn_mixer_v2,
    load_cenn_state_v2,
    reset_stream_state_v2,
    set_alpha_v2,
)


def parse_args():
    p = argparse.ArgumentParser(description="CeNNMixer-v2 progressive Qwen3.5 mixer distillation")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--layers", default="0", help="comma-separated Qwen decoder layer indices")
    p.add_argument("--alphas", default="0,0.05,0.10,0.25,0.50,0.75,1.0")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--train-blocks", type=int, default=768)
    p.add_argument("--val-blocks", type=int, default=32)

    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--cell-dim", type=int, default=48)
    p.add_argument("--graph-steps", type=int, default=1)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--stage-updates", type=int, default=300)
    p.add_argument("--extend-updates", type=int, default=200)
    p.add_argument("--max-stage-updates", type=int, default=3000)
    p.add_argument("--probe-every", type=int, default=50)
    p.add_argument("--patience-probes", type=int, default=8)
    p.add_argument("--min-lr", type=float, default=1.25e-5)

    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--min-top1", type=float, default=0.97)
    p.add_argument("--max-kl", type=float, default=0.03)
    p.add_argument("--max-hidden-mse", type=float, default=0.05)
    p.add_argument("--max-mixer-mse", type=float, default=0.12)

    p.add_argument("--output-dir", default="results/cennmixer_v2_qwen35_08b")
    p.add_argument("--quick-smoke", action="store_true", help="Fast smoke test with reduced data/steps while preserving v2 logic")
    p.add_argument("--seed", type=int, default=8621)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_for(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def blocks(tokenizer, texts, n, seq_len, seed):
    rng = random.Random(seed)
    clean = [x.strip() for x in texts if isinstance(x, str) and len(x.strip()) > 40]
    if not clean:
        raise RuntimeError("No usable text rows")
    out = []
    while len(out) < n:
        s = " ".join(rng.choice(clean) for _ in range(10))
        ids = tokenizer(s, return_tensors="pt", truncation=False).input_ids[0]
        if ids.numel() < seq_len + 1:
            continue
        mx = ids.numel() - (seq_len + 1)
        st = rng.randint(0, mx) if mx > 0 else 0
        out.append(ids[st:st + seq_len + 1].unsqueeze(0))
    return out


def load_data(tokenizer, a):
    from datasets import load_dataset
    tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    va = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    train = blocks(tokenizer, [x["text"] for x in tr], a.train_blocks, a.seq_len, a.seed + 11)
    val = blocks(tokenizer, [x["text"] for x in va], a.val_blocks, a.seq_len, a.seed + 777)
    return train, val


def causal_ce(logits, y):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


def distill_kl(student, teacher):
    s = F.log_softmax(student.float(), dim=-1)
    t = F.softmax(teacher.float(), dim=-1)
    return F.kl_div(s, t, reduction="batchmean") / max(student.shape[1], 1)


def rel_mse(student, teacher):
    t = teacher.float()
    s = student.float()
    den = t.square().mean().clamp_min(1e-12)
    return (s - t).square().mean() / den


def hidden_losses(student_hidden, teacher_hidden, layers):
    hm = student_hidden[0].new_zeros((), dtype=torch.float32)
    dm = student_hidden[0].new_zeros((), dtype=torch.float32)
    for li in layers:
        s = student_hidden[li + 1]
        t = teacher_hidden[li + 1]
        hm = hm + rel_mse(s, t)
        if s.shape[1] > 1:
            dm = dm + rel_mse(s[:, 1:] - s[:, :-1], t[:, 1:] - t[:, :-1])
    n = max(len(layers), 1)
    return hm / n, dm / n


def topk_rank_loss(student_logits, teacher_logits, k=64):
    """Preserve teacher's most competitive next-token ordering and margins."""
    k = min(int(k), int(teacher_logits.shape[-1]))
    with torch.no_grad():
        # Take top-k before casting the whole vocabulary tensor to fp32 to keep
        # memory use modest on Colab GPUs.
        idx = teacher_logits.topk(k, dim=-1).indices
        t = teacher_logits.gather(-1, idx).float()
        t = t - t.mean(dim=-1, keepdim=True)
        scale = t.std(dim=-1, keepdim=True).clamp_min(0.25)
        t = t / scale
    s = student_logits.gather(-1, idx).float()
    s = s - s.mean(dim=-1, keepdim=True)
    s = s / scale
    return F.smooth_l1_loss(s, t)


def stage_targets(alpha, a):
    """Tighten output-preservation targets as CeNN takes over."""
    alpha = float(alpha)
    if alpha <= 0.0:
        return {
            "min_top1": 0.999,
            "max_kl": min(a.max_kl, 1e-4),
            "max_hidden_mse": min(a.max_hidden_mse, 1e-4),
            "max_mixer_mse": a.max_mixer_mse,
        }
    return {
        "min_top1": 1.0 - alpha * (1.0 - a.min_top1),
        "max_kl": max(a.max_kl * alpha, 0.002),
        "max_hidden_mse": max(a.max_hidden_mse * alpha, 0.003),
        "max_mixer_mse": a.max_mixer_mse,
    }


def stage_violation(m, alpha, a):
    t = stage_targets(alpha, a)
    return (
        max(t["min_top1"] - m["top1"], 0.0) / max(1.0 - t["min_top1"], 1e-5)
        + max(m["kl"] - t["max_kl"], 0.0) / max(t["max_kl"], 1e-6)
        + max(m["hidden_mse"] - t["max_hidden_mse"], 0.0) / max(t["max_hidden_mse"], 1e-6)
        + max(m["mixer_mse"] - t["max_mixer_mse"], 0.0) / max(t["max_mixer_mse"], 1e-6)
    )


def stage_ok(m, alpha, a):
    t = stage_targets(alpha, a)
    return (
        m["top1"] >= t["min_top1"]
        and m["kl"] <= t["max_kl"]
        and m["hidden_mse"] <= t["max_hidden_mse"]
        and m["mixer_mse"] <= t["max_mixer_mse"]
    )


def quality_key(m, alpha, a):
    return (
        0 if stage_ok(m, alpha, a) else 1,
        stage_violation(m, alpha, a),
        m["kl"],
        1.0 - m["top1"],
        m["hidden_mse"],
        m["mixer_mse"],
        m["student_ce"],
    )


@torch.no_grad()
def probe(student, teacher, batches, layers, device):
    student.eval()
    teacher.eval()
    totals = {
        "teacher_ce": 0.0,
        "student_ce": 0.0,
        "kl": 0.0,
        "hidden_mse": 0.0,
        "delta_mse": 0.0,
        "mixer_mse": 0.0,
    }
    top = 0
    count = 0

    for cpu in batches:
        ids = cpu.to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        to = teacher(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
        so = student(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
        hm, dm = hidden_losses(so.hidden_states, to.hidden_states, layers)

        totals["teacher_ce"] += float(causal_ce(to.logits, y))
        totals["student_ce"] += float(causal_ce(so.logits, y))
        totals["kl"] += float(distill_kl(so.logits, to.logits))
        totals["hidden_mse"] += float(hm)
        totals["delta_mse"] += float(dm)
        totals["mixer_mse"] += float(direct_mixer_loss_v2(student))

        top += int((so.logits.argmax(-1) == to.logits.argmax(-1)).sum())
        count += int(so.logits.shape[0] * so.logits.shape[1])

    n = max(len(batches), 1)
    for k in totals:
        totals[k] /= n
    totals["ce_gap"] = totals["student_ce"] - totals["teacher_ce"]
    totals["top1"] = top / max(count, 1)
    return totals


@torch.no_grad()
def probe_without_local_teacher(student, teacher, batches, layers, device):
    student.eval()
    teacher.eval()
    totals = {
        "teacher_ce": 0.0,
        "student_ce": 0.0,
        "kl": 0.0,
        "hidden_mse": 0.0,
        "delta_mse": 0.0,
    }
    top = 0
    count = 0
    for cpu in batches:
        ids = cpu.to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        to = teacher(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
        so = student(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
        hm, dm = hidden_losses(so.hidden_states, to.hidden_states, layers)
        totals["teacher_ce"] += float(causal_ce(to.logits, y))
        totals["student_ce"] += float(causal_ce(so.logits, y))
        totals["kl"] += float(distill_kl(so.logits, to.logits))
        totals["hidden_mse"] += float(hm)
        totals["delta_mse"] += float(dm)
        top += int((so.logits.argmax(-1) == to.logits.argmax(-1)).sum())
        count += int(so.logits.shape[0] * so.logits.shape[1])
    n = max(len(batches), 1)
    for k in totals:
        totals[k] /= n
    totals["ce_gap"] = totals["student_ce"] - totals["teacher_ce"]
    totals["top1"] = top / max(count, 1)
    return totals


@torch.no_grad()
def generation_suite(student, teacher, tokenizer, device):
    prompts = [
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "Explain in one sentence what an API is.",
        "Translate 'Good morning' into German. Give only the translation.",
        "The secret number is 81427. Remember it. What is the secret number? Give only the number.",
    ]
    rows = []
    for p in prompts:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(chat, return_tensors="pt").to(device)

        def run(m):
            reset_stream_state_v2(m)
            y = m.generate(
                **enc,
                max_new_tokens=48,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            z = y[0, enc.input_ids.shape[1]:]
            return z.tolist(), tokenizer.decode(z, skip_special_tokens=True).strip()

        ti, tt = run(teacher)
        si, st = run(student)
        aa, bb = set(ti), set(si)
        rows.append({
            "prompt": p,
            "qwen": tt,
            "cenn": st,
            "exact": ti == si,
            "jaccard": len(aa & bb) / max(len(aa | bb), 1),
        })
    return rows


def main():
    a = parse_args()
    if a.quick_smoke:
        # Fast sanity-check defaults. User-supplied values are intentionally
        # overridden so the mode stays genuinely quick and reproducible.
        a.alphas = "0,0.50,1.0"
        a.seq_len = 32
        a.train_blocks = 64
        a.val_blocks = 4
        a.stage_updates = 40
        a.extend_updates = 30
        a.max_stage_updates = 120
        a.probe_every = 20
        a.patience_probes = 4
        a.topk = 16
        print("QUICK_SMOKE enabled:", {
            "alphas": a.alphas,
            "seq_len": a.seq_len,
            "train_blocks": a.train_blocks,
            "val_blocks": a.val_blocks,
            "stage_updates": a.stage_updates,
            "max_stage_updates": a.max_stage_updates,
            "probe_every": a.probe_every,
            "topk": a.topk,
        }, flush=True)
    set_seed(a.seed)

    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    if not alphas or alphas[0] != 0.0 or alphas[-1] != 1.0:
        raise ValueError("--alphas must start at 0 and end at 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_for(device)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("DEVICE", device, "dtype", dtype, "layers", layers, "alphas", alphas, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(a.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        a.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(
        a.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = CeNNMixerV2Config(
        hidden_size=int(student.config.hidden_size),
        groups=a.groups,
        cell_dim=a.cell_dim,
        graph_steps=a.graph_steps,
    )

    layer_kinds = {}
    for li in layers:
        layer = student.model.layers[li]
        layer_kinds[str(li)] = getattr(layer, "block_type", "unknown")
        install_cenn_mixer_v2(student, li, cfg)

    train, val = load_data(tokenizer, a)
    trainable = freeze_all_except_cenn_v2(student)

    replaced_params = 0
    for li in layers:
        tl = teacher.model.layers[li]
        mod = tl.linear_attn if getattr(tl, "block_type", None) == "linear_attention" else tl.self_attn
        replaced_params += sum(p.numel() for p in mod.parameters())
    cenn_params = sum(p.numel() for p in trainable)

    print("CeNN params:", cenn_params, "Qwen mixer params:", replaced_params, flush=True)

    history = []
    stages = []
    global_step = 0
    previous_best = clone_cenn_state_v2(student)

    for alpha in alphas:
        load_cenn_state_v2(student, previous_best)
        set_alpha_v2(student, alpha)
        reset_stream_state_v2(student)

        trainable = freeze_all_except_cenn_v2(student)
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)

        initial = probe(student, teacher, val, layers, device)
        best_m = dict(initial)
        best_state = clone_cenn_state_v2(student)
        best_key = quality_key(initial, alpha, a)
        best_step = 0
        stage_steps = 0
        target_budget = a.stage_updates
        no_improve = 0

        print("\n=== ALPHA", alpha, "===", flush=True)
        print("TARGETS", json.dumps(stage_targets(alpha, a)), flush=True)
        print("INITIAL", json.dumps(initial), flush=True)

        while stage_steps < a.max_stage_updates:
            while stage_steps < min(target_budget, a.max_stage_updates):
                global_step += 1
                stage_steps += 1
                student.train()

                ids = train[(global_step - 1) % len(train)].to(device)
                x, y = ids[:, :-1], ids[:, 1:]

                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    to = teacher(
                        input_ids=x,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                so = student(
                    input_ids=x,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

                mixer = direct_mixer_loss_v2(student)
                hm, dm = hidden_losses(so.hidden_states, to.hidden_states, layers)
                lkl = distill_kl(so.logits, to.logits)
                rank = topk_rank_loss(so.logits, to.logits, a.topk)
                lce = causal_ce(so.logits, y)

                # Mixer supervision is alpha-independent and therefore trains CeNN
                # even when alpha=0 and the externally visible model is exact Qwen.
                loss = (
                    0.35 * mixer
                    + 0.15 * hm
                    + 0.10 * dm
                    + 0.20 * lkl
                    + 0.15 * rank
                    + 0.05 * lce
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()

                if stage_steps == 1 or stage_steps % a.probe_every == 0:
                    vm = probe(student, teacher, val, layers, device)
                    row = {
                        "global_step": global_step,
                        "alpha": alpha,
                        "stage_step": stage_steps,
                        "train_loss": float(loss.detach()),
                        "train_mixer_mse": float(mixer.detach()),
                        "train_kl": float(lkl.detach()),
                        "train_rank_loss": float(rank.detach()),
                        **vm,
                    }
                    history.append(row)
                    print("PROBE", json.dumps(row), flush=True)

                    key = quality_key(vm, alpha, a)
                    if key < best_key:
                        best_key = key
                        best_m = dict(vm)
                        best_state = clone_cenn_state_v2(student)
                        best_step = stage_steps
                        no_improve = 0
                        ck = {
                            "state": best_state,
                            "config": cfg.to_dict(),
                            "layers": layers,
                            "layer_kinds": layer_kinds,
                            "alpha": alpha,
                            "stage_step": best_step,
                            "metrics": best_m,
                        }
                        torch.save(
                            ck,
                            out / f"cennmixer_v2_alpha_{str(alpha).replace('.', 'p')}_best.pt",
                        )
                        print("✓ NEW BEST", json.dumps({
                            "alpha": alpha,
                            "step": best_step,
                            "violation": stage_violation(best_m, alpha, a),
                            "pass": stage_ok(best_m, alpha, a),
                        }), flush=True)
                    else:
                        no_improve += 1

                    if stage_ok(best_m, alpha, a):
                        print("✓ STAGE TARGET FULFILLED", alpha, "at", best_step, flush=True)
                        break

            if stage_ok(best_m, alpha, a):
                break
            if stage_steps >= a.max_stage_updates:
                print("! MAX STAGE UPDATES reached for alpha", alpha, flush=True)
                break

            if no_improve >= a.patience_probes:
                for g in opt.param_groups:
                    g["lr"] = max(float(g["lr"]) * 0.5, a.min_lr)
                no_improve = 0
                print("↘ plateau: LR ->", opt.param_groups[0]["lr"], flush=True)

            old = target_budget
            target_budget = min(target_budget + a.extend_updates, a.max_stage_updates)
            print("↻ extending alpha", alpha, old, "->", target_budget, flush=True)

        load_cenn_state_v2(student, best_state)
        previous_best = clone_cenn_state_v2(student)
        set_alpha_v2(student, alpha)
        restored = probe(student, teacher, val, layers, device)
        gens = generation_suite(student, teacher, tokenizer, device)

        stage_row = {
            "alpha": alpha,
            "best_step": best_step,
            "trained_steps": stage_steps,
            "pass": stage_ok(restored, alpha, a),
            "violation": stage_violation(restored, alpha, a),
            **restored,
            "generation_exact_rate": sum(int(x["exact"]) for x in gens) / max(len(gens), 1),
            "generation_mean_jaccard": sum(x["jaccard"] for x in gens) / max(len(gens), 1),
        }
        stages.append(stage_row)
        print("RESTORED BEST", json.dumps(stage_row), flush=True)

        (out / f"generation_alpha_{str(alpha).replace('.', 'p')}.json").write_text(
            json.dumps(gens, indent=2), encoding="utf-8"
        )

    # Final alpha=1 candidate while the local Qwen mixer is still available.
    set_alpha_v2(student, 1.0)
    alpha1_with_teacher = probe(student, teacher, val, layers, device)
    alpha1_generations = generation_suite(student, teacher, tokenizer, device)

    # Now physically remove the wrapped Qwen mixer from replaced layers.
    finalize_cenn_only_v2(student)
    reset_stream_state_v2(student)
    compact_probe = probe_without_local_teacher(student, teacher, val, layers, device)
    compact_generations = generation_suite(student, teacher, tokenizer, device)

    torch.save(
        {
            "state": clone_cenn_state_v2(student),
            "config": cfg.to_dict(),
            "layers": layers,
            "layer_kinds": layer_kinds,
            "base_model": a.base_model,
            "alpha": 1.0,
            "cenn_only": True,
        },
        out / "cennmixer_v2_final_cenn_only.pt",
    )

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    pd.DataFrame(stages).to_csv(out / "stage_summary.csv", index=False)

    strict_quality = (
        compact_probe["top1"] >= a.min_top1
        and compact_probe["kl"] <= a.max_kl
        and compact_probe["hidden_mse"] <= a.max_hidden_mse
    )

    report = {
        "architecture": "CeNNMixer-v2 progressive takeover",
        "base_model": a.base_model,
        "layers": layers,
        "layer_kinds": layer_kinds,
        "alpha_schedule": alphas,
        "config": cfg.to_dict(),
        "loss": {
            "mixer_mse": 0.35,
            "hidden_mse": 0.15,
            "delta_hidden_mse": 0.10,
            "logit_kl": 0.20,
            "topk_rank": 0.15,
            "causal_ce": 0.05,
            "topk": a.topk,
        },
        "cenn_params": cenn_params,
        "replaced_qwen_mixer_params": replaced_params,
        "mixer_param_reduction_pct": 100.0 * (1.0 - cenn_params / max(replaced_params, 1)),
        "stage_summary": stages,
        "alpha1_before_removing_qwen_mixer": alpha1_with_teacher,
        "alpha1_generation_before_removal": alpha1_generations,
        "final_cenn_only_probe": compact_probe,
        "final_cenn_only_generation": compact_generations,
        "strict_quality_gate": strict_quality,
        "thresholds": {
            "min_top1": a.min_top1,
            "max_kl": a.max_kl,
            "max_hidden_mse": a.max_hidden_mse,
            "max_mixer_mse": a.max_mixer_mse,
        },
        "args": vars(a),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
