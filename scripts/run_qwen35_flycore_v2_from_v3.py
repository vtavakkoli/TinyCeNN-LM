#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import re
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
from huggingface_hub import list_repo_files, snapshot_download

from tinycenn_lm.qwen35_flycore_v2 import (
    FlyVocabV2Config,
    assert_qwen35_flycore_v2,
    choose_hot_tokens,
    factorize_embedding_weight_v2,
    freeze_source_ffn_train_vocab,
    install_fly_vocab_v2,
)
from tinycenn_lm.qwen35_flycore_v2_standalone import (
    export_flycore_v2_standalone,
    upload_flycore_v2_standalone,
)
from tinycenn_lm.qwen35_flyffn_v3 import FlyFFNV3Config, routing_schedule
from tinycenn_lm.qwen35_standalone import load_standalone

_HELPER_PATH = Path(__file__).with_name("run_qwen35_flycore_v1.py")
_spec = importlib.util.spec_from_file_location("flycore_v1_helper", _HELPER_PATH)
helper = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(helper)
base = helper.base

DEFAULT_SOURCE_REPO = "vtava/Qwen35-0.8B-FlyFFN-v3-AllFFN"


def parse_args():
    p = argparse.ArgumentParser(
        description="FlyCore-v2 from trained FlyFFN-v3 standalone"
    )
    p.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO)
    p.add_argument("--source-dir", default="/content/qwen35_v3_source")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--vocab-latent-dim", type=int, default=768)
    p.add_argument("--vocab-graph-mix-init", type=float, default=0.10)
    p.add_argument("--max-fly-scale", type=float, default=0.10)
    p.add_argument("--hot-token-count", type=int, default=8192)
    p.add_argument("--factor-chunk-rows", type=int, default=4096)
    p.add_argument("--output-dir", default="results/flycore_v2_from_v3_qwen35_08b")
    p.add_argument("--standalone-dir", default=None)
    p.add_argument("--upload-hf", action="store_true")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--seed", type=int, default=8621)
    p.add_argument("--max-final-ce-gap", type=float, default=0.15)
    return p.parse_args()


def download_source(repo_id: str, local_dir: str | Path) -> Path:
    local_dir = Path(local_dir)
    token = os.environ.get("HF_TOKEN") or None
    files = list_repo_files(repo_id, repo_type="model", token=token)
    required = {"standalone_state.pt", "flyffn_config.json", "config.json"}
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(f"source standalone missing required files: {missing}")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=[
            "standalone_state.pt",
            "standalone_manifest.json",
            "flyffn_config.json",
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "chat_template*",
            "*.jinja",
        ],
        token=token,
    )
    return Path(path)


def source_ffn_config(source_dir: Path) -> FlyFFNV3Config:
    d = json.loads((source_dir / "flyffn_config.json").read_text(encoding="utf-8"))
    return FlyFFNV3Config(
        fly_nodes=int(d["fly_nodes"]),
        router_rank=int(d["router_rank"]),
        num_shards=int(d["num_shards"]),
        graph_steps=int(d["graph_steps"]),
        graph_mix_init=float(d["graph_mix_init"]),
        router_noise_std=float(d.get("router_noise_std", 5e-3)),
    )


@torch.no_grad()
def probe(student, teacher, batches, device, dtype):
    student.eval()
    teacher.eval()
    tce = sce = kl = emb = head_kl = 0.0
    n = 0
    top1_match = 0
    top1_total = 0

    for cpu_ids in batches:
        ids = cpu_ids.to(device)
        x = ids[:, :-1]
        with base.base.amp_ctx(device, dtype):
            t = teacher(
                input_ids=x,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            s = student(
                input_ids=x,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            te = teacher.model.embed_tokens(x)
            se = student.model.embed_tokens(x)

        tce += float(base.base.causal_ce(t.logits, ids))
        sce += float(base.base.causal_ce(s.logits, ids))
        kl += float(base.base.distill_kl(s.logits, t.logits))
        denom = te.float().square().mean().clamp_min(1e-8)
        emb += float(((se.float() - te.float()).square().mean() / denom).item())

        step = max(1, t.logits.shape[1] // 16)
        th = t.hidden_states[-1][:, ::step]
        with base.base.amp_ctx(device, dtype):
            sh = student.lm_head(th)
        head_kl += float(
            base.base.distill_kl(sh, t.logits[:, ::step])
        )

        tp = t.logits.argmax(dim=-1)
        sp = s.logits.argmax(dim=-1)
        top1_match += int((tp == sp).sum().item())
        top1_total += int(tp.numel())
        n += 1

    return {
        "teacher_ce": tce / n,
        "student_ce": sce / n,
        "ce_gap": (sce - tce) / n,
        "teacher_kl": kl / n,
        "embedding_relative_mse": emb / n,
        "head_only_kl": head_kl / n,
        "top1_logit_agreement": top1_match / max(top1_total, 1),
    }


def make_optimizer(student, device):
    core = student.fly_vocab_core_v2
    trainable = freeze_source_ffn_train_vocab(student)
    groups = [
        {"params": [core.codebook.weight], "lr": 1.0e-5, "weight_decay": 0.0},
        {"params": [core.basis], "lr": 3.0e-6, "weight_decay": 0.0},
        {"params": [core.hot_residual], "lr": 2.0e-5, "weight_decay": 0.0},
        {
            "params": [core.fly_down.weight, core.fly_up.weight],
            "lr": 3.0e-5,
            "weight_decay": 0.0,
        },
        {
            "params": [core.fly_scale_raw, core.graph_mix_logit],
            "lr": 8.0e-5,
            "weight_decay": 0.0,
        },
    ]
    return base.base.make_optimizer(groups, device), trainable


def train_vocab_v2(
    student,
    teacher,
    train_batches,
    probe_batches,
    updates,
    device,
    dtype,
):
    opt, trainable = make_optimizer(student, device)
    scaler = base.base.make_scaler(device, dtype)
    core = student.fly_vocab_core_v2
    teacher_weight = teacher.model.embed_tokens.weight.detach()
    history = []
    warmup = max(20, updates // 20)
    initial_lrs = [g["lr"] for g in opt.param_groups]
    t0 = time.perf_counter()

    student.train()
    teacher.eval()

    for update in range(1, updates + 1):
        ids = train_batches[(update - 1) % len(train_batches)].to(device)
        x = ids[:, :-1]
        opt.zero_grad(set_to_none=True)

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

            emb_den = teacher_emb.float().square().mean().clamp_min(1e-8)
            emb = (
                (student_emb.float() - teacher_emb.float()).square().mean()
                / emb_den
            )

            # Isolate LM-head quality on teacher hidden states.
            step = max(1, t.logits.shape[1] // 16)
            teacher_hidden = t.hidden_states[-1][:, ::step]
            head_logits = student.lm_head(teacher_hidden)
            head_kl = base.base.distill_kl(
                head_logits,
                t.logits[:, ::step],
            )

            # Directly preserve frequent vocabulary rows.
            if core.hot_token_ids.numel():
                k = min(512, core.hot_token_ids.numel())
                offset = ((update - 1) * k) % core.hot_token_ids.numel()
                idx = torch.arange(
                    offset,
                    offset + k,
                    device=device,
                ) % core.hot_token_ids.numel()
                hot_ids = core.hot_token_ids[idx]
                pred_hot = core.effective_hot_weight()[idx]
                ref_hot = teacher_weight[hot_ids].to(
                    pred_hot.device, pred_hot.dtype
                )
                hot_den = ref_hot.float().square().mean().clamp_min(1e-8)
                hot_mse = (
                    (pred_hot.float() - ref_hot.float()).square().mean()
                    / hot_den
                )
            else:
                hot_mse = torch.zeros((), device=device)

            scale_reg = core.fly_scale.float().square()

            loss = (
                0.25 * ce
                + 0.35 * kl
                + 0.20 * head_kl
                + 0.15 * emb
                + 0.05 * hot_mse
                + 0.01 * scale_reg
            )

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)

        if update <= warmup:
            mult = max(update / warmup, 1e-3)
        else:
            p = (update - warmup) / max(updates - warmup, 1)
            mult = 0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g, lr0 in zip(opt.param_groups, initial_lrs):
            g["lr"] = lr0 * mult

        scaler.step(opt)
        scaler.update()

        row = {
            "update": update,
            "loss": float(loss.detach()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "head_kl": float(head_kl.detach()),
            "embedding_relative_mse": float(emb.detach()),
            "hot_mse": float(hot_mse.detach()),
            "fly_scale": float(core.fly_scale.detach()),
        }
        history.append(row)

        if update == 1 or update % 25 == 0 or update == updates:
            elapsed = time.perf_counter() - t0
            print(
                f"VOCAB-V2 {update}/{updates} "
                f"loss={row['loss']:.4f} ce={row['ce']:.4f} kl={row['kl']:.4f} "
                f"head={row['head_kl']:.4f} emb={row['embedding_relative_mse']:.4f} "
                f"hot={row['hot_mse']:.4f} scale={row['fly_scale']:.4f} "
                f"ups={update/max(elapsed,1e-9):.3f}",
                flush=True,
            )

    final_probe = probe(student, teacher, probe_batches, device, dtype)
    return final_probe, history


@torch.no_grad()
def generation_sanity(model, tokenizer, device):
    prompts = [
        "Explain in two sentences why the sky is blue.",
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
    ]
    rows = []
    passed = True
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
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
        toks = ids.tolist()
        unique_ratio = len(set(toks)) / max(len(toks), 1)
        longest = 1
        run = 1
        for i in range(1, len(toks)):
            if toks[i] == toks[i - 1]:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        bad = (
            len(toks) >= 20
            and (unique_ratio < 0.12 or longest >= 12)
        ) or not reply
        passed = passed and not bad
        rows.append({
            "prompt": p,
            "reply": reply,
            "tokens": len(toks),
            "unique_token_ratio": unique_ratio,
            "longest_same_token_run": longest,
            "passed": not bad,
        })
    return passed, rows


def main():
    args = parse_args()
    base.base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.base.choose_dtype(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"DEVICE {device} | dtype={dtype} | "
        f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}",
        flush=True,
    )

    source_dir = download_source(args.source_repo_id, args.source_dir)
    teacher, tokenizer = load_standalone(source_dir, device=device, dtype=dtype)
    student, _ = load_standalone(source_dir, device=device, dtype=dtype)
    teacher.eval()
    student.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    source_schedule = routing_schedule(teacher)
    adjacency = teacher.flyffn_shared_graph.adjacency.detach().float()
    ffn_cfg = source_ffn_config(source_dir)

    if args.run_mode == "quick":
        updates = 500
        eval_count = 10
        probe_count = 4
    else:
        updates = 2500
        eval_count = 24
        probe_count = 8

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STAGE preparing text blocks", flush=True)
    train_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 10,
                updates * args.batch_size,
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

    special_ids = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]
    hot_ids = choose_hot_tokens(
        [*train_batches[: min(256, len(train_batches))], *probe_batches],
        int(teacher.config.vocab_size),
        args.hot_token_count,
        special_ids=special_ids,
    )
    print(f"Selected {len(hot_ids)} frequent hot tokens", flush=True)

    print(
        f"STAGE rank-{args.vocab_latent_dim} factorization of source v3 embedding",
        flush=True,
    )
    fact = factorize_embedding_weight_v2(
        teacher.model.embed_tokens.weight,
        args.vocab_latent_dim,
        args.factor_chunk_rows,
    )
    fact_stats = {k: v for k, v in fact.items() if not isinstance(v, torch.Tensor)}
    print("FACTORIZATION", json.dumps(fact_stats, indent=2), flush=True)

    vocab_cfg = FlyVocabV2Config(
        latent_dim=args.vocab_latent_dim,
        fly_nodes=int(adjacency.shape[0]),
        graph_steps=ffn_cfg.graph_steps,
        graph_mix_init=args.vocab_graph_mix_init,
        max_fly_scale=args.max_fly_scale,
        hot_token_count=len(hot_ids),
    )

    install_fly_vocab_v2(
        student,
        vocab_cfg,
        adjacency,
        hot_ids,
        factorization=fact,
        teacher_weight=teacher.model.embed_tokens.weight.detach(),
    )
    assert_qwen35_flycore_v2(student)

    if routing_schedule(student) != source_schedule:
        raise RuntimeError("source FlyFFN routing changed during vocab installation")
    print("✓ 24/24 source FlyFFN-v3 layers/routing preserved", flush=True)

    initial = probe(student, teacher, probe_batches, device, dtype)
    print("INITIAL PROBE", json.dumps(initial, indent=2), flush=True)

    final_probe, hist = train_vocab_v2(
        student,
        teacher,
        train_batches,
        probe_batches,
        updates,
        device,
        dtype,
    )
    pd.DataFrame(hist).to_csv(out_dir / "vocab_training_history.csv", index=False)
    print("FINAL PROBE", json.dumps(final_probe, indent=2), flush=True)

    if routing_schedule(student) != source_schedule:
        raise RuntimeError("source FlyFFN routing changed during FlyCore-v2 training")

    sanity_ok, sanity_rows = generation_sanity(student, tokenizer, device)
    (out_dir / "chat_samples.json").write_text(
        json.dumps(sanity_rows, indent=2), encoding="utf-8"
    )
    print("GENERATION SANITY", json.dumps(sanity_rows, indent=2), flush=True)
    print("GENERATION SANITY PASSED:", sanity_ok, flush=True)

    # Held-out CE/PPL through the existing evaluator.
    metrics = base.evaluate(
        student,
        teacher,
        eval_batches,
        device,
        dtype,
        args.seq_len,
        args.batch_size,
    )
    test_ids = eval_batches[0][:, :-1].to(device)
    source_bench = base.base.benchmark(teacher, test_ids, device)
    fly_bench = base.base.benchmark(student, test_ids, device)

    vocab_stats = student.fly_vocab_core_v2.parameter_stats()
    report = {
        "architecture": "FlyCore-v2 from trained Qwen3.5 FlyFFN-v3 AllFFN",
        "source_repo_id": args.source_repo_id,
        "source_routing_schedule": source_schedule,
        "final_routing_schedule": routing_schedule(student),
        "source_flyffn_frozen": True,
        "adjoint_consistent_vocab_transform": True,
        "hot_token_residuals": True,
        "factorization": fact_stats,
        "initial_probe": initial,
        "final_probe": final_probe,
        "generation_sanity_passed": sanity_ok,
        "generation_sanity": sanity_rows,
        "vocab_parameter_stats": vocab_stats,
        "source_v3": {
            "ce": metrics["teacher_ce"],
            "perplexity": metrics["teacher_perplexity"],
            **source_bench,
        },
        "flycore_v2": {
            "ce": metrics["ce"],
            "perplexity": metrics["perplexity"],
            "teacher_kl": metrics["teacher_kl"],
            **fly_bench,
        },
        "ce_gap_vs_source_v3": metrics["ce"] - metrics["teacher_ce"],
        "ppl_ratio_vs_source_v3": metrics["perplexity"] / metrics["teacher_perplexity"],
        "parameter_ratio_flycore_over_source_v3": (
            sum(p.numel() for p in student.parameters())
            / sum(p.numel() for p in teacher.parameters())
        ),
        "decode_speed_ratio_flycore_over_source_v3": (
            fly_bench["decode_tokens_s"] / source_bench["decode_tokens_s"]
        ),
        "config": {
            "run_mode": args.run_mode,
            "seq_len": args.seq_len,
            "vocab_latent_dim": args.vocab_latent_dim,
            "vocab_graph_mix_init": args.vocab_graph_mix_init,
            "max_fly_scale": args.max_fly_scale,
            "hot_token_count": len(hot_ids),
            "training_updates": updates,
            "fly_nodes": ffn_cfg.fly_nodes,
            "router_rank": ffn_cfg.router_rank,
            "num_shards": ffn_cfg.num_shards,
            "graph_steps": ffn_cfg.graph_steps,
            "graph_mix_init": ffn_cfg.graph_mix_init,
        },
    }

    rows = {
        "Source FlyFFN-v3 AllFFN": report["source_v3"],
        "FlyCore-v2": report["flycore_v2"],
    }
    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    standalone_dir = (
        Path(args.standalone_dir)
        if args.standalone_dir
        else out_dir / "standalone"
    )
    manifest = export_flycore_v2_standalone(
        student,
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

    upload_allowed = (
        sanity_ok
        and final_probe["ce_gap"] <= args.max_final_ce_gap
    )
    print(
        f"UPLOAD QUALITY GATE: sanity={sanity_ok} "
        f"ce_gap={final_probe['ce_gap']:.4f} <= {args.max_final_ce_gap:.4f} "
        f"=> {upload_allowed}",
        flush=True,
    )

    hf_url = None
    if args.upload_hf:
        if not upload_allowed:
            print(
                "Hugging Face upload skipped because FlyCore-v2 failed quality gate.",
                flush=True,
            )
        else:
            if not args.hf_repo_id:
                raise RuntimeError("--upload-hf requires --hf-repo-id")
            hf_url = upload_flycore_v2_standalone(
                standalone_dir,
                args.hf_repo_id,
                private=args.hf_private,
            )
            print("HUGGINGFACE", hf_url, flush=True)

    print("\nSUMMARY", flush=True)
    print(pd.DataFrame(rows).T, flush=True)
    print("\nVOCAB PARAMETER STATS", json.dumps(vocab_stats, indent=2), flush=True)
    print("\nSaved:", out_dir, flush=True)
    print("Standalone:", standalone_dir, flush=True)
    if hf_url:
        print("Hugging Face:", hf_url, flush=True)


if __name__ == "__main__":
    main()
