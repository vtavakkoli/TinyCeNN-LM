"""Compact training pipeline for the standalone PDelta3 decision model.

The source checkpoint is reconstructed without importing Laya. The final model
replaces every global/full encoder attention layer and every full attention
layer in the typed-decision head, then runs a short teacher-distillation pass.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .laya_lab.pdelta import PDelta3GDN2CLVRAttention
from .standalone_decision import QTYPES, build_sequence, collate_items, normalize_question
from .standalone_pdelta3_decision import (
    PDelta3DecisionHead,
    PDelta3DecisionHeadLayer,
    build_source_model,
    converted_encoder_indices,
    decision_head_adaptation_parameters,
    encoder_full_attention_indices,
    pdelta_core_parameters,
    pdelta_modules,
    pdelta_projection_parameters,
    remaining_full_attention,
)


@dataclass
class PDelta3TrainConfig:
    source_model: str = "convaiinnovations/laya-typed-decisions"
    seed: int = 2026
    feature_dim: int = 96
    local_window: int = 32
    encoder_steps: int = 120
    head_steps: int = 80
    joint_steps: int = 300
    train_cases: int = 900
    val_cases: int = 200
    max_len: int = 384
    batch_size: int = 3
    output_dir: str = "/content/PDelta3_GDN2_Standalone_Decision"


def _device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    return dev, amp, dtype


def _jsonish(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _make_items(tokenizer, source_cfg, cfg, split, limit, seed):
    from datasets import load_dataset

    rows = list(load_dataset("LocalLLaMA/typed-decisions", "all", split=split))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[: min(limit, len(rows))]
    out = []
    hml = int(source_cfg.get("head_max_len", 192))
    ml = min(int(cfg.max_len), int(source_cfg.get("max_len", cfg.max_len)))

    for row in rows:
        state = _jsonish(row["state"])
        questions = _jsonish(row["questions"])
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        state_ids = tokenizer(
            text.replace(tokenizer.mask_token, " "), add_special_tokens=False
        )["input_ids"]
        for qid, qdef in questions.items():
            try:
                q = normalize_question(qdef)
                ids, markers = build_sequence(
                    tokenizer,
                    state,
                    q,
                    max_len=ml,
                    head_max_len=hml,
                    truncate_left=isinstance(state, list),
                    state_ids=state_ids,
                )
                if len(markers) >= 2:
                    out.append(
                        {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "qid": qid}
                    )
            except Exception:
                continue
    rng.shuffle(out)
    return out


def _batch(items, tokenizer, device, offset, size):
    chosen = [items[(offset + i) % len(items)] for i in range(size)]
    batch = collate_items(chosen, int(tokenizer.pad_token_id))
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, batch, device, amp, dtype):
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=amp
    ):
        return model(**batch)


def _recon(pred, target, valid):
    mask = valid[:, :, None].float()
    mse = (((pred.float() - target.float()) ** 2) * mask).sum()
    mse = mse / (mask.sum() * pred.shape[-1]).clamp_min(1)
    power = ((target.float() ** 2) * mask).sum()
    power = power / (mask.sum() * target.shape[-1]).clamp_min(1)
    nmse = mse / power.clamp_min(1e-8)
    cosine = F.cosine_similarity(
        pred.float()[valid], target.float()[valid], dim=-1
    ).mean()
    return nmse + 0.75 * (1 - cosine), nmse, cosine


def _encoder_io(student, idx, batch, device, amp, dtype):
    got = {}
    attn = student.encoder.layers[idx].attn

    def pre(module, args, kwargs):
        got["x"] = (args[0] if args else kwargs["hidden_states"]).detach()
        pos = kwargs.get("position_embeddings")
        got["pos"] = None if pos is None else tuple(x.detach() for x in pos)
        mask = kwargs.get("attention_mask")
        got["mask"] = None if mask is None else mask.detach()

    def post(module, args, kwargs, output):
        got["y"] = output[0].detach()

    h1 = attn.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = attn.register_forward_hook(post, with_kwargs=True)
    try:
        _forward(student, batch, device, amp, dtype)
    finally:
        h1.remove()
        h2.remove()
    return got


def _head_io(student, idx, batch, device, amp, dtype):
    got = {}
    layer = student.head.layers[idx]

    def pre(module, args, kwargs):
        got["x"] = args[0].detach()
        pad = kwargs.get("src_key_padding_mask")
        got["pad"] = None if pad is None else pad.detach()

    def post(module, args, kwargs, output):
        got["y"] = output.detach()

    h1 = layer.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = layer.register_forward_hook(post, with_kwargs=True)
    try:
        _forward(student, batch, device, amp, dtype)
    finally:
        h1.remove()
        h2.remove()
    return got


def _fit_encoder(student, train, val, tokenizer, cfg, device, amp, dtype):
    full = encoder_full_attention_indices(student)
    report = {}
    for idx in full:
        print(f"\n=== encoder full layer {idx} ===")
        probe_b = _batch(val, tokenizer, device, idx * 23, min(6, cfg.batch_size + 2))
        probe = _encoder_io(student, idx, probe_b, device, amp, dtype)
        replacement = PDelta3GDN2CLVRAttention(
            student.encoder.layers[idx].attn,
            feature_dim=cfg.feature_dim,
            conv_kernel=4,
            chunk_size=64,
            local_kernel=5,
            local_window=cfg.local_window,
            local_gate_init=0.72,
        ).to(device)
        replacement.requires_grad_(False)
        params = []
        for name, param in replacement.named_parameters():
            if name.startswith(("Wqkv.", "Wo.", "out_drop.")):
                continue
            param.requires_grad_(True)
            if param.is_floating_point():
                param.data = param.data.float()
            params.append(param)

        opt = torch.optim.AdamW(params, lr=1e-2, betas=(0.9, 0.95), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, max(1, cfg.encoder_steps), eta_min=8e-4
        )
        best, best_score = None, float("inf")
        every = max(10, cfg.encoder_steps // 4)

        for step in range(1, cfg.encoder_steps + 1):
            batch = _batch(
                train, tokenizer, device,
                (step - 1) * cfg.batch_size + idx * 31,
                cfg.batch_size,
            )
            target = _encoder_io(student, idx, batch, device, amp, dtype)
            replacement.train()
            replacement.out_drop.eval()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                pred, _ = replacement(
                    target["x"],
                    position_embeddings=target["pos"],
                    attention_mask=target["mask"],
                )
                loss, _, _ = _recon(
                    pred, target["y"], batch["attention_mask"].bool()
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite encoder loss at layer {idx}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step == 1 or step % every == 0 or step == cfg.encoder_steps:
                replacement.eval()
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=amp
                ):
                    pred, _ = replacement(
                        probe["x"],
                        position_embeddings=probe["pos"],
                        attention_mask=probe["mask"],
                    )
                    ploss, nmse, cosine = _recon(
                        pred, probe["y"], probe_b["attention_mask"].bool()
                    )
                if float(ploss) < best_score:
                    best_score = float(ploss)
                    best = {
                        k: v.detach().cpu().clone()
                        for k, v in replacement.state_dict().items()
                    }
                print(
                    f"{step:03d}/{cfg.encoder_steps} "
                    f"nmse={float(nmse):.4f} cos={float(cosine):.4f}"
                )

        if best is not None:
            replacement.load_state_dict(best)
        replacement.eval()
        student.encoder.layers[idx].attn = replacement
        report[str(idx)] = {"best_local_score": best_score}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if converted_encoder_indices(student) != full:
        raise RuntimeError("not all source full-attention encoder layers were converted")
    return full, report


def _fit_head(student, train, val, tokenizer, cfg, device, amp, dtype):
    if student.head is None:
        return 0, {}
    count = len(student.head.layers)
    student.head = PDelta3DecisionHead(list(student.head.layers))
    report = {}

    for idx in range(count):
        print(f"\n=== decision-head full layer {idx} ===")
        original = student.head.layers[idx]
        probe_b = _batch(val, tokenizer, device, 700 + idx * 29, min(6, cfg.batch_size + 2))
        probe = _head_io(student, idx, probe_b, device, amp, dtype)
        replacement = PDelta3DecisionHeadLayer(
            original,
            idx,
            feature_dim=cfg.feature_dim,
            conv_kernel=4,
            chunk_size=64,
            local_kernel=5,
            local_window=cfg.local_window,
            local_gate_init=0.72,
        ).to(device)
        replacement.requires_grad_(False)
        params = []
        for name, param in replacement.named_parameters():
            if name.startswith("attn.") and not name.startswith(
                ("attn.Wqkv.", "attn.Wo.", "attn.out_drop.")
            ):
                param.requires_grad_(True)
                if param.is_floating_point():
                    param.data = param.data.float()
                params.append(param)

        opt = torch.optim.AdamW(params, lr=8e-3, betas=(0.9, 0.95), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, max(1, cfg.head_steps), eta_min=6e-4
        )
        best, best_score = None, float("inf")
        every = max(8, cfg.head_steps // 4)

        for step in range(1, cfg.head_steps + 1):
            batch = _batch(
                train, tokenizer, device,
                900 + (step - 1) * cfg.batch_size + idx * 37,
                cfg.batch_size,
            )
            target = _head_io(student, idx, batch, device, amp, dtype)
            replacement.train()
            replacement.attn.out_drop.eval()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                pred = replacement(
                    target["x"], src_key_padding_mask=target["pad"]
                )
                loss, _, _ = _recon(
                    pred, target["y"], batch["attention_mask"].bool()
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite head loss at layer {idx}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step == 1 or step % every == 0 or step == cfg.head_steps:
                replacement.eval()
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=amp
                ):
                    pred = replacement(
                        probe["x"], src_key_padding_mask=probe["pad"]
                    )
                    ploss, nmse, cosine = _recon(
                        pred, probe["y"], probe_b["attention_mask"].bool()
                    )
                if float(ploss) < best_score:
                    best_score = float(ploss)
                    best = {
                        k: v.detach().cpu().clone()
                        for k, v in replacement.state_dict().items()
                    }
                print(
                    f"{step:03d}/{cfg.head_steps} "
                    f"nmse={float(nmse):.4f} cos={float(cosine):.4f}"
                )

        if best is not None:
            replacement.load_state_dict(best)
        replacement.eval()
        student.head.layers[idx] = replacement
        report[str(idx)] = {"best_local_score": best_score}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if remaining_full_attention(student) != {"encoder": 0, "decision_head": 0}:
        raise RuntimeError(f"full attention remains: {remaining_full_attention(student)}")
    return count, report


@torch.no_grad()
def _fidelity(teacher, student, tokenizer, items, device, amp, dtype, limit=400, bs=16):
    agree, kls, l1s = [], [], []
    n = min(limit, len(items))
    for start in range(0, n, bs):
        rows = items[start:min(start + bs, n)]
        batch = collate_items(rows, int(tokenizer.pad_token_id))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            teacher_logits, _ = teacher(**batch)
            student_logits, _ = student(**batch)
        mask = batch["marker_mask"].bool()
        tp = torch.softmax(teacher_logits.float().masked_fill(~mask, -1e4), -1)
        sp = torch.softmax(student_logits.float().masked_fill(~mask, -1e4), -1)
        agree.append((tp.argmax(-1) == sp.argmax(-1)).float().cpu())
        kls.append(
            (
                tp * (
                    torch.log(tp.clamp_min(1e-9))
                    - torch.log(sp.clamp_min(1e-9))
                )
            ).sum(-1).cpu()
        )
        l1s.append(((tp - sp).abs() * mask).sum(-1).cpu())
    return {
        "teacher_student_top1_agreement": float(torch.cat(agree).mean()),
        "mean_teacher_kl": float(torch.cat(kls).mean()),
        "mean_probability_l1": float(torch.cat(l1s).mean()),
    }


def _joint(teacher, student, train, val, tokenizer, cfg, device, amp, dtype):
    core = pdelta_core_parameters(student)
    proj = pdelta_projection_parameters(student)
    adapt = decision_head_adaptation_parameters(student)

    student.requires_grad_(False)
    for param in core + proj + adapt:
        param.requires_grad_(True)

    opt = torch.optim.AdamW(
        [
            {"params": core, "lr": 3e-4},
            {"params": proj, "lr": 3e-5},
            {"params": adapt, "lr": 1e-5},
        ],
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, cfg.joint_steps), eta_min=3e-6
    )
    before = _fidelity(
        teacher, student, tokenizer, val, device, amp, dtype,
        limit=min(240, len(val))
    )
    best_eval = before
    best_score = before["mean_teacher_kl"] + 0.35 * (
        1 - before["teacher_student_top1_agreement"]
    )
    best = {
        n: p.detach().cpu().clone()
        for n, p in student.named_parameters()
        if p.requires_grad
    }
    every = max(10, cfg.joint_steps // 9)

    for step in range(1, cfg.joint_steps + 1):
        batch = _batch(
            train, tokenizer, device,
            2000 + (step - 1) * cfg.batch_size,
            cfg.batch_size,
        )
        mask = batch["marker_mask"].bool()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=amp
        ):
            teacher_logits, teacher_act = teacher(**batch)

        # eval() keeps distillation deterministic while gradients stay enabled.
        student.eval()
        for module in pdelta_modules(student):
            module.out_drop.eval()
        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            student_logits, student_act = student(**batch)
            temp = 1.25
            sl = (student_logits.float() / temp).masked_fill(~mask, -1e4)
            tl = (teacher_logits.detach().float() / temp).masked_fill(~mask, -1e4)
            decision_kl = F.kl_div(
                F.log_softmax(sl, -1),
                F.softmax(tl, -1),
                reduction="none",
            ).sum(-1).mean() * (temp * temp)

            m = mask.float()
            count = m.sum(-1, keepdim=True).clamp_min(1)
            sc = student_logits.float() - (
                student_logits.float() * m
            ).sum(-1, keepdim=True) / count
            tc = teacher_logits.detach().float() - (
                teacher_logits.detach().float() * m
            ).sum(-1, keepdim=True) / count
            logit_mse = (((sc - tc) ** 2) * m).sum() / m.sum().clamp_min(1)

            action_kl = F.kl_div(
                F.log_softmax(student_act.float(), -1),
                F.softmax(teacher_act.detach().float(), -1),
                reduction="batchmean",
            )
            loss = decision_kl + 0.20 * logit_mse + 0.03 * action_kl

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite joint loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core + proj + adapt, 0.75)
        opt.step()
        sched.step()

        if step == 1 or step % every == 0 or step == cfg.joint_steps:
            student.eval()
            evaluation = _fidelity(
                teacher, student, tokenizer, val, device, amp, dtype,
                limit=min(240, len(val))
            )
            score = evaluation["mean_teacher_kl"] + 0.35 * (
                1 - evaluation["teacher_student_top1_agreement"]
            )
            if score < best_score:
                best_score = score
                best_eval = evaluation
                best = {
                    n: p.detach().cpu().clone()
                    for n, p in student.named_parameters()
                    if p.requires_grad
                }
            print(
                f"{step:03d}/{cfg.joint_steps} "
                f"loss={float(loss):.5f} "
                f"agreement={evaluation['teacher_student_top1_agreement']:.4f} "
                f"KL={evaluation['mean_teacher_kl']:.5f}"
            )

    named = dict(student.named_parameters())
    with torch.no_grad():
        for name, value in best.items():
            named[name].copy_(value.to(named[name].device, dtype=named[name].dtype))
    student.eval().requires_grad_(False)
    return before, best_eval


@torch.no_grad()
def _latency(model, batch, device, amp, dtype, repeats=20):
    for _ in range(4):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return float(np.mean(values))


def train_and_export(cfg: PDelta3TrainConfig | None = None):
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_model
    from transformers import AutoTokenizer

    cfg = cfg or PDelta3TrainConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device, amp, dtype = _device()
    source_dir = Path(
        snapshot_download(
            cfg.source_model,
            allow_patterns=[
                "rl_agent_config.json", "model.safetensors",
                "encoder/*", "tokenizer/*",
            ],
        )
    )
    teacher, source_cfg = build_source_model(source_dir)
    student, _ = build_source_model(source_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(source_dir / "tokenizer"))
    teacher.to(device).eval().requires_grad_(False)
    student.to(device).eval().requires_grad_(False)

    train = _make_items(tokenizer, source_cfg, cfg, "train", cfg.train_cases, cfg.seed)
    val = _make_items(tokenizer, source_cfg, cfg, "test", cfg.val_cases, cfg.seed + 1)
    if len(train) < 64 or len(val) < 32:
        raise RuntimeError("not enough usable typed-decision rows")
    print("device:", device, "| train:", len(train), "| val:", len(val))
    print("full attention before:", remaining_full_attention(student))

    full_layers, encoder_report = _fit_encoder(
        student, train, val, tokenizer, cfg, device, amp, dtype
    )
    head_count, head_report = _fit_head(
        student, train, val, tokenizer, cfg, device, amp, dtype
    )
    before_joint, best_joint = _joint(
        teacher, student, train, val, tokenizer, cfg, device, amp, dtype
    )
    final = _fidelity(
        teacher, student, tokenizer, val, device, amp, dtype,
        limit=min(600, len(val)), bs=16
    )

    latency_batch = _batch(
        val, tokenizer, device, 3100, min(8, len(val))
    )
    teacher_ms = _latency(teacher, latency_batch, device, amp, dtype)
    student_ms = _latency(student, latency_batch, device, amp, dtype)
    final.update(
        {
            "teacher_mean_ms": teacher_ms,
            "student_mean_ms": student_ms,
            "forward_speedup_vs_teacher": teacher_ms / max(student_ms, 1e-9),
        }
    )

    export_dir = Path(cfg.output_dir) / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (export_dir / "encoder").mkdir(parents=True)
    (export_dir / "tokenizer").mkdir(parents=True)
    save_model(student, str(export_dir / "model.safetensors"))
    student.encoder.config.save_pretrained(str(export_dir / "encoder"))
    tokenizer.save_pretrained(str(export_dir / "tokenizer"))

    metadata = {
        "format": "tinycenn-standalone-pdelta3-decision-v1",
        "source_model": cfg.source_model,
        "requires_laya": False,
        "architecture": "pdelta3_gdn2_clvr_standalone_decision",
        "head_layers": head_count,
        "n_act": int(student.act_head[-1].out_features),
        "max_len": int(source_cfg.get("max_len", 512)),
        "head_max_len": int(source_cfg.get("head_max_len", 192)),
        "converted_full_attention_layers": full_layers,
        "converted_decision_head_layers": list(range(head_count)),
        "kept_sliding_attention": True,
        "remaining_full_attention": remaining_full_attention(student),
        "pdelta3": {
            "feature_dim": cfg.feature_dim,
            "conv_kernel": 4,
            "chunk_size": 64,
            "local_kernel": 5,
            "local_window": cfg.local_window,
            "local_gate_init": 0.72,
            "global_dense_attention": False,
        },
    }
    if metadata["remaining_full_attention"] != {"encoder": 0, "decision_head": 0}:
        raise RuntimeError("refusing to export while full attention remains")

    report = {
        "encoder_transfer": encoder_report,
        "head_transfer": head_report,
        "before_joint": before_joint,
        "best_joint": best_joint,
        "final": final,
    }
    (export_dir / "standalone_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (export_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("final:", json.dumps(final, indent=2))
    print("exported:", export_dir)

    return {
        "student": student,
        "tokenizer": tokenizer,
        "metadata": metadata,
        "report": report,
        "export_dir": export_dir,
        "device": device,
    }


def upload_export(result, repo_id, token=None, private=False, min_agreement=0.90):
    from huggingface_hub import HfApi

    agreement = float(
        result["report"]["final"]["teacher_student_top1_agreement"]
    )
    if agreement < float(min_agreement):
        raise RuntimeError(
            f"upload blocked: agreement={agreement:.4f} < {min_agreement:.4f}"
        )
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id, repo_type="model", private=private, exist_ok=True
    )
    return api.upload_folder(
        folder_path=str(result["export_dir"]),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload standalone all-layer PDelta3 decision model",
    )
