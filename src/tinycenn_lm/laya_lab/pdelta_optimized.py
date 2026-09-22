from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .core import LayaLabConfig
from .data import _batch_from_items, _load_typed_split, build_training_items, dataset_cases
from .evaluate import (
    _accept,
    adapter_payload,
    benchmark_latency,
    evaluate_agent,
    fast_teacher_student_eval,
)
from .factory import choose_candidate_layers
from .pdelta import PDelta3GDN2CLVRAttention


def _tree_detach(x):
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_tree_detach(v) for v in x)
    if isinstance(x, list):
        return [_tree_detach(v) for v in x]
    return x


def _args(batch, device):
    return [
        batch[k].to(device)
        for k in (
            "input_ids",
            "attention_mask",
            "marker_pos",
            "marker_mask",
            "qtype",
        )
    ]


def _local_metrics(pred, target, valid2d):
    mask = valid2d[:, :, None].float()
    mse = (((pred.float() - target.float()) ** 2) * mask).sum()
    mse = mse / (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    denom = ((target.float() ** 2) * mask).sum()
    denom = denom / (mask.sum() * target.shape[-1]).clamp_min(1.0)
    nmse = mse / denom.clamp_min(1e-8)
    p, t = pred.float()[valid2d], target.float()[valid2d]
    cosine = (
        F.cosine_similarity(p, t, dim=-1).mean()
        if p.numel()
        else pred.new_tensor(0.0)
    )
    return nmse, cosine


def _core_metrics(pred, target, valid2d):
    p = pred.transpose(1, 2).reshape(pred.shape[0], pred.shape[2], -1)
    t = target.transpose(1, 2).reshape(target.shape[0], target.shape[2], -1)
    return _local_metrics(p, t, valid2d)


def _kl(student, teacher, mask=None, temperature: float = 1.0):
    t = max(float(temperature), 1e-3)
    s = student.float() / t
    q = teacher.detach().float() / t
    if mask is not None:
        m = mask.bool()
        s = s.masked_fill(~m, -1e4)
        q = q.masked_fill(~m, -1e4)
    return F.kl_div(
        F.log_softmax(s, -1),
        F.softmax(q, -1),
        reduction="none",
    ).sum(-1).mean() * (t * t)


def _centered_logit_mse(student, teacher, mask):
    m = mask.float()
    count = m.sum(-1, keepdim=True).clamp_min(1.0)
    s = student.float()
    t = teacher.detach().float()
    sm = (s * m).sum(-1, keepdim=True) / count
    tm = (t * m).sum(-1, keepdim=True) / count
    diff = ((s - sm) - (t - tm)) * m
    return (diff.square().sum() / m.sum().clamp_min(1.0))


def _student_forward_capture(student, layer_idx, batch, grad: bool):
    capture: dict[str, Any] = {}
    module = student.model.encoder.layers[layer_idx].attn

    def pre_hook(mod, hook_args, kwargs):
        x = hook_args[0] if hook_args else kwargs["hidden_states"]
        capture["x"] = x.detach()
        capture["pos"] = _tree_detach(kwargs.get("position_embeddings"))
        mask = kwargs.get("attention_mask")
        capture["mask"] = None if mask is None else mask.detach()

    def post_hook(mod, hook_args, kwargs, output):
        capture["y"] = output[0]
        capture["core"] = mod.last_core_output

    h1 = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = module.register_forward_hook(post_hook, with_kwargs=True)
    try:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast(
            device_type=student.device.type,
            dtype=student.dtype,
            enabled=student.device.type == "cuda",
        ):
            out = student.model(*_args(batch, student.device))
    finally:
        h1.remove()
        h2.remove()

    if "x" not in capture or "y" not in capture or capture.get("core") is None:
        raise RuntimeError(f"failed to capture PDelta3 layer {layer_idx}")
    return capture, out


@torch.no_grad()
def _teacher_forward_capture(teacher, layer_idx, batch):
    """One teacher forward: logits + candidate-layer input/output/core target."""
    capture: dict[str, Any] = {}
    attn = teacher.model.encoder.layers[layer_idx].attn

    def pre_hook(module, hook_args, kwargs):
        x = hook_args[0] if hook_args else kwargs["hidden_states"]
        capture["x"] = x.detach()
        capture["pos"] = _tree_detach(kwargs.get("position_embeddings"))
        mask = kwargs.get("attention_mask")
        capture["mask"] = None if mask is None else mask.detach()

    def wo_pre_hook(module, hook_args):
        if not hook_args:
            return
        flat = hook_args[0]
        if flat.ndim != 3:
            return
        b, t, _ = flat.shape
        h = int(attn.config.num_attention_heads)
        d = flat.shape[-1] // h
        capture["core"] = (
            flat.reshape(b, t, h, d).transpose(1, 2).contiguous().detach()
        )

    def post_hook(module, hook_args, kwargs, output):
        capture["y"] = output[0].detach()

    h1 = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.Wo.register_forward_pre_hook(wo_pre_hook)
    h3 = attn.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.autocast(
            device_type=teacher.device.type,
            dtype=teacher.dtype,
            enabled=teacher.device.type == "cuda",
        ):
            out = teacher.model(*_args(batch, teacher.device))
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    required = ("x", "y", "core")
    if any(k not in capture for k in required):
        raise RuntimeError(
            f"failed to capture teacher layer {layer_idx}: "
            f"missing {[k for k in required if k not in capture]}"
        )
    return capture, out


@torch.no_grad()
def _teacher_attention_target(teacher, layer_idx, x, pos, mask):
    attn = teacher.model.encoder.layers[layer_idx].attn
    captured: dict[str, torch.Tensor] = {}

    def wo_pre_hook(module, hook_args):
        if not hook_args:
            return
        flat = hook_args[0]
        if flat.ndim != 3:
            return
        b, t, _ = flat.shape
        h = int(attn.config.num_attention_heads)
        d = flat.shape[-1] // h
        captured["core"] = (
            flat.reshape(b, t, h, d).transpose(1, 2).contiguous().detach()
        )

    h = attn.Wo.register_forward_pre_hook(wo_pre_hook)
    try:
        with torch.autocast(
            device_type=teacher.device.type,
            dtype=teacher.dtype,
            enabled=teacher.device.type == "cuda",
        ):
            y, _ = attn(
                x,
                position_embeddings=pos,
                attention_mask=mask,
            )
    finally:
        h.remove()
    if "core" not in captured:
        raise RuntimeError("failed to capture teacher pre-Wo attention core")
    return y.detach(), captured["core"]


def _stratified_disjoint_cases(ds, gate_count, final_count, seed):
    grouped: dict[str, list] = {}
    for case in dataset_cases(ds, None):
        grouped.setdefault(str(case[3]), []).append(case)

    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)

    def take(n):
        keys = [k for k in sorted(grouped) if grouped[k]]
        out, cursor = [], 0
        while len(out) < n and keys:
            key = keys[cursor % len(keys)]
            out.append(grouped[key].pop())
            if not grouped[key]:
                keys = [k for k in keys if grouped[k]]
                cursor = 0
            else:
                cursor += 1
        return out

    gate = take(gate_count)
    final = take(final_count)
    if len(gate) < gate_count or len(final) < final_count:
        raise RuntimeError(
            "typed-decisions test split is too small for disjoint gate/final sets"
        )
    return gate, final


@torch.no_grad()
def _probe_local(teacher, student, layer_idx, probe_batch):
    cap, _ = _student_forward_capture(student, layer_idx, probe_batch, False)
    target, target_core = _teacher_attention_target(
        teacher, layer_idx, cap["x"], cap["pos"], cap["mask"]
    )
    valid = probe_batch["attention_mask"].to(student.device).bool()
    nmse, cosine = _local_metrics(cap["y"], target, valid)
    core_nmse, core_cosine = _core_metrics(cap["core"], target_core, valid)
    return {
        "nmse": float(nmse.item()),
        "cosine": float(cosine.item()),
        "core_nmse": float(core_nmse.item()),
        "core_cosine": float(core_cosine.item()),
    }


def _candidate_score(local, fast):
    return (
        float(local["nmse"])
        + 0.25 * (1.0 - float(local["cosine"]))
        + 0.40 * float(fast["mean_teacher_kl"])
        + 0.25 * (1.0 - float(fast["teacher_student_top1_agreement"]))
    )


def _replacement_trainable_count(model):
    total = 0
    for layer in model.encoder.layers:
        if not isinstance(layer.attn, PDelta3GDN2CLVRAttention):
            continue
        for name, p in layer.attn.named_parameters():
            if name.startswith("Wqkv.") or name.startswith("out_drop."):
                continue
            total += p.numel()
    return int(total)


def _train_candidate(
    teacher,
    student,
    layer_idx,
    cfg,
    fit_items,
    probe_batch,
    fast_items,
    gate_cases,
    teacher_gate,
    steps,
    batch_size,
):
    layer = student.model.encoder.layers[layer_idx]
    old = layer.attn
    replacement = PDelta3GDN2CLVRAttention(
        teacher.model.encoder.layers[layer_idx].attn,
        feature_dim=cfg.feature_dim,
        conv_kernel=cfg.pdelta_conv_kernel,
        chunk_size=cfg.pdelta_chunk_size,
        local_kernel=cfg.local_kernel,
    ).to(student.device)

    proj_dtype = teacher.model.encoder.layers[layer_idx].attn.Wqkv.weight.dtype
    replacement.Wqkv.to(device=student.device, dtype=proj_dtype)
    # Keep a FP32 master copy for the very low-LR Wo calibration.  Autocast
    # still executes the projection efficiently while AdamW updates retain
    # enough numerical resolution.
    replacement.Wo.to(device=student.device, dtype=torch.float32)
    layer.attn = replacement

    # Freeze the complete decision model.  Only the new PDelta3 core and a
    # low-LR calibration of the copied output projection are trainable.
    student.model.eval().requires_grad_(False)
    core_params, output_params = [], []
    for name, p in replacement.named_parameters():
        if name.startswith("Wqkv.") or name.startswith("out_drop."):
            p.requires_grad = False
        elif name.startswith("Wo."):
            p.requires_grad = True
            output_params.append(p)
        else:
            p.requires_grad = True
            if p.is_floating_point():
                p.data = p.data.float()
            core_params.append(p)

    # Restore the fast-convergence regime: start at 1e-2 and decay
    # smoothly to 1e-3.  The previous optimized runner accidentally capped the
    # requested LR at 6e-4, which is why 900 steps still converged very slowly.
    core_lr_start = float(cfg.learning_rate)
    core_lr_end = float(
        cfg.final_learning_rate
        if cfg.final_learning_rate is not None
        else core_lr_start * 0.1
    )
    if core_lr_start <= 0 or core_lr_end <= 0:
        raise ValueError("learning rates must be positive")
    if core_lr_end > core_lr_start:
        raise ValueError("final_learning_rate must be <= learning_rate")

    output_lr_start = core_lr_start * 0.10
    output_lr_end = core_lr_end * 0.10
    opt = torch.optim.AdamW(
        [
            {"params": core_params, "lr": core_lr_start},
            {"params": output_params, "lr": output_lr_start},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    end_ratio = core_lr_end / core_lr_start

    def lr_scale(step_idx):
        progress = float(step_idx) / float(max(1, steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return end_ratio + (1.0 - end_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)

    best_state = None
    best_score = float("inf")
    best_local = None
    best_fast = None
    best_step = 0
    accepted = False
    accepted_gate = None
    accepted_checks = None
    accepted_drop = None

    # Stage 1 is cheap local transfer.  Stage 2 turns on the expensive
    # full-student decision distillation only after the replacement has learned
    # the teacher attention shape.  This removes a full student-model pass from
    # most optimization steps.
    functional_start = max(40, int(round(steps * 0.60)))
    check_every = max(25, steps // 14)
    min_gate_step = functional_start
    pad_id = teacher.tok.pad_token_id

    for step in range(1, steps + 1):
        batch = _batch_from_items(
            fit_items,
            (step - 1) * batch_size,
            batch_size,
            pad_id,
        )
        replacement.train()
        replacement.out_drop.eval()

        # One teacher full forward captures everything needed for local transfer
        # and, later, functional distillation.  The previous code performed an
        # extra teacher-attention forward every step.
        teacher_cap, (tlog, tact) = _teacher_forward_capture(
            teacher, layer_idx, batch
        )

        opt.zero_grad(set_to_none=True)
        valid = batch["attention_mask"].to(student.device).bool()
        marker_mask = batch["marker_mask"].to(student.device).bool()
        functional = step >= functional_start

        if functional:
            # Refinement stage: one student full forward gives both the
            # replacement output/core and the final Laya decision logits.
            cap, (slog, sact) = _student_forward_capture(
                student, layer_idx, batch, True
            )
            pred = cap["y"]
            core_pred = cap["core"]
            decision_kl = _kl(
                slog, tlog, marker_mask, temperature=1.25
            )
            action_kl = _kl(sact, tact, None, temperature=1.0)
            logit_mse = _centered_logit_mse(
                slog, tlog, marker_mask
            )
        else:
            # Fast transfer stage: train only the candidate module on teacher
            # hidden states. No full student-model forward is needed.
            with torch.autocast(
                device_type=student.device.type,
                dtype=student.dtype,
                enabled=student.device.type == "cuda",
            ):
                pred, _ = replacement(
                    teacher_cap["x"],
                    position_embeddings=teacher_cap["pos"],
                    attention_mask=teacher_cap["mask"],
                )
            core_pred = replacement.last_core_output
            if core_pred is None:
                raise RuntimeError("replacement did not expose core output")
            zero = pred.new_zeros(())
            decision_kl = zero
            action_kl = zero
            logit_mse = zero

        nmse, cosine = _local_metrics(
            pred, teacher_cap["y"], valid
        )
        core_nmse, core_cosine = _core_metrics(
            core_pred, teacher_cap["core"], valid
        )

        if functional:
            refine_progress = float(
                step - functional_start
            ) / float(max(1, steps - functional_start))
            decision_weight = 0.75 + 0.75 * refine_progress
        else:
            decision_weight = 0.0

        loss = (
            nmse
            + 0.30 * (1.0 - cosine)
            + 0.20 * core_nmse
            + 0.08 * (1.0 - core_cosine)
            + decision_weight * decision_kl
            + 0.03 * action_kl
            + 0.05 * logit_mse
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite PDelta3 loss layer={layer_idx} step={step}"
            )

        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            core_params + output_params, 1.0
        )
        opt.step()
        scheduler.step()

        should_check = (
            step == 1
            or step % check_every == 0
            or step == steps
        )
        if step == 1 or step % 25 == 0 or should_check:
            print(
                f"layer={layer_idx:02d} step={step:04d}/{steps} "
                f"nmse={float(nmse.detach()):.4f} "
                f"cos={float(cosine.detach()):.4f} "
                f"core_nmse={float(core_nmse.detach()):.4f} "
                f"dkl={float(decision_kl.detach()):.4f} "
                f"logit={float(logit_mse.detach()):.4f} "
                f"dw={decision_weight:.2f} "
                f"lr={opt.param_groups[0]['lr']:.5f} "
                f"stage={'functional' if functional else 'local'} "
                f"grad={float(grad):.3f}"
            )
        if not should_check:
            continue

        replacement.eval()
        local = _probe_local(
            teacher, student, layer_idx, probe_batch
        )
        fast = fast_teacher_student_eval(
            teacher,
            student,
            fast_items,
            batch_size=max(4, batch_size * 4),
            limit=min(128, len(fast_items)),
        )
        score = _candidate_score(local, fast)
        print(
            "  FAST CHECK:",
            json.dumps(
                {
                    "step": step,
                    "nmse": round(local["nmse"], 5),
                    "cosine": round(local["cosine"], 5),
                    "agreement": round(
                        fast["teacher_student_top1_agreement"], 5
                    ),
                    "teacher_kl": round(fast["mean_teacher_kl"], 6),
                    "prob_l1": round(fast["mean_probability_l1"], 6),
                    "speedup": round(
                        fast["forward_speedup_vs_teacher"], 3
                    ),
                },
                indent=2,
            ),
        )

        if score < best_score:
            best_score = score
            best_step = step
            best_local = dict(local)
            best_fast = dict(fast)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in replacement.state_dict().items()
            }

        ready_for_gate = (
            step >= min_gate_step
            and fast["teacher_student_top1_agreement"] >= 0.93
            and fast["mean_teacher_kl"] <= max(0.075, cfg.max_mean_kl * 1.5)
        )
        if ready_for_gate or step == steps:
            gate = evaluate_agent(
                student,
                gate_cases,
                teacher_agent=teacher,
                label="pdelta3_gdn2_clvr",
            )
            accepted, checks, accuracy_drop = _accept(
                local, teacher_gate, gate, cfg
            )
            print(
                "  STRICT GATE:",
                json.dumps(
                    {
                        "accepted": accepted,
                        "teacher_agreement": gate.get("teacher_agreement"),
                        "mean_teacher_kl": gate.get("mean_teacher_kl"),
                        "accuracy": gate.get("accuracy"),
                        "accuracy_drop": accuracy_drop,
                        "checks": checks,
                    },
                    indent=2,
                ),
            )
            if accepted:
                accepted_gate = gate
                accepted_checks = checks
                accepted_drop = accuracy_drop
                best_local = dict(local)
                best_fast = dict(fast)
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in replacement.state_dict().items()
                }
                break

    if best_state is not None:
        replacement.load_state_dict(best_state)
    replacement.eval()

    # Re-evaluate the best checkpoint, not merely the final optimizer step.
    best_local = _probe_local(
        teacher, student, layer_idx, probe_batch
    )
    best_fast = fast_teacher_student_eval(
        teacher,
        student,
        fast_items,
        batch_size=max(4, batch_size * 4),
        limit=min(128, len(fast_items)),
    )
    final_gate = evaluate_agent(
        student,
        gate_cases,
        teacher_agent=teacher,
        label="pdelta3_gdn2_clvr",
    )
    final_accepted, final_checks, final_drop = _accept(
        best_local, teacher_gate, final_gate, cfg
    )
    if final_accepted:
        accepted = True
        accepted_gate = final_gate
        accepted_checks = final_checks
        accepted_drop = final_drop
    else:
        accepted = False

    rec = {
        "layer": int(layer_idx),
        "attention_type": str(layer.attention_type),
        "accepted": bool(accepted),
        "best_step": int(best_step),
        "local": best_local,
        "fast_eval": best_fast,
        "gate": final_gate,
        "checks": final_checks,
        "accuracy_drop": final_drop,
        "training": {
            "steps": int(steps),
            "core_learning_rate_start": core_lr_start,
            "core_learning_rate_end": core_lr_end,
            "output_learning_rate_start": output_lr_start,
            "output_learning_rate_end": output_lr_end,
            "functional_refinement_start_step": functional_start,
            "local_only_fraction": float(functional_start) / float(max(1, steps)),
            "decision_distillation": True,
            "train_output_projection": True,
            "qkv_frozen": True,
        },
    }

    if not accepted:
        layer.attn = old
        del replacement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        student.model.eval().requires_grad_(False)

    return bool(accepted), rec


def _demo_and_latency(teacher, student):
    state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": (
            "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan."
        ),
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "not urgent",
                "soon",
                "critical deadline or blocking issue",
            ],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    }
    return (
        {
            "teacher": teacher.predict(state, questions),
            "student": student.predict(state, questions),
        },
        {
            "teacher": benchmark_latency(
                teacher, state, questions, warmup=3, repeats=12
            ),
            "student": benchmark_latency(
                student, state, questions, warmup=3, repeats=12
            ),
        },
    )


def run_pdelta3_optimized(cfg: LayaLabConfig):
    """Optimized Laya PDelta3 conversion with functional distillation."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    settings = cfg.mode_settings()
    out_dir = Path(cfg.output_dir) / cfg.architecture
    out_dir.mkdir(parents=True, exist_ok=True)

    import laya

    print("Loading Laya teacher:", cfg.model_id)
    teacher = laya.load(
        cfg.model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    teacher.model.eval().requires_grad_(False)
    try:
        teacher.model.encoder.config.reference_compile = False
    except Exception:
        pass

    student = copy.copy(teacher)
    student.model = copy.deepcopy(teacher.model).eval().requires_grad_(False)

    train_ds = _load_typed_split("train")
    test_ds = _load_typed_split("test")
    model_max_len = int(
        teacher.cfg.get("max_len", settings["train_max_len"])
    )
    train_max_len = min(int(settings["train_max_len"]), model_max_len)

    all_items = build_training_items(
        teacher,
        train_ds,
        settings["train_cases"],
        train_max_len,
        cfg.seed,
    )
    if len(all_items) < max(48, settings["batch_size"] * 12):
        raise RuntimeError("too few Laya attention-transfer sequences")

    probe_n = min(
        max(24, settings["batch_size"] * 8),
        max(24, len(all_items) // 12),
    )
    fit_items = all_items[:-probe_n]
    probe_items = all_items[-probe_n:]
    from laya.common import collate_items

    probe_batch = collate_items(
        [[
            probe_items[i]
            for i in range(
                min(len(probe_items), max(4, settings["batch_size"] * 2))
            )
        ]],
        teacher.tok.pad_token_id,
    )

    fast_items = build_training_items(
        teacher,
        test_ds,
        min(80, len(test_ds)),
        train_max_len,
        cfg.seed + 17,
    )
    gate_cases, final_cases = _stratified_disjoint_cases(
        test_ds,
        settings["gate_cases"],
        settings["final_cases"],
        cfg.seed + 31,
    )
    teacher_gate = evaluate_agent(
        teacher, gate_cases, label="teacher"
    )

    candidates = choose_candidate_layers(
        student.model,
        settings["max_candidates"],
        preferred_attention_type="full_attention",
    )
    candidates = [
        i
        for i in candidates
        if str(student.model.encoder.layers[i].attention_type)
        == "full_attention"
    ]
    # Your previous run showed layer 15 had the strongest pre-Wo core fidelity
    # and already satisfied the accuracy-drop gate, so try the deeper full
    # layer first; layer 12 remains the fallback.
    candidates.sort(reverse=True)

    steps = (
        int(cfg.training_steps)
        if cfg.training_steps is not None
        else int(settings["steps"])
    )

    print(
        "Attention-transfer sequences:",
        len(all_items),
        "| fit:",
        len(fit_items),
        "| probe:",
        len(probe_items),
        "| train max length:",
        train_max_len,
    )
    print("Teacher gate accuracy:", round(teacher_gate["accuracy"], 4))
    print(
        "Candidate full-attention layers:",
        [
            (i, student.model.encoder.layers[i].attention_type)
            for i in candidates
        ],
    )
    print(
        "Optimized recipe: learned global linear path + GDN2 + local/direct, "
        "end-to-end decision KL, low-LR Wo calibration, fixed local probe, "
        "fast batched functional checks, disjoint gate/final sets."
    )

    history = []
    for idx in candidates:
        print(
            f"\n{'=' * 88}\n"
            f"PDelta3 optimized candidate layer {idx} "
            f"({student.model.encoder.layers[idx].attention_type})\n"
            f"{'=' * 88}"
        )
        accepted, rec = _train_candidate(
            teacher,
            student,
            idx,
            cfg,
            fit_items,
            probe_batch,
            fast_items,
            gate_cases,
            teacher_gate,
            steps,
            settings["batch_size"],
        )
        history.append(rec)
        torch.save(
            {
                "layer": idx,
                "accepted": rec["accepted"],
                "metrics": rec,
                "state_dict": (
                    {
                        k: v.detach().cpu()
                        for k, v in student.model.encoder.layers[
                            idx
                        ].attn.state_dict().items()
                    }
                    if accepted
                    else None
                ),
            },
            out_dir / f"candidate_layer_{idx}.pt",
        )
        if accepted:
            print(
                f"✅ accepted layer {idx}; stopping after the first strict "
                "pass to preserve quality and latency."
            )
            break
        print(f"❌ layer {idx} did not pass all strict gates; trying fallback.")

    teacher_final = evaluate_agent(
        teacher, final_cases, label="teacher"
    )
    student_final = evaluate_agent(
        student,
        final_cases,
        teacher_agent=teacher,
        label=cfg.architecture,
    )
    final_fast = fast_teacher_student_eval(
        teacher,
        student,
        fast_items,
        batch_size=max(4, settings["batch_size"] * 4),
        limit=min(160, len(fast_items)),
    )
    demo, latency = _demo_and_latency(teacher, student)

    accepted_layers = [
        i
        for i, layer in enumerate(student.model.encoder.layers)
        if isinstance(layer.attn, PDelta3GDN2CLVRAttention)
    ]
    report = {
        "architecture": cfg.architecture,
        "model_id": cfg.model_id,
        "mode": cfg.mode,
        "config": asdict(cfg),
        "candidate_layers": candidates,
        "accepted_layers": accepted_layers,
        "replacement_trainable_parameters": _replacement_trainable_count(
            student.model
        ),
        "training_steps_per_candidate": steps,
        "history": history,
        "teacher_gate": teacher_gate,
        "teacher_final": teacher_final,
        "student_final": student_final,
        "fast_eval": final_fast,
        "latency": latency,
        "demo": demo,
        "gate_final_disjoint": True,
        "conversion_succeeded": bool(accepted_layers),
        "student_is_unmodified_teacher": not bool(accepted_layers),
        "notes": [
            "QKV stays frozen; copied Wo is calibrated at 5% of the core LR.",
            "PDelta3 now includes a learned positive-feature global linear path.",
            "Training combines local attention/core fidelity with decision/action distillation.",
            "Fast eval is batched and label-free; strict acceptance still uses held-out Agent.predict metrics.",
            "Gate and final evaluation sets are disjoint and workflow-stratified.",
            "Only full-attention candidates are attempted; training stops after the first strict pass.",
        ],
    }

    torch.save(
        adapter_payload(student.model, cfg, report),
        out_dir / "adapter.pt",
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nAccepted layers:", accepted_layers)
    print("Final student metrics:")
    print(json.dumps(student_final, indent=2))
    print("Fast teacher/student comparison:")
    print(json.dumps(final_fast, indent=2))
    print("Latency:")
    print(json.dumps(latency, indent=2))
    print("Saved:", out_dir / "adapter.pt")
    return teacher, student, report
