"""Cumulative decision distillation for Integrated Memory V2.2.

Fit/probe items come from disjoint training states. Neither acceptance cases
nor final test cases participate in gradients or checkpoint selection.
"""
from __future__ import annotations

import math
import torch

from .data import _batch_from_items
from .evaluate import fast_teacher_student_eval
from .factory import make_replacement
from .pdelta_optimized import (
    _teacher_forward_capture, _student_forward_capture, _prepare_local_probe,
    _probe_local_cached, _core_metrics, _kl, _centered_logit_mse, _checkpoint_rank,
)
from .train import _local_metrics, _score_metrics


def _integrated_checkpoint_rank(local, fast, cfg):
    gate_rank = _checkpoint_rank(local, fast, cfg)
    return (*gate_rank[:2], -fast["teacher_student_top1_agreement"],
            fast["mean_teacher_kl"], gate_rank[2])


def train_integrated_replacement(teacher, student, layer_idx, cfg,
                                 fit_items, probe_items, steps, batch_size):
    if steps < 1 or batch_size < 1 or cfg.integrated_probe_every < 1:
        raise ValueError("steps, batch size and probe interval must be positive")
    if not fit_items or not probe_items:
        raise ValueError("distinct nonempty fit and probe partitions are required")
    end_lr = cfg.final_learning_rate or cfg.learning_rate * 0.1
    if not 0 < end_lr <= cfg.learning_rate:
        raise ValueError("require 0 < final_learning_rate <= learning_rate")
    layer = student.model.encoder.layers[layer_idx]
    old = layer.attn
    replacement = make_replacement(teacher.model.encoder.layers[layer_idx].attn, cfg).to(student.device)
    # Previous accepted modules and the native backbone stay frozen.
    student.model.eval().requires_grad_(False)
    params = replacement.trainable_core_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ratio = end_lr / cfg.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * min(i / max(1, steps - 1), 1)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=student.device.type == "cuda" and student.dtype == torch.float16)
    pad = teacher.tok.pad_token_id
    # Several microbatches cover the probe without a large padded GPU batch.
    probe_items = probe_items[:64]
    probes = [_prepare_local_probe(teacher, layer_idx, _batch_from_items(
        probe_items, start, min(batch_size, len(probe_items) - start), pad))
        for start in range(0, len(probe_items), batch_size)]
    best_rank, best_state, best_local = None, None, None
    streak = 0
    functional_start = max(1, min(200, steps // 4))
    layer.attn = replacement
    try:
        for step in range(1, steps + 1):
            batch = _batch_from_items(fit_items, (step - 1) * batch_size, batch_size, pad)
            teacher_cap, (tlog, tact) = _teacher_forward_capture(teacher, layer_idx, batch)
            optimizer.zero_grad(set_to_none=True)
            # eval mode disables dropout but does not disable gradients.
            student.model.eval()
            functional = step >= functional_start
            if functional:
                cap, (slog, sact) = _student_forward_capture(student, layer_idx, batch, True)
                pred, core = cap["y"], cap["core"]
                markers = batch["marker_mask"].to(student.device).bool()
                decision = (2.0 * _kl(slog, tlog, markers)
                            + 0.1 * _kl(sact, tact)
                            + 0.1 * _centered_logit_mse(slog, tlog, markers))
            else:
                with torch.autocast(device_type=student.device.type, dtype=student.dtype,
                                    enabled=student.device.type == "cuda"):
                    pred, _ = replacement(teacher_cap["x"], position_embeddings=teacher_cap["pos"],
                                          attention_mask=teacher_cap["mask"])
                core = replacement.last_core_output
                decision = pred.new_zeros(())
            valid = batch["attention_mask"].to(student.device).bool()
            nmse, cosine = _local_metrics(pred, teacher_cap["y"], valid)
            cnmse, ccosine = _core_metrics(core, teacher_cap["core"], valid)
            loss = _score_metrics(nmse, cosine, cnmse, ccosine, cfg.architecture) + decision
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Integrated Memory loss at layer {layer_idx}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step != 1 and step % cfg.integrated_probe_every and step != steps:
                continue
            local = _probe_local_cached(replacement, probes, teacher)
            fast = fast_teacher_student_eval(teacher, student, probe_items,
                                            batch_size=batch_size, limit=len(probe_items))
            rank = _integrated_checkpoint_rank(local, fast, cfg)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_state = {k: v.detach().cpu().clone() for k, v in replacement.state_dict().items()}
                best_local = dict(local, step=step, selection="fixed_local_and_cumulative_decision_probe",
                                  functional_probe=fast)
            print(f"layer={layer_idx:02d} step={step}/{steps} nmse={local['nmse']:.4f} "
                  f"cos={local['cosine']:.4f} agreement={fast['teacher_student_top1_agreement']:.4f} "
                  f"KL={fast['mean_teacher_kl']:.6f} stage={'decision' if functional else 'local'}")
            # Local fidelity alone cannot stop training. Seek near-perfect
            # decision agreement, then still apply the independent gold gate.
            strong = (functional and rank[0] == 0
                      and fast['teacher_student_top1_agreement'] >= max(0.99, cfg.min_teacher_agreement)
                      and fast['mean_teacher_kl'] <= min(0.001, cfg.max_mean_kl))
            streak = streak + 1 if strong else 0
            if cfg.early_stop_local and streak >= 2:
                break
        replacement.load_state_dict(best_state)
        best_local.update(stopped_early=step < steps, max_steps=steps)
        replacement.eval().requires_grad_(False)
        return replacement, best_local
    finally:
        # Leave installation/acceptance to the runner, including on exceptions.
        layer.attn = old
