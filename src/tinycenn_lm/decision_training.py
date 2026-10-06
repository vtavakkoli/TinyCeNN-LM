"""Numerical and checkpoint safeguards shared by the decision notebooks."""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import torch


def require_finite(tensor, context):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{context}: non-finite values; stop and reload the last good checkpoint.")


def gold_checkpoint_key(metrics):
    """Minimize lexicographically: accuracy first, then KL and Brier."""
    values = tuple(float(metrics[k]) for k in ("accuracy", "kl_from_gold", "brier"))
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError("Non-finite dev metrics cannot select a checkpoint")
    return (-values[0], values[1], values[2])


def trainable_snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def recovery_lr(step, total_steps, start_lr, end_lr, warmup_steps=20):
    """Short recovery warmup, followed by cosine decay to the exact configured floor."""
    warmup = min(int(warmup_steps), max(0, total_steps // 10))
    if warmup and step < warmup:
        return float(start_lr * (step + 1) / warmup)
    if total_steps - warmup <= 1:
        return float(end_lr)
    x = min(1., max(0., (step-warmup) / (total_steps-warmup-1)))
    return float(end_lr + .5*(start_lr-end_lr)*(1+math.cos(math.pi*x)))


def guarded_step(loss, optimizer, scaler, groups, context, max_skips=8):
    """Never update from NaN/Inf; let AMP lower its scale for bounded retries.

    `groups` is a sequence of (parameter_list, clipping_norm). All optimizer
    parameters must be covered. Returns False on an AMP overflow (no update).
    """
    require_finite(loss, context + " loss")
    params = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    for p in params:
        require_finite(p, context + " parameter before update")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise RuntimeError(context + ": no trainable gradients")
    if not all(bool(torch.isfinite(g).all()) for g in grads):
        if not scaler.is_enabled():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(context + ": non-finite gradients without AMP scaling")
        old_scale = scaler.get_scale()
        # GradScaler has recorded the Inf/NaN during unscale; step MUST skip.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        skips = getattr(optimizer, '_decision_overflow_skips', 0) + 1
        optimizer._decision_overflow_skips = skips
        warnings.warn(f"{context}: skipped overflowing gradient update; scale {old_scale:g} -> {scaler.get_scale():g}")
        if skips >= max_skips or scaler.get_scale() < 1.:
            raise FloatingPointError(context + ": persistent gradient overflow; restore checkpoint and lower recovery LR")
        return False
    covered = {id(p) for ps, _ in groups for p in ps}
    if any(id(p) not in covered for p in params):
        raise ValueError("Gradient clipping groups must cover all optimizer parameters")
    for ps, limit in groups:
        norm = torch.nn.utils.clip_grad_norm_(ps, limit, error_if_nonfinite=False)
        require_finite(norm, context + " gradient norm")
    scaler.step(optimizer)
    scaler.update()
    optimizer._decision_overflow_skips = 0
    for p in params:
        require_finite(p, context + " parameter after update")
    return True


def decision_distill_loss(slog, tlog, mask, sact, tact, temperature=1.25,
                          rank_margin=.15, rank_weight=.15):
    """FP32 probability/ranking arithmetic, including the AMP action head.

    Finite FP16 logits can still overflow their log-softmax differences; casting
    after log-softmax is too late. Never conceal non-finite model outputs.
    """
    from torch.nn import functional as F
    for name, value in (("student decision", slog), ("teacher decision", tlog),
                        ("student action", sact), ("teacher action", tact)):
        require_finite(value, name + " logits")
    s = slog.float().masked_fill(~mask.bool(), -1e4)
    t = tlog.detach().float().masked_fill(~mask.bool(), -1e4)
    def kl(student, teacher):
        sp = F.log_softmax(student / temperature, -1)
        tp = F.log_softmax(teacher / temperature, -1)
        return (tp.exp() * (tp-sp)).sum(-1).mean() * temperature**2
    target = t.argmax(-1)
    target_score = s.gather(1, target[:, None]).squeeze(1)
    other = s.clone().scatter(1, target[:, None], -1e4).max(-1).values
    rank = F.relu(rank_margin + other - target_score).mean()
    return (kl(s, t) + .30*F.cross_entropy(s, target) + rank_weight*rank
            + .05*kl(sact.float(), tact.detach().float()))


def run_recovery_phase(model, optimizer, groups, total_steps, lr_ranges,
                       loss_fn, evaluate_fn, score_fn, context, eval_every=50,
                       amp_enabled=False, warmup_steps=20, max_restarts=3,
                       backoff=.25, report_fn=None):
    """Train with fresh AMP state and bounded rollback to the best dev model.

    On a numerical failure, restore parameters, discard Adam moments, reset AMP,
    reduce all rates, and retry from the selected checkpoint's next step. Failed
    or AMP-skipped updates never count as completed training steps. If retries
    are exhausted, restore the best model and raise; do not export a failed run.
    """
    if len(lr_ranges) != len(optimizer.param_groups) or total_steps < 1:
        raise ValueError("Invalid recovery schedule/groups")
    if not 0 < backoff < 1 or max_restarts < 0 or eval_every < 1:
        raise ValueError("Invalid recovery retry settings")
    def finite_score(metrics):
        score = score_fn(metrics)
        values = score if isinstance(score, (tuple, list)) else (score,)
        if not all(math.isfinite(float(v)) for v in values):
            raise FloatingPointError(context + ": non-finite dev score")
        return score
    model.eval()  # deterministic recovery; inherited dropout remains disabled
    metrics = evaluate_fn()
    best_score = finite_score(metrics)
    best = trainable_snapshot(model)
    for value in best.values():
        require_finite(value, context + " baseline parameters")
    def restore():
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in best:
                    parameter.copy_(best[name].to(parameter.device, parameter.dtype))
        optimizer.zero_grad(set_to_none=True)
    def fresh_scaler():
        return torch.amp.GradScaler(next(model.parameters()).device.type,
                                   enabled=amp_enabled, init_scale=128.0)
    scaler = fresh_scaler()
    step, best_step, retries, skipped, updates = 0, -1, 0, 0, 0
    factor = 1.0
    history = []
    try:
        while step < total_steps:
            rates = [factor*recovery_lr(step, total_steps, start, end, warmup_steps)
                     for start, end in lr_ranges]
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = loss_fn(step)
                updated = guarded_step(loss, optimizer, scaler, groups, f"{context} step {step+1}")
                if not updated:
                    skipped += 1
                    continue
                updates += 1
                if step == 0 or (step+1) % eval_every == 0 or step+1 == total_steps:
                    candidate = evaluate_fn()
                    score = finite_score(candidate)
                    if score < best_score:
                        best, best_score, best_step = trainable_snapshot(model), score, step
                        metrics = candidate
                    record = {"step": step+1, "loss": float(loss.detach()),
                              "lr": rates, "metrics": candidate}
                    history.append(record)
                    if report_fn is not None:
                        report_fn(record)
                step += 1
            except FloatingPointError as error:
                loss = None  # release a failed forward graph before allocating the retry
                restore()
                optimizer.state.clear()  # moments from rejected trajectory must not survive
                optimizer._decision_overflow_skips = 0
                scaler = fresh_scaler()
                if retries >= max_restarts:
                    raise FloatingPointError(
                        f"{context}: retry budget exhausted; best dev checkpoint restored. {error}"
                    ) from error
                retries += 1
                factor *= backoff
                history.append({"failed_step": step+1, "restart_step": best_step+2,
                                "lr_factor": factor, "reason": str(error)})
                warnings.warn(f"{context}: {error}; restored best checkpoint, LR factor={factor:g}")
                step = best_step+1
    finally:
        restore()  # also preserves the best weights if the cell is interrupted
    return {"metrics": metrics, "best_step": best_step+1, "completed_steps": step,
            "optimizer_updates": updates, "amp_skips": skipped, "restarts": retries,
            "lr_factor": factor, "history": history}


def save_stage(model, path, metadata):
    """Stage boundaries are independently resumable, without optimizer state."""
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    for name, value in state.items():
        require_finite(value, "stage checkpoint " + name)
    temporary = path / 'weights.tmp.safetensors'
    save_file({n: t.detach().cpu().contiguous().clone() for n, t in state.items()}, str(temporary))
    temporary.replace(path / 'model.safetensors')
    (path / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_stage(model, path, metadata):
    from safetensors.torch import load_file
    path = Path(path)
    if json.loads((path / 'metadata.json').read_text()) != metadata:
        raise ValueError("Stage checkpoint configuration differs from this run")
    state = load_file(str(path / 'model.safetensors'))
    for name, value in state.items():
        require_finite(value, "loaded stage checkpoint " + name)
    model.load_state_dict(state, strict=True)


def forward_with_markers(model, batch):
    """Capture CLS/option representations from the same forward, without an extra encoder pass."""
    captured = {}
    def capture(module, args, output):
        h = output.last_hidden_state
        index = batch['marker_pos'][:, :, None].expand(-1, -1, h.shape[-1])
        captured['markers'] = torch.cat((h[:, :1], h.gather(1, index)), dim=1)
    hook = model.encoder.register_forward_hook(capture)
    try:
        logits, actions = model(**batch)
    finally:
        hook.remove()
    return logits, actions, captured['markers']


def marker_alignment_loss(student, teacher, option_mask):
    """Match pretrained representations at decision-relevant positions during recovery."""
    require_finite(student, 'student marker representations')
    require_finite(teacher, 'teacher marker representations')
    valid = torch.cat((torch.ones_like(option_mask[:, :1]), option_mask), dim=1).bool()
    s, t = student.float()[valid], teacher.detach().float()[valid]
    power = t.square().mean().clamp_min(1e-6)
    nmse = (s-t).square().mean() / power
    cosine = torch.nn.functional.cosine_similarity(s, t, dim=-1).mean()
    return nmse + (1-cosine)
