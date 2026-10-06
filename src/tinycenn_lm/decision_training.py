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
        torch.nn.utils.clip_grad_norm_(ps, limit, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    optimizer._decision_overflow_skips = 0
    for p in params:
        require_finite(p, context + " parameter after update")
    return True


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
