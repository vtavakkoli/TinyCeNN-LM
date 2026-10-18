"""Ground-truth-only end-to-end training for a CCM first-attention-layer ablation.

Training targets are actual next tokens, never teacher outputs or teacher agreement.
The frozen pretrained backbone is not trained; only the cellular replacement changes.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from scripts.benchmark_ccm_consensus_smol360 import attention_slot


def gold_training_step(model, native, adapter, batch, optimizer, *, max_grad_norm=1.0):
    """Update the adapter on gold next-token cross entropy. Returns scalar diagnostics.

    A LlamaDecoderLayer adds the attention residual itself; the adapter returns
    only the mixer update. Cached decoding is deliberately disabled.
    """
    if batch.ndim != 2 or batch.shape[1] < 2:
        raise ValueError('batch must be [B,T] with T >= 2')
    if model.model.layers[0].self_attn is not native:
        raise RuntimeError('expected native attention before entering a replacement context')
    model.eval()
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    with attention_slot(model, adapter):
        logits = model(input_ids=batch, use_cache=False, return_dict=True).logits[:, :-1].float()
    gold = batch[:, 1:]
    if logits.shape[:2] != gold.shape:
        raise RuntimeError('label/logit time dimension mismatch')
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite training loss')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.cell.parameters(), max_grad_norm)
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError('nonfinite gradient norm')
    optimizer.step()
    with torch.no_grad():
        gold_accuracy = (logits.detach().argmax(dim=-1) == gold).float().mean().item()
    return {'train_gold_ce': float(loss.detach().item()),
            'train_gold_accuracy': float(gold_accuracy),
            'grad_norm': float(grad_norm.detach().item())}


def check_gold_metrics(metrics):
    """Score only real-label metrics and guard against NaNs."""
    keys = ('teacher_nll', 'student_nll', 'student_gold_accuracy', 'teacher_gold_accuracy')
    if not all(k in metrics for k in keys):
        raise ValueError('missing gold-label evaluation fields')
    if not all(math.isfinite(float(metrics[k])) for k in keys):
        raise FloatingPointError('nonfinite gold-label evaluation')
    return {'gold_nll': float(metrics['student_nll']),
            'gold_accuracy': float(metrics['student_gold_accuracy']),
            'gold_nll_delta': float(metrics['student_nll']-metrics['teacher_nll']),
            'gold_accuracy_drop': float(metrics['teacher_gold_accuracy']-metrics['student_gold_accuracy'])}
