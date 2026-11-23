"""V5.6 research-policy helpers for four-way Qwen35 Delta-CeNN ablations.

A = pretrained native; B = 18 chunk-cell Delta-CeNN + native full attention;
C = B with quality-accepted V5.3 full-attention heads; D = forced 24-layer
comparison. Selection is validation-only and never chooses forced D.
This module intentionally has no torch/CUDA/HF dependency.
"""
from __future__ import annotations
from contextlib import contextmanager
from math import isfinite

ARMS = ('A_native', 'B_delta18', 'C_gated_hybrid', 'D_forced24')


def validate_arm_rows(rows):
    mapping = {str(r['arm']): r for r in rows}
    if set(mapping) != set(ARMS) or len(rows) != len(ARMS):
        raise ValueError('Exactly four independent arm records are required')
    for label, record in mapping.items():
        for name in ('nll', 'agreement', 'speedup'):
            if not isfinite(float(record[name])):
                raise ValueError(f'{label}: nonfinite {name}')
        if not 0 <= float(record['agreement']) <= 1 or float(record['speedup']) <= 0:
            raise ValueError(f'{label}: invalid agreement/speedup')
    return mapping


def select_validation_arm(rows, *, min_agreement=0.98,
                          max_delta_nll=0.02, min_speedup=0.90,
                          max_hybrid_slowdown=0.03):
    """Prefer B when quality holds; C only for a meaningful non-degrading gain.

    D is research-only regardless of its metrics. Even a selected B/C is NOT a
    deployed model unless a separate audit demonstrates >1.05x speed.
    """
    a = validate_arm_rows(rows)
    native = float(a['A_native']['nll'])

    def passes(label):
        row = a[label]
        return (float(row['agreement']) >= min_agreement
                and float(row['nll'])-native <= max_delta_nll
                and float(row['speedup']) >= min_speedup)

    if not passes('B_delta18'):
        return {'selected': 'A_native', 'reason': 'B failed validation quality/latency budget',
                'deploy_approved': False}
    if passes('C_gated_hybrid'):
        b, c = a['B_delta18'], a['C_gated_hybrid']
        improved = (float(c['agreement']) > float(b['agreement']) + 0.002
                    or float(c['nll']) < float(b['nll']) - 0.002)
        not_slower = float(c['speedup']) >= (1-max_hybrid_slowdown)*float(b['speedup'])
        if improved and not_slower:
            return {'selected': 'C_gated_hybrid', 'reason': 'Hybrid improves fidelity without excessive slowdown',
                    'deploy_approved': False}
    return {'selected': 'B_delta18', 'reason': 'B passes and C brings no acceptable gain',
            'deploy_approved': False}


@contextmanager
def temporary_full_attention(layers, native, candidates):
    """Temporarily graft trained full-attention candidates, including rejected.

    Does not touch linear-attention layers or mutate an installed-layer manager.
    Restores exactly the previously present layer objects after exceptions.
    """
    old = {}
    try:
        for i, replacement in sorted(candidates.items()):
            if i not in native or replacement.original is not native[i]:
                raise ValueError(f'Invalid V5.3 full-attention adapter for layer {i}')
            old[i] = layers[i].self_attn
            layers[i].self_attn = replacement
        yield
    finally:
        for i, previous in old.items():
            layers[i].self_attn = previous
