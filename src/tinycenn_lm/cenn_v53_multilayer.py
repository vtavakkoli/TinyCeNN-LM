"""Experimental orchestration helpers for six Qwen3.5 full-attention substitutions.

Only full-attention decoder layers are supported. The 18 native linear/gated
DeltaNet layers are deliberately left unchanged; they require a different API.
No KV-cache integration, production serving, or GPU speedup is implied.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)


def validate_full_attention_targets(layer_types, requested=FULL_ATTENTION_LAYERS):
    """Validate against Qwen's layer_types rather than treating all 24 as SDPA."""
    available = tuple(i for i, kind in enumerate(layer_types) if kind == 'full_attention')
    targets = tuple(int(i) for i in requested)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError('Full-attention target list must be nonempty and unique')
    if tuple(sorted(targets)) != targets:
        raise ValueError('Targets must be ordered from early to late layers')
    if any(i not in available for i in targets):
        raise ValueError(f'Targets {targets} not subset of native full-attention layers {available}')
    return targets


class FullAttentionGraftManager:
    """Temporarily or persistently graft V5.3 adapters, with safe restoration.

    `installed` refers to replacements under test, not quality-gate acceptance.
    Every adapter must retain a reference to its exact original module.
    """

    def __init__(self, layers, targets=FULL_ATTENTION_LAYERS):
        self.layers = layers
        self.targets = tuple(targets)
        self.native = {i: layers[i].self_attn for i in self.targets}
        self.installed = {}

    def install(self, idx, adapter):
        if idx not in self.native or adapter.original is not self.native[idx]:
            raise ValueError('Adapter must wrap native attention in a supported layer')
        current = self.layers[idx].self_attn
        if current is not self.native[idx] and current is not self.installed.get(idx):
            raise RuntimeError(f'Layer {idx} was modified outside this manager')
        self.layers[idx].self_attn = adapter
        self.installed[idx] = adapter

    @contextmanager
    def candidate(self, idx, adapter):
        if idx in self.installed:
            raise RuntimeError('Use a new, not-yet-installed target for candidate evaluation')
        if idx not in self.native or adapter.original is not self.native[idx]:
            raise ValueError('Candidate does not wrap the native attention')
        if self.layers[idx].self_attn is not self.native[idx]:
            raise RuntimeError('Candidate target is not currently native')
        self.layers[idx].self_attn = adapter
        try:
            yield
        finally:
            self.layers[idx].self_attn = self.native[idx]

    @contextmanager
    def native_context(self) -> Iterator[None]:
        """Restore native attention for a baseline run, even on exceptions."""
        original = {i: self.layers[i].self_attn for i in self.targets}
        try:
            for i in self.targets:
                self.layers[i].self_attn = self.native[i]
            yield
        finally:
            for i, module in original.items():
                self.layers[i].self_attn = module

    @property
    def installed_layers(self):
        return tuple(i for i in self.targets if i in self.installed)

    def restore_all(self):
        for i in self.targets:
            self.layers[i].self_attn = self.native[i]
        self.installed.clear()


@dataclass(frozen=True)
class GateResult:
    local_cosine: float
    teacher_agreement: float
    delta_nll: float
    speedup: float
    quality_pass: bool
    deploy_pass: bool


def check_quality(local_cosine, teacher_agreement, delta_nll, speedup,
                  min_cosine=.95, min_agreement=.98, max_delta_nll=.10,
                  min_speedup=1.05):
    values = (local_cosine, teacher_agreement, delta_nll, speedup)
    import math
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Nonfinite validation metrics')
    quality = (local_cosine >= min_cosine and teacher_agreement >= min_agreement
               and delta_nll <= max_delta_nll)
    return GateResult(float(local_cosine), float(teacher_agreement),
                      float(delta_nll), float(speedup), bool(quality),
                      bool(quality and speedup > min_speedup))


def experiment_fingerprint(spec):
    """Stable SHA256 over JSON-compatible config, model/data/source provenance."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def append_jsonl(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()
