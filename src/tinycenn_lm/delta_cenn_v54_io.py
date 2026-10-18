"""Artifact and reload helpers for quality-gated Delta-CeNN V5.4 Qwen research.

These are custom-adapter files, NOT a standard AutoModel checkpoint. The native
Qwen3.5 weights must be fetched separately at the recorded pinned revision.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_adapter_package(dest: str | Path, output: str | Path, *,
                          model_id: str, model_revision: str, git_sha: str,
                          selected_layers, full_layers, replacement_mode: str,
                          source_sha256: str, run_fingerprint: str,
                          final_audit: dict | None, fast_eval: dict | None,
                          experimental: bool) -> dict[str, Any]:
    """Create a reusable *research* adapter artifact without base-model weights.

    Includes selected weights under ``selected/`` and rejected/experimental
    candidate weights under ``candidates/`` (for reproduction, not inference).
    No external I/O, credentials, telemetry or HF upload is performed here.
    """
    dest, output = Path(dest), Path(output)
    if dest.resolve() == output.resolve():
        raise ValueError('Destination cannot overwrite original results')
    dest.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    full = {int(x) for x in full_layers}
    selected = sorted({int(x) for x in selected_layers})
    if not set(selected).issubset(set(range(24))):
        raise ValueError('Selected decoder layer is outside Qwen3.5-0.8B')
    if replacement_mode not in {'one_group', 'both_groups'}:
        raise ValueError('Invalid attention replacement mode')
    candidates = []
    for src in sorted(checkpoints.glob('L??_*_core.pt')):
        if not src.is_file():
            continue
        # It is critical not to mark failed candidates as installed.
        layer = int(src.name[1:3])
        family = 'full_attention' if layer in full else 'linear_attention'
        expected = f'L{layer:02d}_{family}_core.pt'
        if src.name != expected:
            raise ValueError(f'Unexpected checkpoint naming: {src.name}')
        subdir = 'selected' if layer in selected else 'candidates'
        target = dest / subdir / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        candidates.append(dict(layer=layer, family=family, selected=layer in selected,
                               path=str(target.relative_to(dest)), sha256=_sha256(target),
                               bytes=target.stat().st_size))
    seen = {a['layer'] for a in candidates}
    if not set(selected).issubset(seen):
        raise FileNotFoundError(f'Selected layers without saved checkpoints: {set(selected)-seen}')
    for name in ('final_audit.json', 'run_status.json', 'training.log',
                 'training_steps.jsonl', 'cumulative_layer_results.csv',
                 'fast_eval.json', 'parity_diagnostics.json'):
        src = output / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    passed = bool(final_audit and final_audit.get('final_pass') and not experimental)
    manifest = {
        'format': 'tinycenn-qwen35-v54-adapter-v1',
        'model_id': model_id,
        'model_revision': model_revision,
        'source_repo': 'https://github.com/vtavakkoli/TinyCeNN-LM',
        'git_sha': git_sha,
        'source_sha256': source_sha256,
        'run_fingerprint': run_fingerprint,
        'replacement_mode': replacement_mode,
        'full_attention_layers': sorted(full),
        'selected_layers': selected,
        'candidate_checkpoints': candidates,
        'research_only': not passed,
        'deploy_quality_pass': passed,
        'final_audit': final_audit,
        'fast_eval_summary': {k:v for k,v in (fast_eval or {}).items() if k != 'greedy_texts'},
        'loading_note': 'Use tinycenn_lm.delta_cenn_v54_io.load_delta_cenn_bundle; not AutoModel.from_pretrained',
    }
    (dest / 'adapter_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def prepare_hf_model_card(manifest: dict, repo_id: str) -> str:
    status = 'RESEARCH ONLY — failed or incomplete acceptance checks' if manifest['research_only'] else 'Quality gates passed in stated limited evaluation'
    return f'''---
base_model: {manifest['model_id']}
tags:
- pytorch
- delta-cenn
- research
- qwen3.5
---
# Delta-CeNN V5.4 Qwen3.5 research adapter

**Status: {status}.** The checkpoint is a research ablation, not a demonstrated
faster or better model. Native Qwen is the appropriate reference if acceptance fails.

Repository: https://github.com/vtavakkoli/TinyCeNN-LM  
Hugging Face: https://huggingface.co/{repo_id}  
Base model: `{manifest['model_id']}`  
Pinned base revision: `{manifest['model_revision']}`

## What is saved?

Only custom CeNN core weights and metadata. The *selected* directory lists
actually installed layers: `{manifest['selected_layers']}`. The `candidates`
directory contains failed/skipped ablations **not included in the student**.
The base Qwen checkpoint, its token embeddings, unchanged layers and frozen
projections are downloaded separately at the recorded revision.

This repository is **not directly loadable with AutoModel.from_pretrained**.
The experimental hybrid decoder does not support Qwen's cached `.generate()`.

## Load after installing TinyCeNN-LM

```python
from huggingface_hub import snapshot_download
from tinycenn_lm.delta_cenn_v54_io import load_delta_cenn_bundle
path = snapshot_download('{repo_id}')
model, report = load_delta_cenn_bundle(path)
print(report['selected_layers'])
# model: evaluation/prefill only, pass use_cache=False
```

## Evidence and limitations

See `final_audit.json`, `fast_eval.json`, `training_steps.jsonl` and
`cumulative_layer_results.csv`. Model gates measure WikiText-2 on a
small controlled Colab profile; this is not independent production validation.
FP32 eager Delta-CeNN is not a fused kernel; speedups are not assumed.
'''


def load_delta_cenn_bundle(path: str | Path, *, device: str = 'cpu',
                            dtype=None, allow_research: bool = True,
                            model=None):
    """Rehydrate ONLY genuinely installed V5.4 CeNN cores onto pinned Qwen.

    If `model` is provided, it must already be the specified native Qwen model
    at the same revision, not a previously modified adapter instance.
    """
    path = Path(path)
    info = json.loads((path/'adapter_manifest.json').read_text())
    if info.get('format') != 'tinycenn-qwen35-v54-adapter-v1':
        raise ValueError('Unknown adapter artifact format')
    if info['research_only'] and not allow_research:
        raise ValueError('Research-only checkpoint explicitly rejected by loader policy')
    import torch
    from transformers import Qwen3_5ForCausalLM
    from .delta_cenn_qwen35 import Qwen35DeltaCeNNLinear
    from .cenn_head_rescue_v5 import Qwen35HeadRescueV5
    from .cenn_head_rescue_v53 import CeNNHeadRescueV53
    if model is None:
        model = Qwen3_5ForCausalLM.from_pretrained(
            info['model_id'], revision=info['model_revision'],
            dtype=dtype or (torch.bfloat16 if str(device).startswith('cuda') else torch.float32),
            attn_implementation='sdpa').to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    found = {int(e['layer']): e for e in info['candidate_checkpoints'] if e['selected']}
    if sorted(found) != sorted(info['selected_layers']):
        raise ValueError('Installed layer list and checkpoint manifest disagree')
    for idx in sorted(found):
        item = found[idx]
        weights_path = path / item['path']
        if _sha256(weights_path) != item['sha256']:
            raise ValueError(f'Corrupt layer {idx} checkpoint')
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        if int(checkpoint['layer']) != idx:
            raise ValueError('Incorrect layer checkpoint')
        layer = model.model.layers[idx]
        if item['family'] == 'linear_attention':
            coreconf = checkpoint['core_config']
            adapter = Qwen35DeltaCeNNLinear(layer.linear_attn,
                feature_dim=int(coreconf['feature_dim']),
                cell_interval=int(coreconf['cell_interval']),cells=True)
            adapter.core.load_state_dict(checkpoint['core_state'], strict=True)
            layer.linear_attn = adapter.to(device).eval()
        elif item['family'] == 'full_attention':
            conf = checkpoint['core_config']
            core = CeNNHeadRescueV53(**conf)
            core.load_state_dict(checkpoint['core_state'], strict=True)
            exact = (0,) if info['replacement_mode'] == 'one_group' else ()
            layer.self_attn = Qwen35HeadRescueV5(layer.self_attn, core.to(device),
                                                  exact_groups=exact).to(device).eval()
        else:
            raise ValueError('Unknown saved attention family')
    return model, info
