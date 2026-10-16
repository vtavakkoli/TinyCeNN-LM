"""Inference-only, non-mutating structured Laya layer substitutions (V3)."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from torch import nn
from fast_inference import FastLaya

MODES = ('drop_attn', 'drop_mlp', 'drop_block', 'mlp75', 'mlp50')


class NarrowMLP(nn.Module):
    """Preserve paired GLU rows and matching output columns; physically smaller GEMMs."""
    def __init__(self, source, indices):
        super().__init__()
        index = torch.tensor(indices, device=source.Wi.weight.device, dtype=torch.long)
        n = source.Wo.in_features
        if not len(indices) or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= n:
            raise ValueError('Invalid GLU channel indices')
        rows = torch.cat((index, index + n))
        kw = dict(device=source.Wi.weight.device, dtype=source.Wi.weight.dtype)
        self.Wi = nn.Linear(source.Wi.in_features, 2 * len(indices), bias=source.Wi.bias is not None, **kw)
        self.Wo = nn.Linear(len(indices), source.Wo.out_features, bias=source.Wo.bias is not None, **kw)
        with torch.no_grad():
            self.Wi.weight.copy_(source.Wi.weight[rows])
            self.Wo.weight.copy_(source.Wo.weight[:, index])
            if self.Wi.bias is not None:
                self.Wi.bias.copy_(source.Wi.bias[rows])
            if self.Wo.bias is not None:
                self.Wo.bias.copy_(source.Wo.bias)
        self.act = source.act
        self.eval().requires_grad_(False)

    def forward(self, hidden_states):
        value, gate = self.Wi(hidden_states).chunk(2, -1)
        return self.Wo(self.act(value) * gate)


class ChangedLayer(nn.Module):
    def __init__(self, source, spec):
        super().__init__()
        self.mode = spec['mode']
        if self.mode not in MODES:
            raise ValueError(self.mode)
        self.use_attn = self.mode not in ('drop_attn', 'drop_block')
        self.use_mlp = self.mode not in ('drop_mlp', 'drop_block')
        if self.use_attn:
            self.attn_norm, self.attn = source.attn_norm, source.attn
        if self.use_mlp:
            self.mlp_norm = source.mlp_norm
            self.mlp = NarrowMLP(source.mlp, spec['indices']) if self.mode.startswith('mlp') else source.mlp
        self.eval()

    def forward(self, hidden_states, **kwargs):
        if self.use_attn:
            hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), **kwargs)[0]
        if self.use_mlp:
            hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return (hidden_states,)


def apply_plan(original, plan):
    """Fork module containers, share frozen originals, allocate only narrowed FFNs."""
    if original.training or any(p.requires_grad for p in original.parameters()):
        raise ValueError('Expected frozen eval model')
    if original.encoder.config._attn_implementation != 'sdpa':
        raise ValueError('V3 requires the pinned ModernBERT SDPA implementation')
    plan = {int(k): v for k, v in plan.items()}
    if any(i < 0 or i >= len(original.encoder.layers) for i in plan):
        raise ValueError('Layer index outside encoder')
    fork = copy.copy(original)
    fork._modules = original._modules.copy()
    encoder = copy.copy(original.encoder)
    encoder._modules = original.encoder._modules.copy()
    encoder.layers = nn.ModuleList([
        ChangedLayer(layer, plan[i]) if i in plan else layer
        for i, layer in enumerate(original.encoder.layers)
    ])
    fork.encoder = encoder
    result = FastLaya(fork)
    if not result.device_masks:
        raise ValueError('Use transformers==4.57.6 and ModernBertModel with SDPA')
    return result.eval()


def channel_specs(original, scores):
    specs = {}
    for i, layer in enumerate(original.encoder.layers):
        n = layer.mlp.Wo.in_features
        score = scores[i].float()
        if score.shape != (n,) or not torch.isfinite(score).all():
            raise ValueError(f'Invalid calibration for layer {i}')
        modes = {m: {'mode': m} for m in MODES[:3]}
        for name, fraction in (('mlp75', .75), ('mlp50', .5)):
            # Align widths to tensor-core-friendly 64 channels, except tiny tests.
            width = min(n, max(1, int(n * fraction / 64) * 64)) if n >= 128 else max(1, int(n * fraction))
            indices = torch.topk(score, width).indices.sort().values.tolist()
            modes[name] = dict(mode=name, indices=indices)
        specs[i] = modes
    return specs


def load_selected(recipe_path, device='cuda'):
    """Reconstruct a small recipe using its immutable original HF checkpoint."""
    import importlib.util
    import sys
    from huggingface_hub import snapshot_download
    recipe = json.loads(Path(recipe_path).read_text())
    if recipe.get('format') != 'laya-layer-search-v3-recipe':
        raise ValueError('Not a V3 recipe')
    folder = Path(snapshot_download(recipe['model_id'], revision=recipe['model_revision'],
        allow_patterns=['*.json', '*.py', '*.safetensors', 'encoder/*', 'tokenizer/*']))
    spec = importlib.util.spec_from_file_location('laya_v3_checkpoint_runtime', folder / 'modeling_laya_standalone.py')
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    original, tokenizer = runtime.load_model(folder, device=device, compile_head=False)
    return apply_plan(original, recipe['plan']), tokenizer, runtime
