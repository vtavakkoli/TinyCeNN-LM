"""Standalone loader for the Jev_CeNN Bonsai-inspired g128 packed bundle.
Review bundled Python before loading. The source model has a custom runtime.
"""
from pathlib import Path
import importlib.util
import json
import sys


def _import(name, file):
    spec=importlib.util.spec_from_file_location(name,str(file))
    if spec is None or spec.loader is None:raise ImportError(f'Cannot import {file}')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def load_bonsai(path_or_repo, device='cuda', token=None, revision=None):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    root=Path(str(path_or_repo))
    if not (root.is_dir() and (root/'bonsai_meta.json').is_file()):
        root=Path(snapshot_download(repo_id=str(path_or_repo), token=token, revision=revision))
    manifest=json.loads((root/'bundle.json').read_text())
    meta=json.loads((root/'bonsai_meta.json').read_text())
    assert manifest.get('architecture') == 'laya-v31-all-attention'
    assert meta.get('format') == 'bonsai_inspired_q2_g128_v1'
    base_runtime=_import('_bonsai_runtime_base',root/'runtime.py')
    qr=_import('_bonsai_runtime_ops',root/'jev_bonsai_g128_runtime.py')
    model=base_runtime.build_model(root,manifest).to(dtype=torch.float16)
    for spec in meta['quantized_modules']:
        name=spec['name']
        old=model.get_submodule(name)
        if not isinstance(old,torch.nn.Linear):raise TypeError(f'Expected linear at {name}')
        new=qr.TernaryG128Linear(old.in_features,old.out_features,old.bias is not None)
        qr.swap(model,name,new)
    weights=load_file(str(root/'model_bonsai.safetensors'),device='cpu')
    model.load_state_dict(weights,strict=True)
    model.to(device=device).eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(root/'tokenizer'))
    return model,tokenizer,base_runtime
