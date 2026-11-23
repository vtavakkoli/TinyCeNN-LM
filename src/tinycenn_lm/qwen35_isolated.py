"""Qwen3.5-0.8B controlled attention isolation experiments.

Modes:
  delta: replace selected Gated DeltaNet mixers with causal window+anchor;
         preserve all native full-attention mixers.
  full:  replace selected GQA attention mixers with causal window+anchor;
         preserve all native DeltaNet mixers.

The model is text-only, no-cache, full-prefix research architecture.
Not a streaming/KV-cache-compatible implementation. No CeNN components.
"""
import json
from pathlib import Path
import torch
from safetensors.torch import save_model, load_model
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from tinycenn_lm.qwen35_window_anchor_v3 import (
    WindowAnchorDeltaV2, WindowAnchorFull, load_official_text,
)

VERSION="isolated-attention-v1"
SUPPORTED=("delta","full")


def layer_ids(model, mode):
    assert mode in SUPPORTED
    want="linear_attention" if mode=="delta" else "full_attention"
    layers=model.model.layers
    if len(layers)!=24:
        raise ValueError("Expected the 24-layer Qwen3.5-0.8B TEXT backbone")
    return [i for i,layer in enumerate(layers) if layer.block_type==want]


def substitute(model, index, mode, delta_window=128, full_window=512,
               anchors=16, chunk_size=64):
    index=int(index)
    ids=layer_ids(model,mode)
    if index not in ids:
        raise ValueError(f"Layer {index} is not a {mode} layer; available={ids}")
    layer=model.model.layers[index]
    if mode=="delta":
        if isinstance(layer.linear_attn,WindowAnchorDeltaV2):
            return False
        layer.linear_attn=WindowAnchorDeltaV2(
            layer.linear_attn,window=int(delta_window),
            anchors=int(anchors),chunk_size=int(chunk_size))
    else:
        if isinstance(layer.self_attn,WindowAnchorFull):
            return False
        layer.self_attn=WindowAnchorFull(
            layer.self_attn,window=int(full_window),
            anchors=int(anchors),chunk_size=int(chunk_size))
    return True


def install_selected(model, mode, selected_indices, **settings):
    expected=set(layer_ids(model,mode))
    selected=[int(x) for x in selected_indices]
    if len(selected)!=len(set(selected)) or not set(selected)<=expected:
        raise ValueError("Selected indices contain duplicates or incorrect types")
    for i in selected:
        substitute(model,i,mode,**settings)
    return describe(model)


def describe(model):
    native_delta=native_full=replaced_delta=replaced_full=0
    selected_delta=[];selected_full=[]
    for i,layer in enumerate(model.model.layers):
        if layer.block_type=="linear_attention":
            if isinstance(layer.linear_attn,WindowAnchorDeltaV2):
                replaced_delta+=1;selected_delta.append(i)
            else:native_delta+=1
        elif layer.block_type=="full_attention":
            if isinstance(layer.self_attn,WindowAnchorFull):
                replaced_full+=1;selected_full.append(i)
            else:native_full+=1
        else:raise ValueError(f"Unexpected Qwen layer type: {layer.block_type}")
    assert replaced_delta+native_delta==18
    assert replaced_full+native_full==6
    return {
        "replaced_delta":replaced_delta,"native_delta":native_delta,
        "replaced_full":replaced_full,"native_full":native_full,
        "delta_indices":selected_delta,"full_indices":selected_full
    }


def original_ffn_digest(model):
    """Fingerprint original decoder FFNs and input/post-attention norms."""
    import hashlib
    sha=hashlib.sha256()
    for i,layer in enumerate(model.model.layers):
        for name in ("mlp","input_layernorm","post_attention_layernorm"):
            for key,val in getattr(layer,name).state_dict().items():
                sha.update(f"{i}/{name}/{key}".encode())
                sha.update(val.detach().float().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def save_isolated(model,tokenizer,root,manifest):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model.cpu().eval()
    model.config.save_pretrained(str(root))
    tokenizer.save_pretrained(str(root))
    (root/"isolated_manifest.json").write_text(
        json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    save_model(model,str(root/"model.safetensors"))
    (root/"qwen35_isolated.py").write_text(
        Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    from tinycenn_lm import qwen35_window_anchor_v3 as backend
    (root/"qwen35_window_anchor_v3.py").write_text(
        Path(backend.__file__).read_text(encoding="utf-8"),encoding="utf-8")


def load_isolated(root,device="cuda"):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    root=Path(root)
    metadata=json.loads((root/"isolated_manifest.json").read_text())
    config=Qwen3_5TextConfig.from_pretrained(str(root))
    dtype=torch.float16 if metadata["model_dtype"]=="float16" else torch.bfloat16
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model=Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(old)
    install_selected(model,metadata["mode"],metadata["selected_indices"],
                     **metadata["attention_settings"])
    model.to(dtype=dtype)
    load_model(model,str(root/"model.safetensors"),strict=True,device="cpu")
    model.eval().requires_grad_(False).to(device)
    return model,AutoTokenizer.from_pretrained(str(root)),describe(model)
