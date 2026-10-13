"""Jev-CeNN all-28 standalone architecture loader.

Pin transformers==4.57.6, safetensors>=0.4, and install TinyCeNN-LM
from https://github.com/vtavakkoli/TinyCeNN-LM before importing this file.

This module is exported to Hugging Face alongside a FULL safetensors checkpoint.
No original model weights need to be downloaded at inference time.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
from tinycenn_lm.laya_lab.core import BaseLayaReplacementAttention, _valid_tokens
from tinycenn_lm.jev_cenn_pairwindow_superlayer import CellularMultiscale

def attention_sdpa(q,k,v,valid,window,pattern="local",anchors=0):
    """Functional attention with per-example right-padding-aware global anchors."""
    b,h,t,d=q.shape
    pos=torch.arange(t,device=q.device)
    if pattern=="full" or window>=t:
        allowed=valid[:,None,None,:]
    else:
        radius=(int(window)-1)//2
        local=(pos[:,None]-pos[None,:]).abs()<=radius
        allowed=valid[:,None,None,:]&local[None,None,:,:]
        if pattern=="anchors":
            ends=valid.long().sum(-1).clamp_min(1)
            keys=(pos[None,:]<anchors)|(
                pos[None,:]>=(ends[:,None]-anchors).clamp_min(0))
            half=max(1,anchors//2)
            queries=(pos[None,:]<half)|(
                pos[None,:]>=(ends[:,None]-half).clamp_min(0))
            allowed=valid[:,None,None,:]&(
                allowed|keys[:,None,None,:]|queries[:,None,:,None])
    out=F.scaled_dot_product_attention(
        q,k,v,attn_mask=allowed,dropout_p=0.0,is_causal=False)
    return out*valid[:,None,:,None].to(out.dtype)

def chunked_local_sdpa(q,k,v,valid,window,chunk=64):
    """Bound mask memory to O(T*(W+chunk)), rather than O(T*T)."""
    b,h,t,d=q.shape
    if window>=t:return attention_sdpa(q,k,v,valid,window)
    radius=(window-1)//2
    pieces=[]
    for start in range(0,t,chunk):
        stop=min(start+chunk,t)
        first=max(0,start-radius)
        last=min(t,stop+radius)
        qi=torch.arange(start,stop,device=q.device)
        ki=torch.arange(first,last,device=q.device)
        nearby=(qi[:,None]-ki[None,:]).abs()<=radius
        ok=valid[:,None,None,first:last]&nearby[None,None,:,:]
        self_key=(qi[:,None]==ki[None,:])[None,None,:,:]
        ok=ok|(~ok.any(dim=-1,keepdim=True)&self_key)
        piece=F.scaled_dot_product_attention(
            q[:,:,start:stop],k[:,:,first:last],v[:,:,first:last],
            attn_mask=ok,dropout_p=0.0)
        pieces.append(piece)
    out=torch.cat(pieces,dim=-2)
    return out*valid[:,None,:,None].to(out.dtype)

class WindowCeNNAttention(BaseLayaReplacementAttention):
    architecture="jev_cenn_all28_attention_only_frozen_ffn_v4"
    def __init__(self,original,window,pattern,anchors):
        super().__init__(original)
        self.window=int(window);self.pattern=str(pattern);self.anchors=int(anchors)
        self.rotary_emb=copy.deepcopy(original.rotary_emb)
        self.rotary_emb.requires_grad_(False)
        self.cenn=CellularMultiscale(self.hidden_size)
        self.cenn_gain=nn.Parameter(torch.tensor(0.0))
        self.use_cenn=True
        self.backend="dense"
        self.backend_used=None
        self.Wqkv.requires_grad_(True)
        self.Wo.requires_grad_(True)
        self.out_drop.requires_grad_(False)

    def forward(self,hidden_states,position_ids=None,attention_mask=None,**kwargs):
        b,t,d=hidden_states.shape
        if position_ids is None:
            position_ids=torch.arange(t,device=hidden_states.device)[None,:]
        packed=self.Wqkv(hidden_states).view(
            b,t,3,self.num_heads,self.head_dim)
        q,k,v=packed.transpose(1,3).unbind(2)
        cos,sin=self.rotary_emb(packed,position_ids=position_ids)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        valid=_valid_tokens(attention_mask,hidden_states)
        if self.backend=="chunked" and self.pattern=="local":
            result=chunked_local_sdpa(q,k,v,valid,self.window)
            self.backend_used="chunked"
        else:
            result=attention_sdpa(q,k,v,valid,self.window,
                                  self.pattern,self.anchors)
            self.backend_used="dense"
        if self.use_cenn:
            # This pathway can be OFF for free; initial gain = exactly zero.
            flat=result.transpose(1,2).reshape(b,t,d)
            flat=flat+0.1*torch.tanh(self.cenn_gain)*self.cenn(flat,valid)
            result=flat.view(b,t,self.num_heads,self.head_dim).transpose(1,2)
        return self.finish(result,hidden_states)



def build_all28_from_dir(model_dir, *, device="cpu"):
    """Rebuild the original 28-layer encoder with ALL attentions replaced.

    The original pretrained FFNs, LayerNorms, 28 residual paths and task head
    are loaded from the full safetensors checkpoint. No Hub code execution is
    required: import this file as a regular trusted Python module.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors.torch import load_model
    from tinycenn_lm.standalone_decision import (
        StandaloneDecisionModel, StandaloneDecisionRuntime, _apply_rope_compat
    )
    root=Path(model_dir)
    manifest=json.loads((root/"all28_manifest.json").read_text(encoding="utf-8"))
    source_cfg=json.loads((root/"rl_agent_config.json").read_text(encoding="utf-8"))
    enc_cfg=AutoConfig.from_pretrained(str(root/"encoder"))
    _apply_rope_compat(enc_cfg)
    encoder=AutoModel.from_config(enc_cfg,attn_implementation="sdpa")
    model=StandaloneDecisionModel(
        encoder,
        head_layers=int(source_cfg.get("head_layers",2)),
        n_act=len(source_cfg.get("act_costs",{}))+1
    )
    layers=manifest["attention_layers"]
    if len(layers)!=28 or {int(x["layer"]) for x in layers}!=set(range(28)):
        raise ValueError("The published model must replace ALL 28 attention layers")
    for entry in layers:
        i=int(entry["layer"])
        native=model.encoder.layers[i].attn
        conv=WindowCeNNAttention(
            native,entry["window"],entry["pattern"],entry["anchors"])
        conv.use_cenn=bool(entry["cenn_on"])
        conv.eval().requires_grad_(False)
        model.encoder.layers[i].attn=conv
    load_model(model,str(root/"model.safetensors"),strict=True,device="cpu")
    model.encoder.config.reference_compile=False
    model.eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(root/"tokenizer"))
    runtime=StandaloneDecisionRuntime(model,tokenizer,source_cfg,device=device)
    return runtime

def load_all28(repo_or_directory, *, token=None, device=None):
    """Load a complete published model locally OR by Hugging Face model ID.

    Usage:
        from all28_hf import load_all28
        runtime=load_all28("USERNAME/jev-cenn-all28-ffn-frozen",device="cuda")
        result=runtime.decide({"road_direction":"left"},
             ["left","straight","right"])
    """
    from huggingface_hub import snapshot_download
    p=Path(repo_or_directory)
    if not p.exists():
        p=Path(snapshot_download(
            str(repo_or_directory),token=token,
            allow_patterns=[
                "model.safetensors","all28_manifest.json",
                "rl_agent_config.json","encoder/*","tokenizer/*"
            ]))
    return build_all28_from_dir(p,device=device or
        ("cuda" if torch.cuda.is_available() else "cpu"))
