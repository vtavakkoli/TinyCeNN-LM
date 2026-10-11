"""Jev-CeNN experimental 15/16/17 -> 1 ModernBERT superlayer.

Two query-dependent sparse attentions are composed sequentially:
  A1: centered radius R1;
  A2: dilated offsets stride S, radius R2.
With R1=S-1, R2=floor((T_max-1)/S), every token can reach every
other within two hops for T<=T_max. Reachability does NOT imply that
arbitrary full-attention weights can be exactly factorized by A2 A1.

One trainable CeNN multiscale module is coupled into BOTH stages before
stage-2 QKV projection, so CeNN's effect can be measured by truly disabling
its computation. The module also includes ONE FFN instead of three.

Only the 15th entry of encoder.layers is replaced by the new superlayer;
entries 16 and 17 become parameter-free bypass modules that maintain the
ModernBERT encoder-layer forward contract.

This is an experimental architecture and is not guaranteed to converge.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tinycenn_lm.laya_lab.core import _valid_tokens


def pair_reachable(length: int, local_radius: int, stride: int,
                   dilated_radius: int) -> bool:
    """Exact Boolean reachability; deterministic and model-independent."""
    if length < 1 or stride < 1 or local_radius < 0 or dilated_radius < 0:
        raise ValueError("Invalid two-window geometry")
    t=torch.arange(length)
    hops=t[:,None]+stride*torch.arange(
        -dilated_radius,dilated_radius+1)[None,:]
    valid=(hops>=0)&(hops<length)
    neighbors=t[None,None,:]
    cover=(valid[:,:,None]&
           ((hops[:,:,None]-neighbors).abs()<=local_radius)).any(1)
    return bool(cover.all())


def pair_coverage_fraction(length: int, local_radius: int,
                           stride: int, dilated_radius: int) -> float:
    t=torch.arange(length)
    hops=t[:,None]+stride*torch.arange(
        -dilated_radius,dilated_radius+1)[None,:]
    cover=(((hops[:,:,None]-t[None,None,:]).abs()<=local_radius)&
           (hops[:,:,None]>=0)&(hops[:,:,None]<length)).any(1)
    return float(cover.float().mean())


def sparse_attention(q:Tensor,k:Tensor,v:Tensor,valid:Tensor,
                     *,radius:int,stride:int) -> Tensor:
    """Differentiable sparse exact-softmax attention on fixed strided indices.

    O(B*H*T*(2R+1)*D) intermediate memory. Native flash/global attention
    can beat this on T4; the notebook measures actual latency.
    q/k/v: [B,H,T,D]; valid: [B,T].
    """
    if radius < 0 or stride < 1:
        raise ValueError("Invalid sparse pattern")
    b,h,t,d=q.shape
    indices=torch.arange(t,device=q.device)[:,None]+stride*torch.arange(
        -radius,radius+1,device=q.device)[None,:]
    in_bounds=(indices>=0)&(indices<t)
    gather_index=indices.clamp(0,t-1)
    kk=k[:,:,gather_index,:]
    vv=v[:,:,gather_index,:]
    valid_keys=valid[:,gather_index]&in_bounds[None,:,:]
    scores=(q.unsqueeze(3).float()*kk.float()).sum(-1)*d**-0.5
    scores=scores.masked_fill(~valid_keys[:,None,:,:],-1e4)
    probabilities=torch.softmax(scores,dim=-1).to(v.dtype)
    out=(probabilities.unsqueeze(-1)*vv).sum(3)
    return out*valid[:,None,:,None].to(out.dtype)


class CellularMultiscale(nn.Module):
    """Trainable CeNN-inspired local+dilated depthwise neighborhood operator."""

    def __init__(self,hidden:int):
        super().__init__()
        self.norm=nn.LayerNorm(hidden)
        self.filters=nn.ModuleList([
            nn.Conv1d(hidden,hidden,kernel_size=7,padding=3,groups=hidden),
            nn.Conv1d(hidden,hidden,kernel_size=9,padding=16,
                      dilation=4,groups=hidden),
            nn.Conv1d(hidden,hidden,kernel_size=9,padding=64,
                      dilation=16,groups=hidden),
        ])
        self.mix=nn.Linear(3*hidden,hidden,bias=False)
        self.gate=nn.Linear(hidden,hidden)
        # Initialize cellular mixing as a local smooth/identity family so that
        # stage-A training starts from a stable, nontrivial signal.
        with torch.no_grad():
            for conv in self.filters:
                conv.weight.zero_()
                conv.weight[:,0,conv.kernel_size[0]//2]=1.0
                if conv.bias is not None:
                    conv.bias.zero_()
            self.gate.weight.zero_()
            self.gate.bias.zero_()

    def forward(self,x:Tensor,valid:Tensor)->Tensor:
        z=self.norm(x)
        z=z*valid[:,:,None].to(z.dtype)
        packed=z.transpose(1,2).contiguous()
        features=torch.cat([layer(packed).transpose(1,2)
                            for layer in self.filters],dim=-1)
        update=self.mix(features)*torch.sigmoid(self.gate(z))
        return update*valid[:,:,None].to(update.dtype)


class BypassModernBertLayer(nn.Module):
    """Skip one *entire* encoder layer, including attention, MLP, norms/residuals."""

    def forward(self,hidden_states:Tensor,attention_mask=None,
                sliding_window_mask=None,position_ids=None,
                cu_seqlens=None,max_seqlen=None,
                output_attentions=False,**kwargs):
        return (hidden_states,None) if output_attentions else (hidden_states,)


class JevCeNNPairWindowSuperLayer(nn.Module):
    """Compress original encoder layers 15+16+17 into ONE trainable block."""

    architecture="jev_cenn_pair_window_superlayer_v1"

    def __init__(self,teacher_layer15:nn.Module,teacher_layer16:nn.Module,
                 *,local_radius:int=22,stride:int=23,
                 dilated_radius:int=22,max_len:int=512):
        super().__init__()
        for name,layer in (("global 15",teacher_layer15),
                           ("sliding 16",teacher_layer16)):
            if not hasattr(layer,"attn") or not hasattr(layer,"mlp"):
                raise TypeError(f"Expected complete ModernBERT layer: {name}")
        a=teacher_layer15.attn
        b=teacher_layer16.attn
        self.max_len=int(max_len)
        self.local_radius=int(local_radius)
        self.stride=int(stride)
        self.dilated_radius=int(dilated_radius)
        if not pair_reachable(self.max_len,self.local_radius,
                              self.stride,self.dilated_radius):
            raise ValueError("Two-hop coverage is not global")
        self.num_heads=int(a.num_heads)
        self.head_dim=int(a.head_dim)
        hidden=self.num_heads*self.head_dim
        self.norm1=copy.deepcopy(teacher_layer15.attn_norm)
        self.norm2=copy.deepcopy(teacher_layer16.attn_norm)
        self.qkv1=copy.deepcopy(a.Wqkv)
        self.qkv2=copy.deepcopy(b.Wqkv)
        self.out1=copy.deepcopy(a.Wo)
        self.out2=copy.deepcopy(b.Wo)
        self.rope1=copy.deepcopy(a.rotary_emb)
        self.rope2=copy.deepcopy(b.rotary_emb)
        # Trainable cellular mixing enters stage2 attention's Q/K/V.
        self.cenn=CellularMultiscale(hidden)
        self.cenn_gain1=nn.Parameter(torch.tensor(0.0))
        self.cenn_gain2=nn.Parameter(torch.tensor(-0.8))
        # ONE teacher-derived feedforward residual instead of three.
        self.mlp_norm=copy.deepcopy(teacher_layer15.mlp_norm)
        self.mlp=copy.deepcopy(teacher_layer15.mlp)
        self.output_scale=nn.Parameter(torch.tensor(0.0))
        self._use_cenn=True

    def set_cenn_enabled(self,enabled:bool):
        self._use_cenn=bool(enabled)

    def _attention(self,hidden_states,valid,position_ids,
                   *,stage:int,mask=None):
        qkv_proj=self.qkv1 if stage==1 else self.qkv2
        norm=self.norm1 if stage==1 else self.norm2
        wo=self.out1 if stage==1 else self.out2
        rotary=self.rope1 if stage==1 else self.rope2
        b,t,d=hidden_states.shape
        states=norm(hidden_states)
        qkv=qkv_proj(states).view(
            b,t,3,self.num_heads,self.head_dim)
        q,k,v=qkv.transpose(3,1).unbind(2)
        cos,sin=rotary(qkv,position_ids=position_ids)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        if stage==1:
            radius,stride=self.local_radius,1
        else:
            radius,stride=self.dilated_radius,self.stride
        context=sparse_attention(q,k,v,valid,radius=radius,stride=stride)
        projected=wo(context.transpose(1,2).reshape(b,t,d))
        return projected*valid[:,:,None].to(projected.dtype)

    def cellular_only_prediction(self,x:Tensor,attention_mask=None):
        valid=_valid_tokens(attention_mask,x)
        return x+self.cenn(x,valid)

    def forward(self,hidden_states:Tensor,attention_mask=None,
                sliding_window_mask=None,position_ids=None,
                cu_seqlens=None,max_seqlen=None,
                output_attentions=False,**kwargs):
        b,t,d=hidden_states.shape
        if t>self.max_len:
            raise ValueError(f"Input length {t} > configured {self.max_len}")
        valid=_valid_tokens(attention_mask,hidden_states)
        if position_ids is None:
            position_ids=torch.arange(t,device=hidden_states.device)[None,:]
        first=self._attention(hidden_states,valid,position_ids,stage=1)
        # CeNN is part of the first intermediate state and thus directly
        # controls QKV in the second, globally-reaching dilated attention.
        cellular1=self.cenn(hidden_states,valid) if self._use_cenn else 0.0
        mid=hidden_states+first+torch.sigmoid(self.cenn_gain1)*cellular1
        second=self._attention(mid,valid,position_ids,stage=2)
        cellular2=self.cenn(mid,valid) if self._use_cenn else 0.0
        hidden=mid+second+torch.sigmoid(self.cenn_gain2)*cellular2
        hidden=hidden+torch.sigmoid(self.output_scale)*self.mlp(
            self.mlp_norm(hidden))
        return (hidden,None) if output_attentions else (hidden,)


def install_superlayer(student,superlayer,indices=(15,16,17)):
    """Structural 3-to-1 substitution, returning originals for safe rollback."""
    indices=tuple(indices)
    if indices!=(15,16,17):
        raise ValueError("This controlled lab only compresses (15,16,17)")
    layers=student.encoder.layers
    if len(layers)<=17:
        raise ValueError("Encoder must contain at least 18 layers")
    original=tuple(layers[i] for i in indices)
    layers[15]=superlayer
    layers[16]=BypassModernBertLayer()
    layers[17]=BypassModernBertLayer()
    return original


def restore_original_layers(student,original,indices=(15,16,17)):
    for idx,layer in zip(indices,original):
        student.encoder.layers[idx]=layer
