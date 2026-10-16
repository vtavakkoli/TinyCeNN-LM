"""Laya all-layer G128 mixed precision (2/4/8-bit) with native FP16 GEMM.

Research PTQ: packed checkpoint representation, lazy decoded FP16 CUDA weights.
It is NOT a native low-bit compute kernel, nor a claim of inference speedup.
Each linear can use signed ternary 2-bit, symmetric int4, or symmetric int8.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

GROUP=128
QMAX={2:1,4:7,8:127}


@torch.no_grad()
def quantize(weight,bits,clip_factors=(0.72,0.82,0.92,1.0)):
    if bits not in QMAX: raise ValueError("bits must be one of 2, 4, 8")
    if weight.ndim!=2: raise ValueError("Expected 2-dimensional projection")
    out,inp=weight.shape
    padded=((inp+GROUP-1)//GROUP)*GROUP
    w=F.pad(weight.detach().float(),(0,padded-inp)).reshape(out,-1,GROUP)
    valid=(torch.arange(padded,device=w.device)<inp).reshape(1,-1,GROUP)
    best=torch.full(w.shape[:2],float('inf'),dtype=torch.float32,device=w.device)
    scales=torch.ones_like(best)
    codes=torch.zeros_like(w,dtype=torch.int8)
    absolute=w.abs().amax(dim=-1).clamp_min(1e-8)
    hi=QMAX[bits]
    # Per-row per-G128 independent clipping and least-squares scale search.
    for fraction in clip_factors:
        s=(absolute*float(fraction)/hi).clamp_min(1e-8)
        q=(w/s.unsqueeze(-1)).round().clamp(-hi,hi)*valid
        for _ in range(2):
            denominator=q.float().square().sum(-1).clamp_min(1)
            s=((w*q.float()).sum(-1)/denominator).clamp_min(1e-8)
            q=(w/s.unsqueeze(-1)).round().clamp(-hi,hi)*valid
        error=((w-q*s.unsqueeze(-1)).square()*valid).sum(-1)
        select=error<best
        best=torch.where(select,error,best)
        scales=torch.where(select,s,scales)
        codes=torch.where(select.unsqueeze(-1),q.to(torch.int8),codes)
    return codes.reshape(out,padded),scales.half().contiguous()


def pack_codes(q,bits):
    if bits not in QMAX or q.ndim!=2: raise ValueError("Invalid bit width/shape")
    if bits==8: return q.contiguous()
    if bits==4:
        # signed [-7,+7] -> 0..14, two codes per byte
        v=(q.to(torch.int16)+7).reshape(q.shape[0],-1,2)
        return (v[:,:,0] | (v[:,:,1]<<4)).to(torch.uint8).contiguous()
    # signed [-1,0,+1] -> 0,1,2, four codes per byte
    v=(q.to(torch.int16)+1).reshape(q.shape[0],-1,4)
    return (v[:,:,0] | (v[:,:,1]<<2) | (v[:,:,2]<<4) |
            (v[:,:,3]<<6)).to(torch.uint8).contiguous()


def unpack_codes(p,bits,padded):
    if bits==8:return p[:,:padded].to(torch.int8)
    x=p.to(torch.int16)
    if bits==4:
        return torch.stack(((x&15)-7,((x>>4)&15)-7),dim=-1).reshape(p.shape[0],-1)[:,:padded].to(torch.int8)
    if bits==2:
        return torch.stack(tuple(((x>>s)&3)-1 for s in (0,2,4,6)),dim=-1).reshape(p.shape[0],-1)[:,:padded].to(torch.int8)
    raise ValueError(bits)


class PackedMixedLinear(nn.Module):
    """2/4/8-bit serialized weights; cached FP16 native PyTorch F.linear.

    Cache is NON-persistent and counted separately: active VRAM is not low-bit!
    """
    def __init__(self,in_features,out_features,bits,bias,device="cpu"):
        super().__init__()
        if bits not in QMAX:raise ValueError(bits)
        self.in_features=int(in_features)
        self.out_features=int(out_features)
        self.bits=int(bits)
        self.padded=((self.in_features+GROUP-1)//GROUP)*GROUP
        self.groups=self.padded//GROUP
        cols=self.padded//(4 if bits==2 else 2 if bits==4 else 1)
        self.register_buffer("packed_weight",torch.empty((out_features,cols),
                              device=device,dtype=torch.int8 if bits==8 else torch.uint8))
        self.register_buffer("group_scales",torch.empty((out_features,self.groups),
                              device=device,dtype=torch.float16))
        self.register_buffer("bias",torch.zeros(out_features,device=device,dtype=torch.float16)
                              if bias else None)
        self._cached=None

    @classmethod
    @torch.no_grad()
    def from_linear(cls,linear,bits):
        if not isinstance(linear,nn.Linear):raise TypeError(type(linear))
        result=cls(linear.in_features,linear.out_features,bits,
                   linear.bias is not None,device=linear.weight.device)
        q,s=quantize(linear.weight,bits)
        result.packed_weight.copy_(pack_codes(q,bits))
        result.group_scales.copy_(s)
        if linear.bias is not None:result.bias.copy_(linear.bias.detach().half())
        return result

    @torch.no_grad()
    def weight_fp16(self):
        code=unpack_codes(self.packed_weight,self.bits,self.padded).reshape(
            self.out_features,self.groups,GROUP)
        dense=(code.to(torch.float16)*self.group_scales.unsqueeze(-1))
        return dense.reshape(self.out_features,self.padded)[:,:self.in_features].contiguous()

    def release_cache(self):
        self._cached=None

    def cache_bytes(self):
        return self._cached.numel()*self._cached.element_size() if self._cached is not None else 0

    def forward(self,x):
        if self._cached is None or self._cached.device!=x.device or self._cached.dtype!=x.dtype:
            self._cached=self.weight_fp16().to(device=x.device,dtype=x.dtype).contiguous()
        return F.linear(x,self._cached,self.bias.to(x.dtype) if self.bias is not None else None)


def swap(model,fullname,replacement):
    parent,_,name=fullname.rpartition(".")
    module=model.get_submodule(parent) if parent else model
    former=getattr(module,name)
    setattr(module,name,replacement)
    return former


def state_bytes(model):
    return sum(t.numel()*t.element_size() for t in model.state_dict().values())


def live_bytes(model):
    return state_bytes(model)+sum(m.cache_bytes() for m in model.modules()
                                 if isinstance(m,PackedMixedLinear))
