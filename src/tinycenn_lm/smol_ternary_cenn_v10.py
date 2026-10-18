"""SmolLM2 V10: strict ternary CeNN replacement and tiny low-rank recovery.

Uses the V9 {-1,0,+1} signed G128 codebook (2 bits/weight; FP16 group
scales). No cached dense FP16 weight. 'rescue16' adds small FP16 low-rank
residuals (so the complete layer is no longer purely ternary). 'cenn_only'
is a genuinely local 3-neighbour stencil with ternary coefficients, used
as a square projection replacement NEGATIVE CONTROL.

All conversions are post-training inference experiments. Do not call these
native low-bit GEMM or faster without empirical CUDA measurements.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .laya_cenn_ternary_concat_v9 import (
        PackedTernaryLinear,TernaryCeNNStencil,serialized_bytes)
except ImportError:
    from laya_cenn_ternary_concat_v9 import (
        PackedTernaryLinear,TernaryCeNNStencil,serialized_bytes)


class SmolTernaryProjection(nn.Module):
    """Packed G128 ternary + optional SVD residual + optional local CeNN.

    base: 2-bit packed ternary, output chunks preallocated (never torch.cat)
    svd_rank: if >0, two low-rank FP16 buffers approximate ternary errors
    stencil_gain: if >0, CeNN ternary [-1,0,1] stencil adds local context
    """
    def __init__(self,base:PackedTernaryLinear,rank:int=0,
                 stencil_gain:float=0):
        super().__init__()
        if base.mode!="prealloc":
            raise ValueError("Use preallocated direct output, not concat")
        self.base=base
        self.in_features=base.in_features
        self.out_features=base.out_features
        self.svd_rank=int(rank)
        self.register_buffer("residual_left",None)
        self.register_buffer("residual_right",None)
        self.stencil_gain=float(stencil_gain)
        self.stencil=(TernaryCeNNStencil([[1,-1,0],[-1,1,0]])
                     if stencil_gain else None)
        if self.stencil is not None and self.in_features!=self.out_features:
            raise ValueError("CeNN stencil residual requires square projection")

    @classmethod
    @torch.no_grad()
    def from_linear(cls,linear:nn.Linear,chunk_rows=128,rank=0,
                    stencil_gain=0.0,svd_iters=1):
        if not isinstance(linear,nn.Linear):raise TypeError("nn.Linear required")
        base=PackedTernaryLinear.from_linear(
            linear,chunk_rows=int(chunk_rows),mode="prealloc")
        result=cls(base,rank=rank,stencil_gain=stencil_gain)
        if rank>0:
            if rank>=min(linear.weight.shape):
                raise ValueError("rank should be much smaller than matrix dimensions")
            original=linear.weight.detach().float()
            # Decode once at construction for calibration and discard immediately.
            reconstructed=torch.cat([
                base.decode_rows(a,min(base.out_features,a+128),torch.float16)
                for a in range(0,base.out_features,128)],dim=0).float()
            error=(original-reconstructed).contiguous()
            del reconstructed,original
            # GPU randomized SVD; no expensive dense full-rank decomposition.
            try:
                u,s,v=torch.svd_lowrank(error,q=min(rank+6,min(error.shape)),
                                       niter=svd_iters)
            except Exception as exc:
                raise RuntimeError("Low-rank rescue factorization failed") from exc
            root=s[:rank].clamp_min(0).sqrt()
            left=(u[:,:rank]*root.unsqueeze(0)).to(linear.weight.dtype).contiguous()
            right=(v[:,:rank].T*root.unsqueeze(1)).to(
                linear.weight.dtype).contiguous()
            result.residual_left=left
            result.residual_right=right
            del error,u,s,v
        if result.stencil is not None:
            result.stencil=result.stencil.to(linear.weight.device)
        return result

    def forward(self,x):
        y=self.base(x)
        if self.residual_left is not None:
            right=self.residual_right.to(dtype=x.dtype)
            left=self.residual_left.to(dtype=x.dtype)
            correction=F.linear(F.linear(x,right),left)
            y=y+correction
        if self.stencil is not None:
            y=y+self.stencil(x,mode="sum")*self.stencil_gain
        return y


class CeNNOnlyProjection(nn.Module):
    """CeNN cell stands in for a *square* pretrained projection.

    This is a radical architectural replacement; without training, accuracy
    likely fails. Only causal (left/center) taps are used. Recurrent KV-state
    caching for the stencil is NOT implemented, so it is a prefill-only control.
    """
    def __init__(self,width,original_bias=None):
        super().__init__()
        self.in_features=self.out_features=int(width)
        # Causal decoder: right/next-token coefficients MUST be zero.
        # A nonzero right tap would leak future tokens into SmolLM2 logits.
        self.stencil=TernaryCeNNStencil([
            [-1,1,0],[1,-1,0],[0,1,0]
        ])
        if original_bias is None:
            self.register_buffer("bias",None)
        else:
            self.register_buffer("bias",original_bias.detach().half().clone())

    @classmethod
    @torch.no_grad()
    def from_linear(cls,linear):
        if (not isinstance(linear,nn.Linear) or
                linear.in_features!=linear.out_features):
            raise ValueError("CeNN-only replacement must be square nn.Linear")
        out=cls(linear.in_features,linear.bias)
        return out.to(linear.weight.device)

    def forward(self,x):
        y=self.stencil(x,mode="sum")
        return y+self.bias.to(x.dtype) if self.bias is not None else y


def replace(model:nn.Module,name:str,new:nn.Module):
    parent,_,field=name.rpartition(".")
    module=model.get_submodule(parent)
    old=getattr(module,field)
    setattr(module,field,new)
    return old


def model_storage_bytes(model:nn.Module):
    return sum(t.numel()*t.element_size() for t in model.state_dict().values())


def projection_storage_bytes(module:nn.Module):
    return sum(t.numel()*t.element_size() for t in module.state_dict().values())


def candidate_names(model,*,last_layers=6,include_attention=True):
    """Prefer last layers' MLP down projections; optional attention O.

    Exactly one projection is changed per trial, and other weights are frozen.
    """
    layers=model.model.layers
    names=[]
    for i in range(max(0,len(layers)-last_layers),len(layers)):
        for key in (("mlp.down_proj","self_attn.o_proj") if include_attention
                    else ("mlp.down_proj",)):
            path=f"model.layers.{i}.{key}"
            if isinstance(model.get_submodule(path),nn.Linear):
                names.append(path)
    return names


@torch.inference_mode()
def teacher_metrics(student,teacher_last_logits,inputs):
    """Last-token teacher fidelity; inputs batches include masks.

    Cached reference tensors are CPU float32 logits [batch,vocabulary].
    """
    assert len(inputs)==len(teacher_last_logits)>0
    count=errors=0
    total_js=0.0
    new_mask=[]
    for batch,teacher in zip(inputs,teacher_last_logits):
        prediction=student(**batch,use_cache=False).logits[:,-1,:]
        prediction=prediction.detach().float().cpu()
        t=teacher.float().cpu()
        if prediction.shape!=t.shape:raise ValueError("Teacher logits mismatch")
        p=prediction.softmax(-1).clamp_min(1e-9)
        q=t.softmax(-1).clamp_min(1e-9)
        mid=(p+q)/2
        js=0.5*(p*(p/mid).log()).sum(-1)
        js+=0.5*(q*(q/mid).log()).sum(-1)
        wrong=(p.argmax(-1)!=q.argmax(-1))
        new_mask.extend(wrong.tolist())
        total_js+=float(js.sum())
        errors+=int(wrong.sum());count+=int(wrong.numel())
    return {"n":count,"teacher_agreement":1-errors/count,
            "errors":errors,"mean_js":total_js/count,"wrong_mask":new_mask}


def acceptance(metric,previous,base,*,max_new_errors=0,max_js=.008):
    if not (metric["n"]==previous["n"]==base["n"]):return False
    new_from_base=sum(a and not b for a,b in zip(
        metric["wrong_mask"],base["wrong_mask"]))
    new_from_previous=sum(a and not b for a,b in zip(
        metric["wrong_mask"],previous["wrong_mask"]))
    return (new_from_base<=max_new_errors and
            new_from_previous<=max_new_errors and
            metric["errors"]<=previous["errors"]+max_new_errors and
            metric["mean_js"]<=max_js)


def public(metric):
    return {k:v for k,v in metric.items() if k!="wrong_mask"}


@torch.inference_mode()
def benchmark_pair(a,b,batches,repeats=5):
    if not torch.cuda.is_available():raise RuntimeError("CUDA needed")
    import statistics
    def fn(module,batch):
        return module(**batch,use_cache=False).logits
    records=[]
    for batch in batches:
        for _ in range(2):fn(a,batch);fn(b,batch)
        torch.cuda.synchronize()
        aa=[];bb=[]
        for i in range(repeats):
            def once(module):
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record();val=fn(module,batch)
                end.record();end.synchronize()
                del val
                return float(start.elapsed_time(end))
            if i%2:
                b1=once(b);a1=once(a);a2=once(a);b2=once(b)
            else:
                a1=once(a);b1=once(b);b2=once(b);a2=once(a)
            aa.append((a1+a2)/2);bb.append((b1+b2)/2)
        ratios=[x/y for x,y in zip(aa,bb) if y>0]
        records.append(dict(batch=int(batch["input_ids"].shape[0]),
            tokens=int(batch["input_ids"].shape[1]),
            original_ms=float(statistics.median(aa)),
            ternary_ms=float(statistics.median(bb)),
            speedup=float(statistics.median(ratios))))
    return records
