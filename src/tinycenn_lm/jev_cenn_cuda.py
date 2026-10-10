"""Jev_CeNN optional CUDA sliding-attention backend.

A single Triton program processes one query/head, evaluates exactly the
centered ±radius QK attention scores, and reduces V without building global
T×T masks, unfolding halo blocks, or allocating attention probabilities.

The kernel is an inference fast path ONLY. Training/autograd uses the
existing differentiable PyTorch block-SDPA implementation. No parameter names,
mathematics, layer type, checkpoints or Hugging Face weights are changed.

The optional import is deliberately lazy/guarded. CPU and environments without
Triton use the passed reference implementation unconditionally.
"""
from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Portable HF bundles must still work without Triton.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _sliding_forward(
        Q, K, V, MASK, OUT,
        H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,R:tl.constexpr,
        QB:tl.constexpr,QH:tl.constexpr,QT:tl.constexpr,QD:tl.constexpr,
        KB:tl.constexpr,KH:tl.constexpr,KT:tl.constexpr,KD:tl.constexpr,
        VB:tl.constexpr,VH:tl.constexpr,VT:tl.constexpr,VD:tl.constexpr,
        MB:tl.constexpr,MT:tl.constexpr,SCALE:tl.constexpr,
        N:tl.constexpr,DB:tl.constexpr,
    ):
        qi=tl.program_id(0)
        bh=tl.program_id(1)
        b=bh//H
        h=bh%H
        dd=tl.arange(0,DB)
        ks=qi-R+tl.arange(0,N)
        kv=(ks>=0)&(ks<T)&(tl.abs(ks-qi)<=R)
        valid_k=tl.load(MASK+b*MB+ks*MT,mask=kv,other=0)!=0
        q=tl.load(Q+b*QB+h*QH+qi*QT+dd*QD,mask=dd<D,other=0).to(tl.float32)
        keys=tl.load(
            K+b*KB+h*KH+ks[:,None]*KT+dd[None,:]*KD,
            mask=kv[:,None]&(dd[None,:]<D),other=0
        ).to(tl.float32)
        score=tl.sum(keys*q[None,:],axis=1)*SCALE
        score=tl.where(kv&valid_k,score,-1.0e6)
        top=tl.max(score,axis=0)
        weights=tl.exp(score-top)*((kv&valid_k).to(tl.float32))
        denominator=tl.sum(weights,axis=0)
        values=tl.load(
            V+b*VB+h*VH+ks[:,None]*VT+dd[None,:]*VD,
            mask=kv[:,None]&(dd[None,:]<D),other=0
        ).to(tl.float32)
        result=tl.sum(weights[:,None]*values,axis=0)/tl.maximum(denominator,1.0e-20)
        valid_q=tl.load(MASK+b*MB+qi*MT)!=0
        result=tl.where(valid_q,result,0.0)
        tl.store(OUT+((b*H+h)*T+qi)*D+dd,result,mask=dd<D)


def fast_sliding_sdpa(q,k,v,valid,*,radius,block_size=64,reference=None):
    """Same API/shape as original local_sliding_sdpa.

    Accepts all native q/k/v strides (including ModernBERT's interleaved
    QKV views). Never detaches gradients: when needed, delegates to the
    exact reference implementation. Triton errors are deliberately *not*
    swallowed: the Colab preflight must expose failures before deployment.
    """
    if reference is None:
        raise ValueError("reference=local_sliding_sdpa must be supplied")
    supported=(
        triton is not None
        and q.is_cuda and k.is_cuda and v.is_cuda
        and q.dtype in (torch.float16,torch.bfloat16,torch.float32)
        and q.dtype==k.dtype==v.dtype
        and q.ndim==4 and q.shape==k.shape==v.shape
        and valid.ndim==2 and valid.shape==(q.shape[0],q.shape[2])
        and int(radius)>=0 and int(radius)<=256
    )
    # Keep the original, differentiable SDPA route for every training call.
    if not supported or (torch.is_grad_enabled() and
                        (q.requires_grad or k.requires_grad or v.requires_grad)):
        return reference(q,k,v,valid,radius=radius,block_size=block_size)
    b,h,t,d=q.shape
    if d>128:
        return reference(q,k,v,valid,radius=radius,block_size=block_size)
    radius=int(radius)
    output=torch.empty((b,h,t,d),dtype=q.dtype,device=q.device)
    _sliding_forward[(t,b*h)](
        q,k,v,valid,output,
        h,t,d,radius,
        *q.stride(),*k.stride(),*v.stride(),
        *valid.stride(),
        1.0/math.sqrt(d),
        triton.next_power_of_2(2*radius+1),
        triton.next_power_of_2(d),
        num_warps=4,
    )
    return output


def install_fast_sliding_backend(module, *, reference=None):
    """Enable on an existing module namespace, without changing state_dicts.

    Example: import tinycenn_lm.sliding_window_fusion as swf
             install_fast_sliding_backend(swf)
    """
    if reference is None:
        reference=getattr(module,"_jev_original_sliding_sdpa",None)
    if reference is None:
        reference=module.local_sliding_sdpa
    if not hasattr(module,"_jev_original_sliding_sdpa"):
        module._jev_original_sliding_sdpa=reference
    def accelerated(q,k,v,valid,*,radius,block_size=64):
        return fast_sliding_sdpa(
            q,k,v,valid,radius=radius,block_size=block_size,
            reference=module._jev_original_sliding_sdpa
        )
    module.local_sliding_sdpa=accelerated
    return bool(triton is not None and torch.cuda.is_available())


def uninstall_fast_sliding_backend(module):
    if hasattr(module,"_jev_original_sliding_sdpa"):
        module.local_sliding_sdpa=module._jev_original_sliding_sdpa


def triton_available():
    return triton is not None and torch.cuda.is_available()
