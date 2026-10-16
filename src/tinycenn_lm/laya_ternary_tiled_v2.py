"""Laya v2: tiled Triton ternary dequant GEMM and activation-aware G128 scale refinement.

2-bit packed -1/0/+1 codes; fp16 row-group scale per 128 input features.
CUDA: decode BK input-weight tile, multiply scales BEFORE FP16 tensor-core dot,
accumulate output with float32 tl.dot accumulator. Experimental kernel.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

from laya_fp16_ternary_cascade import (
    TernaryCascadeLinear, quantize_groups, pack_ternary,
    unpack_ternary, replace, state_bytes, runtime_bytes,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _packed_tiled_kernel(X,P,S,B,Y,
                             M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                             PSTRIDE:tl.constexpr,GROUPS:tl.constexpr,
                             HAS_BIAS:tl.constexpr,BM:tl.constexpr,
                             BN:tl.constexpr,BK:tl.constexpr):
        rows=tl.program_id(0)*BM+tl.arange(0,BM)
        cols=tl.program_id(1)*BN+tl.arange(0,BN)
        off=tl.arange(0,BK)
        acc=tl.full((BM,BN),0.0,tl.float32)
        for tile in range(tl.cdiv(K,BK)):
            kk=tile*BK+off
            a=tl.load(X+rows[:,None]*K+kk[None,:],
                      (rows[:,None]<M)&(kk[None,:]<K),other=0)
            codebytes=tl.load(P+cols[None,:]*PSTRIDE+(kk[:,None]//4),
                      (cols[None,:]<N)&(kk[:,None]<K),other=0x55)
            code=((codebytes.to(tl.int32)>>((kk[:,None]%4)*2))&3)-1
            groups=kk[:,None]//128
            scale=tl.load(S+cols[None,:]*GROUPS+groups,
                      (cols[None,:]<N)&(groups<GROUPS),other=0).to(tl.float32)
            b=(code.to(tl.float32)*scale).to(a.dtype)
            acc=tl.dot(a,b,acc)
        if HAS_BIAS:
            acc+=tl.load(B+cols,cols<N,other=0).to(tl.float32)[None,:]
        tl.store(Y+rows[:,None]*N+cols[None,:],acc,
                 (rows[:,None]<M)&(cols[None,:]<N))


class TiledTernaryLinear(TernaryCascadeLinear):
    @classmethod
    @torch.no_grad()
    def from_linear(cls,module,backend="triton_tiled",tile_k=128):
        if not isinstance(module,nn.Linear):
            raise TypeError("Expected nn.Linear source")
        obj=cls(module.in_features,module.out_features,module.bias is not None,
                backend="triton",device=module.weight.device)
        q,s=quantize_groups(module.weight)
        obj.packed_weight.copy_(pack_ternary(q))
        obj.group_scales.copy_(s)
        if module.bias is not None:
            obj.bias.copy_(module.bias.detach().half())
        obj.tile_k=int(tile_k)
        obj.set_backend(backend)
        return obj

    def set_backend(self,backend):
        if backend == "triton_tiled":
            self.backend=backend
            self._dense_cache=None
        else:
            super().set_backend(backend)

    def forward(self,x):
        if self.backend!="triton_tiled" or not x.is_cuda:
            if self.backend=="triton_tiled":
                # Local correctness checks can use the CPU reference.
                return F.linear(x,self.dense_weight().to(x.dtype),
                                self.bias.to(x.dtype) if self.bias is not None else None)
            return super().forward(x)
        if triton is None:
            raise RuntimeError("Triton is required on GPU")
        dtype=x.dtype
        src=x.reshape(-1,self.in_features).contiguous()
        if src.dtype not in (torch.float16,torch.bfloat16):
            src=src.half()
        m=int(src.shape[0])
        result=torch.empty((m,self.out_features),device=src.device,dtype=src.dtype)
        bk=int(getattr(self,"tile_k",128))
        if bk not in (64,128,256,512):
            raise ValueError(f"Unsupported tile_k={bk}")
        bm=16 if m<32 else 32
        bn=32 if self.out_features<64 else 64
        _packed_tiled_kernel[(triton.cdiv(m,bm),triton.cdiv(self.out_features,bn))](
            src,self.packed_weight,self.group_scales,
            self.bias if self.bias is not None else self.group_scales,result,
            m,self.out_features,self.in_features,self.kpad//4,self.groups,
            self.bias is not None,bm,bn,bk,num_warps=4)
        return result.to(dtype).reshape(*x.shape[:-1],self.out_features)


@torch.no_grad()
def refine_scales_from_activations(module, original, x, relative_ridge=0.03,
                                  holdout_fraction=0.25, output_chunk=64):
    """Safely refit g128 scales; solve small normal equations on CPU.

    The previous implementation called CUDA torch.linalg.solve() for every
    output row and could fail with CUBLAS_STATUS_NOT_INITIALIZED. This variant
    computes bounded per-output features (GPU native matmul), transfers only
    those features to CPU, and solves at most 64 tiny ridge systems at once.
    A failed chunk retains the original PTQ scales instead of crashing.
    """
    if not isinstance(module,TernaryCascadeLinear):
        raise TypeError("Expected packed ternary module")
    if not isinstance(original,nn.Linear) or x.ndim!=2:
        raise TypeError("Expected original nn.Linear and [tokens,features] inputs")
    if x.shape[-1] != module.in_features or len(x)<48:
        raise ValueError("At least 48 aligned calibration tokens required")
    if not (0.0<holdout_fraction<0.5):
        raise ValueError("holdout_fraction must be in (0,0.5)")
    if output_chunk<1:
        raise ValueError("output_chunk must be >= 1")

    old=module.group_scales.detach().clone()
    x=x.detach().to(device=original.weight.device,dtype=original.weight.dtype)
    rows=module.out_features
    groups=module.groups
    train_count=max(32,min(len(x)-16,int(len(x)*(1-holdout_fraction))))
    # Match the actual original FP16 projection; never allocate all per-output
    # group activations simultaneously.
    teacher=F.linear(x,original.weight,original.bias).detach().float()
    if original.bias is not None:
        teacher-=original.bias.detach().float()[None,:]
    xp=F.pad(x.float(),(0,module.kpad-module.in_features)).reshape(-1,groups,128)
    codes=unpack_ternary(module.packed_weight,module.kpad).reshape(
        rows,groups,128)
    trial_all=old.clone()
    err_before=err_after=0.0
    accepted_rows=0
    failed_chunks=0
    for start in range(0,rows,output_chunk):
        stop=min(rows,start+output_chunk)
        try:
            # Use GPU only for feature extraction (no large CUDA linalg.solve).
            z=torch.einsum(
                "tgi,ogi->otg",xp,codes[start:stop].float()).cpu()
            y=teacher[:,start:stop].transpose(0,1).contiguous().cpu()
            prior=old[start:stop].float().cpu()
            a=z[:,:train_count,:]
            b=y[:,:train_count].unsqueeze(-1)
            gram=torch.bmm(a.transpose(1,2),a)
            rhs=torch.bmm(a.transpose(1,2),b)
            anchor=(gram.diagonal(dim1=-2,dim2=-1).mean(-1)
                    .clamp_min(1e-6)*max(float(relative_ridge),1e-6))
            eye=torch.eye(groups,dtype=torch.float32).unsqueeze(0)
            updated=torch.linalg.solve(
                gram+anchor[:,None,None]*eye,
                rhs+anchor[:,None,None]*prior.unsqueeze(-1)).squeeze(-1)
            updated=updated.clamp_min(0.0)
            updated=torch.minimum(torch.maximum(updated,prior*0.35),
                                  prior*2.2).to(torch.float16).float()
            val=z[:,train_count:,:]
            truth=y[:,train_count:]
            before=(torch.einsum("otg,og->ot",val,prior)-truth).square().mean(-1)
            after=(torch.einsum("otg,og->ot",val,updated)-truth).square().mean(-1)
            better=torch.isfinite(after)&(after<before-1e-8)
            if bool(better.any()):
                trial_all[start:stop][better.to(old.device)]=updated[better].to(
                    device=old.device,dtype=old.dtype)
            accepted_rows+=int(better.sum())
            err_before+=float(before.sum())
            err_after+=float(torch.where(better,after,before).sum())
        except (RuntimeError,ValueError) as exc:
            # Model exploration must be resilient: an unstable calibration
            # chunk never poisons the packed weights.
            failed_chunks+=1
            print(f"Scale-fit chunk {start}:{stop} retained PTQ: {str(exc)[:180]}")
    any_accepted=accepted_rows>0
    if any_accepted:
        module.group_scales.copy_(trial_all)
        module._dense_cache=None
    return {
        "baseline_nmse":err_before/max(rows,1),
        "candidate_nmse":err_after/max(rows,1),
        "accepted":any_accepted,
        "accepted_output_rows":accepted_rows,
        "failed_chunks":failed_chunks,
        "group_count":groups,
        "token_count":len(x),
        "calibration_solver":"cpu_chunked_ridge",
    }


@torch.no_grad()
def collect_calibration_inputs(model,module_names,batch_list,max_tokens=384):
    """Collect exact current activations from each FP16 projection in same layer."""
    samples={name:[] for name in module_names}
    handles=[]
    for name in module_names:
        module=model.get_submodule(name)
        if not isinstance(module,nn.Linear):
            raise TypeError(f"Calibration target is not original Linear: {name}")
        def observe(_module,args,key=name):
            src=args[0].detach().reshape(-1,args[0].shape[-1])
            n=max(1,min(64,len(src),max_tokens))
            indices=torch.linspace(0,len(src)-1,n,device=src.device).long()
            samples[key].append(src.index_select(0,indices).float())
        handles.append(module.register_forward_pre_hook(observe))
    try:
        for batch in batch_list:
            with torch.autocast("cuda",dtype=torch.float16,
                                enabled=next(model.parameters()).is_cuda):
                model(**batch)
    finally:
        for handle in handles:handle.remove()
    result={}
    for name,parts in samples.items():
        if not parts:
            raise RuntimeError(f"No activation samples for {name}")
        xs=torch.cat(parts,dim=0)
        if len(xs)>max_tokens:
            ids=torch.linspace(0,len(xs)-1,max_tokens,device=xs.device).long()
            xs=xs.index_select(0,ids)
        result[name]=xs.contiguous()
    return result
