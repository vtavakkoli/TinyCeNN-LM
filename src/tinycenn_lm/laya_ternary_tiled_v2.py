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
def refine_scales_from_activations(module,original,x,relative_ridge=0.03,
                                  holdout_fraction=0.25):
    """Fit FP16 scale per out-channel/G128; keep original scales on failure.

    Codes are frozen, not trained. Uses calibration inputs and original linear
    teacher outputs (not development or audit). Ridge regression with anchor
    to PTQ scales; accept only lower normalized error on held-out cal tokens.
    """
    if not isinstance(module,TernaryCascadeLinear):
        raise TypeError("Expected packed ternary module")
    if not isinstance(original,nn.Linear) or x.ndim!=2:
        raise TypeError("Expected original nn.Linear and [tokens,features] inputs")
    if len(x)<48 or x.shape[1]!=module.in_features:
        raise ValueError("At least 48 aligned calibration tokens required")
    x=x.detach().float()
    train_count=max(32,int(len(x)*(1-holdout_fraction)))
    if train_count >= len(x):
        train_count=len(x)-16
    old=module.group_scales.detach().clone()
    o=module.out_features
    g=module.groups
    codes=unpack_ternary(module.packed_weight,module.kpad).reshape(o,g,128).float()
    def make_features(part):
        xpad=F.pad(part,(0,module.kpad-module.in_features))
        xt=xpad.reshape(-1,g,128)
        # [out, token, group], with a single output per G128 ternary dot.
        return torch.einsum("tgi,ogi->otg",xt,codes)
    bias=(original.bias.detach().float() if original.bias is not None else None)
    targets=F.linear(x,original.weight.detach().float(),bias).transpose(0,1).contiguous()
    if bias is not None:
        targets=targets-bias[:,None]
    f=make_features(x[:train_count])
    y=targets[:,:train_count].unsqueeze(-1)
    gram=torch.bmm(f.transpose(1,2),f)
    rhs=torch.bmm(f.transpose(1,2),y)
    base=old.float().unsqueeze(-1)
    anchor=(gram.diagonal(dim1=-2,dim2=-1).mean(-1)
            .clamp_min(1e-7)*float(relative_ridge))
    eye=torch.eye(g,device=x.device,dtype=torch.float32).unsqueeze(0)
    solution=torch.linalg.solve(gram+anchor[:,None,None]*eye,
                                rhs+anchor[:,None,None]*base).squeeze(-1)
    trial=solution.clamp(min=0.0)
    # Avoid unstable magnitude explosions in barely-observed groups.
    trial=torch.minimum(torch.maximum(trial,old.float()*0.35),
                        old.float()*2.2).half()
    fv=make_features(x[train_count:])
    truth=targets[:,train_count:]
    denom=truth.square().mean().clamp_min(1e-8)
    pre=(torch.einsum("otg,og->ot",fv,old.float())-truth).square().mean()/denom
    post=(torch.einsum("otg,og->ot",fv,trial.float())-truth).square().mean()/denom
    passed=(torch.isfinite(post).item() and float(post)<=float(pre)-1e-6)
    if passed:
        module.group_scales.copy_(trial)
        module._dense_cache=None
    return dict(baseline_nmse=float(pre),candidate_nmse=float(post),
                accepted=bool(passed),group_count=g,token_count=len(x))


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
