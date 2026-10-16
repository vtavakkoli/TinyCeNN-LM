"""Experimental Laya 1.2 packed ternary G128 -> FP16 cascade runtimes.

This is post-training ternary approximation, not trained BitNet / proprietary Bonsai.
Storage: two-bit slots (-1,0,+1) plus one FP16 scale / 128 input weights.
CUDA fused path: packed bytes -> FP16 Q tile -> FP16 dot -> FP32 accumulate.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

GROUP_SIZE = 128


@torch.no_grad()
def quantize_groups(weight: torch.Tensor, search=(1.0, 1.25, 1.5, 1.8, 2.1), iterations=4):
    if weight.ndim != 2:
        raise ValueError("Expected 2D linear weights")
    rows, cols = weight.shape
    padded = math.ceil(cols / GROUP_SIZE) * GROUP_SIZE
    w = F.pad(weight.detach().float(), (0, padded-cols)).reshape(rows,-1,GROUP_SIZE)
    valid = torch.arange(padded,device=w.device).reshape(1,-1,GROUP_SIZE) < cols
    base = (w.abs()*valid).sum(-1) / valid.sum(-1).clamp_min(1)
    best_error = torch.full_like(base, float("inf"))
    best_scale = torch.ones_like(base)
    best_codes = torch.zeros_like(w, dtype=torch.int8)
    for factor in search:
        s = (base*factor).clamp_min(1e-8)
        for _ in range(iterations):
            q = (w/s.unsqueeze(-1)).round().clamp(-1,1)*valid
            s = ((w*q).sum(-1)/(q.square().sum(-1).clamp_min(1))).clamp_min(1e-8)
        q = (w/s.unsqueeze(-1)).round().clamp(-1,1)*valid
        error = ((w-s.unsqueeze(-1)*q).square()*valid).sum(-1)
        accept = error < best_error
        best_error = torch.where(accept,error,best_error)
        best_scale = torch.where(accept,s,best_scale)
        best_codes = torch.where(accept.unsqueeze(-1),q.to(torch.int8),best_codes)
    return best_codes.reshape(rows,padded), best_scale.to(torch.float16).contiguous()


def pack_ternary(q):
    if q.ndim != 2 or q.shape[1] % 4 != 0:
        raise ValueError("Expected [out, padded_in] with padded_in % 4 == 0")
    if not bool(((q >= -1)&(q <= 1)).all()):
        raise ValueError("Invalid ternary values")
    s = (q.to(torch.int16)+1).reshape(q.shape[0],-1,4)
    return (s[...,0] | (s[...,1]<<2) | (s[...,2]<<4) | (s[...,3]<<6)).to(torch.uint8).contiguous()


def unpack_ternary(p, limit=None):
    a = p.to(torch.int16)
    out = torch.stack(tuple((a>>shift)&3 for shift in (0,2,4,6)), dim=-1).reshape(p.shape[0],-1)-1
    return out[:,:limit].to(torch.int8) if limit is not None else out.to(torch.int8)


try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    @triton.jit
    def _fused_g128(X,P,S,B,Y, M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                    PSTRIDE:tl.constexpr,GROUPS:tl.constexpr,HAS_BIAS:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr):
        row = tl.program_id(0)*BM + tl.arange(0,BM)
        col = tl.program_id(1)*BN + tl.arange(0,BN)
        step = tl.arange(0,128)
        acc = tl.full((BM,BN),0,tl.float32)
        for group in range(GROUPS):
            kk = group*128 + step
            inp = tl.load(X+row[:,None]*K+kk[None,:],
                          (row[:,None]<M)&(kk[None,:]<K),other=0)
            raw = tl.load(P+col[None,:]*PSTRIDE+kk[:,None]//4,
                          (col[None,:]<N)&(kk[:,None]<K),other=0x55)
            codes = ((raw.to(tl.int32) >> ((kk[:,None]%4)*2))&3)-1
            partial = tl.dot(inp,codes.to(inp.dtype))
            scales = tl.load(S+col*GROUPS+group,col<N,other=0).to(tl.float32)
            acc += partial.to(tl.float32)*scales[None,:]
        if HAS_BIAS:
            acc += tl.load(B+col,col<N,other=0).to(tl.float32)[None,:]
        tl.store(Y+row[:,None]*N+col[None,:],acc,
                 (row[:,None]<M)&(col[None,:]<N))


class TernaryCascadeLinear(nn.Module):
    """Packed ternary linear with selectable backend.

    backend='triton' uses fused fp16 tensor-core per-group dots and fp32 accumulation.
    backend='torch_cascade' demonstrates explicit torch FP16 partial-sum cascade.
    backend='cached_fp16' dequantizes ONCE and uses cuBLAS FP16 (no runtime compression).
    """
    def __init__(self,in_features,out_features,has_bias,backend='triton',device='cpu'):
        super().__init__()
        self.in_features=int(in_features)
        self.out_features=int(out_features)
        self.groups=math.ceil(self.in_features/GROUP_SIZE)
        self.kpad=self.groups*GROUP_SIZE
        self.backend=backend
        self.register_buffer('packed_weight',torch.empty(
            self.out_features,self.kpad//4,dtype=torch.uint8,device=device))
        self.register_buffer('group_scales',torch.ones(
            self.out_features,self.groups,dtype=torch.float16,device=device))
        self.register_buffer('bias',torch.zeros(self.out_features,dtype=torch.float16,device=device)
                             if has_bias else None)
        self._dense_cache=None

    @classmethod
    @torch.no_grad()
    def from_linear(cls,module,backend='triton'):
        if not isinstance(module,nn.Linear):
            raise TypeError("Source must be nn.Linear")
        obj=cls(module.in_features,module.out_features,module.bias is not None,
                backend=backend,device=module.weight.device)
        q,scale=quantize_groups(module.weight)
        obj.packed_weight.copy_(pack_ternary(q))
        obj.group_scales.copy_(scale)
        if module.bias is not None:
            obj.bias.copy_(module.bias.detach().half())
        return obj

    def dense_weight(self):
        q=unpack_ternary(self.packed_weight,self.kpad).reshape(
            self.out_features,self.groups,GROUP_SIZE)
        w=q.to(torch.float16)*self.group_scales.unsqueeze(-1)
        return w.reshape(self.out_features,self.kpad)[:,:self.in_features].contiguous()

    def set_backend(self,backend):
        if backend not in ('triton','torch_cascade','cached_fp16'):
            raise ValueError(backend)
        self.backend=backend
        if backend!='cached_fp16':
            self._dense_cache=None

    def cache_bytes(self):
        return 0 if self._dense_cache is None else (
            self._dense_cache.numel()*self._dense_cache.element_size())

    def forward(self,x):
        shape=x.shape
        if shape[-1] != self.in_features:
            raise ValueError("Invalid input width")
        if not x.is_cuda or self.backend=='cached_fp16':
            if self.backend=='cached_fp16':
                if (self._dense_cache is None or self._dense_cache.device!=x.device):
                    self._dense_cache=self.dense_weight().to(x.device).contiguous()
                w=self._dense_cache
            else:
                w=self.dense_weight()
            return F.linear(x,w.to(x.dtype),self.bias.to(x.dtype)
                            if self.bias is not None else None)
        original_dtype=x.dtype
        xx=x.reshape(-1,self.in_features).contiguous()
        if xx.dtype not in (torch.float16,torch.bfloat16):
            xx=xx.half()
        m=xx.shape[0]
        if self.backend=='torch_cascade':
            yy=torch.zeros(m,self.out_features,device=xx.device,dtype=torch.float32)
            for group in range(self.groups):
                start=group*GROUP_SIZE
                stop=min(start+GROUP_SIZE,self.in_features)
                p=self.packed_weight[:,start//4:(group+1)*GROUP_SIZE//4]
                q=unpack_ternary(p)[:,:stop-start]
                w=(q.to(torch.float16)*self.group_scales[:,group:group+1])
                yy+=F.linear(xx[:,start:stop],w,None).float()
            if self.bias is not None:
                yy+=self.bias.float()[None,:]
            return yy.to(original_dtype).reshape(*shape[:-1],self.out_features)
        if self.backend!='triton':
            raise ValueError(f"Unknown backend: {self.backend}")
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton required for packed CUDA backend")
        out=torch.empty(m,self.out_features,device=xx.device,dtype=xx.dtype)
        bm=32
        bn=32 if self.out_features<64 else 64
        _fused_g128[(triton.cdiv(m,bm),triton.cdiv(self.out_features,bn))](
            xx,self.packed_weight,self.group_scales,
            self.bias if self.bias is not None else self.group_scales,out,
            m,self.out_features,self.in_features,self.kpad//4,self.groups,
            self.bias is not None,bm,bn,num_warps=4)
        return out.to(original_dtype).reshape(*shape[:-1],self.out_features)


def replace(model,fullname,child):
    prefix,_,name=fullname.rpartition('.')
    p=model.get_submodule(prefix) if prefix else model
    old=getattr(p,name)
    setattr(p,name,child)
    return old


def state_bytes(model):
    return sum(v.numel()*v.element_size() for v in model.state_dict().values())


def runtime_bytes(model):
    return state_bytes(model)+sum(m.cache_bytes() for m in model.modules()
                                  if isinstance(m,TernaryCascadeLinear))
