"""Bonsai-inspired g128 ternary linear modules for Jev_CeNN experimental ablation.

The format is compatible in principle with Q2_0-style ternary coding: 2-bit
codes 0/1/2 representing -1/0/+1 and one fp16 group scale per 128 weights.
It is NOT the proprietary Bonsai training method or a llama.cpp GGUF loader.
"""
import torch
from torch import nn
from torch.nn import functional as F

GROUP_SIZE = 128


def fit_g128(weight, importance=None, method='activation_mse', search=(0.85,1.0,1.15,1.35,1.6,1.9), iterations=3):
    """Return row-major ternary signed int8 codes (O,Kpad), scales (O,G), and weighted error.

    Search scale initializations and alternating LS scale/ternary-code refits.
    Importance is diagonal E[x_i^2] estimated from calibration data;
    this approximates output reconstruction under uncorrelated activations.
    """
    assert weight.ndim == 2
    o,k=weight.shape
    padded=((k+GROUP_SIZE-1)//GROUP_SIZE)*GROUP_SIZE
    w=F.pad(weight.detach().float(),(0,padded-k)).reshape(o,-1,GROUP_SIZE)
    if importance is None or method=='mse':
        imp=torch.ones((1,1,GROUP_SIZE),device=weight.device,dtype=torch.float32).expand(1,w.shape[1],-1).clone()
    else:
        if importance.numel()!=k:raise ValueError('Calibration feature mismatch')
        imp=F.pad(importance.detach().to(weight.device,dtype=torch.float32).clamp_min(0),(0,padded-k)).reshape(1,-1,GROUP_SIZE)
        imp=imp / imp.mean(-1,keepdim=True).clamp_min(1e-9)
    if padded>k:
        last_valid=k-(padded-GROUP_SIZE)
        imp=imp.clone()
        imp[:,-1,last_valid:]=0
    imp=imp.clamp_min(0)
    mean_abs=w.abs().mean(-1).clamp_min(1e-7)
    best_err=torch.full(mean_abs.shape,float('inf'),dtype=torch.float32,device=w.device)
    best_scale=torch.ones_like(mean_abs)
    best_q=torch.zeros_like(w,dtype=torch.int8)
    factors=search if method != 'naive' else (1.5,)
    for factor in factors:
        scale=(mean_abs*factor).clamp_min(1e-7)
        for _ in range(iterations if method!='naive' else 1):
            q=(w / scale.unsqueeze(-1)).round().clamp(-1,1)
            numerator=(imp*w*q).sum(-1)
            denominator=(imp*q.square()).sum(-1).clamp_min(1e-9)
            scale=(numerator/denominator).clamp_min(1e-7)
        q=(w/scale.unsqueeze(-1)).round().clamp(-1,1)
        err=(imp*(w-q*scale.unsqueeze(-1)).square()).sum(-1)
        take=err<best_err
        best_err=torch.where(take,err,best_err)
        best_scale=torch.where(take,scale,best_scale)
        best_q=torch.where(take.unsqueeze(-1),q.to(torch.int8),best_q)
    return best_q.reshape(o,padded), best_scale.to(torch.float16).contiguous(), float(best_err.sum().item())


def pack(q):
    """Ternary int8 values -1/0/+1 -> 2-bit slots, 4 per byte."""
    if q.ndim!=2 or q.shape[1]%4:raise ValueError('Expected 2D and K divisible by 4')
    if not bool(((q>=-1)&(q<=1)).all()):raise ValueError('Not ternary')
    c=(q.to(torch.int16)+1).reshape(q.shape[0],-1,4)
    return (c[:,:,0] | (c[:,:,1]<<2) | (c[:,:,2]<<4) | (c[:,:,3]<<6)).to(torch.uint8).contiguous()


def unpack(packed, kpad):
    p=packed.to(torch.int16)
    return (torch.stack([(p>>s)&3 for s in (0,2,4,6)],dim=-1).reshape(p.shape[0],-1)[:,:kpad]-1).to(torch.int8)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE=True
except ImportError:
    TRITON_AVAILABLE=False

if TRITON_AVAILABLE:
    @triton.jit
    def _g128_kernel(X,P,S,B,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                     KP:tl.constexpr,G:tl.constexpr,HAS_BIAS:tl.constexpr,
                     BM:tl.constexpr,BN:tl.constexpr):
        rows=tl.program_id(0)*BM+tl.arange(0,BM)
        cols=tl.program_id(1)*BN+tl.arange(0,BN)
        offsets=tl.arange(0,128)
        acc=tl.full((BM,BN),0.0,tl.float32)
        for g in range(tl.cdiv(K,128)):
            kk=g*128+offsets
            xx=tl.load(X+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),0)
            pp=tl.load(P+cols[None,:]*KP+kk[:,None]//4,
                       (cols[None,:]<N)&(kk[:,None]<K),0x55)
            cc=((pp.to(tl.int32)>>((kk[:,None]%4)*2))&3)-1
            dot=tl.dot(xx,cc.to(xx.dtype))
            scale=tl.load(S+cols*G+g,cols<N,0).to(tl.float32)
            acc += dot.to(tl.float32)*scale[None,:]
        if HAS_BIAS:
            acc += tl.load(B+cols,cols<N,0).to(tl.float32)[None,:]
        tl.store(Y+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


class TernaryG128Linear(nn.Module):
    """Real packed on-device weights (O,Kpad/4) and fp16 scale (O,ceil(K/128)).

    CUDA uses a direct Triton matrix multiply that decodes 2-bit codes in the
    kernel without allocating FP16 reconstructed weights. CPU has a reference.
    """
    def __init__(self,in_features,out_features,bias=False,device='cpu'):
        super().__init__()
        self.in_features=int(in_features)
        self.out_features=int(out_features)
        self.kpad=((self.in_features+127)//128)*128
        self.groups=self.kpad//128
        self.register_buffer('packed_weight',torch.empty((self.out_features,self.kpad//4),device=device,dtype=torch.uint8))
        self.register_buffer('group_scales',torch.ones((self.out_features,self.groups),device=device,dtype=torch.float16))
        self.register_buffer('bias',torch.zeros(self.out_features,device=device,dtype=torch.float16) if bias else None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls,module,importance=None,method='activation_mse'):
        assert isinstance(module,nn.Linear)
        result=cls(module.in_features,module.out_features,module.bias is not None,device=module.weight.device)
        q,s,_=fit_g128(module.weight,importance=importance,method=method)
        result.packed_weight.copy_(pack(q))
        result.group_scales.copy_(s)
        if module.bias is not None:result.bias.copy_(module.bias.detach().to(torch.float16))
        return result

    def unpack_weight(self):
        q=unpack(self.packed_weight,self.kpad).reshape(self.out_features,self.groups,128).float()
        return (q*self.group_scales.float().unsqueeze(-1)).reshape(self.out_features,self.kpad)[:,:self.in_features]

    def forward(self,x):
        if not x.is_cuda:
            w=self.unpack_weight().to(x.dtype)
            return F.linear(x,w,self.bias.to(x.dtype) if self.bias is not None else None)
        if not TRITON_AVAILABLE:raise RuntimeError('CUDA ternary operation requires Triton')
        original_dtype=x.dtype
        xc=x.contiguous().reshape(-1,self.in_features)
        if xc.dtype not in (torch.bfloat16,torch.float16):
            xc=xc.to(torch.float16)
        m=xc.shape[0]
        out=torch.empty((m,self.out_features),device=xc.device,dtype=xc.dtype)
        bm=16 if m<32 else 32
        bn=32 if self.out_features<64 else 64
        _g128_kernel[(triton.cdiv(m,bm),triton.cdiv(self.out_features,bn))](
            xc,self.packed_weight,self.group_scales,
            self.bias if self.bias is not None else self.group_scales,out,
            m,self.out_features,self.in_features,self.kpad//4,self.groups,
            self.bias is not None,bm,bn,num_warps=4)
        result=out.reshape(*x.shape[:-1],self.out_features)
        if original_dtype==torch.float32 and not torch.is_autocast_enabled('cuda'):
            return result.float()
        return result


def state_bytes(model):
    return sum(t.numel()*t.element_size() for t in model.state_dict().values())


def get_parent(model,name):
    parent,_,attr=name.rpartition('.')
    return (model.get_submodule(parent) if parent else model),attr


def swap(model,name,replacement):
    parent,attr=get_parent(model,name)
    previous=getattr(parent,attr)
    setattr(parent,attr,replacement)
    return previous
