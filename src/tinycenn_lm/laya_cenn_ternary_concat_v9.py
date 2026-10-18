"""CeNN V9: strict {-1,0,+1} templates and G128 2-bit packed linear ablations.

RESEARCH-ONLY INFERENCE PATHS. No custom compiler or external CUDA wheel.
- Ternary CeNN stencil branches use only add/sub/zero on shifted neighbors.
- Concatenating branches does not compress activations. 'sum' has no cat.
- PackedTernaryLinear has no persistent FP16 decoded weight cache.
- Full G128 packed weights: ~2.125 bits/weight including half scales.
- Per-tile decoding temporarily expands only output-chunk weights to FP16.
  This is not a native ternary tensor-core accelerator and may be slower.

Use torch.cuda.max_memory_allocated to measure peak live allocations, never
confuse a packed state_dict size with total GPU memory or inference speed.
"""
from __future__ import annotations
import math
import time
import statistics
import torch
from torch import nn
from torch.nn import functional as F

GROUP=128


def pack_ternary(codes:torch.Tensor)->torch.Tensor:
    """Map [-1,0,+1] to 2-bit [0,1,2], four values per byte."""
    if codes.ndim!=2:
        raise ValueError("Require [rows, columns] ternary codes")
    if codes.shape[1]%4:
        raise ValueError("Code width must be divisible by 4")
    if codes.numel() and (bool((codes < -1).any()) or bool((codes > 1).any())):
        raise ValueError("Only -1, 0, +1 codes are supported")
    v=(codes.to(torch.int16)+1).reshape(codes.shape[0],-1,4)
    return (v[...,0] | (v[...,1]<<2) | (v[...,2]<<4) |
            (v[...,3]<<6)).to(torch.uint8).contiguous()


def unpack_ternary(packed:torch.Tensor,count:int)->torch.Tensor:
    if packed.ndim!=2 or count<0 or count>4*packed.shape[1]:
        raise ValueError("Invalid packed shape/count")
    v=packed.to(torch.int16)
    return torch.stack([(v>>shift)&3 for shift in (0,2,4,6)],-1).reshape(
        packed.shape[0],-1)[:,:count].to(torch.int8)-1


@torch.no_grad()
def ternarize_g128(weight:torch.Tensor,thresholds=(0.45,0.70,1.0)):
    """Choose per-row/G128 threshold with minimal local MSE.

    Each group's scale is the mean absolute value of active weights.
    Only ternary codes and FP16 group scales are saved. Padding is masked.
    """
    if weight.ndim!=2:
        raise ValueError("Expected 2-dimensional weight matrix")
    n,k=weight.shape
    padded=((k+GROUP-1)//GROUP)*GROUP
    groups=padded//GROUP
    w=F.pad(weight.detach().float(),(0,padded-k)).reshape(n,groups,GROUP)
    valid=(torch.arange(padded,device=w.device)<k).reshape(1,groups,GROUP)
    absw=w.abs()
    meanabs=(absw*valid).sum(-1) / valid.sum(-1).clamp_min(1)
    best=torch.full((n,groups),float("inf"),device=w.device)
    bestcode=torch.zeros_like(w,dtype=torch.int8)
    bestscale=torch.zeros((n,groups),device=w.device)
    for factor in thresholds:
        active=(absw>meanabs.unsqueeze(-1)*factor)&valid
        t=torch.where(active,torch.sign(w),0.0)
        count=active.sum(-1).clamp_min(1)
        scale=(absw*active).sum(-1)/count
        error=(((w-t*scale.unsqueeze(-1)).square())*valid).sum(-1)
        choose=error<best
        best=torch.where(choose,error,best)
        bestscale=torch.where(choose,scale,bestscale)
        bestcode=torch.where(choose.unsqueeze(-1),t.to(torch.int8),bestcode)
    return bestcode.reshape(n,padded),bestscale.half().contiguous()


class PackedTernaryLinear(nn.Module):
    """G128 ternary weight-only, no cached FP16 weight, output-tiled execution.

    'prealloc': direct output slices, avoids output concatenation.
    'cat': holds each output chunk, allocates another tensor for torch.cat.
    Both yield the same numerical result. Neither is native low-bit GEMM.
    """
    def __init__(self,inp,out,bias=True,chunk_rows=64,mode="prealloc",device="cpu"):
        super().__init__()
        if chunk_rows<=0:raise ValueError("chunk_rows must be positive")
        if mode not in ("prealloc","cat"):raise ValueError(mode)
        self.in_features=int(inp)
        self.out_features=int(out)
        self.padded=((inp+GROUP-1)//GROUP)*GROUP
        self.groups=self.padded//GROUP
        self.chunk_rows=int(chunk_rows)
        self.mode=mode
        self.register_buffer("packed_weight",torch.empty(
            (out,self.padded//4),dtype=torch.uint8,device=device))
        self.register_buffer("group_scales",torch.empty(
            (out,self.groups),dtype=torch.float16,device=device))
        self.register_buffer("bias",torch.empty(
            out,dtype=torch.float16,device=device) if bias else None)

    @classmethod
    @torch.no_grad()
    def from_linear(cls,linear,chunk_rows=64,mode="prealloc"):
        if not isinstance(linear,nn.Linear):raise TypeError(type(linear))
        obj=cls(linear.in_features,linear.out_features,
                linear.bias is not None,chunk_rows,mode,linear.weight.device)
        q,s=ternarize_g128(linear.weight)
        obj.packed_weight.copy_(pack_ternary(q))
        obj.group_scales.copy_(s)
        if linear.bias is not None:obj.bias.copy_(linear.bias.detach().half())
        return obj

    @torch.no_grad()
    def decode_rows(self,begin,end,dtype=torch.float16):
        if not (0<=begin<end<=self.out_features):raise ValueError("Row interval")
        code=unpack_ternary(self.packed_weight[begin:end],self.padded).reshape(
            end-begin,self.groups,GROUP)
        weights=(code.to(dtype)*self.group_scales[begin:end].to(
            dtype).unsqueeze(-1)).reshape(end-begin,self.padded)
        return weights[:,:self.in_features].contiguous()

    def forward(self,x):
        if x.shape[-1]!=self.in_features:
            raise ValueError("Input width mismatch")
        with torch.no_grad():
            shape=x.shape[:-1]
            if self.mode=="prealloc":
                result=torch.empty((*shape,self.out_features),
                                   device=x.device,dtype=x.dtype)
            else:
                parts=[]
            for begin in range(0,self.out_features,self.chunk_rows):
                end=min(self.out_features,begin+self.chunk_rows)
                weight=self.decode_rows(begin,end,x.dtype)
                offset=self.bias[begin:end].to(x.dtype) if self.bias is not None else None
                part=F.linear(x,weight,offset)
                if self.mode=="prealloc":
                    result[...,begin:end].copy_(part)
                else:
                    parts.append(part)
            return result if self.mode=="prealloc" else torch.cat(parts,dim=-1)


class TernaryCeNNStencil(nn.Module):
    """Local 1D cellular stencil with EXACT {-1,0,+1} neighbor templates.

    Input [batch,token,feature]. Each branch uses left/center/right samples.
    concat: store branch activations -> cat -> sum.
    sum: accumulate branch maps directly (no cat).
    Both have the same output dimension and arithmetic for fair comparison.
    """
    def __init__(self,templates,branch_scales=None):
        super().__init__()
        t=torch.as_tensor(templates,dtype=torch.int8)
        if t.ndim!=2 or t.shape[1]!=3 or not t.shape[0]:
            raise ValueError("Expected non-empty [branches,3] templates")
        t4=F.pad(t,(0,1))
        self.register_buffer("packed_templates",pack_ternary(t4))
        scales=(torch.ones(t.shape[0],dtype=torch.float16)
                if branch_scales is None else
                torch.as_tensor(branch_scales,dtype=torch.float16))
        if scales.shape!=(t.shape[0],):
            raise ValueError("Expected one scale per branch")
        self.register_buffer("scales",scales)
        self.branches=int(t.shape[0])

    def unpack(self):
        return unpack_ternary(self.packed_templates,4)[:,:3]

    def forward(self,x,mode="sum"):
        if x.ndim!=3 or mode not in ("sum","concat"):
            raise ValueError("Need [B,T,D] and mode sum|concat")
        left=F.pad(x[:,:-1,:],(0,0,1,0))
        center=x
        right=F.pad(x[:,1:,:],(0,0,0,1))
        kernels=self.unpack()
        def branch(i):
            z=torch.zeros_like(x)
            for tap,src in enumerate((left,center,right)):
                code=int(kernels[i,tap].item())
                if code==1:z=z+src
                elif code==-1:z=z-src
            return z*self.scales[i].to(x.dtype)
        if mode=="concat":
            pieces=[branch(i) for i in range(self.branches)]
            joined=torch.cat(pieces,dim=-1)
            return joined.reshape(*x.shape[:2],self.branches,x.shape[-1]).sum(dim=2)
        result=torch.zeros_like(x)
        for i in range(self.branches):
            result=result+branch(i)
        return result


def serialized_bytes(module):
    return sum(v.numel()*v.element_size() for v in module.state_dict().values())


@torch.inference_mode()
def peak_cuda_forward(fn,warmup=3,repeats=5):
    """Real peak incremental CUDA allocated bytes + synchronized time."""
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    peaks=[];durations=[]
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline=torch.cuda.memory_allocated()
        s=torch.cuda.Event(enable_timing=True)
        e=torch.cuda.Event(enable_timing=True)
        s.record();out=fn();e.record();e.synchronize()
        peak=torch.cuda.max_memory_allocated()
        peaks.append(max(0,int(peak-baseline)))
        durations.append(float(s.elapsed_time(e)))
        del out
    return {
        "peak_extra_alloc_mib":statistics.median(peaks)/(1024**2),
        "latency_ms":statistics.median(durations),
        "repeats":int(repeats)
    }


def theoretical_pack_bytes(weight_shapes,group=128):
    """Pure state_dict estimate for all named projections; not full GPU VRAM."""
    packed=dense=0
    for out,inp in weight_shapes:
        padded=((inp+group-1)//group)*group
        packed+=out*(padded//4)+out*(padded//group)*2
        dense+=out*inp*2
    return {"fp16_weight_bytes":dense,"ternary_weight_and_scale_bytes":packed,
            "saving_fraction":1-packed/dense if dense else 0.0}
