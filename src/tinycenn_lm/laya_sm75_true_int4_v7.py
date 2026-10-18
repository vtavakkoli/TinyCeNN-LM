"""Experimental SM75 native INT4xINT4 (signed s4*s4->s32) Laya projection.

Requires the Turing-specific Triton fork exposing tl.reinterpret_as_int4:
https://github.com/Chennesxu/triton-turing (MIT).
Upstream Triton does not support this SM75 INT4 dot path.

Unlike Laya V5 or TorchAO W4A16, BOTH activations and weights are quantized
to signed INT4. Weight scales use group size 128, activation scales are dynamic
per token/row; loss of precision is real and MUST pass independent audits.
No FP16 weight cache or dequantized weight matrix is stored/executed.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

GROUP = 128
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def require_native_turing():
    if not HAS_TRITON:
        raise RuntimeError("Triton unavailable. Install the SM75 Turing fork.")
    if not hasattr(tl, "reinterpret_as_int4"):
        raise RuntimeError(
            "Installed Triton lacks tl.reinterpret_as_int4. "
            "Install Chennesxu/triton-turing SM75 fork (not upstream Triton), "
            "then RESTART the Colab runtime so the new Triton is imported.")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0]!=7:
        raise RuntimeError("This experimental s4xs4 backend targets SM75/Tesla T4.")
    return True


@torch.no_grad()
def pack_signed_nibbles(q: torch.Tensor, axis: int = -1):
    """Bit-exact signed [-8,7] into little-endian int32 (8 values/word)."""
    if q.ndim!=2:
        raise ValueError("Expected 2D q matrix")
    if axis not in (-1,1,0):
        raise ValueError("Only 2D axes 0 and 1 supported")
    transposed=axis==0
    src=q.transpose(0,1).contiguous() if transposed else q.contiguous()
    rows,cols=src.shape
    if cols%8:raise ValueError("Packed dimension must be divisible by 8")
    if q.numel() and (int(q.min())< -8 or int(q.max())>7):
        raise ValueError("Out of signed INT4 bounds")
    groups=(src.to(torch.int32).reshape(rows,cols//8,8)&15)
    shift=(torch.arange(8,device=src.device,dtype=torch.int32)*4)
    packed=(groups<<shift).sum(dim=-1).to(torch.int32)
    return packed.transpose(0,1).contiguous() if transposed else packed.contiguous()


@torch.no_grad()
def unpack_signed_nibbles(packed:torch.Tensor,axis:int=-1):
    if axis not in (-1,0,1):raise ValueError(axis)
    transposed=axis==0
    x=packed.transpose(0,1).contiguous() if transposed else packed
    shifts=torch.arange(8,device=x.device,dtype=torch.int32)*4
    codes=(x.unsqueeze(-1)>>shifts)&15
    signed=torch.where(codes>=8,codes-16,codes).to(torch.int8)
    out=signed.reshape(x.shape[0],-1)
    return out.transpose(0,1).contiguous() if transposed else out


if HAS_TRITON:
    @triton.jit
    def _quant_pack_activations(X,AP,AS,
                                M:tl.constexpr,K:tl.constexpr,
                                BK:tl.constexpr):
        row=tl.program_id(0)
        kk=tl.arange(0,BK)
        x=tl.load(X+row*K+kk,mask=kk<K,other=0).to(tl.float32)
        maxabs=tl.max(tl.abs(x),0)
        scale=tl.maximum(maxabs/7.0,1.0e-8)
        q=tl.floor(x/scale+0.5).to(tl.int32)
        q=tl.minimum(tl.maximum(q,-7),7)
        nib=q & 15
        grouped=tl.reshape(nib,(BK//8,8))
        shifts=tl.arange(0,8)*4
        words=tl.sum(grouped << shifts[None,:],1)
        kp=tl.arange(0,BK//8)
        tl.store(AP+row*(K//8)+kp,words,mask=kp<(K//8))
        tl.store(AS+row,scale)

    @triton.jit
    def _sm75_int4_linear(AP,BP,AS,BS,BIAS,Y,
                         M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                         BM:tl.constexpr,BN:tl.constexpr):
        pm=tl.program_id(0)
        pn=tl.program_id(1)
        rr=pm*BM+tl.arange(0,BM)
        cc=pn*BN+tl.arange(0,BN)
        kp=tl.arange(0,16)
        accum=tl.full((BM,BN),0,tl.float32)
        # Group=128: sixteen packed int32 values per reduction.
        for group in range(K//128):
            a=tl.load(AP+rr[:,None]*(K//8)+(group*16+kp)[None,:],
                      mask=rr[:,None]<M,other=0)
            b=tl.load(BP+(group*16+kp)[:,None]*N+cc[None,:],
                      mask=cc[None,:]<N,other=0)
            # Turing-specific native INT4 Tensor Core instruction m8n8k32:
            a4=tl.reinterpret_as_int4(a,axis=1)
            b4=tl.reinterpret_as_int4(b,axis=0)
            part=tl.dot(a4,b4,out_dtype=tl.int32)
            ws=tl.load(BS+group*N+cc,mask=cc<N,other=0)
            accum+=part.to(tl.float32)*ws[None,:]
        activ_scale=tl.load(AS+rr,mask=rr<M,other=0)
        bias=tl.load(BIAS+cc,mask=cc<N,other=0).to(tl.float32)
        output=accum*activ_scale[:,None]+bias[None,:]
        tl.store(Y+rr[:,None]*N+cc[None,:],output,
                 mask=(rr[:,None]<M)&(cc[None,:]<N))


class TuringINT4Linear(nn.Module):
    """INT4xINT4 native Tensor Core, source FP16 -> per-row activation s4.

    Packed weight: int32[K/8,N]. Group scales: FP32[K/128,N].
    No persistent FP16 dense weights; FP16 *outputs* remain.
    """
    def __init__(self,linear:nn.Linear):
        super().__init__()
        if not isinstance(linear,nn.Linear):raise TypeError(type(linear))
        if linear.in_features%128:
            raise ValueError("INT4 group128 path requires K divisible by 128")
        k=linear.in_features;n=linear.out_features
        w=linear.weight.detach().float().contiguous()
        groups=w.reshape(n,k//GROUP,GROUP)
        scale=(groups.abs().amax(dim=2)/7).clamp_min(1.0e-8)
        code=(groups/scale.unsqueeze(-1)).round().clamp(-7,7).to(torch.int8)
        packed=pack_signed_nibbles(code.reshape(n,k).t().contiguous(),axis=0)
        # packed is [K/8,N] (first-axis reduction packing)
        assert packed.shape==(k//8,n)
        self.register_buffer("packed_weight",packed)
        self.register_buffer("group_scales",scale.t().float().contiguous())
        self.register_buffer("bias",linear.bias.detach().half().clone()
                             if linear.bias is not None else
                             torch.zeros(n,device=w.device,dtype=torch.float16))
        self.in_features=k
        self.out_features=n
        self.is_genuine_turing_s4s4=True

    @torch.inference_mode()
    def forward(self,x:torch.Tensor):
        require_native_turing()
        k=self.in_features;n=self.out_features
        if x.shape[-1]!=k or x.device!=self.packed_weight.device:
            raise RuntimeError("Input dimension or device mismatch")
        original_shape=x.shape[:-1]
        flat=x.reshape(-1,k).contiguous().to(torch.float16)
        m=flat.shape[0]
        ap=torch.empty((m,k//8),device=x.device,dtype=torch.int32)
        scales=torch.empty(m,device=x.device,dtype=torch.float32)
        y=torch.empty((m,n),device=x.device,dtype=torch.float16)
        _quant_pack_activations[(m,)](
            flat,ap,scales,m,k,BK=triton.next_power_of_2(k),
            num_warps=4)
        _sm75_int4_linear[(triton.cdiv(m,32),triton.cdiv(n,64))](
            ap,self.packed_weight,scales,self.group_scales,self.bias,y,
            m,n,k,BM=32,BN=64,num_warps=4,num_stages=2)
        return y.reshape(*original_shape,n)


@torch.inference_mode()
def cpu_reference(x,packed_weight,group_scales,bias):
    """Reference for intended dynamic W4A4 arithmetic; does not need Triton."""
    n=group_scales.shape[1]
    k=packed_weight.shape[0]*8
    wq=unpack_signed_nibbles(packed_weight,axis=0).to(torch.float32).t()
    w=wq.reshape(n,k//128,128)*group_scales.t().unsqueeze(-1)
    x2=x.float().reshape(-1,k)
    s=(x2.abs().amax(-1,keepdim=True)/7).clamp_min(1e-8)
    xq=(x2/s).round().clamp(-7,7)
    return F.linear(xq*s,w.reshape(n,k),bias.float()).reshape(*x.shape[:-1],n)


def capability_report():
    report={
        "backend":"experimental SM75 native s4xs4 tensor core",
        "triton_installed":HAS_TRITON,
        "triton_int4_reinterpret":bool(
            HAS_TRITON and hasattr(tl,"reinterpret_as_int4")),
        "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "sm":list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "fork_required":True,
        "caveats":[
            "Turing fork must be installed and Colab restarted.",
            "Activations quantized per token dynamically to signed int4.",
            "Weights use group128 signed int4 scales; no V5 sparse reconstruction.",
            "Actual accelerated execution, fidelity and end-to-end speed unverified."
        ]
    }
    return report
