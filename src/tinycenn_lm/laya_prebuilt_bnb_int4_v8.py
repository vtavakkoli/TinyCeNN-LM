"""Laya V8: precompiled bitsandbytes >=0.50.0 4-bit CUDA inference.

No source-built Triton, no TorchAO MSLK, no BF16 on Tesla T4.
bitsandbytes 0.50+ includes a fused 4-bit dequantize+GEMM dispatcher
supporting SM75 FP16 Tensor Cores. Its *actual* dispatch may still fall back
to FP16 dequantize+matmul on shapes where that is faster. Report honestly:
the 4-bit weights are physically packed, but individual kernel choice varies.

This differs from native INT4xINT4: this is packed W4A16 fused execution.
"""
from __future__ import annotations

import copy
import statistics
from typing import Any

import torch
from torch import nn


def backend_information() -> dict[str, Any]:
    import bitsandbytes as bnb
    ver=str(getattr(bnb,"__version__","0"))
    digits=ver.split(".")
    if len(digits)<2 or tuple(int(x) for x in digits[:2]) < (0,50):
        raise RuntimeError(
            f"bitsandbytes {ver} is too old. Need 0.50.2 for fused CUDA INT4."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cap=torch.cuda.get_device_capability()
    if cap<(7,5):
        raise RuntimeError("This Laya V8 study targets SM75+ CUDA")
    if not hasattr(torch.ops.bitsandbytes,"gemm_4bit"):
        # torch.ops may lazy-register; no eager assertion here:
        # execute a real module to check kernel support.
        pass
    return {"gpu":torch.cuda.get_device_name(0),"sm":list(cap),
            "bitsandbytes_version":ver,"torch_version":str(torch.__version__),
            "activation_dtype":"fp16", "packed_weight_type":"NF4/FP4",
            "backend":"prebuilt bitsandbytes CUDA 4bit fused GEMM dispatcher",
            "native_int4xint4":False,
            "note":"W4A16 packed inference; selected fused/fallback kernel is shape-dependent."}


@torch.inference_mode()
def as_bnb4(original:nn.Linear,quant_type="nf4",nested=False) -> nn.Module:
    """Convert one FP16 linear to a truly packed BNB Linear4bit on CUDA.

    Builds the FP16 temporary on CPU and quantizes only once on .to(cuda).
    Never keeps a dense weight cache in the returned module.
    """
    import bitsandbytes as bnb
    if quant_type not in ("nf4","fp4"):
        raise ValueError(quant_type)
    if not isinstance(original,nn.Linear):
        raise TypeError(type(original))
    if original.weight.device.type!="cuda":
        raise RuntimeError("CUDA FP16 original linear required")
    new=bnb.nn.Linear4bit(
        original.in_features,original.out_features,
        bias=original.bias is not None,
        compute_dtype=torch.float16,compress_statistics=nested,
        quant_type=quant_type,device="cpu")
    # Constructor's packed Params4bit is still unquantized before .to("cuda").
    # CPU copy avoids allocating another full dense GPU weight matrix.
    original_cpu={k:v.detach().to("cpu").clone()
                  for k,v in original.state_dict().items()}
    new.load_state_dict(original_cpu,strict=True)
    new=new.to(device=original.weight.device)
    new.eval().requires_grad_(False)
    weight=new.weight
    qs=getattr(weight,"quant_state",None)
    if qs is None or not getattr(weight,"bnb_quantized",False):
        raise RuntimeError("bitsandbytes failed to pack CUDA weight")
    if weight.data.dtype!=torch.uint8:
        raise RuntimeError(f"Expected packed uint8 weight, got {weight.data.dtype}")
    if tuple(qs.shape)!=(original.out_features,original.in_features):
        raise RuntimeError("Packed weight metadata shape mismatch")
    return new


def locate_candidates(model):
    """4 named projections in each Laya ModernBERT encoder layer."""
    names=[]
    for layer in range(len(model.encoder.layers)):
        for proj in ("mlp.Wi","mlp.Wo","attn.Wqkv","attn.Wo"):
            fullname=f"encoder.layers.{layer}.{proj}"
            if isinstance(model.get_submodule(fullname),nn.Linear):
                names.append(fullname)
    return sorted(names,key=lambda n:model.get_submodule(n).weight.numel(),reverse=True)


def replace_module(model,fullname,replacement):
    parent,_,key=fullname.rpartition(".")
    owner=model.get_submodule(parent)
    old=getattr(owner,key)
    setattr(owner,key,replacement)
    return old


@torch.inference_mode()
def paired_cuda_ms(a,b,repeats=12,warmup=5):
    """Interleaved ABBA timing, robust lower-decile speedup as safety check."""
    if repeats<4:
        raise ValueError("Use at least 4 timing pairs")
    for _ in range(warmup):
        a();b()
    torch.cuda.synchronize()
    def once(func):
        s=torch.cuda.Event(enable_timing=True)
        e=torch.cuda.Event(enable_timing=True)
        s.record();func();e.record();e.synchronize()
        return float(s.elapsed_time(e))
    at=[];bt=[]
    for i in range(repeats):
        if i%2:
            b1=once(b);a1=once(a);a2=once(a);b2=once(b)
        else:
            a1=once(a);b1=once(b);b2=once(b);a2=once(a)
        at.append((a1+a2)/2);bt.append((b1+b2)/2)
    r=[x/y for x,y in zip(at,bt) if y>0]
    rsort=sorted(r)
    return dict(fp16_median_ms=float(statistics.median(at)),
                bnb4_median_ms=float(statistics.median(bt)),
                speedup=float(statistics.median(r)),
                p10_speedup=float(rsort[max(0,int((len(rsort)-1)*.10))]),
                spread=float(max(r)-min(r)),repeats=len(r))


def operator_benchmark(original,candidate,m_values=(128,512,2048),
                       repeats=10):
    results=[]
    for m in m_values:
        x=torch.randn(m,original.in_features,device=original.weight.device,
                      dtype=original.weight.dtype)
        with torch.inference_mode():
            y=candidate(x)
            ref=original(x)
        if y.shape!=ref.shape or not bool(torch.isfinite(y).all()):
            raise RuntimeError("Invalid 4bit output")
        stats=paired_cuda_ms(lambda:original(x),lambda:candidate(x),
                             repeats=repeats)
        mse=float((y.float()-ref.float()).square().mean())
        results.append(dict(m=m,mse=mse,**stats))
    return results


def speed_gate(rows,min_speed=1.06,target_m=(512,2048)):
    lookup={int(row["m"]):row for row in rows}
    return all(m in lookup and lookup[m]["speedup"]>=min_speed
               and lookup[m]["p10_speedup"]>=1.0
               for m in target_m)


def gpu_checkpoint_bytes(model):
    """Packed state bytes, excluding runtime optimizer caches and CUDA allocator."""
    return sum(t.numel()*t.element_size() for t in model.state_dict().values())
