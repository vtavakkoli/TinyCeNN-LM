"""SmolLM2-360M V12 brute-force strict ternary compression (all 32 blocks).

Seven projections per block: q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj. All are forced to {-1,0,+1} packed
2-bit G128 values. No accuracy gate during replacement. Unlike
V11's selective search, acceptance is always forced and *measured*
quality may collapse. Embeddings, norms, lm_head are left unchanged.

Activation-aware group scales/ternary masks minimize approximate output
error using measured teacher input second moments (diagonal X^T X).
Optional local STE-QAT initialization is in the Colab using V11 runtime.
No dense decoded FP16 weight cache; no source compilation. Tiled
decompression uses FP16 GEMM: weight compression != speedup.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .laya_cenn_ternary_concat_v9 import (
        GROUP,PackedTernaryLinear,pack_ternary,unpack_ternary)
except ImportError:
    from laya_cenn_ternary_concat_v9 import (
        GROUP,PackedTernaryLinear,pack_ternary,unpack_ternary)

ATTENTION=("q_proj","k_proj","v_proj","o_proj")
MLP=("gate_proj","up_proj","down_proj")


def paths_all(model):
    layers=model.model.layers
    paths={}
    for layer in range(len(layers)):
        paths[layer]={"attention":[],"mlp":[]}
        for proj in ATTENTION:
            path=f"model.layers.{layer}.self_attn.{proj}"
            if not isinstance(model.get_submodule(path),nn.Linear):
                raise TypeError(f"Missing dense attention projection: {path}")
            paths[layer]["attention"].append(path)
        for proj in MLP:
            path=f"model.layers.{layer}.mlp.{proj}"
            if not isinstance(model.get_submodule(path),nn.Linear):
                raise TypeError(f"Missing dense MLP projection: {path}")
            paths[layer]["mlp"].append(path)
    return paths


def all_path_sets(model):
    bylayer=paths_all(model)
    attn={name for ps in bylayer.values() for name in ps["attention"]}
    mlp={name for ps in bylayer.values() for name in ps["mlp"]}
    out={name for name in attn if name.endswith(".o_proj")}
    return {"attention_output_only":out,
            "attention_all":attn,
            "mlp_all":mlp,
            "all_7_projections_every_layer":attn|mlp}


@torch.no_grad()
def activation_aware_codes(linear,feature_second_moment,
                           thresholds=(0.30,0.50,0.70,0.90,1.10)):
    """Per-row/G128 ternary quantization minimizing diagonal-Hessian MSE.

    For each sign/threshold candidate, use optimal nonnegative G128 scale:
       s* = sum_j (w_j q_j E[x_j^2]) / sum_j(q_j^2 E[x_j^2])
    Codes are restricted to -1/0/+1. False/padded K entries have zero
    second moment, and are quantized as zero.
    """
    if not isinstance(linear,nn.Linear):raise TypeError("Expected nn.Linear")
    n,k=linear.weight.shape
    moments=feature_second_moment.detach().float().flatten().to(
        linear.weight.device)
    if moments.numel()!=k:raise ValueError(
        f"Activation moment dimension {moments.numel()} != {k}")
    pad=(-k)%GROUP
    w=F.pad(linear.weight.detach().float(),(0,pad))
    w=w.reshape(n,-1,GROUP)
    second=F.pad(moments.clamp_min(0),(0,pad)).reshape(1,-1,GROUP)
    # An epsilon protects completely unused groups, while preserving the
    # exact zero-valued padding mask.
    weighted=(second+1.e-8)*(second>0)
    valid=torch.arange(k+pad,device=w.device).reshape(1,-1,GROUP)<k
    weighted=weighted*valid
    count=valid.sum(-1).clamp_min(1)
    ref=(w.abs()*valid).sum(-1)/count
    best=torch.full((n,w.shape[1]),float("inf"),device=w.device)
    best_codes=torch.zeros_like(w,dtype=torch.int8)
    best_scales=torch.zeros((n,w.shape[1]),device=w.device)
    for threshold in thresholds:
        mask=(w.abs()>ref.unsqueeze(-1)*threshold)&valid
        signs=torch.where(mask,torch.sign(w),0)
        numerator=(w*signs*weighted).sum(-1)
        denominator=(signs.square()*weighted).sum(-1)
        scale=torch.where(denominator>1e-12,
                          numerator/denominator.clamp_min(1e-12),0.)
        estimated=(w-scale.unsqueeze(-1)*signs).square()
        error=(estimated*weighted).sum(-1)
        choose=error<best
        best=torch.where(choose,error,best)
        best_scales=torch.where(choose,scale,best_scales)
        best_codes=torch.where(choose.unsqueeze(-1),
                               signs.to(torch.int8),best_codes)
    return best_codes.reshape(n,k+pad),best_scales.half().contiguous()


@torch.no_grad()
def packed_from_activation_aware(linear,moments,chunk_rows=128):
    """Convert and discard dense FP16 reconstruction; return packed weights."""
    codes,scales=activation_aware_codes(linear,moments)
    new=PackedTernaryLinear(
        linear.in_features,linear.out_features,
        bias=linear.bias is not None,
        chunk_rows=chunk_rows,mode="prealloc",device=linear.weight.device)
    new.packed_weight.copy_(pack_ternary(codes))
    new.group_scales.copy_(scales)
    if linear.bias is not None:
        new.bias.copy_(linear.bias.detach().half())
    assert new.packed_weight.dtype==torch.uint8
    assert not hasattr(new,"_cached")
    return new


def swap(model,path,replacement):
    owner,_,name=path.rpartition(".")
    parent=model.get_submodule(owner)
    old=getattr(parent,name)
    setattr(parent,name,replacement)
    return old


def bytes_of_state(model):
    return sum(t.numel()*t.element_size() for t in model.state_dict().values())


def bytes_of_projection(module):
    return sum(t.numel()*t.element_size() for t in module.state_dict().values())


@torch.inference_mode()
def teacher_fidelity(model,batches,reference_logits,top_positions=8):
    """Return disagreement and JS across held-out last-K positions."""
    if len(batches)!=len(reference_logits):raise ValueError("Bad references")
    errors=count=0;js=0.
    for batch,ref in zip(batches,reference_logits):
        got=model(**batch,use_cache=False).logits[:,-top_positions:,:].float()
        teacher=ref.to(got.device)
        if got.shape!=teacher.shape:raise ValueError("Logit shape changed")
        p=got.softmax(-1).clamp_min(1e-9)
        q=teacher.softmax(-1).clamp_min(1e-9)
        mid=(p+q)/2
        js+=float((0.5*(p*(p/mid).log()).sum(-1)
                   +0.5*(q*(q/mid).log()).sum(-1)).sum())
        errors+=int((got.argmax(-1)!=teacher.argmax(-1)).sum())
        count+=got.shape[0]*got.shape[1]
    return {"positions":count,"teacher_agreement":1-errors/count,
            "errors":errors,"mean_js":js/count}


@torch.inference_mode()
def gold_ppl(model,windows,batch_size=2):
    """Independent gold next-token cross entropy, including final window token."""
    import math
    total=0.;n=0
    for start in range(0,len(windows),batch_size):
        data=torch.tensor(windows[start:start+batch_size],
                          device=next(model.parameters()).device)
        logits=model(input_ids=data[:,:-1],use_cache=False).logits.float()
        targets=data[:,1:]
        total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                       targets.reshape(-1),reduction="sum"))
        n+=int(targets.numel())
    return {"gold_tokens":n,"nll":total/n,
            "ppl":math.exp(min(40,total/n))}


def count_codes(linear):
    """Report exact signed values for audit (not a speed diagnostic)."""
    q=unpack_ternary(linear.packed_weight,linear.padded)
    return {str(i):int((q==i).sum()) for i in (-1,0,1)}
