"""SmolLM2 V11: trainable {-1,0,+1} QAT + activation distillation.

Training: FP32 master ternary *codes* with STE, learnable G128 scales,
optional causal-safe channel-neighbor CeNN residual, optional rank16 FP16
error correction. Evaluation: no master-weight cache; compact packed 2-bit
ternary buffers, dequantized only in output tiles via V9.

Ternary + scales != purely integer operations or native 2-bit acceleration.
The cellular neighborhood is over FEATURES, not future tokens, therefore
it is valid with autoregressive KV-cache decoding. It is not a full
recurrent Chua-Yang CeNN.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .laya_cenn_ternary_concat_v9 import (
        PackedTernaryLinear, pack_ternary, ternarize_g128, GROUP)
except ImportError:
    from laya_cenn_ternary_concat_v9 import (
        PackedTernaryLinear, pack_ternary, ternarize_g128, GROUP)


def feature_cenn(x:torch.Tensor):
    """Fixed {-1,+1} local cellular stencil across hidden channels.

    [-1,+1,0] implements y[...,d] - y[...,d-1]. No future-token
    interaction, no convolution cache or concatenation.
    """
    return x-F.pad(x[...,:-1],(1,0))


class TernaryQATLinear(nn.Module):
    """Learn signed {-1,0,+1} STE weights, G128 scales and optional residual."""
    def __init__(self,linear:nn.Linear,rank=0,feature_residual=True):
        super().__init__()
        if not isinstance(linear,nn.Linear):raise TypeError("nn.Linear required")
        self.in_features=linear.in_features
        self.out_features=linear.out_features
        self.padded=((self.in_features+GROUP-1)//GROUP)*GROUP
        self.groups=self.padded//GROUP
        if rank<0 or rank>=min(linear.weight.shape):
            raise ValueError("Invalid low rank")
        self.rank=int(rank)
        codes,scales=ternarize_g128(linear.weight.detach())
        self.codes=nn.Parameter(codes.float().contiguous())
        self.log_scales=nn.Parameter(
            scales.float().clamp_min(1e-8).log().contiguous())
        self.register_buffer("bias",linear.bias.detach().half().clone()
                            if linear.bias is not None else None)
        self.feature_residual=bool(feature_residual)
        if self.feature_residual:
            self.feature_gain=nn.Parameter(torch.zeros((),device=linear.weight.device))
        else:self.register_parameter("feature_gain",None)
        if rank:
            with torch.no_grad():
                weight=linear.weight.detach().float()
                base=(codes.reshape(self.out_features,self.groups,GROUP).float()
                      *scales.float().unsqueeze(-1)).reshape(
                          self.out_features,self.padded)[:,:self.in_features]
                error=(weight-base).contiguous()
                u,s,v=torch.svd_lowrank(error,
                    q=min(int(rank)+5,min(error.shape)),niter=1)
                factor=s[:rank].clamp_min(0).sqrt()
                left=(u[:,:rank]*factor.unsqueeze(0)).contiguous()
                right=(v[:,:rank].T*factor.unsqueeze(1)).contiguous()
            self.left=nn.Parameter(left)
            self.right=nn.Parameter(right)
        else:
            self.register_parameter("left",None)
            self.register_parameter("right",None)

    def quantized_codes(self):
        return self.codes.detach().round().clamp_(-1,1).to(torch.int8)

    def scales(self):
        return self.log_scales.clamp(-16,6).exp()

    def fake_weight(self):
        real=self.codes.clamp(-1,1)
        hard=real.round()
        q=real+(hard-real).detach() # derivative d(q)/d(codes)=1 within range
        w=q.reshape(self.out_features,self.groups,GROUP)
        w=w*self.scales().unsqueeze(-1)
        return w.reshape(self.out_features,self.padded)[:,:self.in_features]

    def forward(self,x):
        y=F.linear(x,self.fake_weight().to(x.dtype),
                   self.bias.to(x.dtype) if self.bias is not None else None)
        if self.left is not None:
            y=y+F.linear(F.linear(x,self.right.to(x.dtype)),self.left.to(x.dtype))
        if self.feature_gain is not None:
            y=y+self.feature_gain.to(x.dtype)*feature_cenn(y)
        return y

    @torch.no_grad()
    def pack(self,chunk_rows=128):
        """Export inference state (no FP32 master codes, no decode cache)."""
        packed=PackedTernaryLinear(
            self.in_features,self.out_features,self.bias is not None,
            chunk_rows=chunk_rows,mode="prealloc",
            device=self.codes.device)
        packed.packed_weight.copy_(pack_ternary(self.quantized_codes()))
        packed.group_scales.copy_(self.scales().half())
        if self.bias is not None:packed.bias.copy_(self.bias)
        return PackedCeNNLinear(
            packed,
            self.left.detach().half().clone() if self.left is not None else None,
            self.right.detach().half().clone() if self.right is not None else None,
            self.feature_gain.detach().half().clone()
                if self.feature_gain is not None else None)


class PackedCeNNLinear(nn.Module):
    """Inference-only compressed 2-bit ternary, never caches dense FP16."""
    def __init__(self,base:PackedTernaryLinear,left=None,right=None,gain=None):
        super().__init__()
        assert base.mode=="prealloc"
        self.base=base
        self.in_features=base.in_features
        self.out_features=base.out_features
        self.register_buffer("left",left)
        self.register_buffer("right",right)
        self.register_buffer("feature_gain",gain)

    def forward(self,x):
        # V9 packed linear deliberately uses no_grad during tiled reconstruction.
        # That is good for inference but would detach the input gradient from
        # any earlier QAT projection when distilling multiple layers. Preserve
        # activation gradients on the small training-only path. The packed
        # weights are buffers and are NEVER made trainable here.
        if torch.is_grad_enabled() and x.requires_grad:
            shape=(*x.shape[:-1],self.out_features)
            y=torch.empty(shape,device=x.device,dtype=x.dtype)
            for begin in range(0,self.out_features,self.base.chunk_rows):
                end=min(self.out_features,begin+self.base.chunk_rows)
                weight=self.base.decode_rows(begin,end,x.dtype)
                bias=(self.base.bias[begin:end].to(x.dtype)
                      if self.base.bias is not None else None)
                y[...,begin:end]=F.linear(x,weight,bias)
        else:
            y=self.base(x)
        if self.left is not None:
            y=y+F.linear(F.linear(x,self.right.to(x.dtype)),self.left.to(x.dtype))
        if self.feature_gain is not None:
            y=y+self.feature_gain.to(x.dtype)*feature_cenn(y)
        return y


def create_from_metadata(original:nn.Linear,metadata:dict,device):
    """Allocate empty packed buffers to load a saved state_dict in Colab."""
    rank=int(metadata.get("rank",0))
    with_cenn=bool(metadata.get("feature_residual",False))
    base=PackedTernaryLinear(
        original.in_features,original.out_features,
        original.bias is not None,int(metadata.get("chunk_rows",128)),
        mode="prealloc",device=device)
    left=(torch.empty((original.out_features,rank),device=device,
                      dtype=torch.float16) if rank else None)
    right=(torch.empty((rank,original.in_features),device=device,
                       dtype=torch.float16) if rank else None)
    gain=(torch.empty((),device=device,dtype=torch.float16)
          if with_cenn else None)
    return PackedCeNNLinear(base,left,right,gain)


def swap(model,path,sub):
    parent,_,field=path.rpartition(".")
    holder=model.get_submodule(parent)
    previous=getattr(holder,field)
    setattr(holder,field,sub)
    return previous


def candidate_paths(model,mlp_last=8,attention_all=True):
    layers=model.model.layers
    result=[]
    for i in range(len(layers)):
        if attention_all:result.append(f"model.layers.{i}.self_attn.o_proj")
        if i>=len(layers)-int(mlp_last):
            result.append(f"model.layers.{i}.mlp.down_proj")
    result=[path for path in result
            if isinstance(model.get_submodule(path),nn.Linear)]
    return result


def tensor_storage_bytes(model):
    return sum(t.numel()*t.element_size() for t in model.state_dict().values())


@torch.inference_mode()
def last_token_metrics(model,batches,refs,last_positions=16):
    """Measure exact last-K token predictions and full-vocabulary JS."""
    wrong=n=0
    js_sum=0.
    for batch,ref in zip(batches,refs):
        logits=model(**batch,use_cache=False).logits[:,-last_positions:,:].float()
        target=ref.to(logits.device).float()
        if logits.shape!=target.shape:
            raise ValueError("Bad teacher/student logits alignment")
        p=logits.softmax(-1).clamp_min(1e-9)
        q=target.softmax(-1).clamp_min(1e-9)
        mixture=(p+q)/2
        js=.5*(p*(p/mixture).log()).sum(-1)
        js+=.5*(q*(q/mixture).log()).sum(-1)
        wrong+=int((logits.argmax(-1)!=target.argmax(-1)).sum())
        n+=int(logits.shape[0]*logits.shape[1])
        js_sum+=float(js.sum())
        del logits,target,p,q,js,mixture
    return {"n":n,"errors":wrong,
            "agreement":1-wrong/n,"mean_js":js_sum/n}


@torch.inference_mode()
def perplexity(model,windows,batch_size=2):
    """Gold-label WikiText PPL, independent of teacher imitation."""
    import math
    loss=0.;count=0
    for i in range(0,len(windows),batch_size):
        chunk=torch.tensor(windows[i:i+batch_size],device=next(model.parameters()).device)
        logits=model(input_ids=chunk[:,:-1],use_cache=False).logits.float()
        gold=chunk[:,1:]
        loss+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                     gold.reshape(-1),reduction="sum"))
        count+=int(gold.numel())
    return {"tokens":count,"nll":loss/count,"perplexity":math.exp(min(50,loss/count))}


def is_acceptable(dev,guard,baseline_dev_ppl,candidate_dev_ppl,
                  max_errors=4,max_js=.004,max_ppl_increase=.015):
    """Selection uses only train-derived DEV/GUARD; not validation/test."""
    return bool(
        dev["errors"]<=max_errors and guard["errors"]<=max_errors
        and dev["mean_js"]<=max_js and guard["mean_js"]<=max_js
        and candidate_dev_ppl<=baseline_dev_ppl*(1+max_ppl_increase))


def projection_bytes(module):
    return sum(t.numel()*t.element_size() for t in module.state_dict().values())
