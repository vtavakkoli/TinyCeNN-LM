"""Jev_CeNN single-global-layer causal branch ablation controls.

The reference is the TRAINED V3.1 PDeltaV28FastWindowMemory module.
All other encoder attention layers remain original teacher attention.

Ablation semantics:
 - "full" delegates to the trained, unmodified reference, byte-for-byte.
 - "manual_full" reconstructs the same forward graph to catch implementation
    drift before any ablation result is trusted.
 - "minus_X" removes the X token-mixing path from the router, RENORMALIZING
    router weights over surviving branches; the disabled branch is not run.
 - "minus_cenn" eliminates additive cellular-convolution residual.
 - "static_routing" deletes query-conditioned route changes, retains learned
    per-head route logits.
 - "only_X" executes one branch with probability 1 and NO CeNN residual; "only_cenn" runs only
    trained cellular convolution without attention (diagnostic, not a fair
    like-for-like trained standalone model).

Drop-one is a controlled surgical intervention: it is NOT a Shapley-value
decomposition, and interacting branches make marginal effects non-additive.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from tinycenn_lm.laya_lab.core import _depthwise_sequence_conv, _valid_tokens

BRANCHES = ("local", "landmarks", "linear", "value")
ABLATION_MODES = (
    "full", "manual_full",
    "minus_local", "minus_landmarks", "minus_linear", "minus_value",
    "minus_cenn", "static_routing",
    "only_local", "only_landmarks", "only_linear", "only_value",
    "only_cenn",
)


def active_branches(mode: str) -> tuple[str, ...]:
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown Jev-CeNN ablation mode {mode!r}")
    if mode == "only_cenn":
        return ()
    if mode.startswith("only_"):
        return (mode.removeprefix("only_"),)
    if mode.startswith("minus_") and mode != "minus_cenn":
        return tuple(b for b in BRANCHES if b != mode.removeprefix("minus_"))
    return BRANCHES


class JevGlobalBranchAblation(nn.Module):
    """Wrapper for the exact V3.1 global module; unchanged state on 'full'."""

    def __init__(self, trained: nn.Module, mode: str = "full"):
        super().__init__()
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown Jev-CeNN ablation mode: {mode}")
        required = (
            "Wqkv", "Wo", "num_heads", "head_dim", "rotary_emb",
            "_local_attention", "_landmarks", "_global_linear",
            "local_weight", "route_logits", "route_query",
            "local_log_temperature", "conv_gain", "finish",
        )
        absent = [name for name in required if not hasattr(trained,name)]
        if absent:
            raise TypeError(f"Expected trained V3.1 global module, missing {absent}")
        self.base=trained
        self.mode=mode

    def forward(self, hidden_states: Tensor, position_embeddings=None,
                attention_mask=None, position_ids=None, sliding_window_mask=None,
                **kwargs):
        # A reference mode that NEVER changes the trained model computation.
        if self.mode == "full":
            return self.base(
                hidden_states,position_embeddings=position_embeddings,
                attention_mask=attention_mask,position_ids=position_ids,
                sliding_window_mask=sliding_window_mask,**kwargs
            )
        base=self.base
        b,t,_=hidden_states.shape
        qkv=base.Wqkv(hidden_states).reshape(
            b,t,3,base.num_heads,base.head_dim)
        q,k,v=qkv.transpose(3,1).unbind(dim=2)
        if position_embeddings is None:
            cos,sin=base.rotary_emb(qkv,position_ids=position_ids)
        else:
            cos,sin=position_embeddings
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        valid=_valid_tokens(attention_mask,hidden_states)
        mask=valid[:,None,:,None].to(v.dtype)
        q,k,v=q*mask,k*mask,v*mask

        if self.mode=="only_cenn":
            # Baseline-specific cellular-only diagnostic: trained local
            # convolution applied to pretrained value features.
            cellular=_depthwise_sequence_conv(v,base.local_weight,causal=False)
            return base.finish(cellular*mask,hidden_states)

        active=active_branches(self.mode)
        channels={}
        if "local" in active:
            temperature=base.local_log_temperature.clamp(-0.4,0.4).exp()
            channels["local"]=base._local_attention(
                q*temperature[None,:,None,None],k,v,valid)
        if "landmarks" in active:
            channels["landmarks"]=base._landmarks(q,k,v,valid)
        if "linear" in active:
            channels["linear"]=base._global_linear(q,k,v,mask)*mask
        if "value" in active:
            channels["value"]=v
        # This is query-dependent routing over FOUR fixed branch types,
        # not an attention distribution over sequence positions.
        if self.mode=="static_routing":
            dynamic=0.0
        else:
            qnorm=F.normalize(q,dim=-1)
            dynamic=torch.einsum("bhtd,hkd->bhtk",qnorm,base.route_query)
        logits=base.route_logits.to(q.dtype)[None,:,None,:]+dynamic
        if len(active)!=4:
            eligible=torch.tensor(
                [x in active for x in BRANCHES],device=q.device)
            logits=logits.masked_fill(~eligible[None,None,None,:],-1e4)
        mixture=torch.softmax(logits,dim=-1)
        fused=sum(
            mixture[...,idx:idx+1]*channels[name]
            for idx,name in enumerate(BRANCHES) if name in channels
        )
        if self.mode!="minus_cenn" and not self.mode.startswith("only_"):
            cellular=_depthwise_sequence_conv(
                v,base.local_weight,causal=False)
            gain=torch.sigmoid(base.conv_gain).to(v.dtype)[None,:,None,None]
            fused=fused+gain*(cellular-v)
        return base.finish(fused*mask,hidden_states)


@torch.no_grad()
def mean_routing_weights(trained: nn.Module, cache: list[dict]):
    """Mean reference route probabilities across valid tokens and all heads."""
    weighted=torch.zeros(4,dtype=torch.float64)
    token_heads=0
    for sample in cache:
        x=sample["x"]
        b,t,_=x.shape
        qkv=trained.Wqkv(x).reshape(
            b,t,3,trained.num_heads,trained.head_dim)
        q=qkv.transpose(3,1).unbind(2)[0]
        kwargs=sample["call_kwargs"]
        if kwargs.get("position_embeddings") is None:
            cos,sin=trained.rotary_emb(qkv,position_ids=kwargs.get("position_ids"))
        else:
            cos,sin=kwargs["position_embeddings"]
        # RoPE may be interleaved: use the identical teacher transform.
        q=apply_rotary_pos_emb(q,q,cos,sin)[0]
        gate=torch.softmax(
            trained.route_logits.to(q.dtype)[None,:,None,:]+
            torch.einsum("bhtd,hkd->bhtk",F.normalize(q,dim=-1),trained.route_query),
            -1
        )
        mask=sample["valid"].bool()[:,None,:].expand(-1,trained.num_heads,-1)
        weighted+=gate[mask].double().sum(0).cpu()
        token_heads+=int(mask.sum().item())
    return {name:float(weighted[i]/max(token_heads,1))
            for i,name in enumerate(BRANCHES)}
