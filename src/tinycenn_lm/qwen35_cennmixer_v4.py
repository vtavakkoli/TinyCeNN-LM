from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen35_cennmixer_v4_core import CeNNMixerV4, CeNNMixerV4Config


def _primary(result):
    return result[0] if isinstance(result,(tuple,list)) else result


def _replace_primary(result,primary):
    if isinstance(result,tuple): return (primary,*result[1:])
    if isinstance(result,list): return [primary,*result[1:]]
    return primary


class ProgressiveCeNNMixerV4(nn.Module):
    def __init__(self,original:nn.Module,cenn:CeNNMixerV4,kind:str):
        super().__init__()
        self.original=original
        self.cenn=cenn
        self.kind=kind
        self.register_buffer("alpha",torch.tensor(0.0,dtype=torch.float32),persistent=True)
        self.last_original:Optional[Tensor]=None
        self.last_cenn:Optional[Tensor]=None
        for p in self.original.parameters():
            p.requires_grad_(False)

    def set_alpha(self,alpha:float):
        if not 0.0<=float(alpha)<=1.0:
            raise ValueError("alpha must be in [0,1]")
        self.alpha.fill_(float(alpha))

    def forward(self,hidden_states:Tensor,*args,**kwargs):
        # Frozen weights still need input gradients from earlier converted layers.
        original_result=self.original(hidden_states,*args,**kwargs)
        original_y=_primary(original_result)

        streaming=kwargs.get("cache_params") is not None or kwargs.get("past_key_values") is not None
        cenn_y=self.cenn(hidden_states,streaming=streaming)

        self.last_original=original_y.detach()
        self.last_cenn=cenn_y

        a=self.alpha.to(cenn_y.device,cenn_y.dtype)
        mixed=(1.0-a)*original_y.to(cenn_y.dtype)+a*cenn_y
        return _replace_primary(original_result,mixed)


class CeNNOnlyMixerV4(nn.Module):
    def __init__(self,cenn:CeNNMixerV4,kind:str):
        super().__init__()
        self.cenn=cenn
        self.kind=kind

    def forward(self,hidden_states:Tensor,*args,**kwargs):
        streaming=kwargs.get("cache_params") is not None or kwargs.get("past_key_values") is not None
        y=self.cenn(hidden_states,streaming=streaming)
        return (y,None) if self.kind=="full_attention" else y


def mixer_attr(layer:nn.Module):
    bt=getattr(layer,"block_type",None)
    if bt=="linear_attention" and hasattr(layer,"linear_attn"):
        return "linear_attn","linear_attention"
    if bt=="full_attention" and hasattr(layer,"self_attn"):
        return "self_attn","full_attention"
    raise RuntimeError(f"Unsupported Qwen3.5 mixer: {bt!r}")


def install_cenn_mixer_v4(model:nn.Module,layer_idx:int,cfg:CeNNMixerV4Config):
    layer=model.model.layers[layer_idx]
    attr,kind=mixer_attr(layer)
    original=getattr(layer,attr)
    ref=next(original.parameters())
    wrapper=ProgressiveCeNNMixerV4(original,CeNNMixerV4(cfg,device=ref.device),kind)
    setattr(layer,attr,wrapper)
    return wrapper


def iter_wrappers_v4(model:nn.Module):
    for m in model.modules():
        if isinstance(m,ProgressiveCeNNMixerV4):
            yield m


def set_alpha_v4(model:nn.Module,alpha:float):
    for m in iter_wrappers_v4(model):
        m.set_alpha(alpha)


def reset_stream_state_v4(model:nn.Module):
    for m in model.modules():
        if isinstance(m,CeNNMixerV4):
            m.reset_stream_state()


def direct_mixer_losses_v4(model:nn.Module)->dict[str,Tensor]:
    mses=[]; cosines=[]; deltas=[]
    for m in iter_wrappers_v4(model):
        if m.last_original is None or m.last_cenn is None:
            continue
        t=m.last_original.float(); s=m.last_cenn.float()
        den=t.square().mean().clamp_min(1e-12)
        mses.append((s-t).square().mean()/den)

        sf=s.reshape(-1,s.shape[-1]); tf=t.reshape(-1,t.shape[-1])
        cosines.append((1.0-F.cosine_similarity(sf,tf,dim=-1,eps=1e-6)).mean())

        if s.shape[1]>1:
            sd=s[:,1:]-s[:,:-1]
            td=t[:,1:]-t[:,:-1]
            dden=td.square().mean().clamp_min(1e-12)
            deltas.append((sd-td).square().mean()/dden)

    if not mses:
        raise RuntimeError("No CeNN-v4 local outputs available")
    mse=torch.stack(mses).mean()
    return {
        "mse":mse,
        "cosine":torch.stack(cosines).mean(),
        "delta":torch.stack(deltas).mean() if deltas else mse.new_zeros(()),
    }


def freeze_all_except_cenn_v4(model:nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    params=[]; seen=set()
    for m in model.modules():
        if isinstance(m,CeNNMixerV4):
            for p in m.parameters():
                if id(p) not in seen:
                    p.requires_grad_(True)
                    params.append(p); seen.add(id(p))
    return params


def clone_cenn_state_v4(model:nn.Module):
    out={}
    for name,m in model.named_modules():
        if isinstance(m,CeNNMixerV4):
            out[name]={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    return out


def load_cenn_state_v4(model:nn.Module,state:dict):
    mods=dict(model.named_modules())
    for name,sd in state.items():
        mods[name].load_state_dict(sd,strict=True)


def finalize_cenn_only_v4(model:nn.Module):
    for layer in model.model.layers:
        bt=getattr(layer,"block_type",None)
        if bt=="linear_attention" and isinstance(getattr(layer,"linear_attn",None),ProgressiveCeNNMixerV4):
            w=layer.linear_attn
            layer.linear_attn=CeNNOnlyMixerV4(w.cenn,w.kind)
        elif bt=="full_attention" and isinstance(getattr(layer,"self_attn",None),ProgressiveCeNNMixerV4):
            w=layer.self_attn
            layer.self_attn=CeNNOnlyMixerV4(w.cenn,w.kind)
    return model


__all__=[
    "CeNNMixerV4Config","CeNNMixerV4","ProgressiveCeNNMixerV4","CeNNOnlyMixerV4",
    "install_cenn_mixer_v4","set_alpha_v4","reset_stream_state_v4",
    "direct_mixer_losses_v4","freeze_all_except_cenn_v4",
    "clone_cenn_state_v4","load_cenn_state_v4","finalize_cenn_only_v4",
]
