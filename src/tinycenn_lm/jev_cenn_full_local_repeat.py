"""Four-pass full+local CeNN superlayer for controlled 3-to-1 experiments.

Exact attention sequence: GLOBAL -> WINDOW -> GLOBAL -> WINDOW.
"Global" sees all <= max_len tokens (including sequence edges), unlike
an even-size centered W512 local mask. "Window" sees offsets -64..+64.
Each attention uses native PyTorch scaled-dot-product attention (SDPA)
to avoid O(T*W*D) gather intermediates. Two FFNs and two gated cellular
residuals are jointly trained. Four attentions may be slower than three
native layers; the Colab reports measured speed and parameter count.
"""
from __future__ import annotations

import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
from tinycenn_lm.laya_lab.core import _valid_tokens
from tinycenn_lm.jev_cenn_pairwindow_superlayer import CellularMultiscale


def sdpa_full_or_window(q:Tensor,k:Tensor,v:Tensor,valid:Tensor,
                        *,window:int|None)->Tensor:
    """Correct global or centered sliding attention, including padded queries.

    q/k/v [B,H,T,D]; valid [B,T]. Global window=None sees ALL valid keys.
    For odd window=129, allowed keys are i-64...i+64.
    """
    b,h,t,d=q.shape
    if valid.shape!=(b,t):
        raise ValueError("valid shape mismatch")
    valid=valid.bool()
    allowed=valid[:,None,None,:]
    if window is not None:
        if int(window)<1:
            raise ValueError("window must be >=1")
        left=(int(window)-1)//2
        right=int(window)-1-left
        pos=torch.arange(t,device=q.device)
        offset=pos[None,:]-pos[:,None]
        band=(offset>=-left)&(offset<=right)
        allowed=allowed & band[None,None,:,:]
    # Every query, even invalid padded queries, has at least one valid key in
    # normal batches. Zero out invalid queries after the attention call.
    out=F.scaled_dot_product_attention(
        q,k,v,attn_mask=allowed,dropout_p=0.0,is_causal=False)
    return out * valid[:,None,:,None].to(out.dtype)


class FaithfulAttentionStage(nn.Module):
    def __init__(self, layer:nn.Module, *,window:int|None):
        super().__init__()
        if not hasattr(layer,"attn"):
            raise TypeError("Expected a ModernBERT encoder layer")
        self.norm=copy.deepcopy(layer.attn_norm)
        self.qkv=copy.deepcopy(layer.attn.Wqkv)
        self.proj=copy.deepcopy(layer.attn.Wo)
        self.rope=copy.deepcopy(layer.attn.rotary_emb)
        self.num_heads=int(layer.attn.num_heads)
        self.head_dim=int(layer.attn.head_dim)
        self.window=window

    def forward(self,x:Tensor,valid:Tensor,position_ids:Tensor)->Tensor:
        b,t,hidden=x.shape
        packed=self.qkv(self.norm(x)).view(
            b,t,3,self.num_heads,self.head_dim)
        q,k,v=packed.transpose(1,3).unbind(2)
        cos,sin=self.rope(packed,position_ids=position_ids)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        y=sdpa_full_or_window(q,k,v,valid,window=self.window)
        return (self.proj(y.transpose(1,2).reshape(b,t,hidden))
                *valid[:,:,None].to(x.dtype))


class FullLocalRepeatCeNNSuperLayer(nn.Module):
    """4 sequential attentions + 2 FFNs + two CeNN pathways, ONE encoder slot.

    Initial stage weights are cloned as:
      full from original global layer L;
      window from original L+1;
      full from original L (separate weights);
      window from original L+2.
    Each pair is followed by a teacher-initialized FFN. No hidden state is
    shared across distinct candidate groups. The original teacher is frozen.
    """

    architecture="jev_cenn_global_local_x2_complete_3to1_v1"

    def __init__(self,teacher_global:nn.Module,teacher_local1:nn.Module,
                 teacher_local2:nn.Module,*,window:int=129,
                 max_len:int=512,correction_rank:int=64):
        super().__init__()
        if window<1 or max_len<1:
            raise ValueError("Invalid attention sequence length/window")
        self.window=int(window)
        self.max_len=int(max_len)
        self.stages=nn.ModuleList([
            FaithfulAttentionStage(teacher_global,window=None),
            FaithfulAttentionStage(teacher_local1,window=self.window),
            FaithfulAttentionStage(teacher_global,window=None),
            FaithfulAttentionStage(teacher_local2,window=self.window),
        ])
        self.ffn_norm=nn.ModuleList([
            copy.deepcopy(teacher_local1.mlp_norm),
            copy.deepcopy(teacher_local2.mlp_norm),
        ])
        self.ffn=nn.ModuleList([
            copy.deepcopy(teacher_local1.mlp),
            copy.deepcopy(teacher_local2.mlp),
        ])
        hidden=self.stages[0].num_heads*self.stages[0].head_dim
        self.cenn=nn.ModuleList([
            CellularMultiscale(hidden),CellularMultiscale(hidden),
        ])
        self.cenn_gain=nn.Parameter(torch.tensor([-1.8,-1.8]))
        # Gate the additional second global attention at initialization to
        # avoid uncontrolled residual amplification (still differentiable).
        self.attention_gain=nn.Parameter(
            torch.tensor([2.0,2.0,-1.5,2.0]))
        self.correction_rank=int(correction_rank)
        if correction_rank>0:
            self.correct_norm=nn.LayerNorm(hidden)
            self.correct_down=nn.Linear(hidden,correction_rank,bias=False)
            self.correct_up=nn.Linear(correction_rank,hidden,bias=False)
            nn.init.zeros_(self.correct_up.weight)
        elif correction_rank<0:
            raise ValueError("correction_rank must be nonnegative")
        self._use_cenn=True
        self.last_first=None

    def set_cenn_enabled(self,enabled:bool):
        self._use_cenn=bool(enabled)

    def forward(self,hidden_states:Tensor,attention_mask=None,
                sliding_window_mask=None,position_ids=None,
                cu_seqlens=None,max_seqlen=None,
                output_attentions=False,**kwargs):
        b,t,hidden=hidden_states.shape
        if t>self.max_len:
            raise ValueError(f"Sequence length {t}>{self.max_len}")
        valid=_valid_tokens(attention_mask,hidden_states)
        if position_ids is None:
            position_ids=torch.arange(t,device=hidden_states.device)[None,:]
        x=hidden_states
        for stage_index,stage in enumerate(self.stages):
            x=x+torch.sigmoid(self.attention_gain[stage_index])*stage(
                x,valid,position_ids)
            if stage_index in (1,3):
                pair_index=stage_index//2
                if self._use_cenn:
                    x=x+torch.sigmoid(self.cenn_gain[pair_index])*(
                        self.cenn[pair_index](x,valid))
                x=x+self.ffn[pair_index](self.ffn_norm[pair_index](x))
                if pair_index==0:
                    self.last_first=x if self.training else None
        if self.correction_rank>0:
            x=x+self.correct_up(F.gelu(
                self.correct_down(self.correct_norm(x))))
        return (x,None) if output_attentions else (x,)
