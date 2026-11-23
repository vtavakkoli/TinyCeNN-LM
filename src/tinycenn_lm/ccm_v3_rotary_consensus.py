"""CCM-v3: RoPE-aware grouped-head cellular log-mass consensus.

IMPORTANT SCIENTIFIC DISCLOSURE:
This computes mathematically the same normalized weighted value aggregation
as softmax attention (on a finite, causal cell tape). It is a parity-first
research CONTROL, not a novel attention-free computation or a demonstrated
faster model. The module does not call any native Attention, SDPA or KV cache.
The pretrained Llama q/k/v/out projection weights are *copied*, not shared,
so the original attention module is not invoked after substitution.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RotaryConsensusConfig:
    hidden_size: int = 960
    query_heads: int = 15
    key_value_heads: int = 5
    head_dim: int = 64
    slots: int = 256
    query_block: int = 16
    correction_rank: int = 16

    def validate(self):
        if self.query_heads < 1 or self.key_value_heads < 1 or self.query_heads % self.key_value_heads:
            raise ValueError("query heads must be a positive multiple of KV heads")
        if min(self.hidden_size, self.head_dim, self.query_block, self.slots) < 1:
            raise ValueError("dimensions must be positive")
        if self.slots & (self.slots - 1):
            raise ValueError("slots must be a power of two for pairwise cellular reductions")
        if self.correction_rank < 0:
            raise ValueError("correction_rank cannot be negative")


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    if x.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: [B,H,T,D]. cos/sin: [B,T,D] or [T,D]."""
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("Expected RoPE cos/sin [batch,tokens,head_dim]")
    if cos.shape[1] != q.shape[2] or cos.shape[-1] != q.shape[-1]:
        raise ValueError("RoPE dimensions do not match tokens/head_dim")
    cos, sin = cos[:, None, :, :], sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RotaryCellularConsensus(nn.Module):
    """Causal per-head conservative cell merging, positional and pretrained.

    Slots >= context length gives exact algebraic full-attention equivalence
    (subject to floating-point differences), NOT a parameter/speed improvement.
    Slots < context length approximates it by truncating the past window.
    Cached decoding is deliberately unsupported; no native Attention fallback.
    """
    def __init__(self, cfg: RotaryConsensusConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        d, qh, kvh, hd = cfg.hidden_size, cfg.query_heads, cfg.key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(d, qh * hd, bias=False)
        self.k_proj = nn.Linear(d, kvh * hd, bias=False)
        self.v_proj = nn.Linear(d, kvh * hd, bias=False)
        self.o_proj = nn.Linear(qh * hd, d, bias=False)
        if cfg.correction_rank:
            self.correct_down = nn.Linear(d, cfg.correction_rank, bias=False)
            self.correct_up = nn.Linear(cfg.correction_rank, d, bias=False)
            nn.init.zeros_(self.correct_up.weight)
        self._initialized = False

    @classmethod
    def from_pretrained_attention(cls, original: nn.Module, cfg: RotaryConsensusConfig):
        result = cls(cfg)
        pairs = ('q_proj','k_proj','v_proj','o_proj')
        for name in pairs:
            src = getattr(original, name)
            dst = getattr(result, name)
            if src.weight.shape != dst.weight.shape or (src.bias is not None):
                raise ValueError(f"{name}: unsupported bias or projection shape")
            with torch.no_grad():
                dst.weight.copy_(src.weight.detach().to(dst.weight))
            dst.requires_grad_(False)
        result._initialized = True
        return result

    def _merge(self, scores: Tensor, values: Tensor) -> Tensor:
        """Finite logmass + pairwise conductance, not torch.softmax or SDPA.

        scores [B,H,C,K] ; values [B,H,C,K,D]. K is a power of 2.
        """
        mass = scores
        state = values
        while state.shape[-2] > 1:
            left, right = mass[..., 0::2], mass[..., 1::2]
            gate = torch.sigmoid(right-left).unsqueeze(-1)
            state = state[..., 0::2, :] + gate * (state[..., 1::2, :] - state[..., 0::2, :])
            mass = torch.logaddexp(left, right)
        return state[..., 0, :]

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError("CCM-v3 must be initialized using native trained projections")
        if any(kwargs.get(name) is not None for name in ('past_key_value','past_key_values','cache_params')):
            raise NotImplementedError("KV cache is not supported; call with use_cache=False")
        if kwargs.get('use_cache', False):
            raise NotImplementedError("KV cache is not supported")
        if kwargs.get('output_attentions', False):
            raise NotImplementedError("CCM-v3 does not expose attention weights")
        position_embeddings = kwargs.get('position_embeddings')
        if position_embeddings is None or len(position_embeddings) != 2:
            raise ValueError("Llama-provided RoPE position_embeddings=(cos,sin) required")
        if hidden_states.ndim != 3:
            raise ValueError("Expected hidden_states [batch,tokens,hidden]")
        b,t,d = hidden_states.shape
        cfg=self.config
        if not t or d != cfg.hidden_size:raise ValueError("invalid input shape")
        q = self.q_proj(hidden_states).view(b,t,cfg.query_heads,cfg.head_dim).transpose(1,2)
        k = self.k_proj(hidden_states).view(b,t,cfg.key_value_heads,cfg.head_dim).transpose(1,2)
        v = self.v_proj(hidden_states).view(b,t,cfg.key_value_heads,cfg.head_dim).transpose(1,2)
        q,k = rope(q,k,*position_embeddings)
        # Expand grouped KV heads (15 query heads / 5 KV heads for SmolLM2-360M).
        repeat=cfg.query_heads//cfg.key_value_heads
        k = k.repeat_interleave(repeat, dim=1).float()
        v = v.repeat_interleave(repeat, dim=1).float()
        q = q.float()
        attention_mask=kwargs.get('attention_mask')
        if attention_mask is not None and attention_mask.ndim not in (2,4):
            raise ValueError("Unsupported causal attention_mask rank")
        outputs=[]
        for start in range(0,t,cfg.query_block):
            stop=min(t,start+cfg.query_block)
            offsets=torch.arange(cfg.slots,device=q.device)[None,:]
            indices=torch.arange(start,stop,device=q.device)[:,None]-cfg.slots+1+offsets
            validity = indices>=0
            index=indices.clamp(0,t-1)
            # B,H,chunk,K,D; chunking avoids quadratic GPU peak memory.
            kw=k[:,:,index,:]
            vw=v[:,:,index,:]
            score=(q[:,:,start:stop,:].unsqueeze(-2)*kw).sum(-1)*cfg.head_dim**-0.5
            score=score.masked_fill(~validity[None,None,:,:],-1.0e9)
            if attention_mask is not None:
                if attention_mask.ndim==2:
                    if attention_mask.shape[1] < t:
                        raise ValueError("2D attention mask shorter than text")
                    score=score.masked_fill(~attention_mask[:,index].bool()[:,None,:,:],-1.0e9)
                else:
                    if attention_mask.shape[-2] < t or attention_mask.shape[-1] < t:
                        raise ValueError("4D attention mask shorter than text")
                    gathered=attention_mask[:,:,start:stop,:].gather(
                        -1,index[None,None,:,:].expand(b,1,stop-start,cfg.slots))
                    score=score+gathered.float()
                    score=score.masked_fill(score < -1.0e4,-1.0e9)
            outputs.append(self._merge(score,vw))
        merged=torch.cat(outputs,dim=2).transpose(1,2).reshape(b,t,cfg.query_heads*cfg.head_dim)
        result=self.o_proj(merged.to(self.o_proj.weight.dtype))
        if cfg.correction_rank:
            result=result+self.correct_up(F.silu(self.correct_down(hidden_states.float()))).to(result.dtype)
        return result.to(hidden_states.dtype),None
