"""CeNN V5: frozen Qwen QKV plus selective native attention-head rescue.

Research implementation for Qwen3.5 **full-attention** token mixers, without KV
cache. Query heads must be grouped by the original Qwen GQA KV group. Every
query head is routed to either original causal SDPA or a GQA-compatible CeNN
memory core. The CeNN core is instantiated ONLY for approximate KV groups:
exact rescue heads don't incur redundant CeNN recurrence work.

Unlike V4, this is a fidelity-first control rather than a claimed faster kernel.
Benchmark real latency; native Flash/SDPA can be faster at short contexts.
"""
from __future__ import annotations

from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F


class Qwen35HeadRescueV5(nn.Module):
    """Replace Qwen3.5 FULL self_attn with group-wise exact-attention rescue.

    ``original`` is the original pretrained Qwen full-attention module and
    ``core`` a CeNNDeltaCellV3 or CeNNMemoryGraphV4 operating only on the
    NON-rescued groups (or ``None`` if all heads are exact). Q/K/V, QK RMS
    normalizers, RoPE, output gate and output projection are preserved.

    The native layer's attention mask may be None or a standard 4D additive
    causal mask; padded/non-causal inputs and cache-based generation are not
    supported and intentionally rejected. No new positional encoding is used.
    """

    def __init__(self, original: nn.Module, core: nn.Module | None,
                 exact_groups: Iterable[int] = ()): 
        super().__init__()
        self.original = original
        self.core = core
        self.head_dim = int(original.head_dim)
        self.num_key_value_groups = int(original.num_key_value_groups)
        self.kv_heads = int(original.k_proj.out_features // self.head_dim)
        self.heads = int(original.q_proj.out_features // (2 * self.head_dim))
        if self.heads != self.kv_heads * self.num_key_value_groups:
            raise ValueError('Pretrained GQA head dimensions are inconsistent')
        self.exact_groups = tuple(sorted(set(int(x) for x in exact_groups)))
        if any(x < 0 or x >= self.kv_heads for x in self.exact_groups):
            raise ValueError('Invalid exact GQA group')
        self.approx_groups = tuple(i for i in range(self.kv_heads)
                                   if i not in self.exact_groups)
        if self.approx_groups:
            if core is None:
                raise ValueError('A CeNN core is required for approximate groups')
            if (core.num_kv_heads != len(self.approx_groups)
                    or core.num_heads != len(self.approx_groups) * self.num_key_value_groups
                    or core.head_dim != self.head_dim):
                raise ValueError('CeNN core shape must match only approximate GQA groups')
        elif core is not None:
            raise ValueError('Fully exact model must not allocate an unused CeNN core')
        self.config = original.config
        self.layer_idx = original.layer_idx
        self.scaling = original.scaling
        self.is_causal = True
        self.attention_dropout = 0.0

    def _native_group(self, q, k, v, mask):
        # The GQA group contains one KV head and multiple Q heads. Expanding
        # K/V is a view, not a materialized copy; SDPA chooses its GPU kernel.
        kv = k.expand(-1, self.num_key_value_groups, -1, -1)
        vv = v.expand(-1, self.num_key_value_groups, -1, -1)
        return F.scaled_dot_product_attention(
            q, kv, vv, attn_mask=mask, dropout_p=0.0,
            is_causal=(mask is None), scale=self.scaling)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise RuntimeError('V5 is prefill-only: pass use_cache=False')
        if position_embeddings is None:
            raise ValueError('Qwen position_embeddings required')
        if hidden_states.ndim != 3:
            raise ValueError('hidden_states must be [B,T,C]')
        b,t,_=hidden_states.shape
        if t < 1:
            raise ValueError('Empty sequence')
        if attention_mask is not None:
            if (attention_mask.ndim != 4 or attention_mask.shape[-2:] != (t,t)
                    or attention_mask.shape[1] != 1):
                raise ValueError('Only unpadded full-prefix Qwen causal masks supported')
            below=torch.ones((t,t),dtype=torch.bool,device=hidden_states.device).tril()
            if not bool((attention_mask.masked_select(below[None,None])==0).all()):
                raise ValueError('Noncausal or padded attention mask unsupported')
            above=~below
            if not bool((attention_mask.masked_select(above[None,None])<0).all()):
                raise ValueError('Future attention positions must be masked')
        # Qwen's native gate is produced by the SECOND half of q_proj.
        # RoPE and QK norms are identical to Qwen3.5's native attention.
        shape=(b,t,-1,self.head_dim)
        query,gate=torch.chunk(self.original.q_proj(hidden_states).reshape(
            b,t,-1,2*self.head_dim),2,dim=-1)
        gate=gate.reshape(b,t,-1)
        q=self.original.q_norm(query.reshape(*shape)).transpose(1,2)
        k=self.original.k_norm(self.original.k_proj(hidden_states).reshape(*shape)).transpose(1,2)
        v=self.original.v_proj(hidden_states).reshape(*shape).transpose(1,2)
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        group_size=self.num_key_value_groups
        pieces: dict[int,torch.Tensor]={}
        for group in self.exact_groups:
            sl=slice(group*group_size,(group+1)*group_size)
            pieces[group]=self._native_group(q[:,sl],k[:,group:group+1],
                                             v[:,group:group+1],attention_mask)
        if self.approx_groups:
            qi=[i for g in self.approx_groups for i in range(g*group_size,(g+1)*group_size)]
            kp=list(self.approx_groups)
            approximate=self.core(q[:,qi],k[:,kp],v[:,kp]).to(q.dtype)
            for j,g in enumerate(self.approx_groups):
                pieces[g]=approximate[:,j*group_size:(j+1)*group_size]
        # Stack in pretrained query-head order, then preserve original Qwen
        # gated output projection and surrounding decoder residual/MLP.
        out=torch.cat([pieces[g] for g in range(self.kv_heads)],dim=1)
        out=out.transpose(1,2).reshape(b,t,-1).contiguous()
        return self.original.o_proj(out * torch.sigmoid(gate)), None

    @property
    def num_exact_heads(self)->int:
        return len(self.exact_groups)*self.num_key_value_groups

    @property
    def num_approx_heads(self)->int:
        return len(self.approx_groups)*self.num_key_value_groups