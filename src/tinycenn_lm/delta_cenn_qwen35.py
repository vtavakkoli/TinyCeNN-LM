"""Experimental Delta-CeNN substitute for Qwen3.5 gated-delta (linear) layers.

Retains the original Qwen causal convolution, Q/K/V and gate projections,
normalization, output projection, and decoder structure. The native gated-delta
*recurrence* is REPLACED by a transparent fp32 tokenwise cellular state rule.
Optionally couples adjacent learned-key feature cells every N global tokens.
No cache/padding/multimodal support. No claims of GPU speedup; eager reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DeltaCeNNState:
    memory: Tensor
    position: int = 0


def qwen_l2norm(x: Tensor) -> Tensor:
    """Use Qwen/FLA's sqrt(sum(x^2)+1e-6) normalization, not F.normalize."""
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


class DeltaCeNNRule(nn.Module):
    """Causal gated delta state with optional spatial coupling over K cells.

    q,k [B,T,H,Dk], v [B,T,H,Dv], g,beta [B,T,H].
    A full-rank identity basis and cells=False recover the Qwen tokenwise rule.
    A compressed basis (feature_dim<Dk) is a trainable approximation.
    """

    def __init__(self, heads: int, key_dim: int, value_dim: int,
                 feature_dim: int | None = None, cells: bool = True,
                 cell_interval: int = 16):
        super().__init__()
        feature_dim = key_dim if feature_dim is None else feature_dim
        if min(heads,key_dim,value_dim,feature_dim,cell_interval) < 1 or feature_dim > key_dim:
            raise ValueError('Invalid Delta-CeNN dimensions or cell_interval')
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.feature_dim, self.cells, self.cell_interval = feature_dim, bool(cells), cell_interval
        weight = torch.eye(key_dim, dtype=torch.float32)[:feature_dim].expand(heads,-1,-1).clone()
        self.register_buffer('feature_basis', weight, persistent=True)
        # Rate is constrained positive; -12 initializes tiny coupling, close to native parity.
        self.cell_strength_logits = nn.Parameter(torch.full((heads,), -12.0))
        self.cell_neighbor_logits = nn.Parameter(torch.zeros(heads,2))
        self.output_log_gain = nn.Parameter(torch.zeros(heads))
        self.output_bias = nn.Parameter(torch.zeros(heads,value_dim))

    def initial_state(self, batch: int, device: torch.device) -> DeltaCeNNState:
        return DeltaCeNNState(torch.zeros(batch,self.heads,self.feature_dim,self.value_dim,
                                           device=device,dtype=torch.float32))

    def _diffuse(self, memory: Tensor) -> Tensor:
        if not self.cells or self.feature_dim == 1:
            return memory
        # No wraparound: boundary cells retain themselves for unavailable neighbor.
        left = torch.cat((memory[:,:,:1],memory[:,:,:-1]),dim=2)
        right = torch.cat((memory[:,:,1:],memory[:,:,-1:]),dim=2)
        w = self.cell_neighbor_logits.float().softmax(-1)
        neighbor = w[None,:,0,None,None]*left + w[None,:,1,None,None]*right
        rate = self.cell_strength_logits.float().sigmoid()[None,:,None,None]
        return memory + rate*(neighbor-memory)

    def forward(self,q:Tensor,k:Tensor,v:Tensor,g:Tensor,beta:Tensor,
                state:DeltaCeNNState|None=None,return_state:bool=False):
        if q.ndim != 4 or q.shape != k.shape or v.shape[:3]!=q.shape[:3]:
            raise ValueError('Expected q/k [B,T,H,Dk] and v [B,T,H,Dv]')
        b,t,h,dk = q.shape
        if (h!=self.heads or dk!=self.key_dim or v.shape[-1]!=self.value_dim or
              g.shape!=(b,t,h) or beta.shape!=(b,t,h) or t<1):
            raise ValueError('Delta-CeNN input dimensions mismatch')
        if state is None:
            state=self.initial_state(b,q.device)
        if (state.memory.shape!=(b,h,self.feature_dim,self.value_dim) or
            state.memory.dtype!=torch.float32 or state.memory.device!=q.device or state.position<0):
            raise ValueError('Invalid recurrent Delta-CeNN state')
        with torch.autocast(device_type=q.device.type,enabled=False):
            qf=qwen_l2norm(q.float());kf=qwen_l2norm(k.float())
            if self.feature_dim!=self.key_dim:
                w=self.feature_basis.float()
                qf=torch.einsum('bthd,hfd->bthf',qf,w)
                kf=torch.einsum('bthd,hfd->bthf',kf,w)
                # For comparable scale under compression; retrain student on teacher outputs.
                qf=qwen_l2norm(qf)
                kf=qwen_l2norm(kf)
            qf=qf*(self.key_dim**-.5)
            vf=v.float();bf=beta.float();gf=g.float()
            mem=state.memory
            pieces=[]
            pos=int(state.position)
            for i in range(t):
                ki=kf[:,i];qi=qf[:,i]
                mem=mem*gf[:,i].exp()[:,:,None,None]
                prediction=torch.einsum('bhf,bhfv->bhv',ki,mem)
                innovation=(vf[:,i]-prediction)*bf[:,i,:,None]
                mem=mem+ki[...,None]*innovation[:,:,None,:]
                yt=torch.einsum('bhf,bhfv->bhv',qi,mem)
                pieces.append(yt)
                pos+=1
                if pos%self.cell_interval==0:
                    mem=self._diffuse(mem)
            y=torch.stack(pieces,dim=1)
            y=y*self.output_log_gain.float().clamp(-.3,.3).exp()[None,None,:,None]
            y=y+self.output_bias.float()[None,None]
            result=DeltaCeNNState(mem,pos)
        return (y,result) if return_state else y


class Qwen35DeltaCeNNLinear(nn.Module):
    """Drop-in for `Qwen3_5GatedDeltaNet` in full-prefix, unpadded evaluation.

    The *native* delta recurrence is never called: this replaces only that
    operator with `DeltaCeNNRule` and reuses pretrained front/back projections.
    """

    def __init__(self,original:nn.Module,feature_dim:int|None=None,
                 cells:bool=True,cell_interval:int=16,
                 native_backend_oracle:bool=False):
        super().__init__()
        self.original=original
        self.core=DeltaCeNNRule(original.num_v_heads,original.head_k_dim,
            original.head_v_dim,feature_dim,cells,cell_interval)
        self.layer_idx=original.layer_idx
        self.layer_type=original.layer_type
        # Diagnostics ONLY. Never enable this on a trained Delta-CeNN candidate:
        # it calls Qwen's original chunk kernel to isolate adapter wiring from
        # token-vs-chunk accumulation differences in the custom recurrence.
        self.native_backend_oracle=bool(native_backend_oracle)
        if self.native_backend_oracle and (cells or self.core.feature_dim != self.core.key_dim):
            raise ValueError('Native-backend oracle requires full-rank, cells=False')
        if original.num_v_heads % original.num_k_heads:
            raise ValueError('Original Qwen value heads must group evenly over key heads')

    def forward(self,hidden_states:Tensor,cache_params=None,attention_mask=None,**kwargs):
        if cache_params is not None:
            raise RuntimeError('Delta-CeNN prototype supports use_cache=False only')
        if hidden_states.ndim!=3 or hidden_states.shape[-1]!=self.original.hidden_size:
            raise ValueError('Invalid hidden_states')
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2] or not bool(attention_mask.bool().all()):
                raise ValueError('Only unpadded full-prefix text supported')
        o=self.original
        batch,t,_=hidden_states.shape
        mixed=o.in_proj_qkv(hidden_states).transpose(1,2)
        kernel=o.conv1d.weight
        conv=F.conv1d(mixed.to(kernel.dtype),kernel,bias=o.conv1d.bias,
                      groups=o.conv_dim,padding=o.conv_kernel_size-1)[...,:t]
        if o.activation=='silu':
            conv=F.silu(conv)
        elif o.activation=='gelu':
            conv=F.gelu(conv)
        else:
            raise ValueError('Unsupported convolution activation: '+str(o.activation))
        conv=conv.to(hidden_states.dtype).transpose(1,2)
        q,k,v=torch.split(conv,(o.key_dim,o.key_dim,o.value_dim),dim=-1)
        q=q.reshape(batch,t,o.num_k_heads,o.head_k_dim)
        k=k.reshape(batch,t,o.num_k_heads,o.head_k_dim)
        v=v.reshape(batch,t,o.num_v_heads,o.head_v_dim)
        if o.num_v_heads!=o.num_k_heads:
            repeat=o.num_v_heads//o.num_k_heads
            q=q.repeat_interleave(repeat,dim=2)
            k=k.repeat_interleave(repeat,dim=2)
        a=o.in_proj_a(hidden_states)
        beta=o.in_proj_b(hidden_states).sigmoid()
        g=-o.A_log.float().exp()[None,None,:]*F.softplus(a.float()+o.dt_bias)
        if self.native_backend_oracle:
            # This is an explicit integration diagnostic, NOT a Delta-CeNN
            # architecture or speed result. It uses precisely the same public
            # Qwen backend as the original no-cache module on this runtime.
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
            y,_=torch_chunk_gated_delta_rule(
                q,k,v,g=g,beta=beta,output_final_state=False,
                use_qk_l2norm_in_kernel=True)
        else:
            # Every evaluated/trainable Delta-CeNN candidate uses OUR cellular
            # recurrence; it never falls back to Qwen's native delta kernel.
            y=self.core(q,k,v,g,beta)
        y=y.to(hidden_states.dtype)
        z=o.in_proj_z(hidden_states).reshape(batch,t,o.num_v_heads,o.head_v_dim)
        y=o.norm(y.reshape(-1,o.head_v_dim),z.reshape(-1,o.head_v_dim))
        return o.out_proj(y.reshape(batch,t,o.value_dim))
