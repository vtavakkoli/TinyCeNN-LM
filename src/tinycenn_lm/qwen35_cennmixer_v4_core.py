from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CeNNMixerV4Config:
    hidden_size: int = 1024

    # CeNN local branch
    groups: int = 24
    cell_dim: int = 32
    graph_steps: int = 1
    neighbor_offsets: tuple[int, ...] = (1, 2, 4, 8)
    fast_decay_init: float = 0.55
    mid_decay_init: float = 0.88
    slow_decay_init: float = 0.985

    # Compact DeltaCell associative branch
    assoc_heads: int = 8
    key_dim: int = 32
    value_dim: int = 64
    conv_kernel: int = 4
    assoc_decay_init: float = 0.96

    dropout: float = 0.0
    rms_eps: float = 1e-6

    @property
    def state_dim(self) -> int:
        return self.groups * self.cell_dim

    @property
    def q_size(self) -> int:
        return self.assoc_heads * self.key_dim

    @property
    def v_size(self) -> int:
        return self.assoc_heads * self.value_dim

    @property
    def qkv_size(self) -> int:
        return 2 * self.q_size + self.v_size

    def validate(self):
        if self.hidden_size <= 0 or self.groups < 2 or self.cell_dim < 8:
            raise ValueError("invalid local CeNN dimensions")
        if self.assoc_heads < 1 or self.key_dim < 8 or self.value_dim < 8:
            raise ValueError("invalid associative dimensions")
        if self.conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1")
        if self.graph_steps < 1 or not self.neighbor_offsets:
            raise ValueError("invalid sparse topology")
        for p in (
            self.fast_decay_init,self.mid_decay_init,
            self.slow_decay_init,self.assoc_decay_init,
        ):
            if not 0.0 < p < 1.0:
                raise ValueError("decays must be in (0,1)")

    def to_dict(self):
        d=asdict(self)
        d["neighbor_offsets"]=list(self.neighbor_offsets)
        return d


def _logit(p: float) -> float:
    p=min(max(float(p),1e-6),1.0-1e-6)
    return float(torch.log(torch.tensor(p/(1.0-p))))


class CeNNMixerV4(nn.Module):
    """CeNN local dynamics + compact gated-delta associative matrix memory.

    The associative state is [heads, key_dim, value_dim], far smaller than
    Qwen3.5's native Gated-DeltaNet state, while preserving its key mechanisms:
    causal depthwise convolution, normalized q/k, beta, decay, delta correction,
    gated RMS normalization, and content-addressed reads.
    """

    def __init__(self,cfg:CeNNMixerV4Config,*,device=None):
        super().__init__()
        cfg.validate()
        self.cfg=cfg
        H,G,D,S=cfg.hidden_size,cfg.groups,cfg.cell_dim,cfg.state_dim

        # ---------- CeNN local branch ----------
        self.local_in=nn.Linear(H,S,bias=False,device=device,dtype=torch.float32)
        self.local_gate=nn.Linear(H,G*6,bias=True,device=device,dtype=torch.float32)
        self.fast_cell=nn.Linear(D,D,bias=False,device=device,dtype=torch.float32)
        self.mid_cell=nn.Linear(D,D,bias=False,device=device,dtype=torch.float32)
        self.slow_cell=nn.Linear(D,D,bias=False,device=device,dtype=torch.float32)
        self.local_out=nn.Linear(S*3,H,bias=False,device=device,dtype=torch.float32)

        # ---------- Compact associative DeltaCell branch ----------
        C=cfg.qkv_size
        self.in_proj_qkv=nn.Linear(H,C,bias=False,device=device,dtype=torch.float32)
        self.conv_weight=nn.Parameter(torch.empty(C,cfg.conv_kernel,device=device))
        self.conv_bias=nn.Parameter(torch.zeros(C,device=device))
        self.beta_proj=nn.Linear(H,cfg.assoc_heads,bias=True,device=device,dtype=torch.float32)
        self.decay_proj=nn.Linear(H,cfg.assoc_heads,bias=True,device=device,dtype=torch.float32)
        self.z_proj=nn.Linear(H,cfg.v_size,bias=False,device=device,dtype=torch.float32)
        self.assoc_norm_weight=nn.Parameter(torch.ones(cfg.assoc_heads,cfg.value_dim,device=device))
        self.assoc_out=nn.Linear(cfg.v_size,H,bias=False,device=device,dtype=torch.float32)

        n=len(cfg.neighbor_offsets)
        self.neighbor_fast=nn.Parameter(torch.zeros(n,device=device))
        self.neighbor_mid=nn.Parameter(torch.zeros(n,device=device))
        self.neighbor_slow=nn.Parameter(torch.zeros(n,device=device))

        self.fast_decay_logit=nn.Parameter(torch.tensor(_logit(cfg.fast_decay_init),device=device))
        self.mid_decay_logit=nn.Parameter(torch.tensor(_logit(cfg.mid_decay_init),device=device))
        self.slow_decay_logit=nn.Parameter(torch.tensor(_logit(cfg.slow_decay_init),device=device))
        self.assoc_decay_bias=nn.Parameter(
            torch.full((cfg.assoc_heads,),_logit(cfg.assoc_decay_init),device=device)
        )

        # Blend scalars start equally weighted and learn during distillation.
        self.local_gain_raw=nn.Parameter(torch.tensor(0.0,device=device))
        self.assoc_gain_raw=nn.Parameter(torch.tensor(0.0,device=device))

        self._init_weights()

        # streaming state
        self._stream_fast:Optional[Tensor]=None
        self._stream_mid:Optional[Tensor]=None
        self._stream_slow:Optional[Tensor]=None
        self._stream_assoc:Optional[Tensor]=None
        self._stream_conv:Optional[Tensor]=None

    def _init_weights(self):
        nn.init.normal_(self.local_in.weight,std=0.018)
        nn.init.normal_(self.local_gate.weight,std=0.006)
        nn.init.zeros_(self.local_gate.bias)
        nn.init.normal_(self.local_out.weight,std=0.008)
        for m in (self.fast_cell,self.mid_cell,self.slow_cell):
            nn.init.eye_(m.weight); m.weight.data.mul_(0.08)

        nn.init.normal_(self.in_proj_qkv.weight,std=0.018)
        nn.init.zeros_(self.conv_weight)
        # Initialize conv close to identity at current token.
        self.conv_weight.data[:,-1]=1.0
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias,0.0)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.decay_proj.bias)
        nn.init.normal_(self.z_proj.weight,std=0.01)
        nn.init.normal_(self.assoc_out.weight,std=0.008)

    @property
    def local_gain(self):
        return 2.0*torch.sigmoid(self.local_gain_raw)

    @property
    def assoc_gain(self):
        return 2.0*torch.sigmoid(self.assoc_gain_raw)

    def reset_stream_state(self):
        self._stream_fast=self._stream_mid=self._stream_slow=None
        self._stream_assoc=self._stream_conv=None

    def _neighbor_mix(self,state,weights):
        out=state
        for _ in range(self.cfg.graph_steps):
            mixed=out
            for w,off in zip(weights,self.cfg.neighbor_offsets):
                mixed=mixed+0.5*torch.tanh(w)*(
                    torch.roll(out,off,1)+torch.roll(out,-off,1)
                )
            out=mixed
        return out

    def _conv_full(self,proj):
        # proj [B,T,C] -> causal depthwise conv
        x=proj.transpose(1,2)
        x=F.pad(x,(self.cfg.conv_kernel-1,0))
        w=self.conv_weight[:,None,:]
        y=F.conv1d(x,w,self.conv_bias,groups=proj.shape[-1])
        return F.silu(y.transpose(1,2))

    def _conv_step(self,proj_t,buffer):
        # proj_t [B,C], buffer [B,K-1,C]
        if self.cfg.conv_kernel==1:
            window=proj_t[:,None,:]
            new_buffer=buffer
        else:
            window=torch.cat((buffer,proj_t[:,None,:]),dim=1)
            new_buffer=window[:,1:]
        y=(window*self.conv_weight.t().unsqueeze(0)).sum(dim=1)+self.conv_bias
        return F.silu(y),new_buffer

    def _local_step(self,x,sf,sm,ss):
        B=x.shape[0]; G,D=self.cfg.groups,self.cfg.cell_dim
        u=self.local_in(x.float()).view(B,G,D)
        gates=self.local_gate(x.float()).view(B,G,6)
        wf,rf,wm,rm,ws,rs=[torch.sigmoid(gates[...,i:i+1]) for i in range(6)]
        nf=self._neighbor_mix(sf,self.neighbor_fast)
        nm=self._neighbor_mix(sm,self.neighbor_mid)
        ns=self._neighbor_mix(ss,self.neighbor_slow)
        cf=F.silu(u+self.fast_cell(nf))
        cm=F.silu(u+self.mid_cell(nm)+0.12*sf)
        cs=F.silu(u+self.slow_cell(ns)+0.08*sm)
        sf=torch.sigmoid(self.fast_decay_logit)*sf+wf*cf
        sm=torch.sigmoid(self.mid_decay_logit)*sm+wm*cm
        ss=torch.sigmoid(self.slow_decay_logit)*ss+ws*cs
        read=torch.cat((rf*sf,rm*sm,rs*ss),-1).reshape(B,G*D*3)
        return self.local_out(read),sf,sm,ss

    def _assoc_step_from_conv(self,x,conv_t,state):
        B=x.shape[0]
        Hh,dk,dv=self.cfg.assoc_heads,self.cfg.key_dim,self.cfg.value_dim
        qsz=self.cfg.q_size; vsz=self.cfg.v_size

        q=conv_t[:,:qsz].view(B,Hh,dk)
        k=conv_t[:,qsz:2*qsz].view(B,Hh,dk)
        v=conv_t[:,2*qsz:2*qsz+vsz].view(B,Hh,dv)

        q=F.normalize(q,dim=-1,eps=1e-6)/math.sqrt(dk)
        k=F.normalize(k,dim=-1,eps=1e-6)

        beta=torch.sigmoid(self.beta_proj(x.float())).unsqueeze(-1)
        # Positive raw -> negative log decay. Bias keeps initial retention high.
        raw_decay=self.decay_proj(x.float())+self.assoc_decay_bias
        decay=torch.sigmoid(raw_decay).unsqueeze(-1).unsqueeze(-1)
        state=state*decay

        predicted=(state*k.unsqueeze(-1)).sum(dim=-2)
        delta=(v-predicted)*beta
        state=state+k.unsqueeze(-1)*delta.unsqueeze(-2)

        out=(state*q.unsqueeze(-1)).sum(dim=-2)
        z=self.z_proj(x.float()).view(B,Hh,dv)
        rms=out*torch.rsqrt(out.square().mean(-1,keepdim=True)+self.cfg.rms_eps)
        out=rms*self.assoc_norm_weight.unsqueeze(0)*F.silu(z)
        out=self.assoc_out(out.reshape(B,Hh*dv))
        return out,state

    def forward(self,hidden_states:Tensor,*,streaming=False):
        B,T,_=hidden_states.shape
        if T == 0:
            raise ValueError("CeNNMixerV4 requires at least one token")
        dev=hidden_states.device
        G,D=self.cfg.groups,self.cfg.cell_dim
        Hh,dk,dv=self.cfg.assoc_heads,self.cfg.key_dim,self.cfg.value_dim
        C=self.cfg.qkv_size

        reuse=streaming and T==1 and self._stream_fast is not None and self._stream_fast.shape[0]==B
        if reuse:
            sf=self._stream_fast.to(dev); sm=self._stream_mid.to(dev); ss=self._stream_slow.to(dev)
            assoc=self._stream_assoc.to(dev); conv_buf=self._stream_conv.to(dev)
        else:
            sf=torch.zeros(B,G,D,device=dev)
            sm=torch.zeros_like(sf); ss=torch.zeros_like(sf)
            assoc=torch.zeros(B,Hh,dk,dv,device=dev)
            conv_buf=torch.zeros(B,max(self.cfg.conv_kernel-1,0),C,device=dev)

        proj=self.in_proj_qkv(hidden_states.float())
        if reuse:
            conv_seq=None
        else:
            conv_seq=self._conv_full(proj)

        outs=[]
        for t in range(T):
            x=hidden_states[:,t]
            local,sf,sm,ss=self._local_step(x,sf,sm,ss)
            if reuse:
                conv_t,conv_buf=self._conv_step(proj[:,t],conv_buf)
            else:
                conv_t=conv_seq[:,t]
            mem,assoc=self._assoc_step_from_conv(x,conv_t,assoc)
            y=self.local_gain*local+self.assoc_gain*mem
            outs.append(y)

        out=torch.stack(outs,1)
        if self.training and self.cfg.dropout:
            out=F.dropout(out,p=self.cfg.dropout)

        if streaming:
            self._stream_fast=sf.detach(); self._stream_mid=sm.detach(); self._stream_slow=ss.detach()
            self._stream_assoc=assoc.detach()
            if not reuse and self.cfg.conv_kernel>1:
                history=torch.cat((conv_buf,proj),dim=1)
                self._stream_conv=history[:,-(self.cfg.conv_kernel-1):].detach()
            elif not reuse:
                self._stream_conv=conv_buf.detach()
            else:
                self._stream_conv=conv_buf.detach()
        return out.to(hidden_states.dtype)
