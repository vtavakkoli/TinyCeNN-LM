"""WaveFlux-CeNN: an experimental, purely cellular, scan-compatible sequence layer.

Research reference, not an established novelty/quality/speed result.

No attention, KV cache, DeltaNet, LSTM/mLSTM, SSM library, or Transformer module.
All temporal memory lives in locally transported cellular states.

A wave lane with velocity v evolves as
    s[t,g] = a[t,g] * s[t-1,(g-v) mod G] + b[t,g].
The co-moving frame z[t] = roll(s[t], -t*v) converts it into an independent
per-cell affine recurrence, which admits an exact associative parallel scan.
Every wave lane has constant-sized persistent memory independent of context.

The output optionally includes a conservative nearest-neighbor flux that is
restricted to readout, preserving the exact parallel-scan recurrence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WaveFluxConfig:
    hidden_size: int = 128
    groups: int = 16
    cell_width: int = 8
    velocities: tuple[int, ...] = (0, 1, -1, 2, -2)
    initial_decay: float = 0.995
    initial_write: float = 0.10
    flux_strength: float = 0.22
    residual_gain: float = 0.1

    def __post_init__(self):
        if self.hidden_size < 1 or self.groups < 3 or self.cell_width < 1:
            raise ValueError('invalid hidden_size/groups/cell_width')
        if not self.velocities or any(not isinstance(v, int) for v in self.velocities):
            raise ValueError('velocities must be a non-empty tuple of integers')
        if not 0 < self.initial_decay < 1 or not 0 < self.initial_write < 1:
            raise ValueError('initial_decay and initial_write must be in (0, 1)')
        if not 0 <= self.flux_strength <= 0.5:
            raise ValueError('flux_strength must be in [0, 0.5]')


class WaveFluxCeNN(nn.Module):
    """Causal cellular mixer; input/output: [batch, time, hidden_size].

    `state` has shape [B, wave_lanes, groups, cell_width].
    `mode='scan'` does O(log T) parallel prefix steps and is differentiable.
    `mode='recurrent'` is an independent sequential numerical reference.
    Both support arbitrary-chunk exact state carryover and input gradients.
    """

    def __init__(self, cfg: WaveFluxConfig):
        super().__init__()
        self.cfg = cfg
        r, g, c = len(cfg.velocities), cfg.groups, cfg.cell_width
        self.write_value = nn.Linear(cfg.hidden_size, g*c, bias=False)
        self.gates = nn.Linear(cfg.hidden_size, 3*r*g, bias=True)
        self.readout = nn.Linear(r*g*c, cfg.hidden_size, bias=False)
        self.wave_decay_logits = nn.Parameter(torch.full((r,), math.log(cfg.initial_decay/(1-cfg.initial_decay))))
        self.wave_write_bias = nn.Parameter(torch.full((r,), math.log(cfg.initial_write/(1-cfg.initial_write))))
        self.gain = nn.Parameter(torch.tensor(float(cfg.residual_gain)))
        self.register_buffer('wave_velocities', torch.tensor(cfg.velocities, dtype=torch.long), persistent=False)
        nn.init.normal_(self.write_value.weight, std=0.02)
        nn.init.normal_(self.gates.weight, std=0.005)
        nn.init.zeros_(self.gates.bias)
        nn.init.normal_(self.readout.weight, std=0.01)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.zeros(batch, len(self.cfg.velocities), self.cfg.groups, self.cfg.cell_width,
                           device=device, dtype=dtype)

    def _projections(self, x: Tensor):
        b, t, _ = x.shape
        r, g, c = len(self.cfg.velocities), self.cfg.groups, self.cfg.cell_width
        values = torch.tanh(self.write_value(x.float())).view(b,t,1,g,c)
        gates = self.gates(x.float()).view(b,t,3,r,g,1)
        write_logit, read_logit, flux_logit = gates.unbind(dim=2)
        writes = torch.sigmoid(write_logit + self.wave_write_bias.view(1,1,r,1,1))
        decay = torch.sigmoid(self.wave_decay_logits).view(1,1,r,1,1)
        # convex, bounded cellular EMA; each lane is a traveling wave
        a = decay * (1 - writes)
        inc = writes * values
        read = torch.sigmoid(read_logit)
        conductance = self.cfg.flux_strength * torch.sigmoid(flux_logit)
        return a, inc, read, conductance

    def _moving_indices(self, t: int, device: torch.device, inverse: bool) -> Tensor:
        # P_v maps old group g-v to new group g; P_v^(-n) maps g+nv.
        n = torch.arange(1,t+1,device=device,dtype=torch.long).view(t,1,1)
        g = torch.arange(self.cfg.groups,device=device,dtype=torch.long).view(1,1,self.cfg.groups)
        v = self.wave_velocities.to(device).view(1,-1,1)
        sign = -1 if inverse else +1
        return (g + sign*n*v) % self.cfg.groups   # [T,R,G]

    def _move(self, src: Tensor, *, inverse: bool) -> Tensor:
        b,t,r,g,c = src.shape
        indices = self._moving_indices(t,src.device,inverse).view(1,t,r,g,1)
        return src.gather(dim=3, index=indices.expand(b,t,r,g,c))

    @staticmethod
    def _affine_scan(a: Tensor, inc: Tensor, s0: Tensor) -> Tensor:
        # Inclusive prefix-composition of maps f_t(z) = a_t*z + inc_t.
        # Scaling tensors have a singleton channel axis; no unstable
        # division by accumulated decay products is required.
        A, B = a, inc
        step = 1
        t = a.shape[1]
        while step < t:
            new_A = torch.cat([A[:,:step],A[:,step:]*A[:,:-step]],dim=1)
            new_B = torch.cat([B[:,:step],B[:,step:]+A[:,step:]*B[:,:-step]],dim=1)
            A,B = new_A,new_B
            step *= 2
        return B + A * s0.unsqueeze(1)

    def _scan(self, a: Tensor, inc: Tensor, previous: Tensor) -> Tensor:
        # Co-moving frame turns neighboring recurrent transfers into
        # independent per-cell affine maps, even with time-varying a[t,g].
        a_co = self._move(a, inverse=False)
        b_co = self._move(inc, inverse=False)
        z = self._affine_scan(a_co,b_co,previous)
        return self._move(z, inverse=True)

    def _recurrent(self, a: Tensor, inc: Tensor, previous: Tensor) -> Tensor:
        # Recurrent reference: explicit synchronous local neighbor update.
        b,t,r,g,c = inc.shape
        src_idx = ((torch.arange(g,device=inc.device).view(1,g)
                  - self.wave_velocities.to(inc.device).view(r,1)) % g)
        src_idx = src_idx.view(1,r,g,1).expand(b,r,g,c)
        states = []
        s = previous
        for j in range(t):
            s = a[:,j]*s.gather(dim=2,index=src_idx)+inc[:,j]
            states.append(s)
        return torch.stack(states,dim=1)

    @staticmethod
    def flux_readout(states: Tensor, conductance: Tensor) -> Tensor:
        # J_g = k_g (s_g - s_(g+1)); s*_g = s_g + J_(g-1) - J_g.
        # Exact conservation over the periodic group ring for each wave/channel.
        edge = conductance * (states - torch.roll(states,-1,dims=3))
        return states + torch.roll(edge,1,dims=3) - edge

    def forward(self, x: Tensor, state: Optional[Tensor] = None,
                *, return_state: bool = False, mode: str = 'scan'):
        if x.ndim != 3 or x.shape[-1] != self.cfg.hidden_size:
            raise ValueError(f'expected [B,T,{self.cfg.hidden_size}]')
        if x.shape[1] < 1:
            raise ValueError('sequence length must be >= 1')
        if mode not in ('scan','recurrent'):
            raise ValueError('mode must be scan or recurrent')
        previous = self.initial_state(x.shape[0],x.device) if state is None else state
        expected = (x.shape[0],len(self.cfg.velocities),self.cfg.groups,self.cfg.cell_width)
        if previous.shape != expected or previous.device != x.device:
            raise ValueError(f'invalid state: expected {expected} on {x.device}')
        a,inc,read,conductance = self._projections(x)
        states = (self._scan(a,inc,previous.float()) if mode=='scan'
                  else self._recurrent(a,inc,previous.float()))
        cells = self.flux_readout(states,conductance)
        out = self.readout((cells*read).reshape(x.shape[0],x.shape[1],-1))
        out = (self.gain*out).to(x.dtype)
        return (out,states[:,-1]) if return_state else out


class WaveFluxBlock(nn.Module):
    """Standalone residual cellular block; no Transformer/attention submodules."""
    def __init__(self, cfg: WaveFluxConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.cell = WaveFluxCeNN(cfg)

    def forward(self, x: Tensor, state: Optional[Tensor] = None,
                *, return_state=False, mode='scan'):
        y, nxt = self.cell(self.norm(x),state,return_state=True,mode=mode)
        out = x+y
        return (out,nxt) if return_state else out


class WaveFluxCeNNLM(nn.Module):
    """End-to-end pure cellular causal LM, with embedding, cellular blocks, head.

    There is no non-cellular sequence mixer, pretrained teacher, attention,
    delta rule, or LSTM/mLSTM module. The embedding and output projection are
    vocabulary interfaces, not a Transformer backbone.
    """
    def __init__(self, vocab_size: int, cfg: WaveFluxConfig, layers: int = 3):
        super().__init__()
        if vocab_size < 2 or layers < 1:
            raise ValueError('vocab_size >= 2 and layers >= 1 required')
        self.embedding = nn.Embedding(vocab_size,cfg.hidden_size)
        self.blocks = nn.ModuleList(WaveFluxBlock(cfg) for _ in range(layers))
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size,vocab_size,bias=False)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=cfg.hidden_size**-0.5)

    def forward(self, token_ids: Tensor, states: Optional[Sequence[Tensor]] = None,
                *, return_states=False, mode='scan'):
        if token_ids.ndim!=2:
            raise ValueError('tokens should have shape [B,T]')
        if states is not None and len(states)!=len(self.blocks):
            raise ValueError('one state per block required')
        x = self.embedding(token_ids)
        nxt = []
        for i,block in enumerate(self.blocks):
            x,s = block(x,None if states is None else states[i],return_state=True,mode=mode)
            nxt.append(s)
        logits = self.lm_head(self.final_norm(x))
        return (logits,tuple(nxt)) if return_states else logits
