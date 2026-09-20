from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CeNNMixerV3Config:
    hidden_size: int = 1024
    groups: int = 32
    cell_dim: int = 48
    graph_steps: int = 1
    neighbor_offsets: tuple[int, ...] = (1, 2, 4, 8)
    fast_decay_init: float = 0.55
    mid_decay_init: float = 0.88
    slow_decay_init: float = 0.985
    memory_slots: int = 8
    memory_dim: int = 64
    memory_topk: int = 4
    memory_decay_init: float = 0.985
    memory_write_scale: float = 0.35
    dropout: float = 0.0

    @property
    def state_dim(self) -> int:
        return self.groups * self.cell_dim

    def validate(self):
        if self.hidden_size <= 0 or self.groups < 2 or self.cell_dim < 8:
            raise ValueError("invalid CeNN dimensions")
        if self.graph_steps < 1 or not self.neighbor_offsets:
            raise ValueError("invalid graph topology")
        if self.memory_slots < 2 or self.memory_dim < 8:
            raise ValueError("invalid memory dimensions")
        if not 1 <= self.memory_topk <= self.memory_slots:
            raise ValueError("memory_topk must be in [1,memory_slots]")
        for p in (
            self.fast_decay_init, self.mid_decay_init,
            self.slow_decay_init, self.memory_decay_init,
        ):
            if not 0.0 < p < 1.0:
                raise ValueError("decays must be in (0,1)")

    def to_dict(self):
        d = asdict(self)
        d["neighbor_offsets"] = list(self.neighbor_offsets)
        return d


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(torch.log(torch.tensor(p / (1.0 - p))))


class CeNNMixerV3(nn.Module):
    """Sparse multi-timescale CeNN plus fixed content-addressable recurrent memory."""

    def __init__(self, cfg: CeNNMixerV3Config, *, device=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        H, G, D, S = cfg.hidden_size, cfg.groups, cfg.cell_dim, cfg.state_dim
        M, MD = cfg.memory_slots, cfg.memory_dim

        self.in_proj = nn.Linear(H, S, bias=False, device=device, dtype=torch.float32)
        self.gate_proj = nn.Linear(H, G * 9, device=device, dtype=torch.float32)
        self.state_fast = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.state_mid = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.state_slow = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.out_proj = nn.Linear(S * 3, H, bias=False, device=device, dtype=torch.float32)

        self.slot_keys = nn.Parameter(torch.empty(M, MD, device=device))
        self.mem_read_query = nn.Linear(H, MD, bias=False, device=device, dtype=torch.float32)
        self.mem_write_query = nn.Linear(H, MD, bias=False, device=device, dtype=torch.float32)
        self.mem_erase_query = nn.Linear(H, MD, bias=False, device=device, dtype=torch.float32)
        self.mem_value = nn.Linear(H, MD, bias=False, device=device, dtype=torch.float32)
        self.mem_to_state = nn.Linear(MD, S, bias=False, device=device, dtype=torch.float32)
        self.mem_to_hidden = nn.Linear(MD, H, bias=False, device=device, dtype=torch.float32)

        nn.init.normal_(self.in_proj.weight, std=0.018)
        nn.init.normal_(self.gate_proj.weight, std=0.006)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.008)
        nn.init.normal_(self.slot_keys, std=0.08)
        for m in (self.state_fast, self.state_mid, self.state_slow):
            nn.init.eye_(m.weight); m.weight.data.mul_(0.08)
        for m in (
            self.mem_read_query, self.mem_write_query, self.mem_erase_query,
            self.mem_value, self.mem_to_state, self.mem_to_hidden,
        ):
            nn.init.normal_(m.weight, std=0.01)

        self.fast_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.fast_decay_init), device=device))
        self.mid_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.mid_decay_init), device=device))
        self.slow_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.slow_decay_init), device=device))
        self.memory_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.memory_decay_init), device=device))
        self.output_gain_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.global_state_gain_raw = nn.Parameter(torch.tensor(_logit(0.12), device=device))
        self.global_output_gain_raw = nn.Parameter(torch.tensor(_logit(0.10), device=device))

        n = len(cfg.neighbor_offsets)
        self.neighbor_fast = nn.Parameter(torch.zeros(n, device=device))
        self.neighbor_mid = nn.Parameter(torch.zeros(n, device=device))
        self.neighbor_slow = nn.Parameter(torch.zeros(n, device=device))

        self._stream_fast: Optional[Tensor] = None
        self._stream_mid: Optional[Tensor] = None
        self._stream_slow: Optional[Tensor] = None
        self._stream_memory: Optional[Tensor] = None

    @property
    def output_gain(self): return 2.0 * torch.sigmoid(self.output_gain_raw)
    @property
    def global_state_gain(self): return torch.sigmoid(self.global_state_gain_raw)
    @property
    def global_output_gain(self): return torch.sigmoid(self.global_output_gain_raw)

    def reset_stream_state(self):
        self._stream_fast = self._stream_mid = self._stream_slow = None
        self._stream_memory = None

    def _neighbor_mix(self, state, weights):
        out = state
        for _ in range(self.cfg.graph_steps):
            mixed = out
            for w, off in zip(weights, self.cfg.neighbor_offsets):
                mixed = mixed + 0.5 * torch.tanh(w) * (
                    torch.roll(out, off, 1) + torch.roll(out, -off, 1)
                )
            out = mixed
        return out

    def _slot_weights(self, scores):
        if self.cfg.memory_topk < scores.shape[-1]:
            idx = scores.topk(self.cfg.memory_topk, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            scores = scores.masked_fill(~mask, -30.0)
        w = torch.sigmoid(scores)
        return w / w.sum(-1, keepdim=True).clamp_min(1e-6)

    def _memory_read(self, x, memory):
        q = F.normalize(self.mem_read_query(x.float()), dim=-1, eps=1e-6)
        content = F.normalize(memory + self.slot_keys.unsqueeze(0), dim=-1, eps=1e-6)
        score = torch.einsum("bmd,bd->bm", content, q) * math.sqrt(self.cfg.memory_dim)
        w = self._slot_weights(score)
        return torch.einsum("bm,bmd->bd", w, memory)

    def _memory_update(self, x, memory):
        keys = F.normalize(self.slot_keys, dim=-1, eps=1e-6)
        wq = F.normalize(self.mem_write_query(x.float()), dim=-1, eps=1e-6)
        eq = F.normalize(self.mem_erase_query(x.float()), dim=-1, eps=1e-6)
        ws = torch.einsum("bd,md->bm", wq, keys) * math.sqrt(self.cfg.memory_dim)
        es = torch.einsum("bd,md->bm", eq, keys) * math.sqrt(self.cfg.memory_dim)
        write = self._slot_weights(ws)
        erase = torch.sigmoid(es)
        value = torch.tanh(self.mem_value(x.float()))
        decay = torch.sigmoid(self.memory_decay_logit)
        kept = decay * (1.0 - 0.35 * erase.unsqueeze(-1)) * memory
        return kept + self.cfg.memory_write_scale * write.unsqueeze(-1) * value.unsqueeze(1)

    def _step(self, x, sf, sm, ss, memory):
        B = x.shape[0]
        G, D = self.cfg.groups, self.cfg.cell_dim
        global_read = self._memory_read(x, memory)
        global_state = self.mem_to_state(global_read).view(B, G, D)

        u = self.in_proj(x.float()).view(B, G, D)
        gates = self.gate_proj(x.float()).view(B, G, 9)
        ef,wf,rf,em,wm,rm,es,ws,rs = [
            torch.sigmoid(gates[..., i:i+1]) for i in range(9)
        ]

        nf = self._neighbor_mix(sf, self.neighbor_fast)
        nm = self._neighbor_mix(sm, self.neighbor_mid)
        ns = self._neighbor_mix(ss, self.neighbor_slow)
        cf = F.silu(u + self.state_fast(nf))
        cm = F.silu(u + self.state_mid(nm) + 0.12 * sf)
        cs = F.silu(u + self.state_slow(ns) + 0.08 * sm + self.global_state_gain * global_state)

        sf = torch.sigmoid(self.fast_decay_logit) * (1-ef) * sf + wf * cf
        sm = torch.sigmoid(self.mid_decay_logit) * (1-em) * sm + wm * cm
        ss = torch.sigmoid(self.slow_decay_logit) * (1-es) * ss + ws * cs

        local = torch.cat((rf*sf, rm*sm, rs*ss), -1).reshape(B, G*D*3)
        y = self.output_gain * self.out_proj(local)
        y = y + self.global_output_gain * self.mem_to_hidden(global_read)
        memory = self._memory_update(x, memory)
        return y, sf, sm, ss, memory

    def forward(self, hidden_states: Tensor, *, streaming=False):
        B, T, _ = hidden_states.shape
        dev = hidden_states.device
        if streaming and T == 1 and self._stream_fast is not None and self._stream_fast.shape[0] == B:
            sf, sm, ss = self._stream_fast.to(dev), self._stream_mid.to(dev), self._stream_slow.to(dev)
            memory = self._stream_memory.to(dev)
        else:
            shape = (B, self.cfg.groups, self.cfg.cell_dim)
            sf = torch.zeros(shape, device=dev); sm = torch.zeros_like(sf); ss = torch.zeros_like(sf)
            memory = torch.zeros(B, self.cfg.memory_slots, self.cfg.memory_dim, device=dev)

        outs = []
        for t in range(T):
            y, sf, sm, ss, memory = self._step(hidden_states[:, t], sf, sm, ss, memory)
            outs.append(y)
        out = torch.stack(outs, 1)
        if self.training and self.cfg.dropout:
            out = F.dropout(out, p=self.cfg.dropout)
        if streaming:
            self._stream_fast, self._stream_mid, self._stream_slow = sf.detach(), sm.detach(), ss.detach()
            self._stream_memory = memory.detach()
        return out.to(hidden_states.dtype)
