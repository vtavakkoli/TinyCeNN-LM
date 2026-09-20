from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CeNNMixerV1Config:
    hidden_size: int = 1024
    groups: int = 16
    cell_dim: int = 32
    graph_steps: int = 1
    neighbor_offsets: tuple[int, ...] = (1, 2, 4, 8)
    fast_decay_init: float = 0.55
    mid_decay_init: float = 0.85
    slow_decay_init: float = 0.97
    output_scale_init: float = 0.0
    dropout: float = 0.0

    @property
    def state_dim(self) -> int:
        return self.groups * self.cell_dim

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.groups < 2:
            raise ValueError("groups must be >= 2")
        if self.cell_dim < 8:
            raise ValueError("cell_dim must be >= 8")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        for x in (self.fast_decay_init, self.mid_decay_init, self.slow_decay_init):
            if not 0.0 < x < 1.0:
                raise ValueError("decay init values must be in (0,1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["neighbor_offsets"] = list(self.neighbor_offsets)
        return d


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(torch.log(torch.tensor(p / (1.0 - p))))


class CeNNMixerV1(nn.Module):
    """Multi-timescale sparse cellular recurrent sequence mixer.

    The mixer has three state banks (fast/mid/slow), grouped cellular state,
    sparse neighbor communication, and separate erase/write/read gates.

    The residual output scale starts at zero so insertion can begin as identity
    when used in a residual-compatible distillation wrapper.
    """

    def __init__(self, cfg: CeNNMixerV1Config, *, device=None, dtype=None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        H = cfg.hidden_size
        S = cfg.state_dim
        G = cfg.groups
        D = cfg.cell_dim

        self.in_proj = nn.Linear(H, S, bias=False, device=device, dtype=torch.float32)
        self.gate_proj = nn.Linear(H, G * 9, bias=True, device=device, dtype=torch.float32)
        self.state_fast = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.state_mid = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.state_slow = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.out_proj = nn.Linear(S * 3, H, bias=False, device=device, dtype=torch.float32)

        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        for m in (self.state_fast, self.state_mid, self.state_slow):
            nn.init.eye_(m.weight)
            m.weight.data.mul_(0.10)

        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.zeros_(self.gate_proj.bias)

        self.fast_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.fast_decay_init), device=device))
        self.mid_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.mid_decay_init), device=device))
        self.slow_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.slow_decay_init), device=device))
        self.output_scale_raw = nn.Parameter(torch.tensor(float(cfg.output_scale_init), device=device))

        # Static sparse neighbor weights, learned per offset and timescale.
        n = len(cfg.neighbor_offsets)
        self.neighbor_fast = nn.Parameter(torch.zeros(n, device=device, dtype=torch.float32))
        self.neighbor_mid = nn.Parameter(torch.zeros(n, device=device, dtype=torch.float32))
        self.neighbor_slow = nn.Parameter(torch.zeros(n, device=device, dtype=torch.float32))

        self._stream_fast: Optional[Tensor] = None
        self._stream_mid: Optional[Tensor] = None
        self._stream_slow: Optional[Tensor] = None

    @property
    def output_scale(self) -> Tensor:
        return 0.25 * torch.tanh(self.output_scale_raw)

    def reset_stream_state(self) -> None:
        self._stream_fast = self._stream_mid = self._stream_slow = None

    def _neighbor_mix(self, state: Tensor, weights: Tensor) -> Tensor:
        # state: [B,G,D]
        out = state
        for _ in range(self.cfg.graph_steps):
            mixed = out
            for w, off in zip(weights, self.cfg.neighbor_offsets):
                a = torch.roll(out, shifts=off, dims=1)
                b = torch.roll(out, shifts=-off, dims=1)
                mixed = mixed + torch.tanh(w) * 0.5 * (a + b)
            out = mixed
        return out

    def _step(self, x_t: Tensor, sf: Tensor, sm: Tensor, ss: Tensor):
        B = x_t.shape[0]
        G, D = self.cfg.groups, self.cfg.cell_dim

        u = self.in_proj(x_t.float()).view(B, G, D)
        gates = self.gate_proj(x_t.float()).view(B, G, 9)
        ef,wf,rf, em,wm,rm, es,ws,rs = [torch.sigmoid(gates[..., i:i+1]) for i in range(9)]

        df = torch.sigmoid(self.fast_decay_logit)
        dm = torch.sigmoid(self.mid_decay_logit)
        ds = torch.sigmoid(self.slow_decay_logit)

        nf = self._neighbor_mix(sf, self.neighbor_fast)
        nm = self._neighbor_mix(sm, self.neighbor_mid)
        ns = self._neighbor_mix(ss, self.neighbor_slow)

        cf = F.silu(u + self.state_fast(nf.float()))
        cm = F.silu(u + self.state_mid(nm.float()) + 0.15 * sf)
        cs = F.silu(u + self.state_slow(ns.float()) + 0.10 * sm)

        sf = (df * (1.0 - ef)) * sf + wf * cf
        sm = (dm * (1.0 - em)) * sm + wm * cm
        ss = (ds * (1.0 - es)) * ss + ws * cs

        read = torch.cat([rf * sf, rm * sm, rs * ss], dim=-1).reshape(B, G * D * 3)
        y = self.out_proj(read.float())
        return y, sf, sm, ss

    def forward(self, hidden_states: Tensor, *, streaming: bool = False) -> Tensor:
        B,T,H = hidden_states.shape
        dev = hidden_states.device
        if streaming and T == 1 and self._stream_fast is not None and self._stream_fast.shape[0] == B:
            sf = self._stream_fast.to(dev)
            sm = self._stream_mid.to(dev)
            ss = self._stream_slow.to(dev)
        else:
            shape = (B, self.cfg.groups, self.cfg.cell_dim)
            sf = torch.zeros(shape, device=dev, dtype=torch.float32)
            sm = torch.zeros_like(sf)
            ss = torch.zeros_like(sf)

        outs = []
        for t in range(T):
            y,sf,sm,ss = self._step(hidden_states[:,t],sf,sm,ss)
            outs.append(y)
        out = torch.stack(outs,dim=1).to(hidden_states.dtype)

        if streaming:
            self._stream_fast = sf.detach()
            self._stream_mid = sm.detach()
            self._stream_slow = ss.detach()

        if self.training and self.cfg.dropout:
            out = F.dropout(out,p=self.cfg.dropout)

        return self.output_scale.to(out.dtype) * out


class CeNNLinearAttentionAdapter(nn.Module):
    """Drop-in replacement for Qwen3.5 GatedDeltaNet forward signature."""
    def __init__(self, core: CeNNMixerV1):
        super().__init__()
        self.core = core

    def forward(self, hidden_states: Tensor, cache_params=None, attention_mask=None, **kwargs):
        streaming = cache_params is not None
        return self.core(hidden_states, streaming=streaming)


class CeNNFullAttentionAdapter(nn.Module):
    """Drop-in replacement for Qwen3.5 full-attention forward signature."""
    def __init__(self, core: CeNNMixerV1):
        super().__init__()
        self.core = core

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        streaming = past_key_values is not None
        return self.core(hidden_states, streaming=streaming), None


def mixer_attr(layer: nn.Module) -> tuple[str, str]:
    block_type = getattr(layer, "block_type", None)
    if block_type == "linear_attention" and hasattr(layer, "linear_attn"):
        return "linear_attn", "linear_attention"
    if block_type == "full_attention" and hasattr(layer, "self_attn"):
        return "self_attn", "full_attention"
    raise RuntimeError(f"Unsupported Qwen3.5 layer mixer: block_type={block_type!r}")


def install_cenn_mixer_v1(model: nn.Module, layer_idx: int, cfg: CeNNMixerV1Config) -> CeNNMixerV1:
    layer = model.model.layers[layer_idx]
    attr, kind = mixer_attr(layer)
    ref = next(layer.parameters())
    core = CeNNMixerV1(cfg, device=ref.device, dtype=ref.dtype)
    adapter = CeNNLinearAttentionAdapter(core) if kind == "linear_attention" else CeNNFullAttentionAdapter(core)
    setattr(layer, attr, adapter)
    return core


def freeze_all_except_cenn(model: nn.Module) -> list[Tensor]:
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, CeNNMixerV1):
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
    return params


__all__ = [
    "CeNNMixerV1Config",
    "CeNNMixerV1",
    "CeNNLinearAttentionAdapter",
    "CeNNFullAttentionAdapter",
    "mixer_attr",
    "install_cenn_mixer_v1",
    "freeze_all_except_cenn",
]
