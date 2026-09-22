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

    # Local CeNN correction branch.
    groups: int = 24
    cell_dim: int = 32
    graph_steps: int = 1
    neighbor_offsets: tuple[int, ...] = (1, 2, 4, 8)
    fast_decay_init: float = 0.55
    mid_decay_init: float = 0.90
    slow_decay_init: float = 0.985

    # Compact Gated-Delta associative branch. 16 heads deliberately preserves
    # Qwen3.5-0.8B's head partition while halving the 128-d key/value subspace.
    assoc_heads: int = 16
    key_dim: int = 64
    value_dim: int = 64
    conv_kernel: int = 4

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
        for p in (self.fast_decay_init, self.mid_decay_init, self.slow_decay_init):
            if not 0.0 < p < 1.0:
                raise ValueError("local decays must be in (0,1)")

    def to_dict(self):
        d = asdict(self)
        d["neighbor_offsets"] = list(self.neighbor_offsets)
        return d


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class CeNNMixerV4(nn.Module):
    """Compact CeNN + Gated-Delta replacement mixer.

    The associative recurrence is intentionally written in the same mathematical
    form as Qwen3.5 Gated DeltaNet:

        g_t = -exp(A) * softplus(a_t + dt)
        S_t^- = exp(g_t) * S_{t-1}
        e_t = v_t - k_t^T S_t^-
        S_t = S_t^- + beta_t * k_t e_t^T
        o_t = q_t^T S_t

    q and k are L2-normalized and q is scaled by 1/sqrt(d_k).  During full
    sequence training/prefill we use Transformers' chunked implementation when
    available; recurrent fallback is mathematically equivalent.

    The CeNN branch is a small contractive multi-timescale correction.  Its
    normalized neighbor operator and EMA-style state update keep the local state
    bounded over 1024+ token windows.
    """

    def __init__(self, cfg: CeNNMixerV4Config, *, device=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        H, G, D, S = cfg.hidden_size, cfg.groups, cfg.cell_dim, cfg.state_dim

        # ---------- local CeNN correction ----------
        self.local_in = nn.Linear(H, S, bias=False, device=device, dtype=torch.float32)
        self.local_gate = nn.Linear(H, G * 6, bias=True, device=device, dtype=torch.float32)
        self.fast_cell = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.mid_cell = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.slow_cell = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.local_out = nn.Linear(S * 3, H, bias=False, device=device, dtype=torch.float32)

        n = len(cfg.neighbor_offsets)
        self.neighbor_fast = nn.Parameter(torch.zeros(n, device=device))
        self.neighbor_mid = nn.Parameter(torch.zeros(n, device=device))
        self.neighbor_slow = nn.Parameter(torch.zeros(n, device=device))
        self.fast_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.fast_decay_init), device=device))
        self.mid_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.mid_decay_init), device=device))
        self.slow_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.slow_decay_init), device=device))

        # ---------- compact Gated-Delta associative memory ----------
        C = cfg.qkv_size
        self.in_proj_qkv = nn.Linear(H, C, bias=False, device=device, dtype=torch.float32)
        self.conv_weight = nn.Parameter(torch.empty(C, cfg.conv_kernel, device=device))
        self.beta_proj = nn.Linear(H, cfg.assoc_heads, bias=False, device=device, dtype=torch.float32)
        self.decay_proj = nn.Linear(H, cfg.assoc_heads, bias=False, device=device, dtype=torch.float32)
        self.assoc_log_rate = nn.Parameter(torch.zeros(cfg.assoc_heads, device=device))
        self.assoc_dt_bias = nn.Parameter(torch.zeros(cfg.assoc_heads, device=device))
        self.z_proj = nn.Linear(H, cfg.v_size, bias=False, device=device, dtype=torch.float32)

        # Qwen's gated RMS norm is zero-centered: multiplicative factor is 1+w.
        self.assoc_norm_weight = nn.Parameter(torch.zeros(cfg.value_dim, device=device))
        self.assoc_out = nn.Linear(cfg.v_size, H, bias=False, device=device, dtype=torch.float32)

        # Start from the spectrally initialized associative branch.  The CeNN
        # correction is present from step 1, but deliberately small.
        self.local_gain_raw = nn.Parameter(torch.tensor(-4.0, device=device))
        self.assoc_gain_raw = nn.Parameter(torch.tensor(0.0, device=device))

        self._init_weights()

        self._stream_fast: Optional[Tensor] = None
        self._stream_mid: Optional[Tensor] = None
        self._stream_slow: Optional[Tensor] = None
        self._stream_assoc: Optional[Tensor] = None
        self._stream_conv: Optional[Tensor] = None

    def _init_weights(self):
        nn.init.normal_(self.local_in.weight, std=0.018)
        nn.init.normal_(self.local_gate.weight, std=0.004)
        nn.init.zeros_(self.local_gate.bias)
        nn.init.normal_(self.local_out.weight, std=0.004)
        for m in (self.fast_cell, self.mid_cell, self.slow_cell):
            nn.init.eye_(m.weight)
            m.weight.data.mul_(0.05)

        nn.init.normal_(self.in_proj_qkv.weight, std=0.018)
        nn.init.zeros_(self.conv_weight)
        self.conv_weight.data[:, -1] = 1.0
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.normal_(self.z_proj.weight, std=0.01)
        nn.init.normal_(self.assoc_out.weight, std=0.008)

    @property
    def local_gain(self):
        return 2.0 * torch.sigmoid(self.local_gain_raw)

    @property
    def assoc_gain(self):
        return 2.0 * torch.sigmoid(self.assoc_gain_raw)

    def reset_stream_state(self):
        self._stream_fast = self._stream_mid = self._stream_slow = None
        self._stream_assoc = self._stream_conv = None

    def _neighbor_mix(self, state: Tensor, weights: Tensor) -> Tensor:
        # Normalized symmetric graph operator.  The denominator bounds the
        # induced amplification even when learned neighbor weights grow.
        out = state
        for _ in range(self.cfg.graph_steps):
            acc = out
            denom = out.new_tensor(1.0)
            for w, off in zip(weights, self.cfg.neighbor_offsets):
                c = 0.25 * torch.tanh(w)
                acc = acc + c * (torch.roll(out, off, 1) + torch.roll(out, -off, 1))
                denom = denom + 2.0 * c.abs()
            out = acc / denom
        return out

    def _conv_full(self, proj: Tensor) -> Tensor:
        x = proj.transpose(1, 2)
        x = F.pad(x, (self.cfg.conv_kernel - 1, 0))
        w = self.conv_weight[:, None, :]
        y = F.conv1d(x, w, None, groups=proj.shape[-1])
        return F.silu(y.transpose(1, 2))

    def _conv_step(self, proj_t: Tensor, buffer: Tensor):
        if self.cfg.conv_kernel == 1:
            window = proj_t[:, None, :]
            new_buffer = buffer
        else:
            window = torch.cat((buffer, proj_t[:, None, :]), dim=1)
            new_buffer = window[:, 1:]
        y = (window * self.conv_weight.t().unsqueeze(0)).sum(dim=1)
        return F.silu(y), new_buffer

    def _local_step(self, x: Tensor, sf: Tensor, sm: Tensor, ss: Tensor):
        B, G, D = x.shape[0], self.cfg.groups, self.cfg.cell_dim
        u = self.local_in(x.float()).view(B, G, D)
        gates = self.local_gate(x.float()).view(B, G, 6)
        wf, rf, wm, rm, ws, rs = [torch.sigmoid(gates[..., i:i + 1]) for i in range(6)]

        nf = self._neighbor_mix(sf, self.neighbor_fast)
        nm = self._neighbor_mix(sm, self.neighbor_mid)
        ns = self._neighbor_mix(ss, self.neighbor_slow)

        cf = F.silu(u + self.fast_cell(nf))
        cm = F.silu(u + self.mid_cell(nm) + 0.10 * sf)
        cs = F.silu(u + self.slow_cell(ns) + 0.06 * sm)

        df = torch.sigmoid(self.fast_decay_logit)
        dm = torch.sigmoid(self.mid_decay_logit)
        ds = torch.sigmoid(self.slow_decay_logit)

        # Contractive EMA update instead of unconstrained additive accumulation.
        sf = df * sf + (1.0 - df) * wf * cf
        sm = dm * sm + (1.0 - dm) * wm * cm
        ss = ds * ss + (1.0 - ds) * ws * cs

        read = torch.cat((rf * sf, rm * sm, rs * ss), -1).reshape(B, G * D * 3)
        return self.local_out(read), sf, sm, ss

    def _local_sequence(self, hidden_states: Tensor, sf: Tensor, sm: Tensor, ss: Tensor):
        outs = []
        for t in range(hidden_states.shape[1]):
            y, sf, sm, ss = self._local_step(hidden_states[:, t], sf, sm, ss)
            outs.append(y)
        return torch.stack(outs, 1), sf, sm, ss

    def _assoc_params(self, hidden_states: Tensor, conv_seq: Tensor):
        B, T = hidden_states.shape[:2]
        Hh, dk, dv = self.cfg.assoc_heads, self.cfg.key_dim, self.cfg.value_dim
        qsz, vsz = self.cfg.q_size, self.cfg.v_size

        q = conv_seq[..., :qsz].view(B, T, Hh, dk)
        k = conv_seq[..., qsz:2 * qsz].view(B, T, Hh, dk)
        v = conv_seq[..., 2 * qsz:2 * qsz + vsz].view(B, T, Hh, dv)
        beta = torch.sigmoid(self.beta_proj(hidden_states.float()))

        a = self.decay_proj(hidden_states.float())
        # Log-space decay exactly mirrors Qwen3.5's Gated DeltaNet parameterization.
        g = -self.assoc_log_rate.float().exp().view(1, 1, Hh) * F.softplus(
            a.float() + self.assoc_dt_bias.float().view(1, 1, Hh)
        )
        return q, k, v, g, beta

    def _gated_norm_out(self, hidden_states: Tensor, out: Tensor) -> Tensor:
        B, T = hidden_states.shape[:2]
        Hh, dv = self.cfg.assoc_heads, self.cfg.value_dim
        z = self.z_proj(hidden_states.float()).view(B, T, Hh, dv)
        rms = out.float() * torch.rsqrt(out.float().square().mean(-1, keepdim=True) + self.cfg.rms_eps)
        normed = rms * (1.0 + self.assoc_norm_weight.view(1, 1, 1, dv))
        gated = normed * F.silu(z)
        return self.assoc_out(gated.reshape(B, T, Hh * dv))

    def _assoc_recurrent(self, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, state: Tensor):
        B, T, Hh, dk = q.shape
        q = F.normalize(q.float(), dim=-1, eps=1e-6) / math.sqrt(dk)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        outs = []
        for t in range(T):
            state = state * g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
            pred = (state * k[:, t].unsqueeze(-1)).sum(dim=-2)
            delta = (v[:, t].float() - pred) * beta[:, t].unsqueeze(-1)
            state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((state * q[:, t].unsqueeze(-1)).sum(dim=-2))
        return torch.stack(outs, 1), state

    def _assoc_full(self, hidden_states: Tensor, proj: Tensor, initial_state: Optional[Tensor] = None):
        B, T = hidden_states.shape[:2]
        Hh, dk, dv = self.cfg.assoc_heads, self.cfg.key_dim, self.cfg.value_dim
        q, k, v, g, beta = self._assoc_params(hidden_states, self._conv_full(proj))

        # Use HF's chunk-parallel Gated Delta rule when available.  It is the
        # same recurrence as the fallback but is much more practical at T=1024.
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
            out, state = torch_chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            out = out.float()
        except Exception:
            if initial_state is None:
                initial_state = torch.zeros(B, Hh, dk, dv, device=hidden_states.device, dtype=torch.float32)
            out, state = self._assoc_recurrent(q, k, v, g, beta, initial_state.float())

        return self._gated_norm_out(hidden_states, out), state

    def _assoc_step(self, x: Tensor, conv_t: Tensor, state: Tensor):
        B = x.shape[0]
        Hh, dk, dv = self.cfg.assoc_heads, self.cfg.key_dim, self.cfg.value_dim
        qsz, vsz = self.cfg.q_size, self.cfg.v_size

        q = conv_t[:, :qsz].view(B, Hh, dk)
        k = conv_t[:, qsz:2 * qsz].view(B, Hh, dk)
        v = conv_t[:, 2 * qsz:2 * qsz + vsz].view(B, Hh, dv)

        q = F.normalize(q.float(), dim=-1, eps=1e-6) / math.sqrt(dk)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.beta_proj(x.float())).unsqueeze(-1)
        g = -self.assoc_log_rate.float().exp().view(1, Hh) * F.softplus(
            self.decay_proj(x.float()) + self.assoc_dt_bias.float().view(1, Hh)
        )
        state = state * g.exp().unsqueeze(-1).unsqueeze(-1)

        pred = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v.float() - pred) * beta
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)

        z = self.z_proj(x.float()).view(B, Hh, dv)
        rms = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + self.cfg.rms_eps)
        out = rms * (1.0 + self.assoc_norm_weight.view(1, 1, dv)) * F.silu(z)
        return self.assoc_out(out.reshape(B, Hh * dv)), state

    def forward(self, hidden_states: Tensor, *, streaming: bool = False):
        B, T, _ = hidden_states.shape
        dev = hidden_states.device
        G, D = self.cfg.groups, self.cfg.cell_dim
        Hh, dk, dv = self.cfg.assoc_heads, self.cfg.key_dim, self.cfg.value_dim
        C = self.cfg.qkv_size

        reuse = (
            streaming and T == 1 and self._stream_fast is not None
            and self._stream_fast.shape[0] == B
        )
        if reuse:
            sf = self._stream_fast.to(dev)
            sm = self._stream_mid.to(dev)
            ss = self._stream_slow.to(dev)
            assoc = self._stream_assoc.to(dev)
            conv_buf = self._stream_conv.to(dev)
        else:
            sf = torch.zeros(B, G, D, device=dev)
            sm = torch.zeros_like(sf)
            ss = torch.zeros_like(sf)
            assoc = torch.zeros(B, Hh, dk, dv, device=dev, dtype=torch.float32)
            conv_buf = torch.zeros(B, max(self.cfg.conv_kernel - 1, 0), C, device=dev)

        local, sf, sm, ss = self._local_sequence(hidden_states, sf, sm, ss)
        proj = self.in_proj_qkv(hidden_states.float())

        if reuse:
            conv_t, conv_buf = self._conv_step(proj[:, 0], conv_buf)
            mem_t, assoc = self._assoc_step(hidden_states[:, 0], conv_t, assoc)
            mem = mem_t[:, None, :]
        else:
            mem, assoc = self._assoc_full(hidden_states, proj, initial_state=None)
            if self.cfg.conv_kernel > 1:
                conv_buf = proj[:, -(self.cfg.conv_kernel - 1):].detach()

        y = self.local_gain * local + self.assoc_gain * mem
        if self.training and self.cfg.dropout:
            y = F.dropout(y, p=self.cfg.dropout)

        if streaming:
            self._stream_fast = sf.detach()
            self._stream_mid = sm.detach()
            self._stream_slow = ss.detach()
            self._stream_assoc = assoc.detach()
            self._stream_conv = conv_buf.detach()

        return y.to(hidden_states.dtype)
