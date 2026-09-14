"""Frontier-inspired PDelta3 research layers.

Independent TinyCeNN adaptations for a controlled frozen-attention replacement lab:

* ``conv4_pdelta_f96`` keeps the proven Conv4 + PDelta2 recurrence;
* ``conv4_channel_decay_f96`` replaces the homogeneous forget path with a
  KDA/GDN2-style content-dependent channel decay;
* ``conv4_gdn2_f96`` uses the Gated DeltaNet-2 update structure with independent
  key-channel erase and value-channel write gates;
* ``conv4_gdn2_clvr_f96`` additionally routes the previous layer's aligned value
  representation into the current write target (a direct one-hop CLVR test).

The implementation is written from the published recurrence rather than copied
from any external kernel. It is a PyTorch research reference, not a claim of
kernel-level reproduction or speed parity with fused frontier implementations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.pdelta2_er import causal_value_conv_stream
from tinycenn_lm.pdelta2_features import PDelta2Core, PDeltaState
from tinycenn_lm.research_layers import delta_recurrence


VARIANTS = (
    "conv4_pdelta_f96",
    "conv4_channel_decay_f96",
    "conv4_gdn2_f96",
    "conv4_gdn2_clvr_f96",
)


@dataclass
class FrontierState:
    memory: Tensor
    curvature: Tensor | None = None
    q_tail: Tensor | None = None
    k_tail: Tensor | None = None
    v_tail: Tensor | None = None


def _inverse_softplus(x: Tensor) -> Tensor:
    return x + torch.log(-torch.expm1(-x))


def _orthogonal_maps(heads: int, feature_dim: int, head_dim: int) -> Tensor:
    base = torch.empty(heads, feature_dim, head_dim)
    for head in range(heads):
        if feature_dim == head_dim:
            base[head] = torch.eye(head_dim)
        else:
            nn.init.orthogonal_(base[head])
    return base


class FrontierPDelta3Layer(nn.Module):
    """Conv4 recurrent attention replacement with four controlled variants.

    GDN2-style variants implement, in feature space,

        S_t = (I - k_t (b_t * k_t)^T) D_t S_{t-1}
              + k_t (w_t * v_t)^T

    where ``D_t`` is channel-wise decay, ``b_t`` is key-channel erase, and
    ``w_t`` is value-channel write.  The CLVR variant adds a learned aligned
    value from the immediately preceding Transformer layer to the write target;
    it does not add another temporal memory matrix.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        feature_dim: int = 96,
        variant: str = "conv4_pdelta_f96",
        chunk_size: int = 32,
        conv_kernel: int = 4,
        state_dtype: str = "fp16",
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if min(num_heads, num_kv_heads, head_dim, feature_dim, conv_kernel) < 1:
            raise ValueError("dimensions must be positive")
        if not 1 <= chunk_size <= 32:
            raise ValueError("chunk_size must be in [1,32]")

        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.variant = variant
        self.chunk_size = int(chunk_size)
        self.conv_kernel = int(conv_kernel)
        self.state_dtype = state_dtype
        self.groups = self.num_heads // self.num_kv_heads

        self.is_pdelta = variant == "conv4_pdelta_f96"
        self.is_channel_decay = variant == "conv4_channel_decay_f96"
        self.is_gdn2 = variant in {"conv4_gdn2_f96", "conv4_gdn2_clvr_f96"}
        self.use_clvr = variant == "conv4_gdn2_clvr_f96"

        if self.is_pdelta or self.is_channel_decay:
            self.pdelta = PDelta2Core(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                feature_dim=self.feature_dim,
                chunk_size=self.chunk_size,
            )
        else:
            self.pdelta = None

        if self.is_gdn2:
            kv_map = _orthogonal_maps(self.num_kv_heads, self.feature_dim, self.head_dim)
            self.wk = nn.Parameter(kv_map)
            self.wq = nn.Parameter(
                kv_map.repeat_interleave(self.groups, dim=0).clone()
            )
            self.erase_w = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.feature_dim, self.head_dim)
            )
            self.erase_b = nn.Parameter(torch.full((self.num_kv_heads, self.feature_dim), -1.0))
            self.write_w = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.head_dim, self.head_dim)
            )
            self.write_b = nn.Parameter(torch.full((self.num_kv_heads, self.head_dim), -1.0))
            self.log_gain = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.register_parameter("wk", None)
            self.register_parameter("wq", None)
            self.register_parameter("erase_w", None)
            self.register_parameter("erase_b", None)
            self.register_parameter("write_w", None)
            self.register_parameter("write_b", None)
            self.register_parameter("log_gain", None)

        # KDA/GDN2-style channel decay.  It is used by the channel-decay and
        # GDN2 variants; the PDelta control keeps its original forget path.
        if not self.is_pdelta:
            self.decay_w = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.feature_dim, self.head_dim)
            )
            # Initial step sizes span short-to-long retention channels.
            dt = torch.exp(torch.linspace(
                math.log(0.001), math.log(0.05), self.feature_dim
            )).clamp_min(1e-4)
            self.dt_bias = nn.Parameter(
                _inverse_softplus(dt)[None].expand(self.num_kv_heads, -1).clone()
            )
            self.A_log = nn.Parameter(torch.zeros(self.num_kv_heads, self.feature_dim))
        else:
            self.register_parameter("decay_w", None)
            self.register_parameter("dt_bias", None)
            self.register_parameter("A_log", None)

        # The proven control uses Conv4 on V.  GDN2-style candidates use the
        # frontier-model Q/K/V short-convolution pattern.
        self.q_conv_weight = self._make_conv_weight(self.num_heads) if self.is_gdn2 else None
        self.k_conv_weight = self._make_conv_weight(self.num_kv_heads) if self.is_gdn2 else None
        self.v_conv_weight = self._make_conv_weight(self.num_kv_heads)
        if self.q_conv_weight is not None:
            self.q_conv_weight = nn.Parameter(self.q_conv_weight)
            self.k_conv_weight = nn.Parameter(self.k_conv_weight)
        else:
            self.register_parameter("q_conv_weight", None)
            self.register_parameter("k_conv_weight", None)
        self.v_conv_weight = nn.Parameter(self.v_conv_weight)

        if self.use_clvr:
            route = torch.eye(self.head_dim)[None].repeat(self.num_kv_heads, 1, 1)
            self.route_proj = nn.Parameter(route)
            self.route_gate_w = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.head_dim, self.head_dim)
            )
            # Nearly off at initialization; learning decides when lower-layer V helps.
            self.route_gate_b = nn.Parameter(torch.full((self.num_kv_heads, self.head_dim), -4.0))
        else:
            self.register_parameter("route_proj", None)
            self.register_parameter("route_gate_w", None)
            self.register_parameter("route_gate_b", None)

    def _make_conv_weight(self, heads: int) -> Tensor:
        channels = heads * self.head_dim
        kernel = torch.zeros(channels, 1, self.conv_kernel)
        kernel[:, 0, -1] = 1.0
        return kernel

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "variant": self.variant,
            "chunk_size": self.chunk_size,
            "conv_kernel": self.conv_kernel,
            "state_dtype": self.state_dtype,
        }

    @property
    def storage_dtype(self):
        return torch.float16 if self.state_dtype == "fp16" else torch.float32

    def _conv(self, x: Tensor, weight: Tensor | None, tail: Tensor | None):
        if weight is None:
            return x.float(), None
        return causal_value_conv_stream(x.float(), weight, tail)

    @staticmethod
    def _project(x: Tensor, weight: Tensor, bias: Tensor | None = None):
        y = torch.einsum("bhtd,hfd->bhtf", x, weight)
        return y if bias is None else y + bias[None, :, None]

    def _channel_decay(self, k: Tensor) -> Tensor:
        if self.decay_w is None:
            raise RuntimeError("channel decay is not configured for this variant")
        kn = F.normalize(k.float(), dim=-1)
        raw_dt = self._project(kn, self.decay_w, self.dt_bias)
        dt = F.softplus(raw_dt.float())
        rate = self.A_log.float().clamp(-4, 2).exp()[None, :, None, :] * dt
        # The existing bounded chunk solver remains numerically safe in this range.
        return -rate.clamp(1e-5, 0.25)

    def _working_state(self, state: FrontierState | None):
        if state is None:
            return None
        dtype = next(self.parameters()).dtype
        return FrontierState(
            memory=state.memory.to(dtype),
            curvature=None if state.curvature is None else state.curvature.to(dtype),
            q_tail=state.q_tail,
            k_tail=state.k_tail,
            v_tail=state.v_tail,
        )

    def _store_state(self, memory: Tensor, curvature: Tensor | None,
                     q_tail: Tensor | None, k_tail: Tensor | None, v_tail: Tensor | None):
        return FrontierState(
            memory=memory.to(self.storage_dtype),
            curvature=None if curvature is None else curvature.float(),
            q_tail=None if q_tail is None else q_tail.to(self.storage_dtype),
            k_tail=None if k_tail is None else k_tail.to(self.storage_dtype),
            v_tail=None if v_tail is None else v_tail.to(self.storage_dtype),
        )

    def _initial_memory(self, q: Tensor):
        return q.new_zeros(
            q.shape[0], self.num_kv_heads, self.feature_dim, self.head_dim
        )

    def _run_pdelta(self, q: Tensor, k: Tensor, v: Tensor, state: FrontierState | None,
                    channel_decay: bool):
        v_conv, v_tail = self._conv(v, self.v_conv_weight, None if state is None else state.v_tail)
        pstate = None
        if state is not None:
            pstate = PDeltaState(state.memory.to(self.pdelta.wq.dtype), state.curvature.to(self.pdelta.wq.dtype))
        if not channel_decay:
            output, new = self.pdelta(q, k, v_conv, state=pstate, return_state=True)
        else:
            qf, kf, z, erase, _ = self.pdelta.features(
                q.to(self.pdelta.wq.dtype), k.to(self.pdelta.wq.dtype), v_conv.to(self.pdelta.wq.dtype)
            )
            curvature = (
                qf.new_ones(qf.shape[0], self.num_kv_heads, self.feature_dim)
                if pstate is None else pstate.curvature
            )
            memory = self._initial_memory(qf) if pstate is None else pstate.memory
            kpre, curvature = self.pdelta.precondition_keys(kf, curvature)
            log_decay = self._channel_decay(k)
            output, memory = delta_recurrence(
                qf, kpre, z, erase, log_decay, memory,
                self.groups, self.chunk_size,
            )
            output = output * self.pdelta.log_gain.clamp(-4, 4).exp()[None, :, None, None]
            new = PDeltaState(memory, curvature)
        stored = self._store_state(new.memory, new.curvature, None, None, v_tail)
        return output, stored

    def _run_gdn2(self, q: Tensor, k: Tensor, v: Tensor, routed_v: Tensor | None,
                   state: FrontierState | None):
        q_conv, q_tail = self._conv(q, self.q_conv_weight, None if state is None else state.q_tail)
        k_conv, k_tail = self._conv(k, self.k_conv_weight, None if state is None else state.k_tail)
        v_conv, v_tail = self._conv(v, self.v_conv_weight, None if state is None else state.v_tail)

        qn, kn = F.normalize(q_conv, dim=-1), F.normalize(k_conv, dim=-1)
        qf = F.normalize(self._project(qn, self.wq), dim=-1)
        kf = F.normalize(self._project(kn, self.wk), dim=-1)
        log_decay = self._channel_decay(k_conv)

        erase_gate = self._project(kn, self.erase_w, self.erase_b).sigmoid()
        write_gate = self._project(F.normalize(v_conv, dim=-1), self.write_w, self.write_b).sigmoid()
        write_value = v_conv

        if self.use_clvr:
            if routed_v is None:
                routed_v = torch.zeros_like(v_conv)
            if routed_v.shape != v_conv.shape:
                raise ValueError("routed_v must match current V shape")
            aligned = torch.einsum("bhtd,hde->bhte", routed_v.float(), self.route_proj.float())
            route_gate = self._project(kn, self.route_gate_w, self.route_gate_b).sigmoid()
            write_value = write_value + route_gate * aligned

        z = write_gate * write_value
        erase = erase_gate * kf
        memory = self._initial_memory(qf) if state is None else state.memory.to(qf.dtype)
        output, memory = delta_recurrence(
            qf, kf, z, erase, log_decay, memory,
            self.groups, self.chunk_size,
        )
        output = output * self.log_gain.clamp(-4, 4).exp()[None, :, None, None]
        stored = self._store_state(memory, None, q_tail, k_tail, v_tail)
        return output, stored

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: FrontierState | None = None, return_state: bool = False,
                implementation: str = "chunk", routed_v: Tensor | None = None):
        if implementation != "chunk":
            raise ValueError("FrontierPDelta3Layer supports the bounded chunk implementation")
        if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
            raise ValueError("expected [B,H,T,D] Q/K/V")
        if self.use_clvr and routed_v is not None and routed_v.shape != v.shape:
            raise ValueError("CLVR routed value shape must match V")

        state = self._working_state(state)
        if self.is_pdelta:
            output, new_state = self._run_pdelta(q, k, v, state, channel_decay=False)
        elif self.is_channel_decay:
            output, new_state = self._run_pdelta(q, k, v, state, channel_decay=True)
        else:
            output, new_state = self._run_gdn2(q, k, v, routed_v, state)
        return (output, new_state) if return_state else output

    def recurrent_state_bytes(self, batch_size: int = 1, context: int | None = None):
        del context
        memory_bytes = 2 if self.state_dtype == "fp16" else 4
        total = self.num_kv_heads * self.feature_dim * self.head_dim * memory_bytes
        if self.is_pdelta or self.is_channel_decay:
            total += self.num_kv_heads * self.feature_dim * 4  # curvature stays fp32
            conv_heads = self.num_kv_heads
        else:
            conv_heads = self.num_heads + 2 * self.num_kv_heads
        if self.conv_kernel > 1:
            total += (self.conv_kernel - 1) * conv_heads * self.head_dim * memory_bytes
        return batch_size * total

    def decay_statistics(self):
        if self.is_pdelta:
            return {}
        with torch.no_grad():
            dt = F.softplus(self.dt_bias.float())
            rate = self.A_log.float().clamp(-4, 2).exp() * dt
            half_life = math.log(2.0) / rate.clamp_min(1e-8)
            return {
                "decay_half_life_min": float(half_life.min()),
                "decay_half_life_median": float(half_life.median()),
                "decay_half_life_max": float(half_life.max()),
            }

    def gate_statistics(self):
        result = {}
        if self.is_gdn2:
            result["erase_bias_mean"] = float(self.erase_b.detach().sigmoid().mean())
            result["write_bias_mean"] = float(self.write_b.detach().sigmoid().mean())
        if self.use_clvr:
            result["route_bias_mean"] = float(self.route_gate_b.detach().sigmoid().mean())
        return result
