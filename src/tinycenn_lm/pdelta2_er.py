"""Error-residual PDelta2 layer for closing the remaining Transformer NLL gap.

The layer keeps the strongest previous TinyCeNN direction (PDelta2 F96 + causal
Conv4) and tests three focused ingredients:

1. learnable per-feature retention initialized to a broad half-life spectrum;
2. a small secondary recurrent state trained on the teacher-minus-base residual;
3. teacher-error-directed weighting that emphasizes the tokens on which the
   replacement diverges most from exact Transformer attention.

The residual memory is deliberately small and sees the original values while the
main memory sees Conv4 values.  Its output gain starts at exactly zero, so adding
it is function preserving before training.  Persistent memory matrices are stored
as FP16 between streaming calls while curvature remains FP32.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.pdelta2_features import PDelta2Core, PDeltaState


@dataclass
class ErrorResidualState:
    base: PDeltaState
    residual: PDeltaState | None = None
    conv_tail: Tensor | None = None


def initialize_retention_spectrum(core: PDelta2Core, minimum: float = 8.0,
                                  maximum: float = 2048.0) -> None:
    """Initialize feature channels with logarithmically spaced decay half-lives."""
    if minimum <= 0 or maximum <= minimum:
        raise ValueError("retention half-life range must satisfy 0 < minimum < maximum")
    with torch.no_grad():
        half_life = torch.exp(torch.linspace(
            math.log(minimum), math.log(maximum), core.feature_dim,
            device=core.forget_b.device, dtype=core.forget_b.dtype,
        ))
        log_decay = math.log(0.5) / half_life
        probability = (-log_decay / 0.25).clamp(1e-5, 1.0 - 1e-5)
        raw = torch.logit(probability)
        core.forget_b.copy_(raw[None].expand(core.num_kv_heads, -1))
        core.forget_w.zero_()


def retention_half_lives(core: PDelta2Core) -> Tensor:
    """Return the bias-only half-life represented by each KV-head/feature channel."""
    with torch.no_grad():
        log_decay = -0.25 * core.forget_b.sigmoid()
        return math.log(0.5) / log_decay.clamp_max(-1e-7)


def teacher_error_weights(prediction: Tensor, target: Tensor, hard_fraction: float = 0.25,
                          hard_boost: float = 3.0) -> Tensor:
    """Return normalized token weights that upweight the current hardest tokens.

    Error is averaged over heads and value dimensions, leaving [batch,time].
    The top ``hard_fraction`` tokens receive ``1 + hard_boost`` weight.
    The returned weights have mean one so the loss scale stays comparable.
    """
    if not 0.0 < hard_fraction < 1.0:
        raise ValueError("hard_fraction must be in (0,1)")
    if hard_boost < 0:
        raise ValueError("hard_boost must be non-negative")
    error = (prediction.detach() - target.detach()).square().mean(dim=(-1, 1))
    threshold = torch.quantile(error, 1.0 - hard_fraction, dim=-1, keepdim=True)
    weights = 1.0 + hard_boost * (error >= threshold).to(error.dtype)
    return weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-8)


def causal_value_conv_stream(v: Tensor, weight: Tensor | None, tail: Tensor | None = None):
    """Causal depthwise value convolution with a correct streaming prefix tail."""
    if weight is None:
        return v.float(), None
    if weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError("weight must be [channels,1,kernel]")
    b, h, t, d = v.shape
    kernel = weight.shape[-1]
    channels = h * d
    if weight.shape[0] != channels:
        raise ValueError("convolution channel count does not match values")
    if tail is None:
        tail = v.new_zeros(b, h, kernel - 1, d)
    if tail.shape != (b, h, kernel - 1, d):
        raise ValueError("streaming convolution tail has an incompatible shape")
    history = torch.cat((tail.to(v.dtype), v), dim=2)
    x = history.transpose(1, 2).reshape(b, history.shape[2], channels).transpose(1, 2)
    y = F.conv1d(x.float(), weight.float(), groups=channels)
    y = y.transpose(1, 2).reshape(b, t, h, d).transpose(1, 2)
    new_tail = history[:, :, -(kernel - 1):].clone() if kernel > 1 else None
    return y, new_tail


class ErrorResidualPDelta2Layer(nn.Module):
    """Conv4 PDelta2 with retention spectrum and a compact residual-error memory."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 96, residual_dim: int = 0, chunk_size: int = 32,
                 conv_kernel: int = 4, retention_spectrum: bool = False,
                 retention_min: float = 8.0, retention_max: float = 2048.0,
                 state_dtype: str = "fp16"):
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if conv_kernel < 1 or residual_dim < 0:
            raise ValueError("conv_kernel must be positive and residual_dim non-negative")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.residual_dim = int(residual_dim)
        self.chunk_size = int(chunk_size)
        self.conv_kernel = int(conv_kernel)
        self.retention_spectrum = bool(retention_spectrum)
        self.retention_min = float(retention_min)
        self.retention_max = float(retention_max)
        self.state_dtype = state_dtype
        self.groups = self.num_heads // self.num_kv_heads

        self.base = PDelta2Core(
            self.num_heads, self.num_kv_heads, self.head_dim,
            feature_dim=self.feature_dim, chunk_size=self.chunk_size,
        )
        if self.retention_spectrum:
            initialize_retention_spectrum(self.base, self.retention_min, self.retention_max)

        if self.conv_kernel > 1:
            channels = self.num_kv_heads * self.head_dim
            kernel = torch.zeros(channels, 1, self.conv_kernel)
            kernel[:, 0, -1] = 1.0
            self.conv_weight = nn.Parameter(kernel)
        else:
            self.register_parameter("conv_weight", None)

        if self.residual_dim:
            self.residual = PDelta2Core(
                self.num_heads, self.num_kv_heads, self.head_dim,
                feature_dim=self.residual_dim, chunk_size=self.chunk_size,
            )
            if self.retention_spectrum:
                initialize_retention_spectrum(
                    self.residual,
                    max(4.0, self.retention_min / 2.0),
                    self.retention_max * 2.0,
                )
            # Zero makes the whole residual branch exactly inactive at initialization.
            self.residual_gain = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.residual = None
            self.register_parameter("residual_gain", None)

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "residual_dim": self.residual_dim,
            "chunk_size": self.chunk_size,
            "conv_kernel": self.conv_kernel,
            "retention_spectrum": self.retention_spectrum,
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
            "state_dtype": self.state_dtype,
        }

    @property
    def storage_dtype(self):
        return torch.float16 if self.state_dtype == "fp16" else torch.float32

    def _working_state(self, state: PDeltaState | None, core: PDelta2Core):
        if state is None:
            return None
        dtype = core.wq.dtype
        return PDeltaState(state.memory.to(dtype), state.curvature.to(dtype))

    def _stored_state(self, state: PDeltaState):
        return PDeltaState(
            state.memory.to(self.storage_dtype),
            state.curvature.float(),
        )

    def _run(self, q: Tensor, k: Tensor, v: Tensor, state: ErrorResidualState | None):
        tail = None if state is None else state.conv_tail
        conv_v, new_tail = causal_value_conv_stream(v.float(), self.conv_weight, tail)
        base_state = None if state is None else state.base
        base_out, new_base = self.base(
            q, k, conv_v,
            state=self._working_state(base_state, self.base),
            return_state=True,
        )

        residual_raw = None
        new_residual = None
        output = base_out
        if self.residual is not None:
            residual_state = None if state is None else state.residual
            residual_raw, residual_state_out = self.residual(
                q, k, v.float(),
                state=self._working_state(residual_state, self.residual),
                return_state=True,
            )
            gain = self.residual_gain.clamp(-1.5, 1.5)[None, :, None, None]
            output = base_out + gain * residual_raw
            new_residual = self._stored_state(residual_state_out)

        new_state = ErrorResidualState(
            base=self._stored_state(new_base),
            residual=new_residual,
            conv_tail=(None if new_tail is None else new_tail.to(self.storage_dtype)),
        )
        components = {
            "base": base_out,
            "residual_raw": residual_raw,
            "output": output,
        }
        return output, new_state, components

    def components(self, q: Tensor, k: Tensor, v: Tensor,
                   state: ErrorResidualState | None = None):
        return self._run(q, k, v, state)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: ErrorResidualState | None = None,
                return_state: bool = False, implementation: str = "chunk"):
        if implementation != "chunk":
            raise ValueError("ErrorResidualPDelta2Layer supports the chunk implementation")
        output, new_state, _ = self._run(q, k, v, state)
        return (output, new_state) if return_state else output

    def recurrent_state_bytes(self, batch_size: int = 1, context: int | None = None):
        del context
        memory_bytes = 2 if self.state_dtype == "fp16" else 4
        base_memory = self.num_kv_heads * self.feature_dim * self.head_dim * memory_bytes
        base_curvature = self.num_kv_heads * self.feature_dim * 4
        total = base_memory + base_curvature
        if self.residual_dim:
            total += self.num_kv_heads * self.residual_dim * self.head_dim * memory_bytes
            total += self.num_kv_heads * self.residual_dim * 4
        if self.conv_kernel > 1:
            total += (self.conv_kernel - 1) * self.num_kv_heads * self.head_dim * memory_bytes
        return batch_size * total

    def retention_statistics(self):
        values = retention_half_lives(self.base).float().reshape(-1)
        result = {
            "base_half_life_min": float(values.min()),
            "base_half_life_median": float(values.median()),
            "base_half_life_max": float(values.max()),
        }
        if self.residual is not None:
            rv = retention_half_lives(self.residual).float().reshape(-1)
            result.update(
                residual_half_life_min=float(rv.min()),
                residual_half_life_median=float(rv.median()),
                residual_half_life_max=float(rv.max()),
            )
        return result
