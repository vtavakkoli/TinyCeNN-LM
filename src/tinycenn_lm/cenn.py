from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CeNNConfig:
    """Configuration for the causal 1-D Cellular Neural Network adapter."""

    hidden_size: int = 192
    kernel_size: int = 3
    expansion: int = 4
    steps: int = 4
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if self.expansion <= 0:
            raise ValueError("expansion must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not self.dilations or any(d <= 0 for d in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["dilations"] = list(self.dilations)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CeNNConfig":
        data = dict(data)
        if "dilations" in data:
            data["dilations"] = tuple(data["dilations"])
        return cls(**data)


class StableRMSNorm(nn.Module):
    """Small RMSNorm with fp32 variance accumulation for training stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(dtype=dtype)
        return x * self.weight.to(dtype=dtype)


class CausalDepthwiseNeighborhood(nn.Module):
    """A causal, channel-wise cellular neighborhood operator.

    The same kernel is reused at every recurrent step. Only the dilation changes,
    which grows the receptive field while keeping the parameter count fixed.
    """

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: Tensor, dilation: int) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, seq, hidden], got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {x.shape[-1]}"
            )
        left_pad = dilation * (self.kernel_size - 1)
        y = x.transpose(1, 2)
        y = F.pad(y, (left_pad, 0))
        y = F.conv1d(
            y,
            self.weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=self.hidden_size,
        )
        return y.transpose(1, 2)


class SharedCeNNCell(nn.Module):
    """Shared recurrent CeNN cell using causal local mixing + gated SwiGLU update."""

    def __init__(self, config: CeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        inner = config.hidden_size * config.expansion

        self.norm = StableRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.neighborhood = CausalDepthwiseNeighborhood(
            config.hidden_size, config.kernel_size
        )
        self.in_proj = nn.Linear(config.hidden_size, inner * 2, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(self, state: Tensor, dilation: int, step_scale: float) -> Tensor:
        x = self.norm(state)
        local = self.neighborhood(x, dilation=dilation)
        a, b = self.in_proj(local).chunk(2, dim=-1)
        update = F.silu(a) * b
        update = self.out_proj(update)
        update = self.dropout(update)
        gate = torch.sigmoid(self.gate_proj(local))
        return state + (step_scale * gate * update)


class FastCeNNCore(nn.Module):
    """Iterates one shared CeNN cell multiple times with a dilation schedule."""

    def __init__(self, config: CeNNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.cell = SharedCeNNCell(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        initial = hidden_states
        state = hidden_states
        step_scale = self.config.steps ** -0.5
        for step in range(self.config.steps):
            dilation = self.config.dilations[step % len(self.config.dilations)]
            state = self.cell(state, dilation=dilation, step_scale=step_scale)
        return state - initial

    @property
    def receptive_field(self) -> int:
        radius = sum(
            self.config.dilations[i % len(self.config.dilations)]
            for i in range(self.config.steps)
        )
        return 1 + (self.config.kernel_size - 1) * radius
