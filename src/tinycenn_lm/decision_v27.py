"""V2.7 hybrid Laya attention replacements.

Full attention uses PDelta3-GDN2-CLVR with exact bidirectional Local32.
Sliding attention uses a query-conditioned AttentionCeNN that preserves the
pretrained ModernBERT Q/K/V and output projections while replacing pairwise
sliding softmax with a contractive cellular recurrence.

The module deliberately contains no training loop. Notebooks/runners should
train against an unchanged teacher and gate every installed replacement.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .laya_lab.core import BaseLayaReplacementAttention, _valid_tokens
from .laya_lab.pdelta import PDelta3GDN2CLVRAttention


class SlidingAttentionCeNN(BaseLayaReplacementAttention):
    """Query-conditioned bidirectional CeNN replacement for sliding attention.

    The recurrent state covers the same approximate local radius as the source
    sliding attention without materializing a token-token score matrix. The
    pretrained Wqkv projection supplies Q/K/V features. A learned local key
    summary modulates how much recurrent cellular context each query consumes,
    while a second query-dependent gate retains a strong direct-value path.

    Complexity is O(T * D * K) for fixed local kernel width K and memory remains
    linear in sequence length. The cellular feedback template is normalized to
    an L1 norm below one, so the recurrent update remains contractive.
    """

    architecture = "sliding_attention_cenn_v27"

    def __init__(
        self,
        original: nn.Module,
        window: int = 128,
        steps: int = 2,
        feedback_strength: float = 0.90,
    ):
        super().__init__(original)
        if steps < 1:
            raise ValueError("steps must be positive")
        if window < 2 * steps or window % (2 * steps):
            raise ValueError("window must be a positive multiple of 2*steps")
        if not 0.0 < feedback_strength < 1.0:
            raise ValueError("feedback_strength must be in (0, 1)")

        self.window = int(window)
        self.steps = int(steps)
        self.feedback_strength = float(feedback_strength)
        self.cell_radius = self.window // (2 * self.steps)
        self.gate_radius = self.window // 2
        channels = self.num_heads * self.head_dim

        cell_width = 2 * self.cell_radius + 1
        self.input_template = nn.Parameter(
            torch.full((channels, 1, cell_width), 1.0 / cell_width)
        )
        self.feedback_template = nn.Parameter(
            torch.full((channels, 1, cell_width), 0.25 / cell_width)
        )
        self.input_gain = nn.Parameter(torch.ones(channels))
        self.leak_logit = nn.Parameter(torch.zeros(channels))
        self.cell_bias = nn.Parameter(torch.zeros(channels))

        gate_width = 2 * self.gate_radius + 1
        self.key_template = nn.Parameter(
            torch.full((channels, 1, gate_width), 1.0 / gate_width)
        )
        self.qk_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.qk_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.direct_gate_w = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim)
        )
        self.direct_gate_b = nn.Parameter(
            torch.full((self.num_heads,), -0.50)
        )
        self.output_log_gain = nn.Parameter(torch.zeros(self.num_heads))

        for p in self.Wo.parameters():
            p.requires_grad = True

    @staticmethod
    def _flatten_heads(x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.permute(0, 1, 3, 2).reshape(b, h * d, t)

    def _unflatten_heads(self, x: Tensor, batch: int, tokens: int) -> Tensor:
        return x.reshape(
            batch, self.num_heads, self.head_dim, tokens
        ).permute(0, 1, 3, 2)

    @staticmethod
    def _normalized_template(weight: Tensor, max_l1: float | None = None) -> Tensor:
        w = weight.float()
        if max_l1 is not None:
            w = w.tanh()
            denom = w.abs().sum(-1, keepdim=True).clamp_min(1.0)
            return max_l1 * w / denom
        return w / w.abs().sum(-1, keepdim=True).clamp_min(1.0)

    def core_parameters(self) -> list[nn.Parameter]:
        blocked = (
            {id(p) for p in self.Wqkv.parameters()}
            | {id(p) for p in self.Wo.parameters()}
            | {id(p) for p in self.out_drop.parameters()}
        )
        return [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in blocked
        ]

    def output_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.Wo.parameters() if p.requires_grad]

    def trainable_core_parameters(self) -> list[nn.Parameter]:
        return self.core_parameters()

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings=None,
        attention_mask=None,
        **kwargs: Any,
    ):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        b, _, t, _ = q.shape
        token_mask = valid[:, None, :].to(q.dtype)
        head_mask = valid[:, None, :, None].to(q.dtype)

        q = q * head_mask
        k = k * head_mask
        v = v * head_mask

        kf = self._flatten_heads(k)
        vf = self._flatten_heads(v)

        input_kernel = self._normalized_template(self.input_template)
        feedback_kernel = self._normalized_template(
            self.feedback_template,
            self.feedback_strength,
        )
        key_kernel = self._normalized_template(self.key_template)

        drive = F.conv1d(
            vf,
            input_kernel.to(vf.dtype),
            padding=self.cell_radius,
            groups=vf.shape[1],
        ).float()
        drive = (
            drive * self.input_gain.float()[None, :, None]
            + self.cell_bias.float()[None, :, None]
        ) * token_mask.float()

        eta = self.leak_logit.float().sigmoid()[None, :, None]
        state = torch.zeros_like(drive)
        for _ in range(self.steps):
            feedback = F.conv1d(
                state.to(vf.dtype),
                feedback_kernel.to(vf.dtype),
                padding=self.cell_radius,
                groups=vf.shape[1],
            ).float()
            state = (
                (1.0 - eta) * state
                + eta * torch.tanh(feedback + drive)
            ) * token_mask.float()

        key_summary = F.conv1d(
            kf,
            key_kernel.to(kf.dtype),
            padding=self.gate_radius,
            groups=kf.shape[1],
        )
        key_summary = self._unflatten_heads(key_summary, b, t)
        state_h = self._unflatten_heads(state.to(v.dtype), b, t)
        drive_h = self._unflatten_heads(drive.to(v.dtype), b, t)

        qn = F.normalize(q.float(), dim=-1)
        kn = F.normalize(key_summary.float(), dim=-1)
        qk = (qn * kn).sum(-1) / math.sqrt(max(1, self.head_dim))
        qk_scale = self.qk_log_scale.float().clamp(-2.0, 2.0).exp()
        context_gate = torch.sigmoid(
            qk * qk_scale[None, :, None]
            + self.qk_bias.float()[None, :, None]
        )[..., None]
        cellular = context_gate * state_h.float() + (1.0 - context_gate) * drive_h.float()

        direct_gate = torch.sigmoid(
            torch.einsum(
                "bhtd,hd->bht",
                qn,
                self.direct_gate_w.float(),
            )
            + self.direct_gate_b.float()[None, :, None]
        )[..., None]
        out = direct_gate * v.float() + (1.0 - direct_gate) * cellular

        gain = self.output_log_gain.float().clamp(-2.0, 2.0).exp()
        out = out * gain[None, :, None, None] * head_mask.float()
        return self.finish(out.to(hidden_states.dtype), hidden_states)

    def config_dict(self) -> dict[str, Any]:
        d = super().config_dict()
        d.update(
            architecture=self.architecture,
            window=self.window,
            steps=self.steps,
            cell_radius=self.cell_radius,
            gate_radius=self.gate_radius,
            feedback_strength=self.feedback_strength,
            query_key_conditioning=True,
            exact_pairwise_attention=False,
            train_output_projection=True,
            complexity="O(T*D*K)",
        )
        return d


def make_v27_replacement(
    original: nn.Module,
    attention_type: str,
    *,
    full_feature_dim: int = 96,
    full_local_window: int = 32,
    full_chunk_size: int = 32,
    sliding_window: int = 128,
    sliding_steps: int = 2,
) -> nn.Module:
    """Build the V2.7 replacement selected for one ModernBERT attention type."""
    kind = str(attention_type)
    if kind == "full_attention":
        replacement = PDelta3GDN2CLVRAttention(
            original,
            feature_dim=int(full_feature_dim),
            conv_kernel=4,
            chunk_size=int(full_chunk_size),
            local_kernel=5,
            local_window=int(full_local_window),
            local_gate_init=0.72,
        )
        for p in replacement.Wo.parameters():
            p.requires_grad = True
        return replacement
    if kind == "sliding_attention":
        return SlidingAttentionCeNN(
            original,
            window=int(sliding_window),
            steps=int(sliding_steps),
        )
    raise ValueError(f"Unsupported attention type: {kind!r}")


def v27_replacement_kind(module: nn.Module) -> str | None:
    if isinstance(module, PDelta3GDN2CLVRAttention):
        return "pdelta3_local32"
    if isinstance(module, SlidingAttentionCeNN):
        return "attention_cenn"
    return None
