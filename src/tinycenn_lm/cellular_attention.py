"""Causal 1-D Cellular Attention for TinyCeNN-LM research experiments.

The layer keeps attention sparse and local at each cellular step, but changes
the neighborhood across steps. Power-of-two dilations create an exponentially
growing receptive field without constructing a T-by-T attention matrix.

Advanced variants add token/head-specific routing, old-school max/mean pooling,
gated RMS normalization, and a strictly-causal two-level encoder-decoder path.
The U-AMP path uses only current/past states during downsampling and nearest-left
upsampling, so no future token can enter an earlier output.

This is a research reference implementation. It favors clarity and auditable
causality over fused-kernel speed.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


VARIANTS = (
    "cellular_local3",
    "cellular_dilated3",
    "cellular_dilated5",
    "cellular_multiscale5",
    "cellular_shifted8",
    "cellular_adaptive_multiscale5",
    "cellular_multiscale5_maxpool",
    "cellular_adaptive_maxpool5",
    "cellular_adaptive_maxpool5_rms",
    "cellular_adaptive_mixedpool5_rms",
    "cellular_uamp5",
    "cellular_uamp5_channelgate",
    "cellular_uamp5_varlatent",
)

_ADAPTIVE_VARIANTS = {
    "cellular_adaptive_multiscale5",
    "cellular_adaptive_maxpool5",
    "cellular_adaptive_maxpool5_rms",
    "cellular_adaptive_mixedpool5_rms",
    "cellular_uamp5",
    "cellular_uamp5_channelgate",
    "cellular_uamp5_varlatent",
}
_MAXPOOL_VARIANTS = {
    "cellular_multiscale5_maxpool",
    "cellular_adaptive_maxpool5",
    "cellular_adaptive_maxpool5_rms",
}
_MIXEDPOOL_VARIANTS = {
    "cellular_adaptive_mixedpool5_rms",
    "cellular_uamp5",
    "cellular_uamp5_channelgate",
    "cellular_uamp5_varlatent",
}
_RMS_VARIANTS = {
    "cellular_adaptive_maxpool5_rms",
    "cellular_adaptive_mixedpool5_rms",
    "cellular_uamp5",
    "cellular_uamp5_channelgate",
    "cellular_uamp5_varlatent",
}
_UNET_VARIANTS = {
    "cellular_uamp5",
    "cellular_uamp5_channelgate",
    "cellular_uamp5_varlatent",
}
_CHANNEL_GATE_VARIANTS = {"cellular_uamp5_channelgate"}
_VARLATENT_VARIANTS = {"cellular_uamp5_varlatent"}
_MULTISCALE_VARIANTS = {
    "cellular_multiscale5",
    "cellular_adaptive_multiscale5",
    "cellular_multiscale5_maxpool",
    "cellular_adaptive_maxpool5",
    "cellular_adaptive_maxpool5_rms",
    "cellular_adaptive_mixedpool5_rms",
    "cellular_uamp5",
    "cellular_uamp5_channelgate",
    "cellular_uamp5_varlatent",
}


class CellularAttentionLayer(nn.Module):
    """Sparse causal attention with optional U-AMP multi-resolution refinement."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        feature_dim: int = 64,
        variant: str = "cellular_dilated3",
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        shifted_window: int = 8,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if min(num_heads, num_kv_heads, head_dim, feature_dim, shifted_window) < 1:
            raise ValueError("all dimensions must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        dilations = tuple(int(d) for d in dilations)
        if not dilations or min(dilations) < 1:
            raise ValueError("dilations must contain positive integers")

        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.groups = self.num_heads // self.num_kv_heads
        self.variant = variant
        self.dilations = dilations
        self.shifted_window = int(shifted_window)
        self.latent_dim = max(8, self.head_dim // 2)
        self._last_aux_loss: Tensor | None = None

        q_base = torch.zeros(self.num_heads, self.feature_dim, self.head_dim)
        k_base = torch.zeros(self.num_kv_heads, self.feature_dim, self.head_dim)
        for h in range(self.num_heads):
            nn.init.orthogonal_(q_base[h])
        for h in range(self.num_kv_heads):
            nn.init.orthogonal_(k_base[h])
        self.wq = nn.Parameter(q_base)
        self.wk = nn.Parameter(k_base)

        self.state_q = nn.Parameter(torch.zeros(
            self.num_heads, self.feature_dim, self.head_dim
        ))
        self.state_q_gate = nn.Parameter(torch.full(
            (len(self.dilations), self.num_heads), -2.0
        ))

        max_neighbors = max(
            self._max_neighbors_for_variant(variant), self.shifted_window
        )
        self.relative_bias = nn.Parameter(torch.zeros(
            len(self.dilations), self.num_heads, max_neighbors
        ))
        self.log_temperature = nn.Parameter(torch.zeros(
            len(self.dilations), self.num_heads
        ))
        self.step_gate = nn.Parameter(torch.zeros(
            len(self.dilations), self.num_heads
        ))

        if self.uses_adaptive_routing():
            self.route_key = nn.Parameter(torch.empty(
                len(self.dilations), self.num_heads, max_neighbors, self.feature_dim
            ))
            nn.init.normal_(self.route_key, mean=0.0, std=0.02)
            self.route_prior = nn.Parameter(torch.zeros(
                len(self.dilations), self.num_heads, max_neighbors
            ))
            self.route_strength = nn.Parameter(torch.zeros(
                len(self.dilations), self.num_heads
            ))
        else:
            self.register_parameter("route_key", None)
            self.register_parameter("route_prior", None)
            self.register_parameter("route_strength", None)

        if self.uses_maxpool_branch():
            self.pool_mix_logit = nn.Parameter(torch.full(
                (len(self.dilations), self.num_heads), -1.5
            ))
            self.log_pool_gain = nn.Parameter(torch.zeros(
                len(self.dilations), self.num_heads
            ))
        else:
            self.register_parameter("pool_mix_logit", None)
            self.register_parameter("log_pool_gain", None)

        if self.uses_mixedpool_branch():
            # [attention, max, mean], initialized to preserve the attention path.
            initial = torch.tensor([2.0, -1.0, -1.0]).view(1, 1, 3)
            self.mixed_pool_logits = nn.Parameter(
                initial.expand(len(self.dilations), self.num_heads, 3).clone()
            )
            self.log_mixed_pool_gain = nn.Parameter(torch.zeros(
                len(self.dilations), self.num_heads, 2
            ))
        else:
            self.register_parameter("mixed_pool_logits", None)
            self.register_parameter("log_mixed_pool_gain", None)

        if self.uses_rms_refinement():
            self.pre_rms_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
            self.post_rms_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
            self.pre_rms_gate = nn.Parameter(torch.full((self.num_heads,), -2.0))
            self.post_rms_gate = nn.Parameter(torch.full((self.num_heads,), -2.0))
        else:
            self.register_parameter("pre_rms_weight", None)
            self.register_parameter("post_rms_weight", None)
            self.register_parameter("pre_rms_gate", None)
            self.register_parameter("post_rms_gate", None)

        if self.uses_unet_refinement():
            # Causal two-level U-Net. Downsampling endpoints are 0,2,4,...;
            # nearest-left upsampling never uses a future endpoint.
            self.unet_pool_logits = nn.Parameter(torch.zeros(2, self.num_heads, 2))
            self.unet_encoder = nn.Parameter(torch.empty(
                self.num_heads, self.latent_dim, self.head_dim
            ))
            self.unet_decoder = nn.Parameter(torch.empty(
                self.num_heads, self.head_dim, self.latent_dim
            ))
            self.unet_swiglu_gate = nn.Parameter(torch.empty(
                self.num_heads, self.latent_dim, self.latent_dim
            ))
            self.unet_swiglu_value = nn.Parameter(torch.empty(
                self.num_heads, self.latent_dim, self.latent_dim
            ))
            for tensor in (
                self.unet_encoder, self.unet_decoder,
                self.unet_swiglu_gate, self.unet_swiglu_value,
            ):
                for h in range(self.num_heads):
                    nn.init.orthogonal_(tensor[h])
            self.unet_skip1_gate = nn.Parameter(torch.full((self.num_heads,), -1.5))
            self.unet_skip0_gate = nn.Parameter(torch.full((self.num_heads,), -1.5))
            self.unet_output_gate = nn.Parameter(torch.full((self.num_heads,), -2.0))
        else:
            self.register_parameter("unet_pool_logits", None)
            self.register_parameter("unet_encoder", None)
            self.register_parameter("unet_decoder", None)
            self.register_parameter("unet_swiglu_gate", None)
            self.register_parameter("unet_swiglu_value", None)
            self.register_parameter("unet_skip1_gate", None)
            self.register_parameter("unet_skip0_gate", None)
            self.register_parameter("unet_output_gate", None)

        if self.uses_channel_gate():
            rank = max(4, self.head_dim // 4)
            self.channel_down = nn.Parameter(torch.empty(
                self.num_heads, rank, self.head_dim
            ))
            self.channel_up = nn.Parameter(torch.empty(
                self.num_heads, self.head_dim, rank
            ))
            for h in range(self.num_heads):
                nn.init.orthogonal_(self.channel_down[h])
                nn.init.orthogonal_(self.channel_up[h])
            self.channel_strength = nn.Parameter(torch.full((self.num_heads,), -2.0))
        else:
            self.register_parameter("channel_down", None)
            self.register_parameter("channel_up", None)
            self.register_parameter("channel_strength", None)

        if self.uses_variational_latent():
            self.var_logvar = nn.Parameter(torch.empty(
                self.num_heads, self.latent_dim, self.head_dim
            ))
            for h in range(self.num_heads):
                nn.init.normal_(self.var_logvar[h], mean=0.0, std=0.01)
            self.var_logvar_bias = nn.Parameter(torch.full(
                (self.num_heads, self.latent_dim), -3.0
            ))
            self.var_kl_weight = 1e-4
        else:
            self.register_parameter("var_logvar", None)
            self.register_parameter("var_logvar_bias", None)
            self.var_kl_weight = 0.0

        eye = torch.eye(self.head_dim).expand(self.num_heads, -1, -1).clone()
        self.out_proj = nn.Parameter(eye)
        self.final_gate = nn.Parameter(torch.full((self.num_heads,), 2.0))
        self.log_gain = nn.Parameter(torch.zeros(self.num_heads))

    def uses_adaptive_routing(self) -> bool:
        return self.variant in _ADAPTIVE_VARIANTS

    def uses_maxpool_branch(self) -> bool:
        return self.variant in _MAXPOOL_VARIANTS

    def uses_mixedpool_branch(self) -> bool:
        return self.variant in _MIXEDPOOL_VARIANTS

    def uses_rms_refinement(self) -> bool:
        return self.variant in _RMS_VARIANTS

    def uses_unet_refinement(self) -> bool:
        return self.variant in _UNET_VARIANTS

    def uses_channel_gate(self) -> bool:
        return self.variant in _CHANNEL_GATE_VARIANTS

    def uses_variational_latent(self) -> bool:
        return self.variant in _VARLATENT_VARIANTS

    @staticmethod
    def _max_neighbors_for_variant(variant: str) -> int:
        if variant in ("cellular_local3", "cellular_dilated3"):
            return 3
        if variant in ("cellular_dilated5", *_MULTISCALE_VARIANTS):
            return 5
        if variant == "cellular_shifted8":
            return 8
        raise ValueError(variant)

    @property
    def config(self) -> dict:
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "variant": self.variant,
            "dilations": list(self.dilations),
            "shifted_window": self.shifted_window,
        }

    def _offsets(self, step: int) -> tuple[int, ...]:
        d = self.dilations[step]
        if self.variant == "cellular_local3":
            return (0, 1, 2)
        if self.variant == "cellular_dilated3":
            return (0, d, 2 * d)
        if self.variant == "cellular_dilated5":
            return (0, d, 2 * d, 3 * d, 4 * d)
        if self.variant in _MULTISCALE_VARIANTS:
            return tuple(dict.fromkeys((0, 1, d, 2 * d, 4 * d)))
        if self.variant == "cellular_shifted8":
            return tuple(range(self.shifted_window))
        raise ValueError(self.variant)

    def _valid_mask(self, length: int, step: int, device) -> tuple[Tensor, Tensor]:
        offsets = torch.tensor(self._offsets(step), device=device, dtype=torch.long)
        pos = torch.arange(length, device=device, dtype=torch.long)
        index = pos[:, None] - offsets[None, :]
        valid = index >= 0

        if self.variant == "cellular_shifted8":
            shift = 0 if step % 2 == 0 else self.shifted_window // 2
            block_start = ((pos + shift) // self.shifted_window) * self.shifted_window - shift
            valid = valid & (index >= block_start[:, None])

        return index.clamp_min(0), valid

    def _features(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        qf = F.normalize(torch.einsum("bhtd,hfd->bhtf", q, self.wq), dim=-1)
        kf = F.normalize(torch.einsum("bhtd,hfd->bhtf", k, self.wk), dim=-1)
        return qf, kf.repeat_interleave(self.groups, dim=1)

    @staticmethod
    def _rms(x: Tensor, weight: Tensor) -> Tensor:
        normed = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
        return normed * weight[None, :, None, :]

    def _adaptive_route_log_prior(
        self, query: Tensor, valid: Tensor, step: int, width: int,
    ) -> Tensor:
        assert self.route_key is not None
        assert self.route_prior is not None
        assert self.route_strength is not None
        prototypes = self.route_key[step, :, :width, :]
        route_logits = torch.einsum("bhtf,hwf->bhtw", query, prototypes)
        route_logits = route_logits / math.sqrt(self.feature_dim)
        route_logits = route_logits + self.route_prior[step, :, :width][None, :, None, :]
        route_logits = route_logits.masked_fill(
            ~valid[None, None, :, :], float("-inf")
        )
        route_log_prob = route_logits.log_softmax(dim=-1)
        route_log_prob = route_log_prob.masked_fill(
            ~valid[None, None, :, :], 0.0
        )
        strength = F.softplus(self.route_strength[step])[None, :, None, None]
        return route_log_prob * strength

    @staticmethod
    def _pool_messages(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        mask = valid[None, None, :, :, None]
        maximum = values.masked_fill(~mask, float("-inf")).amax(dim=-2)
        count = mask.sum(dim=-2).clamp_min(1).to(values.dtype)
        mean = values.masked_fill(~mask, 0.0).sum(dim=-2) / count
        return maximum, mean

    def _cellular_step(
        self, qf: Tensor, kf: Tensor, state: Tensor, step: int,
    ) -> Tensor:
        length = state.shape[2]
        index, valid = self._valid_mask(length, step, state.device)
        keys = kf[:, :, index, :]
        values = state[:, :, index, :]

        dynamic_q = torch.einsum("bhtd,hfd->bhtf", state, self.state_q)
        mix = self.state_q_gate[step].sigmoid()[None, :, None, None]
        query = F.normalize(qf + mix * dynamic_q, dim=-1)

        scores = torch.einsum("bhtf,bhtwf->bhtw", query, keys)
        scores = scores / math.sqrt(self.feature_dim)
        scores = scores * self.log_temperature[step].clamp(-3, 3).exp()[None, :, None, None]
        width = index.shape[1]
        scores = scores + self.relative_bias[step, :, :width][None, :, None, :]
        scores = scores.masked_fill(~valid[None, None, :, :], float("-inf"))

        if self.uses_adaptive_routing():
            scores = scores + self._adaptive_route_log_prior(query, valid, step, width)

        weights = scores.softmax(dim=-1)
        attention_message = torch.einsum("bhtw,bhtwd->bhtd", weights, values)
        message = attention_message

        if self.uses_mixedpool_branch():
            assert self.mixed_pool_logits is not None
            assert self.log_mixed_pool_gain is not None
            maximum, mean = self._pool_messages(values, valid)
            gains = self.log_mixed_pool_gain[step].clamp(-3, 3).exp()
            maximum = maximum * gains[:, 0][None, :, None, None]
            mean = mean * gains[:, 1][None, :, None, None]
            mixture = self.mixed_pool_logits[step].softmax(dim=-1)
            message = (
                mixture[:, 0][None, :, None, None] * attention_message
                + mixture[:, 1][None, :, None, None] * maximum
                + mixture[:, 2][None, :, None, None] * mean
            )
        elif self.uses_maxpool_branch():
            assert self.pool_mix_logit is not None
            assert self.log_pool_gain is not None
            maximum, _ = self._pool_messages(values, valid)
            gain = self.log_pool_gain[step].clamp(-3, 3).exp()[None, :, None, None]
            pooled = maximum * gain
            pool_mix = self.pool_mix_logit[step].sigmoid()[None, :, None, None]
            message = message + pool_mix * (pooled - message)

        gate = self.step_gate[step].sigmoid()[None, :, None, None]
        return state + gate * (message - state)

    def _causal_stride2_pool(self, x: Tensor, level: int) -> Tensor:
        assert self.unet_pool_logits is not None
        length = x.shape[2]
        endpoints = torch.arange(0, length, 2, device=x.device)
        previous = (endpoints - 1).clamp_min(0)
        pair = torch.stack((x[:, :, previous, :], x[:, :, endpoints, :]), dim=-2)
        maximum = pair.amax(dim=-2)
        mean = pair.mean(dim=-2)
        mix = self.unet_pool_logits[level].softmax(dim=-1)
        return (
            mix[:, 0][None, :, None, None] * maximum
            + mix[:, 1][None, :, None, None] * mean
        )

    @staticmethod
    def _causal_upsample(x: Tensor, target_length: int) -> Tensor:
        # Reduced element j represents an endpoint <= 2*j. floor(t/2) is
        # therefore always current/past relative to target token t.
        index = torch.arange(target_length, device=x.device) // 2
        return x[:, :, index.clamp_max(x.shape[2] - 1), :]

    def _unet_refine(self, state: Tensor) -> Tensor:
        assert self.unet_encoder is not None
        assert self.unet_decoder is not None
        assert self.unet_swiglu_gate is not None
        assert self.unet_swiglu_value is not None
        assert self.unet_skip1_gate is not None
        assert self.unet_skip0_gate is not None
        assert self.unet_output_gate is not None

        e0 = state
        e1 = self._causal_stride2_pool(e0, 0)
        e2 = self._causal_stride2_pool(e1, 1)
        mu = torch.einsum("bhtd,hld->bhtl", e2, self.unet_encoder)

        if self.uses_variational_latent():
            assert self.var_logvar is not None
            assert self.var_logvar_bias is not None
            logvar = torch.einsum("bhtd,hld->bhtl", e2, self.var_logvar)
            logvar = (logvar + self.var_logvar_bias[None, :, None, :]).clamp(-8, 4)
            # Deterministic mean path keeps evaluation reproducible; KL still
            # regularizes a variational latent family during fitting.
            self._last_aux_loss = self.var_kl_weight * 0.5 * (
                mu.square() + logvar.exp() - 1.0 - logvar
            ).mean()
        else:
            self._last_aux_loss = None

        gate_part = torch.einsum("bhtl,hlm->bhtm", mu, self.unet_swiglu_gate)
        value_part = torch.einsum("bhtl,hlm->bhtm", mu, self.unet_swiglu_value)
        latent = F.silu(gate_part) * value_part
        decoded = torch.einsum("bhtl,hdl->bhtd", latent, self.unet_decoder)

        up1 = self._causal_upsample(decoded, e1.shape[2])
        g1 = self.unet_skip1_gate.sigmoid()[None, :, None, None]
        d1 = e1 + g1 * (up1 - e1)
        up0 = self._causal_upsample(d1, e0.shape[2])
        g0 = self.unet_skip0_gate.sigmoid()[None, :, None, None]
        d0 = e0 + g0 * (up0 - e0)
        gout = self.unet_output_gate.sigmoid()[None, :, None, None]
        return state + gout * (d0 - state)

    def _channel_gate(self, state: Tensor) -> Tensor:
        assert self.channel_down is not None
        assert self.channel_up is not None
        assert self.channel_strength is not None
        time = torch.arange(1, state.shape[2] + 1, device=state.device, dtype=state.dtype)
        prefix_mean = state.cumsum(dim=2) / time[None, None, :, None]
        hidden = F.silu(torch.einsum("bhtd,hrd->bhtr", prefix_mean, self.channel_down))
        logits = torch.einsum("bhtr,hdr->bhtd", hidden, self.channel_up)
        strength = self.channel_strength.sigmoid()[None, :, None, None]
        return state * (1.0 + strength * torch.tanh(logits))

    def auxiliary_loss(self) -> Tensor:
        if self._last_aux_loss is None:
            return self.wq.sum() * 0.0
        return self._last_aux_loss

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("expected Q/K/V as [batch, heads, time, dim]")
        if k.shape != v.shape:
            raise ValueError("K and V must have the same shape")
        if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError("Q/K/V batch, time and head_dim must match")
        if q.shape[1] != self.num_heads or k.shape[1] != self.num_kv_heads:
            raise ValueError("Q/K head counts do not match layer configuration")
        if q.shape[-1] != self.head_dim or q.shape[2] < 1:
            raise ValueError("head_dim mismatch or empty sequence")

        self._last_aux_loss = None
        q, k, v = (x.to(self.wq.dtype) for x in (q, k, v))
        qf, kf = self._features(q, k)
        base = v.repeat_interleave(self.groups, dim=1)
        state = base

        if self.uses_rms_refinement():
            assert self.pre_rms_weight is not None and self.pre_rms_gate is not None
            normed = self._rms(state, self.pre_rms_weight)
            gate = self.pre_rms_gate.sigmoid()[None, :, None, None]
            state = state + gate * (normed - state)

        for step in range(len(self.dilations)):
            state = self._cellular_step(qf, kf, state, step)

        if self.uses_unet_refinement():
            state = self._unet_refine(state)
        if self.uses_channel_gate():
            state = self._channel_gate(state)

        if self.uses_rms_refinement():
            assert self.post_rms_weight is not None and self.post_rms_gate is not None
            normed = self._rms(state, self.post_rms_weight)
            gate = self.post_rms_gate.sigmoid()[None, :, None, None]
            state = state + gate * (normed - state)

        projected = torch.einsum("bhtd,hde->bhte", state, self.out_proj)
        final_gate = self.final_gate.sigmoid()[None, :, None, None]
        output = base + final_gate * (projected - base)
        return output * self.log_gain.clamp(-4, 4).exp()[None, :, None, None]

    def receptive_field_tokens(self) -> int:
        reach = 0
        for step in range(len(self.dilations)):
            reach += max(self._offsets(step))
        return reach + 1

    def max_score_pairs(self, context: int) -> int:
        total = 0
        device = self.wq.device
        for step in range(len(self.dilations)):
            _, valid = self._valid_mask(context, step, device)
            total += int(valid.sum().item())
        return total

    def max_neighbors_per_step(self) -> int:
        return max(len(self._offsets(step)) for step in range(len(self.dilations)))
