from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .core import (
    BaseLayaReplacementAttention,
    _orthogonal_maps,
    _identity_conv_weight,
    _valid_tokens,
    _depthwise_sequence_conv,
)


class PDelta3GDN2CLVRAttention(BaseLayaReplacementAttention):
    """Bidirectional encoder adaptation of PDelta3-GDN2-CLVR.

    The recurrent GDN2 memory handles long-range content while a learned
    non-attention local/direct value mixer handles the strong diagonal/local
    component found in ModernBERT attention. This avoids asking the recurrent
    state to reproduce both regimes with one path.
    """

    architecture = "pdelta3_gdn2_clvr"

    def __init__(
        self,
        original: nn.Module,
        feature_dim: int = 48,
        conv_kernel: int = 4,
        chunk_size: int = 32,
        local_kernel: int = 5,
    ):
        super().__init__(original)
        self.feature_dim = int(feature_dim)
        self.conv_kernel = int(conv_kernel)
        self.chunk_size = int(chunk_size)
        self.local_kernel = int(local_kernel)

        self.wq = nn.Parameter(
            _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        )
        self.wk = nn.Parameter(
            _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        )

        self.decay_w = nn.Parameter(
            torch.zeros(self.num_heads, self.feature_dim, self.head_dim)
        )
        dt = torch.exp(
            torch.linspace(math.log(0.001), math.log(0.05), self.feature_dim)
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(
            torch.log(torch.expm1(dt))[None].repeat(self.num_heads, 1)
        )

        self.erase_w = nn.Parameter(
            torch.zeros(self.num_heads, self.feature_dim, self.head_dim)
        )
        self.erase_b = nn.Parameter(
            torch.full((self.num_heads, self.feature_dim), -1.0)
        )
        self.write_w = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim, self.head_dim)
        )
        self.write_b = nn.Parameter(
            torch.full((self.num_heads, self.head_dim), -1.0)
        )

        self.route_proj = nn.Parameter(
            torch.eye(self.head_dim)[None].repeat(self.num_heads, 1, 1)
        )
        self.route_gate_w = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim, self.head_dim)
        )
        self.route_gate_b = nn.Parameter(
            torch.full((self.num_heads, self.head_dim), -4.0)
        )

        self.q_conv = nn.Parameter(
            _identity_conv_weight(
                self.num_heads, self.head_dim, self.conv_kernel, causal=True
            )
        )
        self.k_conv = nn.Parameter(
            _identity_conv_weight(
                self.num_heads, self.head_dim, self.conv_kernel, causal=True
            )
        )
        self.v_conv = nn.Parameter(
            _identity_conv_weight(
                self.num_heads, self.head_dim, self.conv_kernel, causal=True
            )
        )

        # Strong self/local attention is common in encoder layers. The previous
        # Laya adaptation had no local path and plateaued at poor local fidelity.
        # This branch is still non-attention: depthwise value mixing plus direct V.
        self.local_weight = nn.Parameter(
            _identity_conv_weight(
                self.num_heads, self.head_dim, self.local_kernel, causal=False
            )
        )
        init_mix = torch.tensor([0.0, 0.15, 0.35], dtype=torch.float32)
        self.mix_logits = nn.Parameter(init_mix[None].repeat(self.num_heads, 1))

        self.direction_logits = nn.Parameter(torch.zeros(self.num_heads, 2))
        self.log_gain = nn.Parameter(torch.zeros(self.num_heads))

    def _project(self, x: Tensor, w: Tensor, bias: Tensor | None = None):
        y = torch.einsum("bhtd,hfd->bhtf", x.float(), w.float())
        return y if bias is None else y + bias[None, :, None, :]

    def _scan(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        routed: Tensor,
        valid: Tensor,
    ):
        q = _depthwise_sequence_conv(q, self.q_conv, causal=True)
        k = _depthwise_sequence_conv(k, self.k_conv, causal=True)
        v = _depthwise_sequence_conv(v, self.v_conv, causal=True)

        qn = F.normalize(q.float(), dim=-1)
        kn = F.normalize(k.float(), dim=-1)
        qf = F.normalize(self._project(qn, self.wq), dim=-1)
        kf = F.normalize(self._project(kn, self.wk), dim=-1)

        raw_dt = self._project(kn, self.decay_w, self.dt_bias)
        decay = torch.exp(-F.softplus(raw_dt).clamp(1e-5, 0.25))
        erase = torch.sigmoid(self._project(kn, self.erase_w, self.erase_b))
        write = torch.sigmoid(
            torch.einsum(
                "bhtd,hde->bhte",
                F.normalize(v.float(), dim=-1),
                self.write_w.float(),
            )
            + self.write_b[None, :, None, :]
        )

        aligned = torch.einsum(
            "bhtd,hde->bhte", routed.float(), self.route_proj.float()
        )
        route_gate = torch.sigmoid(
            torch.einsum(
                "bhtd,hde->bhte", kn, self.route_gate_w.float()
            )
            + self.route_gate_b[None, :, None, :]
        )
        write_value = v.float() + route_gate * aligned

        b, h, t, d = v.shape
        state = torch.zeros(
            b,
            h,
            self.feature_dim,
            d,
            device=v.device,
            dtype=torch.float32,
        )
        outs = []
        valid4 = valid[:, None, :, None].float()

        for i in range(t):
            m = valid4[:, :, i]
            old = state
            decayed = state * decay[:, :, i].unsqueeze(-1)

            # S_t = Sbar_t + k_t (z_t - e_t^T Sbar_t)^T
            erase_vec = torch.einsum(
                "bhf,bhfd->bhd",
                erase[:, :, i] * kf[:, :, i],
                decayed,
            )
            residual = write[:, :, i] * write_value[:, :, i] - erase_vec
            updated = (
                decayed
                + kf[:, :, i].unsqueeze(-1) * residual.unsqueeze(-2)
            )

            state = updated * m.unsqueeze(-1) + old * (1.0 - m.unsqueeze(-1))
            out = torch.einsum("bhf,bhfd->bhd", qf[:, :, i], state)
            outs.append(out * m)

            # chunk_size=0 enables full BPTT for short encoder sequences.
            if (
                self.training
                and self.chunk_size > 0
                and (i + 1) % self.chunk_size == 0
            ):
                state = state.detach()

        return torch.stack(outs, dim=2)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings=None,
        attention_mask=None,
        **kwargs,
    ):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].float()

        routed = hidden_states.float().view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        fwd = self._scan(q, k, v, routed, valid)
        rev = self._scan(
            q.flip(2),
            k.flip(2),
            v.flip(2),
            routed.flip(2),
            valid.flip(1),
        ).flip(2)

        direction = torch.softmax(self.direction_logits.float(), dim=-1)
        memory_out = (
            direction[:, 0][None, :, None, None] * fwd
            + direction[:, 1][None, :, None, None] * rev
        )

        local_out = _depthwise_sequence_conv(
            v.float(), self.local_weight, causal=False
        ) * mask
        direct_out = v.float() * mask

        mix = torch.softmax(self.mix_logits.float(), dim=-1)
        out = (
            mix[:, 0][None, :, None, None] * memory_out
            + mix[:, 1][None, :, None, None] * local_out
            + mix[:, 2][None, :, None, None] * direct_out
        )
        out = (
            out
            * self.log_gain.clamp(-2, 2).exp()[None, :, None, None]
            * mask
        )
        return self.finish(out, hidden_states)

    def config_dict(self):
        d = super().config_dict()
        d.update(
            feature_dim=self.feature_dim,
            conv_kernel=self.conv_kernel,
            chunk_size=self.chunk_size,
            local_kernel=self.local_kernel,
            bidirectional=True,
            clvr_source="previous_encoder_representation",
            local_value_residual=True,
        )
        return d
