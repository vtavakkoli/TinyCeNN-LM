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
    """Fast bidirectional encoder adaptation of PDelta3-GDN2-CLVR.

    Bidirectional/global context is supplied by the learned linear-attention
    branch.  GDN2 remains as a causal recurrent residual, so we avoid the old
    forward+reverse Python scan that roughly doubled sequential inference cost.
    Core math follows the surrounding autocast dtype instead of forcing FP32.
    """

    architecture = "pdelta3_gdn2_clvr"

    def __init__(
        self,
        original: nn.Module,
        feature_dim: int = 48,
        conv_kernel: int = 4,
        chunk_size: int = 32,
        local_kernel: int = 5,
        local_window: int = 0,
        local_gate_init: float = 0.72,
    ):
        super().__init__(original)
        self.feature_dim = int(feature_dim)
        self.conv_kernel = int(conv_kernel)
        self.chunk_size = int(chunk_size)
        self.local_kernel = int(local_kernel)
        self.local_window = int(local_window)
        if self.local_window < 0 or not 0 < local_gate_init < 1:
            raise ValueError("local_window must be nonnegative and local_gate_init in (0, 1)")
        if self.local_window:
            self.local_gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.local_gate_b = nn.Parameter(torch.full(
                (self.num_heads,), math.log(local_gate_init / (1 - local_gate_init))
            ))

        self.wq = nn.Parameter(
            _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        )
        self.wk = nn.Parameter(
            _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        )

        # A second learned positive feature map supplies true bidirectional
        # global token mixing in O(T) memory/time.  The recurrent GDN2 path is
        # retained, but no longer has to imitate full softmax attention alone.
        global_init = _orthogonal_maps(
            self.num_heads, self.feature_dim, self.head_dim
        )
        self.global_wq = nn.Parameter(global_init.clone())
        self.global_wk = nn.Parameter(global_init.clone())
        self.global_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.global_input_scale = float(self.head_dim ** -0.25)

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
        # [GDN2 recurrent, learned global kernel, local convolution, direct V].
        # Start global-dominant for full-attention ModernBERT layers while
        # keeping every branch active so all routes receive gradient.
        init_mix = torch.tensor([0.0, 2.0, -0.5, -1.5], dtype=torch.float32)
        self.mix_logits = nn.Parameter(init_mix[None].repeat(self.num_heads, 1))
        # Zero-initialized token-dependent correction preserves the stable
        # global-dominant warm start but lets each query choose its best route.
        self.mix_gate_w = nn.Parameter(
            torch.zeros(self.num_heads, 4, self.head_dim)
        )

        self.log_gain = nn.Parameter(torch.zeros(self.num_heads))

    def _project(self, x: Tensor, w: Tensor, bias: Tensor | None = None):
        # Let CUDA autocast choose the fast matmul dtype. Parameters stay FP32
        # master weights for AdamW, but activations are no longer forced to FP32.
        y = torch.einsum("bhtd,hfd->bhtf", x, w)
        if bias is None:
            return y
        return y + bias.to(y.dtype)[None, :, None, :]

    def _global_linear(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
    ):
        """Learned positive-feature approximation of bidirectional attention."""
        scale = self.global_log_scale.to(q.dtype).clamp(-1.5, 1.5).exp()
        qz = torch.einsum(
            "bhtd,hfd->bhtf", q, self.global_wq
        )
        kz = torch.einsum(
            "bhtd,hfd->bhtf", k, self.global_wk
        )
        qz = qz * self.global_input_scale * scale[None, :, None, None]
        kz = kz * self.global_input_scale * scale[None, :, None, None]

        # Full-space positive map gives a much closer softmax-kernel warm start
        # than unrelated signed random features.
        qf = torch.cat(
            (torch.softmax(qz, dim=-1), torch.softmax(-qz, dim=-1)),
            dim=-1,
        ).clamp_min(1e-6)
        kf = torch.cat(
            (torch.softmax(kz, dim=-1), torch.softmax(-kz, dim=-1)),
            dim=-1,
        ).clamp_min(1e-6) * mask

        vf = v * mask
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, vf)
        normalizer = kf.sum(dim=2)
        numerator = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        denominator = torch.einsum(
            "bhtf,bhf->bht", qf, normalizer
        ).unsqueeze(-1)
        return numerator / denominator.clamp_min(1e-6)

    def _scan(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        routed: Tensor,
        valid: Tensor,
    ):
        # Padding must be zero BEFORE convolution: masking only the output
        # allows arbitrary padded values to contaminate nearby valid tokens.
        mask = valid[:, None, :, None].to(q.dtype)
        q, k, v = q * mask, k * mask, v * mask
        q = _depthwise_sequence_conv(q, self.q_conv, causal=True)
        k = _depthwise_sequence_conv(k, self.k_conv, causal=True)
        v = _depthwise_sequence_conv(v, self.v_conv, causal=True)

        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        qf = F.normalize(self._project(qn, self.wq), dim=-1)
        kf = F.normalize(self._project(kn, self.wk), dim=-1)

        raw_dt = self._project(kn, self.decay_w, self.dt_bias)
        decay = torch.exp(-F.softplus(raw_dt).clamp(1e-5, 0.25))
        erase = torch.sigmoid(self._project(kn, self.erase_w, self.erase_b))
        write = torch.sigmoid(
            torch.einsum(
                "bhtd,hde->bhte",
                F.normalize(v, dim=-1),
                self.write_w,
            )
            + self.write_b.to(v.dtype)[None, :, None, :]
        )

        aligned = torch.einsum(
            "bhtd,hde->bhte", routed, self.route_proj
        )
        route_gate = torch.sigmoid(
            torch.einsum(
                "bhtd,hde->bhte", kn, self.route_gate_w
            )
            + self.route_gate_b.to(kn.dtype)[None, :, None, :]
        )
        write_value = v + route_gate * aligned

        b, h, t, d = v.shape
        state = torch.zeros(
            b,
            h,
            self.feature_dim,
            d,
            device=v.device,
            dtype=v.dtype,
        )
        outs = []
        valid4 = valid[:, None, :, None].to(v.dtype)

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

    def _local_attention(self, q, k, v, valid):
        """Exact bidirectional LocalW, with O(T W) rather than T² scores.

        A window of 32 contains self, 15 left and 16 right neighbours.
        Batch query blocks together with overlapping key/value halos so SDPA
        can execute them in one call. This is a hybrid attention replacement.
        """
        b, h, t, d = q.shape
        w = self.local_window
        left, right = (w - 1) // 2, w // 2
        blocks = (t + w - 1) // w
        tail = blocks * w - t
        qp = F.pad(q, (0, 0, 0, tail)).reshape(b, h, blocks, w, d)

        def windows(x):
            return F.pad(x, (0, 0, left, right + tail)).unfold(
                2, 2 * w - 1, w
            ).transpose(-1, -2)

        kp, vp = windows(k), windows(v)
        key_valid = F.pad(valid, (left, right + tail)).unfold(1, 2 * w - 1, w)
        qi = torch.arange(w, device=q.device)[:, None]
        ki = torch.arange(2 * w - 1, device=q.device)[None, :]
        band = (ki >= qi) & (ki < qi + w)
        allowed = band[None, None, None] & key_valid[:, None, :, None, :]
        def batch_blocks(x):
            return x.transpose(1, 2).reshape(b * blocks, h, x.shape[-2], d)
        allowed = allowed.transpose(1, 2).reshape(b * blocks, 1, w, 2 * w - 1)
        out = F.scaled_dot_product_attention(
            batch_blocks(qp), batch_blocks(kp), batch_blocks(vp),
            attn_mask=allowed, dropout_p=0.0,
        )
        out = out.reshape(b, blocks, h, w, d).transpose(1, 2)
        return out.reshape(b, h, blocks * w, d)[:, :, :t] * valid[:, None, :, None]

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings=None,
        attention_mask=None,
        **kwargs,
    ):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].to(q.dtype)

        routed = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # One causal GDN2 residual scan.  The global linear branch below is
        # already bidirectional, so a second reverse recurrence only duplicates
        # sequential work without providing unique full-context coverage.
        memory_out = self._scan(q, k, v, routed, valid)
        global_out = self._global_linear(q, k, v, mask) * mask

        local_out = _depthwise_sequence_conv(
            v * mask, self.local_weight, causal=False
        ) * mask
        direct_out = v * mask

        dynamic_mix = torch.einsum(
            "bhtd,hkd->bhtk",
            F.normalize(q, dim=-1),
            self.mix_gate_w,
        )
        mix = torch.softmax(
            self.mix_logits.to(q.dtype)[None, :, None, :] + dynamic_mix,
            dim=-1,
        )
        out = (
            mix[..., 0:1] * memory_out
            + mix[..., 1:2] * global_out
            + mix[..., 2:3] * local_out
            + mix[..., 3:4] * direct_out
        )
        if self.local_window:
            local_attention = self._local_attention(q, k, v, valid)
            gate = torch.sigmoid(
                torch.einsum("bhtd,hd->bht", q, self.local_gate_w)
                + self.local_gate_b.to(q.dtype)[None, :, None]
            ).unsqueeze(-1)
            out = gate * local_attention + (1 - gate) * out
        out = (
            out
            * self.log_gain.to(out.dtype).clamp(-2, 2).exp()[None, :, None, None]
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
            local_window=self.local_window,
            local_attention_bidirectional=bool(self.local_window),
            hybrid_attention=bool(self.local_window),
            bidirectional=True,
            recurrent_bidirectional=False,
            bidirectional_context_source="global_linear_attention",
            autocast_core=True,
            clvr_source="previous_encoder_representation",
            local_value_residual=True,
            global_linear_attention=True,
            global_feature_map="learned_softmax_dim_fullspace",
            effective_global_feature_dim=2 * self.feature_dim,
            token_dependent_fusion=True,
        )
        return d
