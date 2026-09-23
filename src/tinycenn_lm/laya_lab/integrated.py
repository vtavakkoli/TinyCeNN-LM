from __future__ import annotations

import torch
from torch import Tensor, nn

from .core import (
    BaseLayaReplacementAttention,
    _depthwise_sequence_conv,
    _identity_conv_weight,
    _orthogonal_maps,
    _valid_tokens,
)


class IntegratedMemoryV22Attention(BaseLayaReplacementAttention):
    """Bidirectional ModernBERT attention-transfer adapter.

    The pretrained ModernBERT Wqkv/Wo projections stay frozen.  A learned
    positive feature map approximates the pretrained softmax-attention kernel,
    while CeNN-style local mixing and a direct-value route provide residual
    correction paths.
    """

    architecture = "integrated_memory_v22"

    def __init__(
        self,
        original: nn.Module,
        feature_dim: int = 64,
        local_kernel: int = 5,
    ):
        super().__init__(original)
        self.sliding_window = getattr(original, "sliding_window", None)
        if self.sliding_window is not None:
            radius = getattr(original.config, "sliding_window", None)
            if radius is None and hasattr(original.config, "local_attention"):
                radius = original.config.local_attention // 2
            self.sliding_window = int(radius if radius is not None else self.sliding_window)
            if self.sliding_window < 0:
                raise ValueError("sliding window radius must be nonnegative")
        self.feature_dim = int(feature_dim)
        self.local_kernel = int(local_kernel)

        # Q and K must begin in the same feature space.  The previous version
        # used unrelated random maps, so its initial dot-product kernel had no
        # reason to resemble the teacher's pretrained softmax kernel.
        warm_start = _orthogonal_maps(
            self.num_heads,
            self.feature_dim,
            self.head_dim,
        )
        self.wq = nn.Parameter(warm_start.clone())
        self.wk = nn.Parameter(warm_start.clone())

        # ModernBERT uses q·k / sqrt(d). Split that scale symmetrically across
        # q and k before the learned positive feature map.
        self.feature_input_scale = float(self.head_dim ** -0.25)
        self.feature_log_scale = nn.Parameter(torch.zeros(self.num_heads))

        self.local_weight = nn.Parameter(
            _identity_conv_weight(
                self.num_heads,
                self.head_dim,
                self.local_kernel,
                causal=False,
            )
        )

        # Begin global-memory dominant, while retaining non-zero gradients for
        # the local and direct residual routes.
        self.mix_logits = nn.Parameter(
            torch.tensor([2.0, -0.5, -1.5]).repeat(self.num_heads, 1)
        )
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads))

    def _project_features(self, x: Tensor, w: Tensor) -> Tensor:
        z = torch.einsum("bhtd,hfd->bhtf", x.float(), w.float())
        z = z * self.feature_input_scale

        scale = self.feature_log_scale.float().clamp(-1.5, 1.5).exp()
        z = z * scale[None, :, None, None]

        # Positive full-space feature map:
        # [softmax(z), softmax(-z)].  Unlike the previous unrelated ELU maps,
        # q/k are trained in a shared positive kernel space suitable for
        # attention transfer.
        positive = torch.softmax(z, dim=-1)
        negative = torch.softmax(-z, dim=-1)
        return torch.cat((positive, negative), dim=-1).clamp_min(1e-6)

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

        qf = self._project_features(q, self.wq)
        kf = self._project_features(k, self.wk) * mask
        vf = v.float() * mask

        if self.sliding_window is None:
            # Full-attention memory retains linear sequence complexity.
            memory = kf.transpose(-1, -2) @ vf
            numerator = qf @ memory
            denominator = (qf * kf.sum(dim=2, keepdim=True)).sum(-1, keepdim=True)
            global_out = numerator / denominator.clamp_min(1e-6)
        else:
            # Bounded query chunks: no T*T or T*F*D allocation. Respect the
            # inclusive native window and any additional supplied pair mask.
            outputs = []
            t, radius = q.shape[2], self.sliding_window
            for start in range(0, t, 64):
                end = min(start + 64, t)
                lo, hi = max(0, start - radius), min(t, end + radius)
                weights = qf[:, :, start:end] @ kf[:, :, lo:hi].transpose(-1, -2)
                qp = torch.arange(start, end, device=q.device)
                kp = torch.arange(lo, hi, device=q.device)
                allowed = ((qp[:, None] - kp[None, :]).abs() <= radius)[None, None]
                if attention_mask is not None and attention_mask.ndim == 4:
                    pair_mask = attention_mask[..., start:end, lo:hi]
                    allowed = allowed & (pair_mask if pair_mask.dtype == torch.bool else pair_mask > -1e4)
                weights = weights * allowed
                outputs.append((weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)) @ vf[:, :, lo:hi])
            global_out = torch.cat(outputs, dim=2)

        # Mask BEFORE convolution: padded values must not affect valid tokens.
        local_weight = self.local_weight
        if self.sliding_window is not None:
            offsets = torch.arange(self.local_kernel, device=v.device) - (self.local_kernel - 1) // 2
            local_weight = local_weight * (offsets.abs() <= self.sliding_window)
        local_out = _depthwise_sequence_conv(vf, local_weight, causal=False) * mask
        direct_out = v.float() * mask

        mix = torch.softmax(self.mix_logits.float(), dim=-1)
        out = (
            mix[:, 0][None, :, None, None] * global_out
            + mix[:, 1][None, :, None, None] * local_out
            + mix[:, 2][None, :, None, None] * direct_out
        )

        gain = self.output_gain.float().clamp(-2.0, 2.0).exp()
        out = out * gain[None, :, None, None] * mask
        return self.finish(out, hidden_states)

    def config_dict(self):
        d = super().config_dict()
        d.update(
            sliding_window=self.sliding_window,
            feature_dim=self.feature_dim,
            effective_feature_dim=2 * self.feature_dim,
            local_kernel=self.local_kernel,
            feature_map="learned_softmax_dim_fullspace",
        )
        return d
