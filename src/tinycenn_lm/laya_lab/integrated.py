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

        # Bidirectional linear attention:
        # phi(Q) [phi(K)^T V] / phi(Q) [phi(K)^T 1]
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, vf)
        normalizer = kf.sum(dim=2)
        numerator = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        denominator = torch.einsum(
            "bhtf,bhf->bht",
            qf,
            normalizer,
        ).unsqueeze(-1)
        global_out = numerator / denominator.clamp_min(1e-6)

        local_out = _depthwise_sequence_conv(
            v.float(),
            self.local_weight,
            causal=False,
        ) * mask
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
            feature_dim=self.feature_dim,
            effective_feature_dim=2 * self.feature_dim,
            local_kernel=self.local_kernel,
            feature_map="learned_softmax_dim_fullspace",
        )
        return d
