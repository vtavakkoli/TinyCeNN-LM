from __future__ import annotations
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


class IntegratedMemoryV22Attention(BaseLayaReplacementAttention):
    """Bidirectional ModernBERT attention-transfer adapter.

    V2.2 keeps Laya's pretrained Wqkv/Wo frozen, but learns a positive feature
    map that approximates the pretrained softmax kernel.  The feature map uses
    the full-space softmax construction popularized by learned linear-attention
    conversion work: [softmax(z), softmax(-z)].  A small CeNN-style local path
    and direct-value path remain available as residual corrections.
    """

    architecture = "integrated_memory_v22"

    def __init__(self, original: nn.Module, feature_dim: int = 64, local_kernel: int = 5):
        super().__init__(original)
        self.feature_dim = int(feature_dim)
        self.local_kernel = int(local_kernel)

        # Use the SAME orthogonal warm start for q/k.  The old implementation
        # initialized unrelated q/k maps, so the initial kernel had no reason to
        # resemble the teacher's q·k softmax kernel.
        base = _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        self.wq = nn.Parameter(base.clone())
        self.wk = nn.Parameter(base.clone())

        # Split the standard 1/sqrt(d) attention scale symmetrically across q/k.
        self.feature_input_scale = float(self.head_dim ** -0.25)
        self.feature_log_scale = nn.Parameter(torch.zeros(self.num_heads))

        self.local_weight = nn.Parameter(
            _identity_conv_weight(
                self.num_heads, self.head_dim, self.local_kernel, causal=False
            )
        )

        # Start mostly on the learned global kernel, but leave non-zero gradient
        # routes through local/direct paths.
        self.mix_logits = nn.Parameter(
            torch.tensor([2.0, -0.5, -1.5]).repeat(self.num_heads, 1)
        )
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads))

    def _project_features(self, x: Tensor, w: Tensor) -> Tensor:
        z = torch.einsum("bhtd,hfd->bhtf", x.float(), w.float())
        z = z * self.feature_input_scale
        scale = self.feature_log_scale.float().clamp(-1.5, 1.5).exp()
        z = z * scale[None, :, None, None]

        # Positive, normalized learned feature map.  Concatenating the two
        # half-spaces is substantially more expressive than ELU(x)+1 while
        # retaining linear-attention factorization.
        pos = torch.softmax(z, dim=-1)
        neg = torch.softmax(-z, dim=-1)
        return torch.cat((pos, neg), dim=-1).clamp_min(1e-6)

    def forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None, **kwargs):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].float()

        qf = self._project_features(q, self.wq)
        kf = self._project_features(k, self.wk) * mask
        vf = v.float() * mask

        # Bidirectional linear attention:
        #   phi(Q) [phi(K)^T V] / phi(Q) [phi(K)^T 1]
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, vf)
        normalizer = kf.sum(dim=2)
        num = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        den = torch.einsum("bhtf,bhf->bht", qf, normalizer)
        global_out = num / den.unsqueeze(-1).clamp_min(1e-6)

        local_out = _depthwise_sequence_conv(
            v.float(), self.local_weight, causal=False
        ) * mask
        direct = v.float() * mask

        weights = torch.softmax(self.mix_logits.float(), dim=-1)
        out = (
            weights[:, 0][None, :, None, None] * global_out
            + weights[:, 1][None, :, None, None] * local_out
            + weights[:, 2][None, :, None, None] * direct
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
