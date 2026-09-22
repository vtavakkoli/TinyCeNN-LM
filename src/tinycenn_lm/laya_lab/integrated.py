from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .core import BaseLayaReplacementAttention, _orthogonal_maps, _identity_conv_weight, _valid_tokens, _depthwise_sequence_conv

class IntegratedMemoryV22Attention(BaseLayaReplacementAttention):
    """Bidirectional encoder adaptation of TinyCeNN Integrated Memory V2.2."""

    architecture = "integrated_memory_v22"

    def __init__(self, original: nn.Module, feature_dim: int = 64, local_kernel: int = 5):
        super().__init__(original)
        self.feature_dim = int(feature_dim)
        self.local_kernel = int(local_kernel)
        self.wq = nn.Parameter(_orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim))
        self.wk = nn.Parameter(_orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim))
        self.local_weight = nn.Parameter(_identity_conv_weight(
            self.num_heads, self.head_dim, self.local_kernel, causal=False
        ))
        self.mix_logits = nn.Parameter(torch.tensor([0.5, 0.0, -0.5]).repeat(self.num_heads, 1))
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads))

    def _project_features(self, x: Tensor, w: Tensor):
        z = torch.einsum("bhtd,hfd->bhtf", x.float(), w.float())
        return F.elu(z) + 1.0

    def forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None, **kwargs):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].float()
        qf = self._project_features(q, self.wq)
        kf = self._project_features(k, self.wk) * mask
        vf = v.float() * mask
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, vf)
        normalizer = kf.sum(dim=2)
        num = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        den = torch.einsum("bhtf,bhf->bht", qf, normalizer).unsqueeze(-1).clamp_min(1e-4)
        global_out = num / den
        local_out = _depthwise_sequence_conv(v.float(), self.local_weight, causal=False) * mask
        direct = v.float() * mask
        w = torch.softmax(self.mix_logits.float(), dim=-1)
        out = (w[:, 0][None, :, None, None] * global_out +
               w[:, 1][None, :, None, None] * local_out +
               w[:, 2][None, :, None, None] * direct)
        out = out * self.output_gain.clamp(-2, 2).exp()[None, :, None, None] * mask
        return self.finish(out, hidden_states)

    def config_dict(self):
        d = super().config_dict()
        d.update(feature_dim=self.feature_dim, local_kernel=self.local_kernel)
        return d
