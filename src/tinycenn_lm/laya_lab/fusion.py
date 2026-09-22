from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .core import BaseLayaReplacementAttention, _orthogonal_maps, _identity_conv_weight, _valid_tokens

class MemoryFusionAttention(BaseLayaReplacementAttention):
    """Bidirectional Laya MemoryFusion: multiscale CeNN + linear global + slot memory."""

    architecture = "memory_fusion"

    def __init__(self, original: nn.Module, feature_dim: int = 64, memory_rank: int = 32,
                 local_kernel: int = 5):
        super().__init__(original)
        self.feature_dim = int(feature_dim)
        self.memory_rank = int(memory_rank)
        self.local_kernel = int(local_kernel)
        self.wq = nn.Parameter(_orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim))
        self.wk = nn.Parameter(_orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim))
        self.slot_q = nn.Parameter(_orthogonal_maps(self.num_heads, self.memory_rank, self.head_dim))
        self.slot_k = nn.Parameter(_orthogonal_maps(self.num_heads, self.memory_rank, self.head_dim))
        self.dilations = (1, 2, 4, 8)
        self.local_weights = nn.ParameterList([
            nn.Parameter(_identity_conv_weight(self.num_heads, self.head_dim, self.local_kernel, causal=False))
            for _ in self.dilations
        ])
        self.local_mix = nn.Parameter(torch.zeros(self.num_heads, len(self.dilations)))
        self.fuse = nn.Linear(self.head_dim, 3, bias=True)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads))

    def _linear_global(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        qf = F.elu(torch.einsum("bhtd,hfd->bhtf", q.float(), self.wq.float())) + 1.0
        kf = F.elu(torch.einsum("bhtd,hfd->bhtf", k.float(), self.wk.float())) + 1.0
        kf = kf * mask
        vf = v.float() * mask
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, vf)
        z = kf.sum(dim=2)
        num = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        den = torch.einsum("bhtf,bhf->bht", qf, z).unsqueeze(-1).clamp_min(1e-4)
        return num / den

    def _multiscale_local(self, v: Tensor):
        outs = []
        b, h, t, d = v.shape
        base = v.float().permute(0, 1, 3, 2).reshape(b, h * d, t)
        for dilation, weight in zip(self.dilations, self.local_weights):
            effective = dilation * (self.local_kernel - 1) + 1
            left = (effective - 1) // 2
            right = effective - 1 - left
            y = F.conv1d(F.pad(base, (left, right)), weight, dilation=dilation, groups=h * d)
            outs.append(y.reshape(b, h, d, t).permute(0, 1, 3, 2))
        alpha = torch.softmax(self.local_mix.float(), -1)
        out = 0.0
        for i, y in enumerate(outs):
            out = out + alpha[:, i][None, :, None, None] * y
        return out

    def _slot_memory(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        qs = torch.einsum("bhtd,hrd->bhtr", q.float(), self.slot_q.float())
        ks = torch.einsum("bhtd,hrd->bhtr", k.float(), self.slot_k.float())
        q_weight = torch.softmax(qs, dim=-1)
        ks = ks.masked_fill(mask == 0, torch.finfo(ks.dtype).min)
        k_weight = torch.softmax(ks, dim=2) * mask
        slots = torch.einsum("bhtr,bhtd->bhrd", k_weight, v.float())
        return torch.einsum("bhtr,bhrd->bhtd", q_weight, slots)

    def forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None, **kwargs):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].float()
        global_out = self._linear_global(q, k, v, mask)
        local_out = self._multiscale_local(v) * mask
        slot_out = self._slot_memory(q, k, v, mask) * mask
        gate = torch.softmax(self.fuse(q.float()), dim=-1)
        out = gate[..., 0:1] * local_out + gate[..., 1:2] * global_out + gate[..., 2:3] * slot_out
        out = out * self.output_gain.clamp(-2, 2).exp()[None, :, None, None] * mask
        return self.finish(out, hidden_states)

    def config_dict(self):
        d = super().config_dict()
        d.update(feature_dim=self.feature_dim, memory_rank=self.memory_rank,
                 local_kernel=self.local_kernel, dilations=list(self.dilations))
        return d
