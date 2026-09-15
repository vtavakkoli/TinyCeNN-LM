"""Research-only global-memory augmentations for TinyCeNN-LM.

The module keeps the proven adaptive+MaxPool Cellular Attention path as a local/
multiscale branch and adds optional global causal memories inspired by recent
efficient sequence models:

* Hedgehog: learned positive feature maps for softmax-mimicking linear attention.
* Kimi Delta Attention (KDA): fine-grained per-channel forgetting plus delta updates.
* Gated DeltaNet-2: KDA-like decay with decoupled erase and write gates.
* xLSTM/mLSTM: normalized matrix memory with gated covariance-style updates.
* Differential attention: subtracts a second learned linear-attention map.
* Memory fusion: token-wise mixture of sparse Cellular, Hedgehog and GDN2 paths.

These are deliberately small, auditable reference implementations for controlled
ablation inside this repository. They are inspired by the papers, not drop-in
copies of the authors' optimized kernels.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tinycenn_lm.cellular_attention import CellularAttentionLayer


VARIANTS = (
    "cellular_adaptive_maxpool5",
    "cellular_hedgehog_global",
    "cellular_kda_global",
    "cellular_gdn2_global",
    "cellular_xlstm_global",
    "cellular_diff_hedgehog",
    "cellular_memory_fusion",
)

_HEDGEHOG_VARIANTS = {
    "cellular_hedgehog_global",
    "cellular_diff_hedgehog",
    "cellular_memory_fusion",
}
_DIFF_VARIANTS = {"cellular_diff_hedgehog"}
_KDA_VARIANTS = {"cellular_kda_global"}
_GDN2_VARIANTS = {"cellular_gdn2_global", "cellular_memory_fusion"}
_XLSTM_VARIANTS = {"cellular_xlstm_global"}
_GLOBAL_VARIANTS = set(VARIANTS) - {"cellular_adaptive_maxpool5"}


class MemoryAugmentedCellularLayer(nn.Module):
    """Adaptive+MaxPool Cellular Attention plus an optional global causal memory."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        feature_dim: int = 32,
        variant: str = "cellular_adaptive_maxpool5",
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        shifted_window: int = 8,
        memory_rank: int = 16,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if memory_rank < 4:
            raise ValueError("memory_rank must be >= 4")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.groups = self.num_heads // self.num_kv_heads
        self.variant = variant
        self.dilations = tuple(int(x) for x in dilations)
        self.shifted_window = int(shifted_window)
        self.memory_rank = int(memory_rank)

        # Local branch: preserve the strongest result already measured in the repo.
        self.local = CellularAttentionLayer(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            feature_dim=self.feature_dim,
            variant="cellular_adaptive_maxpool5",
            dilations=self.dilations,
            shifted_window=self.shifted_window,
        )

        # Alternate branches always begin as small perturbations of the local winner.
        if self.has_global_memory() and self.variant != "cellular_memory_fusion":
            self.branch_mix_logit = nn.Parameter(torch.full((self.num_heads,), -2.0))
            self.branch_log_gain = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.register_parameter("branch_mix_logit", None)
            self.register_parameter("branch_log_gain", None)

        # Shared low-rank Q/K maps for recurrent matrix memories.
        if self.uses_kda() or self.uses_gdn2() or self.uses_xlstm():
            self.mem_q = nn.Parameter(torch.empty(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.mem_k = nn.Parameter(torch.empty(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            for h in range(self.num_heads):
                nn.init.orthogonal_(self.mem_q[h])
                nn.init.orthogonal_(self.mem_k[h])
        else:
            self.register_parameter("mem_q", None)
            self.register_parameter("mem_k", None)

        # KDA/GDN2-style fine-grained forgetting and token-wise write rate.
        if self.uses_kda() or self.uses_gdn2():
            self.decay_w = nn.Parameter(torch.zeros(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.decay_bias = nn.Parameter(torch.full(
                (self.num_heads, self.memory_rank), 4.0
            ))
            self.beta_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.beta_bias = nn.Parameter(torch.full((self.num_heads,), -1.5))
        else:
            self.register_parameter("decay_w", None)
            self.register_parameter("decay_bias", None)
            self.register_parameter("beta_w", None)
            self.register_parameter("beta_bias", None)

        # GDN2-inspired decoupled key-side erase and value-side write controls.
        if self.uses_gdn2():
            self.erase_w = nn.Parameter(torch.zeros(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.erase_bias = nn.Parameter(torch.full(
                (self.num_heads, self.memory_rank), -0.5
            ))
            self.write_scale = nn.Parameter(torch.zeros(
                self.num_heads, self.head_dim
            ))
            self.write_bias = nn.Parameter(torch.full(
                (self.num_heads, self.head_dim), -0.5
            ))
        else:
            self.register_parameter("erase_w", None)
            self.register_parameter("erase_bias", None)
            self.register_parameter("write_scale", None)
            self.register_parameter("write_bias", None)

        # mLSTM-inspired matrix-memory gates.
        if self.uses_xlstm():
            self.x_forget_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.x_forget_bias = nn.Parameter(torch.full((self.num_heads,), 3.0))
            self.x_input_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.x_input_bias = nn.Parameter(torch.full((self.num_heads,), -1.0))
        else:
            self.register_parameter("x_forget_w", None)
            self.register_parameter("x_forget_bias", None)
            self.register_parameter("x_input_w", None)
            self.register_parameter("x_input_bias", None)

        # Hedgehog-inspired trainable positive feature maps. The softmax over the
        # learned feature axis enforces positivity and can become low-entropy/spiky.
        if self.uses_hedgehog():
            self.hedge_q = nn.Parameter(torch.empty(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.hedge_k = nn.Parameter(torch.empty(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.hedge_q_bias = nn.Parameter(torch.zeros(
                self.num_heads, self.memory_rank
            ))
            self.hedge_k_bias = nn.Parameter(torch.zeros(
                self.num_heads, self.memory_rank
            ))
            self.hedge_log_sharpness = nn.Parameter(torch.zeros(self.num_heads))
            for h in range(self.num_heads):
                nn.init.orthogonal_(self.hedge_q[h])
                nn.init.orthogonal_(self.hedge_k[h])
        else:
            self.register_parameter("hedge_q", None)
            self.register_parameter("hedge_k", None)
            self.register_parameter("hedge_q_bias", None)
            self.register_parameter("hedge_k_bias", None)
            self.register_parameter("hedge_log_sharpness", None)

        if self.uses_differential():
            self.hedge2_q = nn.Parameter(torch.empty(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.hedge2_k = nn.Parameter(torch.empty(
                self.num_heads, self.memory_rank, self.head_dim
            ))
            self.hedge2_q_bias = nn.Parameter(torch.zeros(
                self.num_heads, self.memory_rank
            ))
            self.hedge2_k_bias = nn.Parameter(torch.zeros(
                self.num_heads, self.memory_rank
            ))
            self.hedge2_log_sharpness = nn.Parameter(torch.zeros(self.num_heads))
            self.diff_lambda_logit = nn.Parameter(torch.full((self.num_heads,), -1.0))
            for h in range(self.num_heads):
                nn.init.orthogonal_(self.hedge2_q[h])
                nn.init.orthogonal_(self.hedge2_k[h])
        else:
            self.register_parameter("hedge2_q", None)
            self.register_parameter("hedge2_k", None)
            self.register_parameter("hedge2_q_bias", None)
            self.register_parameter("hedge2_k_bias", None)
            self.register_parameter("hedge2_log_sharpness", None)
            self.register_parameter("diff_lambda_logit", None)

        # The fusion candidate makes the branch choice input-dependent.
        if self.variant == "cellular_memory_fusion":
            self.fusion_gate_w = nn.Parameter(torch.zeros(
                self.num_heads, 3, self.head_dim
            ))
            prior = torch.tensor([2.0, -1.0, -1.0])
            self.fusion_gate_bias = nn.Parameter(
                prior[None, :].expand(self.num_heads, -1).clone()
            )
        else:
            self.register_parameter("fusion_gate_w", None)
            self.register_parameter("fusion_gate_bias", None)

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
            "memory_rank": self.memory_rank,
        }

    def has_global_memory(self) -> bool:
        return self.variant in _GLOBAL_VARIANTS

    def uses_hedgehog(self) -> bool:
        return self.variant in _HEDGEHOG_VARIANTS

    def uses_differential(self) -> bool:
        return self.variant in _DIFF_VARIANTS

    def uses_kda(self) -> bool:
        return self.variant in _KDA_VARIANTS

    def uses_gdn2(self) -> bool:
        return self.variant in _GDN2_VARIANTS

    def uses_xlstm(self) -> bool:
        return self.variant in _XLSTM_VARIANTS

    def _repeat_kv(self, x: Tensor) -> Tensor:
        return x.repeat_interleave(self.groups, dim=1)

    @staticmethod
    def _project(x: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bhtd,hrd->bhtr", x, weight)

    def _memory_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        assert self.mem_q is not None and self.mem_k is not None
        kh = self._repeat_kv(k)
        qm = F.normalize(self._project(q, self.mem_q), dim=-1)
        km = F.normalize(self._project(kh, self.mem_k), dim=-1)
        return qm, km, kh

    def _hedgehog_features(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        log_sharpness: Tensor,
    ) -> Tensor:
        logits = self._project(x, weight) + bias[None, :, None, :]
        sharpness = log_sharpness.clamp(-1.4, 2.1).exp()[None, :, None, None]
        # sqrt(rank) keeps q.k magnitudes from vanishing as rank grows.
        return logits.mul(sharpness).softmax(dim=-1) * math.sqrt(self.memory_rank)

    def _hedgehog_linear(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        second: bool = False,
    ) -> Tensor:
        kh, vh = self._repeat_kv(k), self._repeat_kv(v)
        if second:
            assert self.hedge2_q is not None and self.hedge2_k is not None
            assert self.hedge2_q_bias is not None and self.hedge2_k_bias is not None
            assert self.hedge2_log_sharpness is not None
            qf = self._hedgehog_features(
                q, self.hedge2_q, self.hedge2_q_bias, self.hedge2_log_sharpness
            )
            kf = self._hedgehog_features(
                kh, self.hedge2_k, self.hedge2_k_bias, self.hedge2_log_sharpness
            )
        else:
            assert self.hedge_q is not None and self.hedge_k is not None
            assert self.hedge_q_bias is not None and self.hedge_k_bias is not None
            assert self.hedge_log_sharpness is not None
            qf = self._hedgehog_features(
                q, self.hedge_q, self.hedge_q_bias, self.hedge_log_sharpness
            )
            kf = self._hedgehog_features(
                kh, self.hedge_k, self.hedge_k_bias, self.hedge_log_sharpness
            )

        kv = torch.einsum("bhtr,bhtd->bhtrd", kf, vh).cumsum(dim=2)
        kz = kf.cumsum(dim=2)
        numerator = torch.einsum("bhtr,bhtrd->bhtd", qf, kv)
        denominator = torch.einsum("bhtr,bhtr->bht", qf, kz)
        return numerator / denominator.clamp_min(1e-6)[..., None]

    def _delta_memory(self, q: Tensor, k: Tensor, v: Tensor, *, gdn2: bool) -> Tensor:
        qm, km, kh = self._memory_qk(q, k)
        vh = self._repeat_kv(v)
        assert self.decay_w is not None and self.decay_bias is not None
        assert self.beta_w is not None and self.beta_bias is not None

        decay = torch.sigmoid(
            torch.einsum("bhtd,hrd->bhtr", kh, self.decay_w)
            + self.decay_bias[None, :, None, :]
        )
        beta = torch.sigmoid(
            torch.einsum("bhtd,hd->bht", kh, self.beta_w)
            + self.beta_bias[None, :, None]
        )

        if gdn2:
            assert self.erase_w is not None and self.erase_bias is not None
            assert self.write_scale is not None and self.write_bias is not None
            erase = torch.sigmoid(
                torch.einsum("bhtd,hrd->bhtr", kh, self.erase_w)
                + self.erase_bias[None, :, None, :]
            )
            write = torch.sigmoid(
                vh * self.write_scale[None, :, None, :]
                + self.write_bias[None, :, None, :]
            )
        else:
            erase = None
            write = None

        b, h, t, _ = qm.shape
        state = torch.zeros(
            b, h, self.memory_rank, self.head_dim,
            device=q.device, dtype=q.dtype,
        )
        outputs = []
        for i in range(t):
            state = state * decay[:, :, i, :, None]
            pred = torch.einsum("bhr,bhrd->bhd", km[:, :, i], state)
            error = vh[:, :, i] - pred
            key_write = km[:, :, i]
            if gdn2:
                assert erase is not None and write is not None
                key_write = key_write * erase[:, :, i]
                error = error * write[:, :, i]
            update = torch.einsum("bhr,bhd->bhrd", key_write, error)
            state = state + beta[:, :, i, None, None] * update
            outputs.append(torch.einsum("bhr,bhrd->bhd", qm[:, :, i], state))
        return torch.stack(outputs, dim=2)

    def _xlstm_memory(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        qm, km, kh = self._memory_qk(q, k)
        vh = self._repeat_kv(v)
        assert self.x_forget_w is not None and self.x_forget_bias is not None
        assert self.x_input_w is not None and self.x_input_bias is not None

        forget = torch.sigmoid(
            torch.einsum("bhtd,hd->bht", kh, self.x_forget_w)
            + self.x_forget_bias[None, :, None]
        )
        inp = torch.sigmoid(
            torch.einsum("bhtd,hd->bht", kh, self.x_input_w)
            + self.x_input_bias[None, :, None]
        )
        b, h, t, _ = qm.shape
        memory = torch.zeros(
            b, h, self.memory_rank, self.head_dim,
            device=q.device, dtype=q.dtype,
        )
        normalizer = torch.zeros(
            b, h, self.memory_rank, device=q.device, dtype=q.dtype
        )
        outputs = []
        for i in range(t):
            f = forget[:, :, i, None, None]
            ii = inp[:, :, i, None, None]
            outer = torch.einsum("bhr,bhd->bhrd", km[:, :, i], vh[:, :, i])
            memory = f * memory + ii * outer
            normalizer = (
                forget[:, :, i, None] * normalizer
                + inp[:, :, i, None] * km[:, :, i]
            )
            numerator = torch.einsum("bhr,bhrd->bhd", qm[:, :, i], memory)
            denominator = torch.einsum(
                "bhr,bhr->bh", qm[:, :, i], normalizer
            ).abs().clamp_min(1.0)
            outputs.append(numerator / denominator[..., None])
        return torch.stack(outputs, dim=2)

    def _merge(self, local: Tensor, branch: Tensor) -> Tensor:
        assert self.branch_mix_logit is not None and self.branch_log_gain is not None
        gate = self.branch_mix_logit.sigmoid()[None, :, None, None]
        gain = self.branch_log_gain.clamp(-2, 2).exp()[None, :, None, None]
        branch = branch * gain
        return local + gate * (branch - local)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("expected Q/K/V as [batch, heads, time, dim]")
        local = self.local(q, k, v)
        if self.variant == "cellular_adaptive_maxpool5":
            return local

        q = q.to(local.dtype)
        k = k.to(local.dtype)
        v = v.to(local.dtype)

        if self.variant == "cellular_hedgehog_global":
            return self._merge(local, self._hedgehog_linear(q, k, v))

        if self.variant == "cellular_diff_hedgehog":
            assert self.diff_lambda_logit is not None
            first = self._hedgehog_linear(q, k, v)
            second = self._hedgehog_linear(q, k, v, second=True)
            lam = 0.5 * self.diff_lambda_logit.sigmoid()[None, :, None, None]
            return self._merge(local, first - lam * second)

        if self.variant == "cellular_kda_global":
            return self._merge(local, self._delta_memory(q, k, v, gdn2=False))

        if self.variant == "cellular_gdn2_global":
            return self._merge(local, self._delta_memory(q, k, v, gdn2=True))

        if self.variant == "cellular_xlstm_global":
            return self._merge(local, self._xlstm_memory(q, k, v))

        if self.variant == "cellular_memory_fusion":
            assert self.fusion_gate_w is not None and self.fusion_gate_bias is not None
            hedge = self._hedgehog_linear(q, k, v)
            gdn2 = self._delta_memory(q, k, v, gdn2=True)
            logits = (
                torch.einsum("bhtd,hcd->bhtc", q, self.fusion_gate_w)
                + self.fusion_gate_bias[None, :, None, :]
            )
            weights = logits.softmax(dim=-1)
            return (
                weights[..., 0, None] * local
                + weights[..., 1, None] * hedge
                + weights[..., 2, None] * gdn2
            )

        raise ValueError(self.variant)

    def max_score_pairs(self, context: int) -> int:
        # This counts only the sparse Cellular softmax branch. Global memories
        # are O(T * memory_rank * head_dim), not pairwise T^2 score matrices.
        return self.local.max_score_pairs(context)

    def receptive_field_tokens(self) -> int:
        return self.local.receptive_field_tokens()

    def max_neighbors_per_step(self) -> int:
        return self.local.max_neighbors_per_step()
