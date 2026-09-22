from types import SimpleNamespace

import torch
from torch import nn

from tinycenn_lm.laya_lab import (
    IntegratedMemoryV22Attention,
    MemoryFusionAttention,
    PDelta3GDN2CLVRAttention,
)
from tinycenn_lm.laya_lab.core import LayaLabConfig
from tinycenn_lm.laya_lab.factory import choose_candidate_layers


class DummyConfig:
    hidden_size = 64
    num_attention_heads = 4


class DummyModernBertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.layer_idx = 3
        self.head_dim = 16
        self.Wqkv = nn.Linear(64, 3 * 64)
        self.Wo = nn.Linear(64, 64)
        self.out_drop = nn.Dropout(0.0)


def _run(module):
    x = torch.randn(2, 13, 64)
    mask = torch.ones(2, 13, dtype=torch.long)
    mask[1, -3:] = 0
    y, weights = module(x, attention_mask=mask)
    assert weights is None
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert module.trainable_core_parameters()


def test_integrated_memory_v22_shape_and_finite():
    module = IntegratedMemoryV22Attention(
        DummyModernBertAttention(), feature_dim=16, local_kernel=3
    )
    _run(module)

    x = torch.randn(2, 4, 7, 16)
    phi = module._project_features(x, module.wq)
    assert phi.shape == (2, 4, 7, 32)
    assert torch.isfinite(phi).all()
    assert torch.all(phi > 0)


def test_integrated_memory_v22_qk_maps_share_warm_start():
    module = IntegratedMemoryV22Attention(
        DummyModernBertAttention(), feature_dim=16, local_kernel=3
    )
    assert torch.allclose(module.wq, module.wk)
    assert module.wq.data_ptr() != module.wk.data_ptr()


def test_integrated_memory_prefers_full_attention_candidates():
    types = [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            layers=[SimpleNamespace(attention_type=t) for t in types]
        )
    )
    chosen = choose_candidate_layers(
        model, 3, preferred_attention_type="full_attention"
    )
    assert chosen == [3, 1, 5]
    assert all(types[i] == "full_attention" for i in chosen)


def test_integrated_memory_balanced_mode_has_real_transfer_budget():
    cfg = LayaLabConfig(architecture="integrated_memory_v22", mode="balanced")
    settings = cfg.mode_settings()
    assert settings["steps"] >= 600
    assert settings["train_cases"] >= 400
    assert settings["train_max_len"] >= 512


def test_memory_fusion_shape_and_finite():
    _run(
        MemoryFusionAttention(
            DummyModernBertAttention(),
            feature_dim=16,
            memory_rank=8,
            local_kernel=3,
        )
    )


def test_pdelta3_gdn2_clvr_shape_and_finite():
    _run(
        PDelta3GDN2CLVRAttention(
            DummyModernBertAttention(),
            feature_dim=16,
            conv_kernel=4,
            chunk_size=8,
        )
    )
