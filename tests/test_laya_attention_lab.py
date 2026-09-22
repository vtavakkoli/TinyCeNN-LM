import torch
from torch import nn

from tinycenn_lm.laya_lab import (
    IntegratedMemoryV22Attention,
    MemoryFusionAttention,
    PDelta3GDN2CLVRAttention,
)
from tinycenn_lm.laya_lab.core import LayaLabConfig
from tinycenn_lm.laya_lab.factory import choose_candidate_layers
from tinycenn_lm.laya_lab.train import _score_metrics


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


def test_integrated_memory_v22_shape_and_positive_feature_map():
    module = IntegratedMemoryV22Attention(
        DummyModernBertAttention(),
        feature_dim=16,
        local_kernel=3,
    )
    _run(module)

    x = torch.randn(2, 4, 7, 16)
    phi = module._project_features(x, module.wq)
    assert phi.shape == (2, 4, 7, 32)
    assert torch.isfinite(phi).all()
    assert torch.all(phi > 0)


def test_integrated_memory_v22_qk_maps_share_warm_start_but_not_storage():
    module = IntegratedMemoryV22Attention(
        DummyModernBertAttention(),
        feature_dim=16,
        local_kernel=3,
    )
    assert torch.allclose(module.wq, module.wk)
    assert module.wq.data_ptr() != module.wk.data_ptr()


def test_integrated_memory_v22_prefers_four_full_attention_layers():
    class Layer:
        def __init__(self, attention_type):
            self.attention_type = attention_type

    class Encoder:
        layers = [
            Layer("sliding_attention"),
            Layer("full_attention"),
            Layer("sliding_attention"),
            Layer("full_attention"),
            Layer("sliding_attention"),
            Layer("full_attention"),
            Layer("sliding_attention"),
            Layer("full_attention"),
            Layer("sliding_attention"),
        ]

    class Model:
        encoder = Encoder()

    chosen = choose_candidate_layers(
        Model(),
        4,
        preferred_attention_type="full_attention",
    )
    assert chosen == [3, 5, 1, 7]
    assert all(
        Model.encoder.layers[i].attention_type == "full_attention"
        for i in chosen
    )


def test_integrated_memory_v22_balanced_transfer_budget():
    cfg = LayaLabConfig(
        architecture="integrated_memory_v22",
        mode="balanced",
    )
    settings = cfg.mode_settings()
    assert settings["train_cases"] >= 400
    assert settings["steps"] >= 600
    assert settings["train_max_len"] >= 512


def test_other_laya_architectures_keep_original_balanced_budget():
    cfg = LayaLabConfig(
        architecture="memory_fusion",
        mode="balanced",
    )
    settings = cfg.mode_settings()
    assert settings["train_cases"] == 160
    assert settings["steps"] == 160
    assert settings["batch_size"] == 3


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
            local_kernel=3,
        )
    )


def test_integrated_memory_v22_cosine_heavy_transfer_objective():
    output_nmse = torch.tensor(0.24)
    output_cosine = torch.tensor(0.85)
    core_nmse = torch.tensor(0.20)
    core_cosine = torch.tensor(0.89)
    score = _score_metrics(
        output_nmse,
        output_cosine,
        core_nmse,
        core_cosine,
        "integrated_memory_v22",
    )
    expected = (
        0.75 * output_nmse
        + 1.00 * (1.0 - output_cosine)
        + 0.35 * core_nmse
        + 0.30 * (1.0 - core_cosine)
    )
    assert torch.allclose(score, expected)


def test_laya_config_records_explicit_target_layer():
    cfg = LayaLabConfig(
        architecture="integrated_memory_v22",
        mode="extended",
        target_layers=(18,),
        feature_dim=128,
        learning_rate=4e-4,
        weight_decay=1e-4,
        training_steps=2400,
    )
    assert cfg.target_layers == (18,)
    assert cfg.feature_dim == 128
    assert cfg.training_steps == 2400
