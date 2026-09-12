from types import SimpleNamespace

import torch
from torch import nn

from tinycenn_lm.smollm2_amcenn_v2 import (
    AMCeNNAttentionV2,
    AdaptivePositiveSoftmaxFeatures,
    SmolAMCeNNV2Config,
)


class DummyAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int):
        super().__init__()
        head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)


def model_cfg(hidden=16, inner=32, heads=4, kv_heads=2):
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inner,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
    )


def test_antithetic_features_are_paired_and_delta_starts_zero():
    features = AdaptivePositiveSoftmaxFeatures(
        head_dim=8,
        feature_dim=16,
        seed=9,
        antithetic=True,
        learnable_correction=True,
    )
    torch.testing.assert_close(features.base_projection[:8], -features.base_projection[8:])
    assert features.delta_projection is not None
    assert torch.count_nonzero(features.delta_projection) == 0
    torch.testing.assert_close(features.projection, features.base_projection)


def test_adaptive_features_remain_positive_and_receive_gradient():
    torch.manual_seed(5)
    features = AdaptivePositiveSoftmaxFeatures(
        head_dim=8,
        feature_dim=32,
        seed=11,
        antithetic=True,
        learnable_correction=True,
    )
    x = torch.randn(2, 3, 4, 8)
    out = features(x)
    assert out.shape == (2, 3, 4, 32)
    assert torch.isfinite(out).all()
    assert (out > 0).all()
    out.mean().backward()
    assert features.delta_projection.grad is not None
    assert torch.isfinite(features.delta_projection.grad).all()


def test_amcenn_v2_is_causal():
    torch.manual_seed(17)
    cfg = model_cfg()
    original = DummyAttention(16, 4, 2)
    module = AMCeNNAttentionV2(
        original,
        cfg,
        SmolAMCeNNV2Config(feature_dim=32, num_shards=8, top_k=2),
        layer_idx=3,
    )
    x = torch.randn(1, 10, 16)
    y1, weights1 = module(x, use_cache=False)
    changed = x.clone()
    changed[:, 7:] += 20 * torch.randn_like(changed[:, 7:])
    y2, weights2 = module(changed, use_cache=False)
    torch.testing.assert_close(y1[:, :7], y2[:, :7], rtol=1e-5, atol=1e-6)
    assert weights1 is None and weights2 is None


def test_v2_config_rejects_odd_antithetic_feature_count():
    cfg = SmolAMCeNNV2Config(feature_dim=31, antithetic_features=True)
    try:
        cfg.validate(model_cfg())
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("odd antithetic feature count should fail")
