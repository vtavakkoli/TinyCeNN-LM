from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from tinycenn_lm.smollm2_amcenn import (
    AMCeNNAttention,
    PositiveSoftmaxFeatures,
    ShardedTop2LlamaMLP,
    SmolAMCeNNConfig,
)


class DummyAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int):
        super().__init__()
        head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)


class DummyMLP(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def cfg(hidden=16, inner=32, heads=4, kv_heads=2):
    return SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inner,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
    )


def test_positive_softmax_features_are_positive_and_finite():
    features = PositiveSoftmaxFeatures(head_dim=8, feature_dim=16, seed=7)
    out = features(torch.randn(2, 3, 4, 8))
    assert out.shape == (2, 3, 4, 16)
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_amcenn_attention_is_causal():
    torch.manual_seed(11)
    model_cfg = cfg()
    original = DummyAttention(16, 4, 2)
    module = AMCeNNAttention(
        original,
        model_cfg,
        SmolAMCeNNConfig(feature_dim=16, num_shards=8, top_k=2),
        layer_idx=0,
    )
    x = torch.randn(1, 9, 16)
    y1, _ = module(x, use_cache=False)
    changed = x.clone()
    changed[:, 6:] += torch.randn_like(changed[:, 6:]) * 20
    y2, _ = module(changed, use_cache=False)
    torch.testing.assert_close(y1[:, :6], y2[:, :6], rtol=1e-5, atol=1e-6)


def test_sharded_top2_ffn_is_dense_ffn_at_zero_mix():
    torch.manual_seed(23)
    model_cfg = cfg()
    dense = DummyMLP(16, 32)
    sharded = ShardedTop2LlamaMLP(
        dense,
        model_cfg,
        SmolAMCeNNConfig(feature_dim=16, num_shards=8, top_k=2),
    )
    x = torch.randn(3, 7, 16)
    expected = dense(x)
    actual = sharded(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert float(sharded.route_mix) == 0.0
    torch.testing.assert_close(
        sharded.last_router_stats["shard_fraction"].sum(), torch.tensor(1.0)
    )


def test_sharded_ffn_adds_only_router_and_mix_parameters():
    model_cfg = cfg(hidden=16, inner=32)
    dense = DummyMLP(16, 32)
    sharded = ShardedTop2LlamaMLP(
        dense,
        model_cfg,
        SmolAMCeNNConfig(feature_dim=16, num_shards=8, top_k=2),
    )
    dense_params = sum(p.numel() for p in dense.parameters())
    sharded_params = sum(p.numel() for p in sharded.parameters())
    assert sharded_params == dense_params + 16 * 8 + 1


def test_amcenn_has_linear_state_shape_not_token_pair_matrix():
    torch.manual_seed(3)
    model_cfg = cfg(hidden=16, inner=32, heads=4, kv_heads=2)
    original = DummyAttention(16, 4, 2)
    module = AMCeNNAttention(
        original,
        model_cfg,
        SmolAMCeNNConfig(feature_dim=12, num_shards=8, top_k=2),
        layer_idx=2,
    )
    x = torch.randn(2, 13, 16)
    y, weights = module(x, use_cache=False)
    assert y.shape == x.shape
    assert weights is None
    assert module.features.projection.shape == (12, 4)
