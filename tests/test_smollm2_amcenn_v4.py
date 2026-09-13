from types import SimpleNamespace

import torch
from torch import nn

from tinycenn_lm.smollm2_amcenn_v4 import (
    AdaptiveHybridAMCeNNAttentionV4,
    SmolAMCeNNV4Config,
)


class DummyAttention(nn.Module):
    def __init__(self, hidden=32, heads=4, kv_heads=2):
        super().__init__()
        head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


def dummy_config():
    return SimpleNamespace(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)


def compact_v4_config():
    return SmolAMCeNNV4Config(
        easy_feature_dim=32,
        medium_feature_dim=32,
        hard_feature_dim=32,
        critical_feature_dim=32,
        easy_window=4,
        medium_window=4,
        hard_window=4,
        critical_window=4,
        anchor_tokens=2,
        gate_init=0.03,
    )


def test_v4_default_profile_matches_v3_difficulty_map():
    cfg = SmolAMCeNNV4Config()
    assert cfg.profile_for_layer(0).tier == "easy"
    assert cfg.profile_for_layer(7).tier == "hard"
    assert cfg.profile_for_layer(18).tier == "critical"
    assert cfg.profile_for_layer(20).local_window == 96
    assert cfg.profile_for_layer(13).tier == "medium"


def test_v4_attention_is_causal():
    torch.manual_seed(1)
    cfg = compact_v4_config()
    module = AdaptiveHybridAMCeNNAttentionV4(DummyAttention(), dummy_config(), cfg, 3).eval()
    x = torch.randn(1, 10, 32)
    y1 = module(x, use_cache=False)[0]
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:]) * 9.0
    y2 = module(x2, use_cache=False)[0]
    torch.testing.assert_close(y1[:, :7], y2[:, :7], atol=2e-5, rtol=2e-5)


def test_v4_global_gate_has_no_effect_before_old_memory_exists():
    torch.manual_seed(2)
    cfg = compact_v4_config()
    module = AdaptiveHybridAMCeNNAttentionV4(DummyAttention(), dummy_config(), cfg, 3).eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        module.gate_proj.weight.zero_()
        module.gate_proj.bias.fill_(5.0)
    high = module(x, use_cache=False)[0]
    with torch.no_grad():
        module.gate_proj.bias.fill_(-12.0)
    low = module(x, use_cache=False)[0]
    # With local_window=4, positions 0..3 have no older middle-memory state.
    torch.testing.assert_close(high[:, :4], low[:, :4], atol=2e-5, rtol=2e-5)
    assert not torch.allclose(high[:, 6:], low[:, 6:])


def test_v4_feature_and_token_gate_receive_gradients():
    torch.manual_seed(3)
    cfg = compact_v4_config()
    module = AdaptiveHybridAMCeNNAttentionV4(DummyAttention(), dummy_config(), cfg, 3)
    x = torch.randn(2, 10, 32, requires_grad=True)
    loss = module(x, use_cache=False)[0].float().square().mean()
    loss.backward()
    assert module.features.delta_projection is not None
    assert module.features.delta_projection.grad is not None
    assert torch.isfinite(module.features.delta_projection.grad).all()
    assert module.gate_proj.weight.grad is not None
    assert torch.isfinite(module.gate_proj.weight.grad).all()
