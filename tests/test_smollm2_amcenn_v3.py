from types import SimpleNamespace

import torch
from torch import nn

from tinycenn_lm.smollm2_amcenn_v3 import (
    HybridLocalAMCeNNAttention,
    SmolAMCeNNV3Config,
)


class DummyAttention(nn.Module):
    def __init__(self, hidden=16, heads=4, kv_heads=2):
        super().__init__()
        head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)


def cfg(hidden=16, heads=4, kv_heads=2):
    return SimpleNamespace(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
    )


def test_v3_is_causal_with_local_and_global_paths():
    torch.manual_seed(3)
    module = HybridLocalAMCeNNAttention(
        DummyAttention(),
        cfg(),
        SmolAMCeNNV3Config(feature_dim=32, local_window=4, global_gate_init=0.1),
        layer_idx=0,
    )
    x = torch.randn(1, 10, 16)
    y1, _ = module(x, use_cache=False)
    x2 = x.clone()
    x2[:, 7:] += 10.0 * torch.randn_like(x2[:, 7:])
    y2, _ = module(x2, use_cache=False)
    torch.testing.assert_close(y1[:, :7], y2[:, :7], rtol=1e-5, atol=1e-6)


def test_v3_before_window_is_independent_of_global_gate():
    torch.manual_seed(4)
    module = HybridLocalAMCeNNAttention(
        DummyAttention(),
        cfg(),
        SmolAMCeNNV3Config(feature_dim=32, local_window=8, global_gate_init=0.05),
        layer_idx=0,
    )
    x = torch.randn(1, 6, 16)
    y1, _ = module(x, use_cache=False)
    with torch.no_grad():
        module.global_gate_logit.fill_(8.0)
    y2, _ = module(x, use_cache=False)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)


def test_v3_feature_and_gate_receive_gradient_on_long_context():
    torch.manual_seed(5)
    module = HybridLocalAMCeNNAttention(
        DummyAttention(),
        cfg(),
        SmolAMCeNNV3Config(feature_dim=32, local_window=4, global_gate_init=0.1),
        layer_idx=0,
    )
    x = torch.randn(2, 9, 16)
    y, _ = module(x, use_cache=False)
    y.square().mean().backward()
    assert module.features.delta_projection is not None
    assert module.features.delta_projection.grad is not None
    assert torch.isfinite(module.features.delta_projection.grad).all()
    assert module.global_gate_logit.grad is not None
    assert torch.isfinite(module.global_gate_logit.grad).all()
