import torch
from torch import nn

from tinycenn_lm import CeNNConfig, FastCeNNCore, HybridDecoderLayer


def test_cenn_is_exact_noop_at_initialization():
    torch.manual_seed(0)
    config = CeNNConfig(hidden_size=16, steps=4, expansion=2)
    core = FastCeNNCore(config)
    x = torch.randn(2, 12, 16)
    delta = core(x)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)


def test_cenn_is_strictly_causal_after_randomizing_output_projection():
    torch.manual_seed(1)
    config = CeNNConfig(hidden_size=8, steps=3, expansion=2, dilations=(1, 2, 4))
    core = FastCeNNCore(config).eval()
    nn.init.normal_(core.cell.out_proj.weight, std=0.02)

    x1 = torch.randn(1, 16, 8)
    x2 = x1.clone()
    x2[:, 10:, :] = torch.randn_like(x2[:, 10:, :]) * 10

    y1 = core(x1)
    y2 = core(x2)
    torch.testing.assert_close(y1[:, :10], y2[:, :10], atol=1e-5, rtol=1e-5)


def test_parameter_count_does_not_depend_on_recurrent_steps():
    c1 = FastCeNNCore(CeNNConfig(hidden_size=16, steps=1, expansion=2))
    c8 = FastCeNNCore(CeNNConfig(hidden_size=16, steps=8, expansion=2))
    p1 = sum(p.numel() for p in c1.parameters())
    p8 = sum(p.numel() for p in c8.parameters())
    assert p1 == p8
    assert c8.receptive_field > c1.receptive_field


class DummyLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return (hidden_states * 2, "cache")


class ParameterizedDummyLayer(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False, dtype=dtype)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states),)


def test_hybrid_wrapper_preserves_base_output_initially():
    torch.manual_seed(2)
    config = CeNNConfig(hidden_size=8, steps=4, expansion=2)
    layer = HybridDecoderLayer(DummyLayer(), config)
    x = torch.randn(2, 5, 8)
    outputs = layer(x)
    torch.testing.assert_close(outputs[0], x * 2, atol=0, rtol=0)
    assert outputs[1] == "cache"


def test_hybrid_wrapper_inherits_base_layer_bfloat16_dtype():
    config = CeNNConfig(hidden_size=8, steps=2, expansion=2)
    base = ParameterizedDummyLayer(torch.bfloat16)
    layer = HybridDecoderLayer(base, config)

    floating_params = [p for p in layer.cenn.parameters() if p.is_floating_point()]
    assert floating_params
    assert all(p.dtype == torch.bfloat16 for p in floating_params)
    assert layer.residual_scale.dtype == torch.bfloat16


def test_hybrid_wrapper_rejects_transformer_only_kv_cache():
    config = CeNNConfig(hidden_size=8, steps=2, expansion=2)
    layer = HybridDecoderLayer(DummyLayer(), config)
    x = torch.randn(1, 3, 8)
    try:
        layer(x, use_cache=True)
    except RuntimeError as exc:
        assert "requires use_cache=False" in str(exc)
    else:
        raise AssertionError("use_cache=True must not silently bypass CeNN history")
