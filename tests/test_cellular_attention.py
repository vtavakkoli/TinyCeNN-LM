import pytest
import torch

from tinycenn_lm.cellular_attention import VARIANTS, CellularAttentionLayer


def make_inputs(length=64):
    torch.manual_seed(123)
    q = torch.randn(2, 4, length, 16)
    k = torch.randn(2, 2, length, 16)
    v = torch.randn(2, 2, length, 16)
    return q, k, v


@pytest.mark.parametrize("variant", VARIANTS)
def test_cellular_attention_shapes_and_gradients(variant):
    q, k, v = make_inputs(48)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant=variant,
        dilations=(1, 2, 4, 8), shifted_window=8,
    )
    out = layer(q, k, v)
    assert out.shape == (2, 4, 48, 16)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_cellular_attention_is_strictly_causal(variant):
    q, k, v = make_inputs(48)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant=variant,
        dilations=(1, 2, 4, 8), shifted_window=8,
    ).eval()
    with torch.no_grad():
        reference = layer(q, k, v)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, 30:] += 100 * torch.randn_like(k2[:, :, 30:])
        v2[:, :, 30:] += 100 * torch.randn_like(v2[:, :, 30:])
        changed = layer(q, k2, v2)
    torch.testing.assert_close(
        reference[:, :, :30],
        changed[:, :, :30],
        atol=1e-5,
        rtol=1e-5,
    )


def test_dilated_receptive_field_grows_exponentially():
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant="cellular_dilated3",
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    assert layer.receptive_field_tokens() == 511


@pytest.mark.parametrize(
    "variant",
    ["cellular_dilated3", "cellular_dilated5", "cellular_multiscale5", "cellular_shifted8"],
)
def test_sparse_score_pairs_are_below_dense_attention(variant):
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant=variant,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        shifted_window=8,
    )
    context = 512
    sparse = layer.max_score_pairs(context)
    dense = context * (context + 1) // 2
    assert sparse < dense


def test_checkpoint_config_reconstructs_exactly():
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=20, variant="cellular_multiscale5",
        dilations=(1, 2, 4, 8), shifted_window=6,
    )
    clone = CellularAttentionLayer(**layer.config)
    assert clone.config == layer.config
