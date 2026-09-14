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
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


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
        reference[:, :, :30], changed[:, :, :30], atol=1e-5, rtol=1e-5
    )


def test_dilated_receptive_field_grows_exponentially():
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant="cellular_dilated3",
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    assert layer.receptive_field_tokens() == 511


@pytest.mark.parametrize(
    "variant",
    [v for v in VARIANTS if v != "cellular_local3"],
)
def test_sparse_score_pairs_are_below_dense_attention(variant):
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant=variant,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128), shifted_window=8,
    )
    context = 512
    assert layer.max_score_pairs(context) < context * (context + 1) // 2


@pytest.mark.parametrize(
    "variant",
    [v for v in VARIANTS if "multiscale5" in v or "maxpool5" in v or "uamp5" in v],
)
def test_multiscale_family_keeps_attention_score_budget(variant):
    baseline = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant="cellular_multiscale5",
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    candidate = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant=variant,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    assert candidate.receptive_field_tokens() == baseline.receptive_field_tokens() == 1021
    assert candidate.max_score_pairs(512) == baseline.max_score_pairs(512)
    assert candidate.max_neighbors_per_step() == 5


def test_adaptive_routing_parameters_receive_gradients():
    q, k, v = make_inputs(40)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12,
        variant="cellular_adaptive_multiscale5", dilations=(1, 2, 4, 8),
    )
    layer(q, k, v).square().mean().backward()
    for parameter in (layer.route_key, layer.route_prior, layer.route_strength):
        assert parameter is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_maxpool_branch_parameters_receive_gradients():
    q, k, v = make_inputs(40)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12,
        variant="cellular_multiscale5_maxpool", dilations=(1, 2, 4, 8),
    )
    layer(q, k, v).square().mean().backward()
    for parameter in (layer.pool_mix_logit, layer.log_pool_gain):
        assert parameter is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_mixedpool_and_rms_parameters_receive_gradients():
    q, k, v = make_inputs(40)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12,
        variant="cellular_adaptive_mixedpool5_rms", dilations=(1, 2, 4, 8),
    )
    layer(q, k, v).square().mean().backward()
    for parameter in (
        layer.mixed_pool_logits, layer.log_mixed_pool_gain,
        layer.pre_rms_weight, layer.post_rms_weight,
        layer.pre_rms_gate, layer.post_rms_gate,
    ):
        assert parameter is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_uamp_encoder_decoder_parameters_receive_gradients():
    q, k, v = make_inputs(41)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12,
        variant="cellular_uamp5", dilations=(1, 2, 4, 8),
    )
    layer(q, k, v).square().mean().backward()
    for parameter in (
        layer.unet_pool_logits, layer.unet_encoder, layer.unet_decoder,
        layer.unet_swiglu_gate, layer.unet_swiglu_value,
        layer.unet_skip1_gate, layer.unet_skip0_gate, layer.unet_output_gate,
    ):
        assert parameter is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_channel_gate_is_trainable_and_causal_via_full_layer_test():
    q, k, v = make_inputs(37)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12,
        variant="cellular_uamp5_channelgate", dilations=(1, 2, 4, 8),
    )
    layer(q, k, v).square().mean().backward()
    for parameter in (layer.channel_down, layer.channel_up, layer.channel_strength):
        assert parameter is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_variational_latent_auxiliary_is_finite():
    q, k, v = make_inputs(39)
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=12,
        variant="cellular_uamp5_varlatent", dilations=(1, 2, 4, 8),
    )
    out = layer(q, k, v)
    aux = layer.auxiliary_loss()
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux)
    assert aux >= 0


def test_maxpool_branch_can_fall_back_to_multiscale_attention():
    torch.manual_seed(77)
    baseline = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant="cellular_multiscale5",
        dilations=(1, 2, 4, 8),
    ).eval()
    torch.manual_seed(77)
    candidate = CellularAttentionLayer(
        4, 2, 16, feature_dim=12, variant="cellular_multiscale5_maxpool",
        dilations=(1, 2, 4, 8),
    ).eval()
    candidate.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        candidate.pool_mix_logit.fill_(-30.0)
        q, k, v = make_inputs(32)
        expected = baseline(q, k, v)
        actual = candidate(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "variant",
    ["cellular_adaptive_maxpool5", "cellular_adaptive_mixedpool5_rms", "cellular_uamp5"],
)
def test_checkpoint_config_reconstructs_exactly(variant):
    layer = CellularAttentionLayer(
        4, 2, 16, feature_dim=20, variant=variant,
        dilations=(1, 2, 4, 8), shifted_window=6,
    )
    clone = CellularAttentionLayer(**layer.config)
    assert clone.config == layer.config
