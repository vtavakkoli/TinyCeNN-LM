import pytest
import torch

from tinycenn_lm.memory_attention import VARIANTS, MemoryAugmentedCellularLayer


def make_inputs(length=24):
    torch.manual_seed(1234)
    q = torch.randn(2, 4, length, 16)
    k = torch.randn(2, 2, length, 16)
    v = torch.randn(2, 2, length, 16)
    return q, k, v


@pytest.mark.parametrize("variant", VARIANTS)
def test_memory_attention_shapes_finiteness_and_gradients(variant):
    q, k, v = make_inputs()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    layer = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant=variant,
        dilations=(1, 2, 4, 8),
        memory_rank=8,
    )
    out = layer(q, k, v)
    assert out.shape == (2, 4, 24, 16)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for grad in (q.grad, k.grad, v.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_memory_attention_is_strictly_causal(variant):
    q, k, v = make_inputs(28)
    layer = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant=variant,
        dilations=(1, 2, 4, 8),
        memory_rank=8,
    ).eval()
    with torch.no_grad():
        reference = layer(q, k, v)
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        q2[:, :, 17:] += 100 * torch.randn_like(q2[:, :, 17:])
        k2[:, :, 17:] += 100 * torch.randn_like(k2[:, :, 17:])
        v2[:, :, 17:] += 100 * torch.randn_like(v2[:, :, 17:])
        changed = layer(q2, k2, v2)
    torch.testing.assert_close(
        reference[:, :, :17],
        changed[:, :, :17],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "variant",
    [
        "cellular_hedgehog_global",
        "cellular_kda_global",
        "cellular_gdn2_global",
        "cellular_xlstm_global",
        "cellular_diff_hedgehog",
        "cellular_memory_fusion",
    ],
)
def test_new_memory_parameters_receive_finite_nonzero_gradients(variant):
    q, k, v = make_inputs(18)
    layer = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant=variant,
        dilations=(1, 2, 4),
        memory_rank=8,
    )
    layer(q, k, v).square().mean().backward()

    checked = 0
    for name, parameter in layer.named_parameters():
        if name.startswith("local."):
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("rank", [8, 12])
def test_checkpoint_config_reconstructs(rank):
    layer = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant="cellular_memory_fusion",
        dilations=(1, 2, 4),
        shifted_window=6,
        memory_rank=rank,
    )
    clone = MemoryAugmentedCellularLayer(**layer.config)
    assert clone.config == layer.config


@pytest.mark.parametrize("variant", VARIANTS)
def test_sparse_pair_budget_is_unchanged_by_global_memory(variant):
    baseline = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant="cellular_adaptive_maxpool5",
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        memory_rank=8,
    )
    candidate = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant=variant,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        memory_rank=8,
    )
    assert candidate.max_score_pairs(512) == baseline.max_score_pairs(512)
    assert candidate.max_neighbors_per_step() == baseline.max_neighbors_per_step() == 5
    assert candidate.receptive_field_tokens() == baseline.receptive_field_tokens() == 1021


def test_baseline_wrapper_matches_existing_cellular_path():
    torch.manual_seed(77)
    wrapper = MemoryAugmentedCellularLayer(
        4, 2, 16,
        feature_dim=12,
        variant="cellular_adaptive_maxpool5",
        dilations=(1, 2, 4),
        memory_rank=8,
    ).eval()
    q, k, v = make_inputs(16)
    with torch.no_grad():
        expected = wrapper.local(q, k, v)
        actual = wrapper(q, k, v)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
