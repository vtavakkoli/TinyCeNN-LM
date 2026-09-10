import torch

from tinycenn_lm.moe import FastMoECeNNCore, MoECeNNConfig


def make_config(steps=3):
    return MoECeNNConfig(
        hidden_size=16,
        kernel_size=3,
        expansion=2,
        steps=steps,
        dilations=(1, 2, 4),
        num_experts=8,
        top_k=2,
        router_noise_std=1e-3,
    )


def test_moe_cenn_is_zero_delta_at_initialization():
    torch.manual_seed(0)
    core = FastMoECeNNCore(make_config())
    x = torch.randn(2, 12, 16)
    y = core(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7, rtol=0)


def test_router_uses_top2_and_fractions_sum_to_one():
    torch.manual_seed(1)
    core = FastMoECeNNCore(make_config())
    x = torch.randn(2, 10, 16)
    core(x)
    stats = core.last_router_stats
    assert torch.isfinite(stats["load_balance"])
    assert torch.isfinite(stats["z_loss"])
    assert torch.isfinite(stats["entropy"])
    assert torch.allclose(stats["expert_fraction"].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(stats["probability_fraction"].sum(), torch.tensor(1.0), atol=1e-5)


def test_moe_cenn_is_causal_after_nonzero_experts():
    torch.manual_seed(2)
    core = FastMoECeNNCore(make_config())
    for expert in core.cell.experts:
        torch.nn.init.normal_(expert.out_proj.weight, std=0.02)
    x1 = torch.randn(1, 16, 16)
    x2 = x1.clone()
    x2[:, 10:] = torch.randn_like(x2[:, 10:])
    y1 = core(x1)
    y2 = core(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-5, rtol=1e-5)


def test_recurrent_steps_do_not_change_unique_parameter_count():
    p3 = sum(p.numel() for p in FastMoECeNNCore(make_config(3)).parameters())
    p7 = sum(p.numel() for p in FastMoECeNNCore(MoECeNNConfig(
        hidden_size=16, kernel_size=3, expansion=2, steps=7,
        dilations=(1, 2, 4, 8, 16, 32, 64), num_experts=8, top_k=2,
    )).parameters())
    assert p3 == p7
