import torch

from tinycenn_lm.cenn import CeNNConfig, FastCeNNCore, SharedCeNNCell
from tinycenn_lm.sharded_moe import (
    FastShardedMoECeNNCore,
    ShardedMoECeNNConfig,
    ShardedMoESharedCeNNCell,
)


def _copy_dense_cell_into_shards(dense: SharedCeNNCell, sharded: ShardedMoESharedCeNNCell) -> None:
    cfg = sharded.config
    inner = cfg.dense_inner
    shard = cfg.shard_inner
    with torch.no_grad():
        sharded.norm.weight.copy_(dense.norm.weight)
        sharded.neighborhood.weight.copy_(dense.neighborhood.weight)
        sharded.gate_proj.weight.copy_(dense.gate_proj.weight)
        sharded.gate_proj.bias.copy_(dense.gate_proj.bias)
        for expert_id in range(cfg.num_shards):
            lo = expert_id * shard
            hi = lo + shard
            sharded.ffn.in_weight[expert_id, :shard].copy_(dense.in_proj.weight[lo:hi])
            sharded.ffn.in_weight[expert_id, shard:].copy_(dense.in_proj.weight[inner + lo : inner + hi])
            sharded.ffn.out_weight[expert_id].copy_(dense.out_proj.weight[:, lo:hi])
        sharded.ffn.route_mix.zero_()


def test_sharded_ffn_is_exact_dense_ffn_at_zero_route_mix():
    torch.manual_seed(7)
    dense_cfg = CeNNConfig(hidden_size=16, kernel_size=3, expansion=4, steps=1, dilations=(1,))
    shard_cfg = ShardedMoECeNNConfig(
        hidden_size=16,
        kernel_size=3,
        expansion=4,
        steps=1,
        dilations=(1,),
        num_shards=8,
        top_k=2,
    )
    dense = SharedCeNNCell(dense_cfg)
    sharded = ShardedMoESharedCeNNCell(shard_cfg)
    # Avoid the default zero output projection so equivalence tests real values.
    torch.nn.init.normal_(dense.out_proj.weight, std=0.02)
    _copy_dense_cell_into_shards(dense, sharded)

    x = torch.randn(2, 11, 16)
    dense_out = dense(x, dilation=1, step_scale=1.0)
    sharded_out, stats = sharded(x, dilation=1, step_scale=1.0)
    torch.testing.assert_close(sharded_out, dense_out, rtol=1e-5, atol=1e-6)
    assert float(stats["shard_fraction"].sum()) == 1.0


def test_sharded_model_adds_only_router_and_mix_parameters():
    dense_cfg = CeNNConfig(hidden_size=192, expansion=4, steps=7, dilations=(1, 2, 4, 8, 16, 32, 64))
    shard_cfg = ShardedMoECeNNConfig(
        hidden_size=192,
        expansion=4,
        steps=7,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        num_shards=8,
        top_k=2,
    )
    dense = FastCeNNCore(dense_cfg)
    sharded = FastShardedMoECeNNCore(shard_cfg)
    dense_params = sum(p.numel() for p in dense.parameters())
    sharded_params = sum(p.numel() for p in sharded.parameters())
    assert dense_params == 480_192
    assert sharded_params == dense_params + (192 * 8) + 1
    assert (sharded_params - dense_params) / dense_params < 0.01


def test_sharded_router_is_top2_and_balanced_stats_are_normalized():
    cfg = ShardedMoECeNNConfig(
        hidden_size=16,
        expansion=4,
        steps=1,
        dilations=(1,),
        num_shards=8,
        top_k=2,
    )
    core = FastShardedMoECeNNCore(cfg)
    _ = core(torch.randn(3, 9, 16))
    stats = core.last_router_stats
    torch.testing.assert_close(stats["shard_fraction"].sum(), torch.tensor(1.0))
    torch.testing.assert_close(stats["probability_fraction"].sum(), torch.tensor(1.0))
    assert stats["shard_fraction"].shape == (8,)


def test_sharded_cenn_is_causal():
    torch.manual_seed(11)
    cfg = ShardedMoECeNNConfig(
        hidden_size=16,
        expansion=4,
        steps=3,
        dilations=(1, 2, 4),
        num_shards=8,
        top_k=2,
    )
    core = FastShardedMoECeNNCore(cfg)
    torch.nn.init.normal_(core.cell.ffn.out_weight, std=0.02)
    x = torch.randn(1, 12, 16)
    y1 = core(x)
    changed = x.clone()
    changed[:, 8:] += torch.randn_like(changed[:, 8:]) * 10
    y2 = core(changed)
    torch.testing.assert_close(y1[:, :8], y2[:, :8], rtol=1e-5, atol=1e-6)
