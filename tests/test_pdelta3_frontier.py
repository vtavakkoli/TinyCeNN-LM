import subprocess
import sys
from pathlib import Path

import torch

from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer, VARIANTS


def make_layer(variant, state_dtype="fp32"):
    return FrontierPDelta3Layer(
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        feature_dim=16,
        variant=variant,
        chunk_size=8,
        conv_kernel=4,
        state_dtype=state_dtype,
    )


def test_all_variants_round_trip_config_and_finite_output():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 12, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    route = torch.randn_like(v)
    for variant in VARIANTS:
        layer = make_layer(variant)
        out = layer(q, k, v, routed_v=route if layer.use_clvr else None)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()
        clone = FrontierPDelta3Layer(**layer.config)
        clone.load_state_dict(layer.state_dict())
        assert clone.config == layer.config


def test_gdn2_streaming_matches_full_sequence():
    torch.manual_seed(2)
    layer = make_layer("conv4_gdn2_f96", state_dtype="fp32").eval()
    q = torch.randn(1, 4, 17, 8)
    k = torch.randn(1, 2, 17, 8)
    v = torch.randn(1, 2, 17, 8)
    full = layer(q, k, v)
    state = None
    pieces = []
    for t in range(q.shape[2]):
        out, state = layer(
            q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
            state=state, return_state=True,
        )
        pieces.append(out)
    streamed = torch.cat(pieces, dim=2)
    torch.testing.assert_close(full, streamed, rtol=2e-4, atol=2e-5)


def test_clvr_streaming_matches_full_sequence_and_route_changes_output():
    torch.manual_seed(3)
    layer = make_layer("conv4_gdn2_clvr_f96", state_dtype="fp32").eval()
    with torch.no_grad():
        layer.route_gate_b.fill_(0.0)
    q = torch.randn(1, 4, 15, 8)
    k = torch.randn(1, 2, 15, 8)
    v = torch.randn(1, 2, 15, 8)
    route = torch.randn_like(v)
    full = layer(q, k, v, routed_v=route)
    no_route = layer(q, k, v, routed_v=torch.zeros_like(route))
    assert not torch.allclose(full, no_route)

    state = None
    pieces = []
    for t in range(q.shape[2]):
        out, state = layer(
            q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
            state=state, return_state=True, routed_v=route[:, :, t:t + 1],
        )
        pieces.append(out)
    streamed = torch.cat(pieces, dim=2)
    torch.testing.assert_close(full, streamed, rtol=2e-4, atol=2e-5)


def test_channel_decay_has_broad_initial_half_lives():
    layer = make_layer("conv4_channel_decay_f96")
    stats = layer.decay_statistics()
    assert stats["decay_half_life_max"] > stats["decay_half_life_median"]
    assert stats["decay_half_life_median"] > stats["decay_half_life_min"]


def test_clvr_adds_no_temporal_state_matrix():
    gdn2 = make_layer("conv4_gdn2_f96", state_dtype="fp16")
    clvr = make_layer("conv4_gdn2_clvr_f96", state_dtype="fp16")
    assert gdn2.recurrent_state_bytes() == clvr.recurrent_state_bytes()
    pdelta = make_layer("conv4_pdelta_f96", state_dtype="fp16")
    assert pdelta.recurrent_state_bytes() > 0


def test_benchmark_cli_absolute_path_help():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_pdelta3_frontier_layer.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path("/tmp"), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "frontier-inspired" in result.stdout
