import torch

from scripts.benchmark_tiny_llm_pdelta3 import candidate_specs, route_for
from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer


def test_tiny_llm_candidate_set_and_route_proxy():
    specs = {spec["name"]: spec for spec in candidate_specs()}
    assert set(specs) == {
        "conv4_pdelta_f96",
        "conv4_channel_decay_f96",
        "conv4_gdn2_f96",
        "conv4_gdn2_inputroute_f96",
    }
    v = torch.randn(1, 1, 5, 8)
    assert route_for(specs["conv4_gdn2_f96"], v) is None
    assert route_for(specs["conv4_gdn2_inputroute_f96"], v) is v


def test_one_layer_value_route_proxy_runs_and_streams():
    torch.manual_seed(7)
    core = FrontierPDelta3Layer(
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        feature_dim=8,
        variant="conv4_gdn2_clvr_f96",
        chunk_size=4,
        conv_kernel=4,
        state_dtype="fp32",
    )
    q = torch.randn(1, 2, 9, 8)
    k = torch.randn(1, 1, 9, 8)
    v = torch.randn(1, 1, 9, 8)
    full = core(q, k, v, routed_v=v)

    state = None
    pieces = []
    for t in range(q.shape[2]):
        out, state = core(
            q[:, :, t:t+1],
            k[:, :, t:t+1],
            v[:, :, t:t+1],
            routed_v=v[:, :, t:t+1],
            state=state,
            return_state=True,
        )
        pieces.append(out)
    streamed = torch.cat(pieces, dim=2)
    torch.testing.assert_close(streamed, full, atol=2e-5, rtol=2e-5)
