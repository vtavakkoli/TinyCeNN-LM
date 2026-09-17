import types

import torch
from torch import nn

from tinycenn_lm.smollm2_flydelta_hybrid import (
    FlyDeltaHybridAttention,
    FlyDeltaHybridConfig,
)


class _OriginalAttention(nn.Module):
    def __init__(self, hidden=64, heads=4, kv_heads=2):
        super().__init__()
        d = hidden // heads
        self.q_proj = nn.Linear(hidden, heads * d, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * d, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * d, bias=False)
        self.o_proj = nn.Linear(heads * d, hidden, bias=False)


def _config():
    return types.SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
    )


def _run(layer_idx):
    torch.manual_seed(7)
    cfg = FlyDeltaHybridConfig(
        fly_nodes=32,
        local_window=8,
        anchor_window=16,
        anchor_every=4,
    )
    attn = FlyDeltaHybridAttention(
        _OriginalAttention(),
        _config(),
        cfg,
        layer_idx,
        torch.eye(32),
    )
    x = torch.randn(2, 12, 64)

    attn.set_streaming(False, reset=True)
    full = attn(x)[0][:, -1]

    attn.set_streaming(True, reset=True)
    attn(x[:, :-1])
    streamed = attn(x[:, -1:])[0][:, -1]

    assert torch.allclose(full, streamed, atol=1e-5, rtol=1e-5)


def test_hybrid_streaming_matches_full():
    _run(layer_idx=0)


def test_anchor_streaming_matches_full():
    _run(layer_idx=3)
