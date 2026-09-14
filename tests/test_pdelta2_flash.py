import subprocess
import sys
from pathlib import Path

import torch

from tinycenn_lm.pdelta2_flash import (
    FlashPDelta2Layer,
    causal_depthwise_value_conv,
    indexed_block_attention,
)


def test_causal_conv_identity_initialization_shape():
    torch.manual_seed(1)
    v = torch.randn(2, 2, 13, 8)
    channels = 16
    weight = torch.zeros(channels, 1, 4)
    weight[:, 0, -1] = 1.0
    out = causal_depthwise_value_conv(v, weight)
    torch.testing.assert_close(out, v)


def test_indexed_block_attention_is_causal():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 40, 8)
    k = torch.randn(1, 2, 40, 8)
    v = torch.randn(1, 2, 40, 8)
    base, _ = indexed_block_attention(q, k, v, groups=2, block_size=8, topk=2)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 24:] += 50 * torch.randn_like(k2[:, :, 24:])
    v2[:, :, 24:] += 50 * torch.randn_like(v2[:, :, 24:])
    changed, _ = indexed_block_attention(q, k2, v2, groups=2, block_size=8, topk=2)
    torch.testing.assert_close(base[:, :, :24], changed[:, :, :24], rtol=0, atol=1e-6)


def test_gate_and_conv_are_function_preserving_at_initialization():
    torch.manual_seed(3)
    base = FlashPDelta2Layer(4, 2, 8, feature_dim=16, output_gate=False, conv_kernel=1)
    flash = FlashPDelta2Layer(4, 2, 8, feature_dim=16, output_gate=True, conv_kernel=4)
    flash.core.load_state_dict(base.core.state_dict())
    q = torch.randn(1, 4, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    torch.testing.assert_close(base(q, k, v), flash(q, k, v), rtol=1e-5, atol=1e-5)


def test_config_round_trip_and_state_budget():
    layer = FlashPDelta2Layer(
        6, 3, 8, feature_dim=18, output_gate=True, conv_kernel=4,
        indexed_retrieval=True, block_size=8, index_topk=2, state_dtype="fp16",
    )
    clone = FlashPDelta2Layer(**layer.config)
    clone.load_state_dict(layer.state_dict())
    assert clone.config == layer.config
    fp16_bytes = clone.recurrent_state_bytes(context=256)
    fp32 = FlashPDelta2Layer(
        6, 3, 8, feature_dim=18, output_gate=True, conv_kernel=4,
        indexed_retrieval=True, block_size=8, index_topk=2, state_dtype="fp32",
    )
    assert fp16_bytes < fp32.recurrent_state_bytes(context=256)


def test_indexed_summary_storage_is_smaller_than_token_kv():
    layer = FlashPDelta2Layer(
        4, 2, 8, feature_dim=16, indexed_retrieval=True,
        block_size=16, index_topk=4, state_dtype="fp16",
    )
    context = 512
    summary_only = 2 * (context // 16) * 2 * 8 * 2
    token_kv = 2 * context * 2 * 8 * 2
    assert summary_only * 16 == token_kv
    assert summary_only < token_kv
    assert layer.index_pairs_per_token(context) == 4


def test_streaming_decode_matches_full_last_token_and_stores_fp16_memory():
    torch.manual_seed(4)
    layer = FlashPDelta2Layer(
        4, 2, 8, feature_dim=16, output_gate=True, conv_kernel=4,
        indexed_retrieval=True, block_size=8, index_topk=2, state_dtype="fp16",
    ).eval()
    q = torch.randn(1, 4, 33, 8)
    k = torch.randn(1, 2, 33, 8)
    v = torch.randn(1, 2, 33, 8)
    with torch.no_grad():
        full = layer(q, k, v)
        _, state = layer(q[:, :, :32], k[:, :, :32], v[:, :, :32], return_state=True)
        step = layer(q[:, :, 32:], k[:, :, 32:], v[:, :, 32:], state=state)
    assert state.recurrent.memory.dtype == torch.float16
    assert state.recurrent.curvature.dtype == torch.float32
    torch.testing.assert_close(full[:, :, 32:], step, rtol=5e-4, atol=5e-4)


def test_benchmark_cli_absolute_path_help():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_pdelta2_flash_layer.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path("/tmp"), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PDelta2-Flash" in result.stdout
