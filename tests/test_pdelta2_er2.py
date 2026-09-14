import subprocess
import sys
from pathlib import Path

import torch

from tinycenn_lm.pdelta2_er2 import (
    SelectiveCompressedPDelta2Layer,
    hard_error_mask,
    normalized_hard_weights,
)


def tensors():
    torch.manual_seed(42)
    q = torch.randn(1, 4, 24, 8)
    k = torch.randn(1, 2, 24, 8)
    v = torch.randn(1, 2, 24, 8)
    return q, k, v


def test_compressed_branch_is_function_preserving_at_zero_gain():
    q, k, v = tensors()
    base = SelectiveCompressedPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=0, code_rank=0,
        conv_kernel=4, retention_spectrum=True, retention_max=512,
        residual_mode="none", state_dtype="fp32",
    )
    er2 = SelectiveCompressedPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=4, code_rank=3,
        conv_kernel=4, retention_spectrum=True, retention_max=512,
        residual_mode="compressed", state_dtype="fp32",
    )
    er2.base.load_state_dict(base.base.state_dict())
    er2.conv_weight.data.copy_(base.conv_weight.data)
    torch.testing.assert_close(base(q, k, v), er2(q, k, v), rtol=1e-5, atol=1e-5)


def test_low_rank_projection_and_orthogonality():
    layer = SelectiveCompressedPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=4, code_rank=3,
        residual_mode="compressed", state_dtype="fp32",
    )
    x = torch.randn(2, 4, 11, 8)
    code = layer.project_to_code(x)
    restored = layer.reconstruct_code(code)
    assert code.shape == (2, 4, 11, 3)
    assert restored.shape == x.shape
    assert float(layer.orthogonality_penalty()) < 1e-5


def test_hard_error_mask_and_weights_are_normalized():
    torch.manual_seed(1)
    base = torch.zeros(2, 4, 20, 8)
    target = torch.randn_like(base)
    mask, _ = hard_error_mask(base, target, fraction=0.25)
    assert 0.20 <= float(mask.float().mean()) <= 0.30
    weights = normalized_hard_weights(mask, boost=3.0)
    torch.testing.assert_close(weights.mean(dim=(-1, -2)), torch.ones(2))


def test_streaming_matches_full_sequence_in_fp32_state():
    q, k, v = tensors()
    layer = SelectiveCompressedPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=4, code_rank=3,
        chunk_size=8, conv_kernel=4, retention_spectrum=True,
        retention_max=512, residual_mode="compressed", state_dtype="fp32",
    )
    with torch.no_grad():
        layer.residual_gain.fill_(0.4)
        layer.error_gate_b.fill_(-0.2)
    full = layer(q, k, v)
    state = None
    pieces = []
    for t in range(q.shape[2]):
        out, state = layer(
            q[:, :, t:t + 1],
            k[:, :, t:t + 1],
            v[:, :, t:t + 1],
            state=state,
            return_state=True,
        )
        pieces.append(out)
    streamed = torch.cat(pieces, dim=2)
    torch.testing.assert_close(full, streamed, rtol=3e-4, atol=3e-4)


def test_config_roundtrip_and_fp16_state_budget():
    layer = SelectiveCompressedPDelta2Layer(
        6, 3, 8, feature_dim=18, residual_dim=5, code_rank=4,
        retention_spectrum=True, retention_max=1024,
        residual_mode="compressed", state_dtype="fp16",
    )
    clone = SelectiveCompressedPDelta2Layer(**layer.config)
    clone.load_state_dict(layer.state_dict())
    assert clone.config == layer.config
    fp16_bytes = clone.recurrent_state_bytes()
    fp32 = SelectiveCompressedPDelta2Layer(
        **{**layer.config, "state_dtype": "fp32"}
    )
    assert fp16_bytes < fp32.recurrent_state_bytes()


def test_auxiliary_losses_are_finite():
    q, k, v = tensors()
    layer = SelectiveCompressedPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=4, code_rank=3,
        retention_spectrum=True, retention_max=512,
        residual_mode="compressed", state_dtype="fp32",
    )
    teacher = torch.randn_like(q)
    losses = layer.auxiliary_losses(q, k, v, teacher, hard_fraction=0.25)
    required = {
        "hard_teacher_nmse", "residual_code_nmse",
        "residual_reconstruction_nmse", "gate_bce", "orthogonality",
    }
    assert required <= set(losses)
    assert all(torch.isfinite(losses[key]) for key in required)


def test_benchmark_cli_absolute_path_help():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_pdelta2_er2_layer.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path("/tmp"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PDelta2-ER2" in result.stdout
