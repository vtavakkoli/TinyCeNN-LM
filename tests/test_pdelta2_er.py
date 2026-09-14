import subprocess
import sys
from pathlib import Path

import torch

from tinycenn_lm.pdelta2_er import (
    ErrorResidualPDelta2Layer,
    initialize_retention_spectrum,
    retention_half_lives,
    teacher_error_weights,
)
from tinycenn_lm.pdelta2_features import PDelta2Core


def test_retention_spectrum_spans_short_and_long_half_lives():
    core = PDelta2Core(4, 2, 8, feature_dim=32, chunk_size=8)
    initialize_retention_spectrum(core, 8.0, 2048.0)
    half = retention_half_lives(core)
    assert float(half.min()) < 9.0
    assert float(half.max()) > 1900.0
    assert float(half.std()) > 100.0


def test_teacher_error_weights_focus_hard_tokens_and_normalize():
    target = torch.zeros(1, 2, 8, 4)
    prediction = torch.zeros_like(target)
    prediction[:, :, -2:] = 10.0
    weights = teacher_error_weights(prediction, target, hard_fraction=0.25, hard_boost=3.0)
    assert weights.shape == (1, 8)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    assert float(weights[0, -1]) > float(weights[0, 0])
    assert float(weights[0, -2]) > float(weights[0, 1])


def test_zero_residual_gain_is_function_preserving():
    torch.manual_seed(3)
    base = ErrorResidualPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=0, chunk_size=8,
        conv_kernel=4, retention_spectrum=True, state_dtype="fp32",
    )
    residual = ErrorResidualPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=6, chunk_size=8,
        conv_kernel=4, retention_spectrum=True, state_dtype="fp32",
    )
    residual.base.load_state_dict(base.base.state_dict())
    residual.conv_weight.data.copy_(base.conv_weight.data)
    q = torch.randn(1, 4, 17, 8)
    k = torch.randn(1, 2, 17, 8)
    v = torch.randn(1, 2, 17, 8)
    torch.testing.assert_close(base(q, k, v), residual(q, k, v), rtol=1e-5, atol=1e-5)


def test_streaming_conv_and_recurrence_match_full_sequence_fp32():
    torch.manual_seed(4)
    layer = ErrorResidualPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=4, chunk_size=8,
        conv_kernel=4, retention_spectrum=True, state_dtype="fp32",
    ).eval()
    q = torch.randn(1, 4, 19, 8)
    k = torch.randn(1, 2, 19, 8)
    v = torch.randn(1, 2, 19, 8)
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
    torch.testing.assert_close(streamed, full, rtol=2e-4, atol=2e-4)


def test_fp16_persistent_memory_and_fp32_curvature():
    torch.manual_seed(5)
    layer = ErrorResidualPDelta2Layer(
        4, 2, 8, feature_dim=16, residual_dim=4,
        conv_kernel=4, retention_spectrum=True, state_dtype="fp16",
    )
    q = torch.randn(1, 4, 9, 8)
    k = torch.randn(1, 2, 9, 8)
    v = torch.randn(1, 2, 9, 8)
    _, state = layer(q, k, v, return_state=True)
    assert state.base.memory.dtype == torch.float16
    assert state.base.curvature.dtype == torch.float32
    assert state.residual.memory.dtype == torch.float16
    assert state.residual.curvature.dtype == torch.float32
    assert state.conv_tail.dtype == torch.float16


def test_config_round_trip_and_state_budget():
    layer = ErrorResidualPDelta2Layer(
        6, 3, 8, feature_dim=18, residual_dim=5, chunk_size=8,
        conv_kernel=4, retention_spectrum=True, retention_min=6, retention_max=1024,
        state_dtype="fp16",
    )
    clone = ErrorResidualPDelta2Layer(**layer.config)
    clone.load_state_dict(layer.state_dict())
    assert clone.config == layer.config
    base_only = ErrorResidualPDelta2Layer(
        6, 3, 8, feature_dim=18, residual_dim=0, chunk_size=8,
        conv_kernel=4, retention_spectrum=True, state_dtype="fp16",
    )
    assert layer.recurrent_state_bytes() > base_only.recurrent_state_bytes()
    assert layer.recurrent_state_bytes() < 2 * base_only.recurrent_state_bytes()


def test_benchmark_cli_absolute_path_help():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_pdelta2_er_layer.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=Path("/tmp"), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PDelta2-ER" in result.stdout or "retention" in result.stdout.lower()
