"""Regression tests for FP16 typed-decision auxiliary action head.

A converted FP16 model must run without torch.autocast, including from the
standalone inference runtime and GPU latency benchmarking. Previously the
action head was Half while its concatenated probability/hidden features were
Float, causing: mat1 and mat2 must have the same dtype.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tinycenn_lm.standalone_decision import StandaloneDecisionModel


class TinyStubEncoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.embedding = nn.Embedding(16, dim)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _batch(device):
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], device=device),
        "attention_mask": torch.ones((1, 4), device=device, dtype=torch.long),
        "marker_pos": torch.tensor([[1, 2]], device=device),
        "marker_mask": torch.ones((1, 2), device=device, dtype=torch.bool),
        "qtype": torch.zeros((1,), device=device, dtype=torch.long),
    }


@pytest.mark.parametrize("dtype", [torch.float32])
def test_action_head_float32_no_autocast(dtype):
    model = StandaloneDecisionModel(TinyStubEncoder(), head_layers=0).to(dtype)
    model.eval()
    with torch.inference_mode():
        logits, action_logits = model(**_batch("cpu"))
    assert logits.shape == (1, 2)
    assert action_logits.shape == (1, 2)
    assert action_logits.dtype == dtype
    assert torch.isfinite(action_logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_action_head_low_precision_no_autocast_cuda(dtype):
    # Intentionally *no* autocast: test true native model.half()/bfloat16.
    model = StandaloneDecisionModel(TinyStubEncoder(), head_layers=0).to(
        device="cuda", dtype=dtype
    ).eval()
    with torch.inference_mode():
        logits, action_logits = model(**_batch("cuda"))
    assert logits.shape == (1, 2)
    assert logits.dtype == torch.float32  # decision scores intentionally fp32
    assert action_logits.shape == (1, 2)
    assert action_logits.dtype == dtype
    assert torch.isfinite(action_logits).all()
