"""CPU smoke tests for Laya V5 codec and fidelity decision gates."""
import torch
from torch import nn

from tinycenn_lm.laya_mixedbit_sparse_v4 import (
    PackedSparseLinear, pack_codes, unpack_codes,
)
from tinycenn_lm.laya_mixedbit_accuracy_v5 import (
    capture_input_moments, collect_decisions, public_metrics,
    quantize_aware, safe_accept,
)


def test_quantize_weighted_sparse_roundtrip_cpu():
    torch.manual_seed(5)
    projection = nn.Linear(193, 29, bias=True).half()
    moments = {"rms": torch.linspace(0.4, 2.0, 193),
               "mean": torch.zeros(193)}
    x = torch.randn(6, 193).half()
    for bits in (2, 4, 8):
        for style in ("plain", "weighted"):
            module = quantize_aware(projection, bits, 16, moments, style)
            assert isinstance(module, PackedSparseLinear)
            codes = unpack_codes(module.packed_weight, bits, module.padded)
            assert torch.equal(pack_codes(codes, bits), module.packed_weight)
            with torch.inference_mode():
                target = torch.nn.functional.linear(
                    x, module.weight_fp16(), module.bias)
                result = module(x)
                native = module.to_native_fp16()(x)
            torch.testing.assert_close(result, target, rtol=0.02, atol=0.02)
            torch.testing.assert_close(native, target, rtol=0.02, atol=0.02)


def test_safe_accept_rejects_new_disagreements_and_large_js():
    base = {"n": 4, "errors": 1, "error_mask": [False, True, False, False],
            "mean_js": 0.00001, "margin_erosion": 0.02}
    same = dict(base)
    assert safe_accept(same, base, base, limit_js=0.00010)
    introduced = dict(base, error_mask=[True, False, False, False])
    assert not safe_accept(introduced, base, base, limit_js=0.00010)
    drift = dict(base, mean_js=0.0005)
    assert not safe_accept(drift, base, base, limit_js=0.00010)
    eroded = dict(base, margin_erosion=0.5)
    assert not safe_accept(eroded, base, base, limit_js=0.00010)
    assert "error_mask" not in public_metrics(base)


def test_moments_and_teacher_decisions_cpu():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(5, 4)

        def forward(self, input_ids, marker_mask):
            return self.proj(input_ids.float())

    model = Toy()
    batches = [
        {"input_ids": torch.randn(3, 5),
         "marker_mask": torch.ones(3, 4, dtype=torch.bool)}
    ]
    moments = capture_input_moments(model, ["proj"], batches, stride=1)
    assert moments["proj"]["count"] == 3
    assert len(moments["proj"]["rms"]) == 5
    with torch.inference_mode():
        ref = [model(**batches[0]).detach()]
    self_metrics = collect_decisions(model, batches, ref)
    assert self_metrics["errors"] == 0
    assert self_metrics["agreement"] == 1.0
    assert self_metrics["mean_js"] < 1e-8
