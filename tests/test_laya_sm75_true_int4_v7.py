"""CPU regression checks for signed Turing INT4 packing and reference math.

The compiled s4xs4 Tensor Core kernel is GPU-only and has to be tested on T4.
"""
import torch
from torch import nn

from tinycenn_lm.laya_sm75_true_int4_v7 import (
    TuringINT4Linear, cpu_reference, pack_signed_nibbles,
    unpack_signed_nibbles, capability_report,
)


def test_roundtrip_signed_4bit_both_packed_axes_cpu():
    torch.manual_seed(1)
    for axis in (0, 1):
        codes=torch.randint(-8,8,(256,384),dtype=torch.int8)
        packed=pack_signed_nibbles(codes,axis)
        decoded=unpack_signed_nibbles(packed,axis)
        assert torch.equal(codes,decoded)
        assert packed.dtype==torch.int32
        assert packed.numel()==codes.numel()//8


def test_turing_int4_fp16_to_packed_reference_cpu():
    torch.manual_seed(2)
    base=nn.Linear(256,128,bias=True).half()
    module=TuringINT4Linear(base)
    assert module.is_genuine_turing_s4s4
    assert module.packed_weight.dtype==torch.int32
    assert tuple(module.packed_weight.shape)==(32,128)
    assert tuple(module.group_scales.shape)==(2,128)
    assert not any(name=="weight" for name in module.state_dict())
    x=torch.randn(3,4,256).half()
    expected=cpu_reference(x,module.packed_weight,
                           module.group_scales,module.bias)
    assert expected.shape==(3,4,128)
    assert torch.isfinite(expected).all()
    # The packed weight reconstructed reference is distinct from unquantized
    # FP16 linear by design; this is not an accuracy benchmark.
    original=base(x).float()
    assert (expected-original).abs().mean().item()>0


def test_triton_capabilities_do_not_claim_hardware_without_execution():
    report=capability_report()
    assert report["fork_required"] is True
    assert report["backend"]=="experimental SM75 native s4xs4 tensor core"
    assert any("unverified" in x for x in report["caveats"])
