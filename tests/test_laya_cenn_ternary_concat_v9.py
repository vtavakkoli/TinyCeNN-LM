"""CPU tests for strict CeNN ternary {-1,0,+1} and concat memory ablations.

CUDA peak-memory/latency must be measured by the separate V9 Colab.
"""
import torch
from torch import nn

from tinycenn_lm.laya_cenn_ternary_concat_v9 import (
    PackedTernaryLinear, TernaryCeNNStencil,
    pack_ternary, unpack_ternary, serialized_bytes, theoretical_pack_bytes,
)


def test_pack_exact_values_and_density():
    for rows,cols in ((1,4),(5,128),(16,256)):
        q=torch.randint(-1,2,(rows,cols),dtype=torch.int8)
        packed=pack_ternary(q)
        assert packed.dtype==torch.uint8
        assert packed.shape==(rows,cols//4)
        assert torch.equal(unpack_ternary(packed,cols),q)
        assert packed.numel()*4==q.numel()
    try:
        pack_ternary(torch.tensor([[-2,0,1,1]],dtype=torch.int8))
        assert False,"Invalid out-of-range codes accepted"
    except ValueError:
        pass


def test_linear_concat_and_prealloc_numerical_parity_no_dense_cache():
    torch.manual_seed(3)
    dense=nn.Linear(256,96,bias=True).float().eval()
    tiled=PackedTernaryLinear.from_linear(dense,chunk_rows=32,mode="prealloc")
    concat=PackedTernaryLinear.from_linear(dense,chunk_rows=32,mode="cat")
    x=torch.randn(2,13,256,dtype=torch.float32)
    with torch.inference_mode():
        a=tiled(x)
        b=concat(x)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(tiled.packed_weight,concat.packed_weight)
    assert not hasattr(tiled,"_cached")
    assert not any(v.dtype.is_floating_point and v.numel()==256*96
                   for v in tiled.state_dict().values())
    assert serialized_bytes(tiled)<serialized_bytes(dense)*.20
    assert tiled.packed_weight.numel()==96*(256//4)


def test_cellular_branch_concatenation_is_not_weight_compression():
    block=TernaryCeNNStencil([
        (-1,0,1),(1,-1,0),(0,1,-1),(1,0,0)
    ],[1.0,0.5,0.5,1.0])
    weights=block.unpack()
    assert torch.all((weights>=-1)&(weights<=1))
    assert torch.equal(weights[0],torch.tensor([-1,0,1],dtype=torch.int8))
    assert block.packed_templates.shape==(4,1)
    assert serialized_bytes(block)==4+4*2
    x=torch.randn(3,17,128)
    with torch.inference_mode():
        ysum=block(x,"sum")
        ycat=block(x,"concat")
    torch.testing.assert_close(ysum,ycat,rtol=0,atol=1e-5)
    assert ycat.shape==x.shape


def test_estimate_excludes_activations_and_padded_scales():
    result=theoretical_pack_bytes([(2304,768),(3072,768),(768,3072)])
    assert result["fp16_weight_bytes"]>result["ternary_weight_and_scale_bytes"]
    assert .85 < result["saving_fraction"] < .88
    assert result["ternary_weight_and_scale_bytes"]>0
