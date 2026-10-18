"""CPU-only regression suite for SmolLM2 V11 packed ternary QAT and CeNN.

CUDA-performance, WikiText fidelity, and full causal-LM checkpoints are
validated by the V11 Colab. These tests make NO hardware speed claims.
"""
import torch
from torch import nn
from tinycenn_lm.smol_ternary_cenn_v11 import (
    TernaryQATLinear,PackedCeNNLinear,feature_cenn,
    create_from_metadata,is_acceptable,projection_bytes,candidate_paths,swap)


def test_ste_gradients_codes_group_scales_and_pack_roundtrip():
    torch.manual_seed(3)
    baseline=nn.Linear(256,128,bias=False).float()
    qat=TernaryQATLinear(baseline,rank=0,feature_residual=True)
    x=torch.randn(2,11,256)
    target=baseline(x).detach()
    loss=(qat(x)-target).float().square().mean()
    loss.backward()
    assert qat.codes.grad is not None
    assert qat.log_scales.grad is not None
    assert torch.isfinite(qat.codes.grad).all()
    assert torch.isfinite(qat.log_scales.grad).all()
    packed=qat.pack(chunk_rows=32)
    assert packed.base.packed_weight.dtype==torch.uint8
    assert packed.base.packed_weight.numel()==128*256//4
    assert not hasattr(packed.base,"_cached")
    assert projection_bytes(packed)<projection_bytes(baseline)
    with torch.no_grad():
        actual=packed(x)
        expected=qat(x)
    torch.testing.assert_close(actual,expected,rtol=0.004,atol=0.004)


def test_packed_module_keeps_autograd_to_earlier_qat_layer():
    torch.manual_seed(7)
    a=nn.Linear(256,256,bias=False)
    b=nn.Linear(256,128,bias=False)
    previous=TernaryQATLinear(a,feature_residual=False)
    later=TernaryQATLinear(b,feature_residual=False).pack(64)
    z=torch.randn(4,256)
    loss=later(previous(z)).float().square().mean()
    loss.backward()
    assert previous.codes.grad is not None
    assert previous.codes.grad.abs().sum()>0


def test_reloaded_inference_state_does_not_retain_master_weight():
    base=nn.Linear(256,128,bias=False)
    trained=TernaryQATLinear(base,rank=0,feature_residual=True)
    packed=trained.pack(chunk_rows=32)
    metadata={"rank":0,"feature_residual":True,"chunk_rows":32}
    loaded=create_from_metadata(base,metadata,device="cpu")
    loaded.load_state_dict(packed.state_dict(),strict=True)
    inp=torch.randn(2,5,256)
    with torch.no_grad():
        torch.testing.assert_close(loaded(inp),packed(inp),rtol=0,atol=0)
    assert "codes" not in loaded.state_dict()
    assert "log_scales" not in loaded.state_dict()


def test_cenn_feature_stencil_never_looks_at_future_token():
    torch.manual_seed(9)
    x=torch.randn(2,8,32)
    changed=x.clone()
    changed[:,5:,:]+=100
    torch.testing.assert_close(feature_cenn(x)[:,:5],
                               feature_cenn(changed)[:,:5],rtol=0,atol=0)
    torch.testing.assert_close(feature_cenn(x)[...,:1],x[...,:1])


def test_selection_gate_and_complete_candidate_paths():
    dev={"n":384,"errors":2,"mean_js":.0003}
    guard={"n":384,"errors":3,"mean_js":.0004}
    assert is_acceptable(dev,guard,20.,20.15,
                         max_errors=4,max_js=.004,
                         max_ppl_increase=.015)
    assert not is_acceptable(dev,guard,20.,20.6,
                             max_errors=4,max_js=.004,
                             max_ppl_increase=.015)
    assert not is_acceptable(dict(dev,errors=5),guard,20.,20.15,
                             max_errors=4,max_js=.004)
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=nn.Module()
            self.self_attn.o_proj=nn.Linear(16,16)
            self.mlp=nn.Module()
            self.mlp.down_proj=nn.Linear(48,16)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=nn.Module()
            self.model.layers=nn.ModuleList([Block() for _ in range(4)])
    model=Model()
    names=candidate_paths(model,mlp_last=2,attention_all=True)
    assert len(names)==6
    before=model.get_submodule(names[0])
    new=nn.Linear(16,16)
    assert swap(model,names[0],new) is before
    assert swap(model,names[0],before) is new
