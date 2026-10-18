"""CPU smoke tests for SmolLM2 {-1,0,+1} CeNN replacement and gating.

These tests do not claim real SmolLM2 accuracy. Colab handles GPU/model tests.
"""
import torch
from torch import nn
from tinycenn_lm.smol_ternary_cenn_v10 import (
    SmolTernaryProjection, CeNNOnlyProjection,
    acceptance, public, candidate_names, projection_storage_bytes, replace,
)


def test_ternary_smol_projection_and_svd_rescue_shapes():
    torch.manual_seed(5)
    base=nn.Linear(256,128).eval()
    ternary=SmolTernaryProjection.from_linear(base,chunk_rows=32,rank=0)
    rescued=SmolTernaryProjection.from_linear(base,chunk_rows=32,rank=8)
    input=torch.randn(2,9,256)
    with torch.no_grad():
        a=ternary(input)
        b=rescued(input)
    assert a.shape==b.shape==(2,9,128)
    assert projection_storage_bytes(ternary)<projection_storage_bytes(base)
    assert projection_storage_bytes(rescued)<projection_storage_bytes(base)
    assert rescued.residual_left.shape==(128,8)
    assert rescued.residual_right.shape==(8,256)
    assert not hasattr(ternary.base,"_cached")


def test_cenn_only_is_causal_and_ternary():
    linear=nn.Linear(64,64)
    cell=CeNNOnlyProjection.from_linear(linear)
    codes=cell.stencil.unpack()
    assert set(codes.reshape(-1).tolist()).issubset({-1,0,1})
    assert torch.count_nonzero(codes[:,-1])==0, "Right-tap future leak"
    x=torch.randn(2,10,64)
    shifted=x.clone()
    shifted[:,7:,:]+=50
    with torch.no_grad():
        original=cell(x)
        perturbed=cell(shifted)
    torch.testing.assert_close(original[:,:7,:],perturbed[:,:7,:])


def test_ternary_selection_rejects_new_teacher_mistakes():
    base={"n":3,"errors":0,"wrong_mask":[False,False,False],
          "mean_js":0.0,"teacher_agreement":1.0}
    good=dict(base,mean_js=0.0004)
    assert acceptance(good,base,base,max_new_errors=0,max_js=.01)
    bad=dict(base,errors=1,wrong_mask=[True,False,False],
             teacher_agreement=2/3,mean_js=.0004)
    assert not acceptance(bad,base,base,max_new_errors=0,max_js=.01)
    assert not acceptance(dict(good,mean_js=.05),base,base,
                          max_new_errors=0,max_js=.01)
    assert "wrong_mask" not in public(good)


def test_candidate_site_paths_and_safe_replace_cpu():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp=nn.Module()
            self.self_attn=nn.Module()
            self.mlp.down_proj=nn.Linear(64,64)
            self.self_attn.o_proj=nn.Linear(64,64)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=nn.Module()
            self.model.layers=nn.ModuleList([Block(),Block(),Block()])
    model=Toy()
    names=candidate_names(model,last_layers=2)
    assert len(names)==4 and names[0].startswith("model.layers.1.")
    old=model.get_submodule(names[0])
    replacement=nn.Linear(64,64)
    assert replace(model,names[0],replacement) is old
    assert model.get_submodule(names[0]) is replacement
    replace(model,names[0],old)
