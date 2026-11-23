"""CPU regression tests: forced all-block ternary {-1,0,+1} compression.

Actual SmolLM2 quality, GPU model speed and isolated runtime memory require
the V12 Colab. These tests make no low-bit acceleration claim.
"""
import torch
from torch import nn

from tinycenn_lm.smol_ternary_bruteforce_v12 import (
    paths_all,all_path_sets,activation_aware_codes,
    packed_from_activation_aware,swap,bytes_of_projection,count_codes,
)


class ToyBlock(nn.Module):
    def __init__(self,hidden=256):
        super().__init__()
        self.self_attn=nn.Module()
        for name in ("q_proj","k_proj","v_proj","o_proj"):
            setattr(self.self_attn,name,nn.Linear(hidden,hidden,bias=False))
        self.mlp=nn.Module()
        self.mlp.gate_proj=nn.Linear(hidden,hidden*2,bias=False)
        self.mlp.up_proj=nn.Linear(hidden,hidden*2,bias=False)
        self.mlp.down_proj=nn.Linear(hidden*2,hidden,bias=False)


class ToyLM(nn.Module):
    def __init__(self,depth=3):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([ToyBlock() for _ in range(depth)])


def test_bruteforce_paths_covers_all_seven_dense_projections_per_layer():
    model=ToyLM(depth=3)
    groups=paths_all(model)
    assert len(groups)==3
    assert all(len(g["attention"])==4 and len(g["mlp"])==3
               for g in groups.values())
    sets=all_path_sets(model)
    assert {k:len(v) for k,v in sets.items()}=={
        "attention_output_only":3,"attention_all":12,
        "mlp_all":9,"all_7_projections_every_layer":21}


def test_activation_aware_pack_enforces_strict_signed_ternary():
    torch.manual_seed(12)
    for inp,out in ((256,256),(512,128),(257,192)):
        dense=nn.Linear(inp,out,bias=True).half()
        moments=torch.linspace(.1,2.,steps=inp)
        codes,scales=activation_aware_codes(dense,moments)
        expected=((inp+127)//128)*128
        assert codes.shape==(out,expected)
        assert scales.shape==(out,expected//128)
        assert set(codes.unique().tolist()).issubset({-1,0,1})
        packed=packed_from_activation_aware(dense,moments,chunk_rows=48)
        stats=count_codes(packed)
        assert sum(stats.values())==codes.numel()
        assert packed.packed_weight.dtype==torch.uint8
        assert torch.isfinite(packed.group_scales.float()).all()
        assert bytes_of_projection(packed)<bytes_of_projection(dense)
        with torch.inference_mode():
            actual=packed(torch.randn(2,5,inp).half())
        assert tuple(actual.shape)==(2,5,out)
        assert torch.isfinite(actual).all()
        assert not hasattr(packed,"_cached")


def test_all_layers_replaced_and_restored_without_quality_gates():
    model=ToyLM(depth=3)
    sets=all_path_sets(model)
    original={}
    replacement={}
    for p in sets["all_7_projections_every_layer"]:
        old=model.get_submodule(p)
        original[p]=old
        replacement[p]=packed_from_activation_aware(
            old,torch.ones(old.in_features),chunk_rows=64)
    for p in replacement:
        assert swap(model,p,replacement[p]) is original[p]
    assert all(not isinstance(model.get_submodule(p),nn.Linear)
               for p in replacement)
    for p in sets["attention_all"]:
        assert swap(model,p,original[p]) is replacement[p]
    assert all(isinstance(model.get_submodule(p),nn.Linear)
               for p in sets["attention_all"])
    assert all(not isinstance(model.get_submodule(p),nn.Linear)
               for p in sets["mlp_all"])


def test_moments_change_the_ternary_optimization_weights():
    torch.manual_seed(42)
    linear=nn.Linear(256,128,bias=False).float()
    # Exact shape and numerical stability across extremely imbalanced inputs.
    for second in (torch.ones(256),torch.zeros(256),
                   torch.tensor([1e-9]*128+[1e4]*128)):
        codes,scales=activation_aware_codes(linear,second)
        assert torch.isfinite(scales).all()
        assert set(codes.unique().tolist()).issubset({-1,0,1})
