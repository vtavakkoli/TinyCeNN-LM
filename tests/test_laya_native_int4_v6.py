"""CPU-only safety checks for native INT4 qualification. GPU kernels run in Colab."""
from torch import nn

from tinycenn_lm.laya_native_int4_v6 import (
    speed_gate, candidate_names, swap_module,
)


def test_speed_gate_requires_both_real_encoder_sizes():
    rows=[
        dict(m=128,speedup=5.0,int4_ms=0.1),
        dict(m=512,speedup=1.23,int4_ms=0.8),
        dict(m=2048,speedup=1.12,int4_ms=2.0),
    ]
    assert speed_gate(rows,min_gain=1.10,target_m=(512,2048))
    assert not speed_gate(rows,min_gain=1.15,target_m=(512,2048))
    assert not speed_gate(rows[:2],min_gain=1.10,target_m=(512,2048))
    # Enormous single-token speedup never substitutes for prefill speed.
    assert not speed_gate([dict(m=1,speedup=100.,int4_ms=0.01)])


def test_swap_and_candidate_names_no_cuda():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wqkv=nn.Linear(8,24)
            self.Wo=nn.Linear(8,8)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wi=nn.Linear(8,32)
            self.Wo=nn.Linear(32,8)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn=Attn()
            self.mlp=MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Module()
            self.encoder.layers=nn.ModuleList([Block(),Block()])

    model=Model()
    names=candidate_names(model)
    assert len(names)==8
    assert names[0].endswith(("mlp.Wi","mlp.Wo"))
    location="encoder.layers.0.attn.Wo"
    before=model.get_submodule(location)
    replacement=nn.Linear(8,8)
    old=swap_module(model,location,replacement)
    assert old is before
    assert model.get_submodule(location) is replacement
    swap_module(model,location,old)
    assert model.get_submodule(location) is before
