"""Fast CPU-only safety tests for prebuilt 4-bit Laya V8 research runtime.

The W4A16 packed CUDA GEMM must be validated separately on Tesla T4.
No bitsandbytes import or CUDA is needed for these arithmetic safety checks.
"""
from torch import nn

from tinycenn_lm.laya_prebuilt_bnb_int4_v8 import (
    speed_gate, locate_candidates, replace_module,
)


def test_real_encoder_speed_gate_requires_two_workload_sizes():
    rows=[
        {"m":128, "speedup":2.2, "p10_speedup":1.4},
        {"m":512, "speedup":1.08, "p10_speedup":1.02},
        {"m":2048,"speedup":1.07,"p10_speedup":1.01},
    ]
    assert speed_gate(rows,min_speed=1.06)
    assert not speed_gate(rows,min_speed=1.09)
    assert not speed_gate(rows[:2],min_speed=1.06)
    slow_tail=[dict(x) for x in rows]
    slow_tail[-1]["p10_speedup"]=0.95
    assert not speed_gate(slow_tail,min_speed=1.06)
    assert not speed_gate(
        [{"m":1,"speedup":100.,"p10_speedup":10.}])


def test_model_module_selection_and_revert_cpu():
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn=nn.Module()
            self.mlp=nn.Module()
            self.attn.Wqkv=nn.Linear(8,24)
            self.attn.Wo=nn.Linear(8,8)
            self.mlp.Wi=nn.Linear(8,32)
            self.mlp.Wo=nn.Linear(32,8)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Module()
            self.encoder.layers=nn.ModuleList([TinyBlock(),TinyBlock()])

    model=TinyModel()
    names=locate_candidates(model)
    assert len(names)==8
    assert names[0].endswith(("mlp.Wi","mlp.Wo"))
    name="encoder.layers.0.attn.Wo"
    original=model.get_submodule(name)
    replacement=nn.Linear(8,8)
    assert replace_module(model,name,replacement) is original
    assert model.get_submodule(name) is replacement
    assert replace_module(model,name,original) is replacement
