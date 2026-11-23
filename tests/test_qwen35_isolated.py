"""Qwen3.5 V4 isolated substitution tests (small CPU models, no HF weights)."""
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from torch import nn
from types import SimpleNamespace
from tinycenn_lm.qwen35_isolated import layer_ids,describe,install_selected,substitute
from tinycenn_lm.qwen35_window_anchor_v3 import WindowAnchorDeltaV2,WindowAnchorFull

class TinyNorm(nn.Module):
    def forward(self,x,z=None):
        return x if z is None else x*torch.sigmoid(z)

class NativeDelta(nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.layer_idx=idx
        self.num_k_heads=2;self.num_v_heads=2
        self.head_k_dim=8;self.head_v_dim=8
        self.key_dim=16;self.value_dim=16
        self.conv1d=nn.Conv1d(48,48,3,groups=48,padding=2,bias=False)
        self.in_proj_qkv=nn.Linear(16,48,bias=False)
        self.in_proj_z=nn.Linear(16,16,bias=False)
        self.in_proj_b=nn.Linear(16,2,bias=False)
        self.norm=TinyNorm()
        self.out_proj=nn.Linear(16,16,bias=False)
        self.activation="silu"

class NativeFull(nn.Module):
    def __init__(self,idx):
        super().__init__()
        self.layer_idx=idx;self.head_dim=8;self.num_key_value_groups=1
        self.q_proj=nn.Linear(16,32,bias=False)
        self.k_proj=nn.Linear(16,16,bias=False)
        self.v_proj=nn.Linear(16,16,bias=False)
        self.o_proj=nn.Linear(16,16,bias=False)
        self.q_norm=nn.Identity();self.k_norm=nn.Identity()

class Layer(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.block_type="full_attention" if i%4==3 else "linear_attention"
        if self.block_type=="full_attention":self.self_attn=NativeFull(i)
        else:self.linear_attn=NativeDelta(i)

class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([Layer(i) for i in range(24)])

def test_delta_only_keeps_all_six_full_native():
    model=TinyText()
    original_full={i:id(model.model.layers[i].self_attn)
                   for i in layer_ids(model,"full")}
    selected=layer_ids(model,"delta")[:3]
    got=install_selected(model,"delta",selected,delta_window=8,
                         full_window=32,anchors=2,chunk_size=4)
    assert got["replaced_delta"]==3 and got["native_delta"]==15
    assert got["replaced_full"]==0 and got["native_full"]==6
    assert all(id(model.model.layers[i].self_attn)==before
               for i,before in original_full.items())
    for i in selected:assert isinstance(model.model.layers[i].linear_attn,WindowAnchorDeltaV2)

def test_full_only_keeps_all_18_delta_native():
    model=TinyText()
    original_delta={i:id(model.model.layers[i].linear_attn)
                    for i in layer_ids(model,"delta")}
    selected=layer_ids(model,"full")
    got=install_selected(model,"full",selected,delta_window=8,
                         full_window=32,anchors=2,chunk_size=4)
    assert got["replaced_delta"]==0 and got["native_delta"]==18
    assert got["replaced_full"]==6 and got["native_full"]==0
    assert all(id(model.model.layers[i].linear_attn)==before
               for i,before in original_delta.items())
    for i in selected:assert isinstance(model.model.layers[i].self_attn,WindowAnchorFull)

def test_wrong_type_and_duplicate_layers_rejected():
    m=TinyText()
    with pytest.raises(ValueError):
        install_selected(m,"delta",[3],delta_window=8,full_window=32)
    with pytest.raises(ValueError):
        install_selected(m,"full",[3,3],delta_window=8,full_window=32)
