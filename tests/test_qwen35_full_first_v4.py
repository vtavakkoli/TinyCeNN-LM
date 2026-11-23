"""Qwen V4 staged-loader tests; no model downloads required."""
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from torch import nn
from tinycenn_lm.qwen35_window_anchor_v4 import (
    install_selected,stage_indices,WindowAnchorDeltaV2,WindowAnchorFull)

class ToyDelta(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.layer_idx=i
        self.num_k_heads=2;self.num_v_heads=2
        self.head_k_dim=8;self.head_v_dim=8
        self.key_dim=16;self.value_dim=16
        self.in_proj_qkv=nn.Linear(16,48,bias=False)
        self.in_proj_z=nn.Linear(16,16,bias=False)
        self.in_proj_b=nn.Linear(16,2,bias=False)
        self.out_proj=nn.Linear(16,16,bias=False)
        self.conv1d=nn.Conv1d(48,48,3,padding=2,groups=48,bias=False)
        self.norm=nn.LayerNorm(8)
        self.activation="silu"

class ToyFull(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.layer_idx=i;self.head_dim=8;self.num_key_value_groups=2
        self.q_proj=nn.Linear(16,32,bias=False)
        self.k_proj=nn.Linear(16,8,bias=False)
        self.v_proj=nn.Linear(16,8,bias=False)
        self.o_proj=nn.Linear(16,16,bias=False)
        self.q_norm=nn.LayerNorm(8);self.k_norm=nn.LayerNorm(8)

class ToyLayer(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.block_type="full_attention" if i%4==3 else "linear_attention"
        if self.block_type=="full_attention":self.self_attn=ToyFull(i)
        else:self.linear_attn=ToyDelta(i)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(ToyLayer(i) for i in range(24))

def test_full_first_keeps_all_native_delta():
    m=ToyModel()
    full=[3,7,11,15,19,23]
    counts=install_selected(m,full_indices=full)
    assert counts=={"full":full,"delta":[]}
    assert all(isinstance(m.model.layers[i].self_attn,WindowAnchorFull)
               for i in full)
    assert all(isinstance(m.model.layers[i].linear_attn,ToyDelta)
               for i in range(24) if i not in full)

def test_progressive_delta_only_changes_requested_layers():
    m=ToyModel()
    full=[3,7,11,15,19,23]
    install_selected(m,full_indices=full)
    install_selected(m,delta_indices=[0,4])
    assert stage_indices(m)=={"full":full,"delta":[0,4]}
    assert isinstance(m.model.layers[0].linear_attn,WindowAnchorDeltaV2)
    assert isinstance(m.model.layers[4].linear_attn,WindowAnchorDeltaV2)
    assert isinstance(m.model.layers[1].linear_attn,ToyDelta)
    assert install_selected(m,full_indices=full,delta_indices=[0,4])=={
        "full":full,"delta":[0,4]}

def test_reject_full_index_as_delta():
    m=ToyModel()
    with pytest.raises(ValueError):
        install_selected(m,delta_indices=[3])
