import pytest
import torch
from torch import nn
from scripts.benchmark_ccm_consensus_smol360 import (
    ConsensusAttentionAdapter, attention_slot, logits_positions, get_prompts,
)
from tinycenn_lm.competitive_consensus_v2 import ConsensusConfig, CompetitiveConsensusCeNN


class CountingNative(nn.Module):
    def __init__(self):super().__init__();self.count=0
    def forward(self,hidden_states,**kwargs):
        self.count+=1
        return hidden_states*0+5,None


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Module()]);
        self.model.layers[0].self_attn=CountingNative()


def test_replaces_native_without_calling_native_and_restores():
    model=ToyModel();native=model.model.layers[0].self_attn
    cell=CompetitiveConsensusCeNN(ConsensusConfig(hidden_size=24,slots=8,tag_width=8,cell_width=12))
    x=torch.randn(1,5,24)
    adapter=ConsensusAttentionAdapter(cell)
    with attention_slot(model,adapter):
        y,_=model.model.layers[0].self_attn(hidden_states=x,use_cache=False)
        torch.testing.assert_close(y,cell(x))
        assert native.count==0
        with pytest.raises(NotImplementedError): adapter(x,use_cache=True)
    assert model.model.layers[0].self_attn is native


def test_restores_on_exception():
    model=ToyModel();native=model.model.layers[0].self_attn
    with pytest.raises(RuntimeError):
        with attention_slot(model,nn.Identity()): raise RuntimeError('fail')
    assert model.model.layers[0].self_attn is native


def test_chat_prompt_contract_and_positions():
    assert logits_positions(128)[-1]==126
    assert len(logits_positions(128))>=8
    with pytest.raises(ValueError):get_prompts('/dev/null')
    assert len(get_prompts(None))>=4
