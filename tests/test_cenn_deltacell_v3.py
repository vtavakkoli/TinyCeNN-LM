"""Tests of CeNN-DeltaCell V3, independent of any pretrained weights."""
import copy
import sys
import types

import pytest
import torch
from torch import nn

from tinycenn_lm.cenn_deltacell_v3 import CeNNDeltaCellV3, Qwen35DeltaCellAttention


@pytest.mark.parametrize('mode', ['none','fixed','adaptive'])
@pytest.mark.parametrize('window', [0, 5])
def test_streaming_chunk_partition_invariance(mode, window):
    torch.manual_seed(6)
    layer = CeNNDeltaCellV3(6, 2, 8, feature_dim=5, chunk_size=7,
                            mode=mode, local_window=window)
    q = torch.randn(2, 6, 23, 8)
    k = torch.randn(2, 2, 23, 8)
    v = torch.randn(2, 2, 23, 8)
    full, whole_state = layer(q, k, v, return_state=True)
    pieces=[]; state=None
    for a,b in [(0,1),(1,3),(3,8),(8,16),(16,17),(17,23)]:
        y, state=layer(q[:,:,a:b],k[:,:,a:b],v[:,:,a:b],state,True)
        pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces,2),full,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(state.memory,whole_state.memory,atol=2e-5,rtol=2e-5)
    assert whole_state.position == 23 and state.position == 23
    assert whole_state.nbytes == layer.recurrent_state_bytes(batch_size=2)


@pytest.mark.parametrize('mode', ['none','fixed','adaptive'])
def test_future_causality(mode):
    torch.manual_seed(13)
    layer=CeNNDeltaCellV3(4,2,8,feature_dim=8,chunk_size=4,mode=mode,local_window=5)
    q,k,v=torch.randn(1,4,16,8),torch.randn(1,2,16,8),torch.randn(1,2,16,8)
    pred=layer(q,k,v)
    q[:,:,10:]+=99;k[:,:,10:]+=99;v[:,:,10:]+=99
    newer=layer(q,k,v)
    torch.testing.assert_close(pred[:,:,:10],newer[:,:,:10],atol=1e-5,rtol=1e-5)


def test_fixed_and_adaptive_can_represent_identical_diffusion():
    torch.manual_seed(17)
    a=CeNNDeltaCellV3(4,2,8,feature_dim=6,chunk_size=5,mode='adaptive',local_window=0)
    b=CeNNDeltaCellV3(4,2,8,feature_dim=6,chunk_size=5,mode='fixed',local_window=0)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.cell_logits[:]=torch.log(torch.tensor([.25,.5,.25]))
        a.cell_router.weight.zero_()
    q,k,v=torch.randn(1,4,14,8),torch.randn(1,2,14,8),torch.randn(1,2,14,8)
    torch.testing.assert_close(a(q,k,v),b(q,k,v),atol=2e-5,rtol=2e-5)


def test_adaptive_cell_transition_receives_gradients():
    torch.manual_seed(19)
    layer=CeNNDeltaCellV3(4,2,8,feature_dim=6,chunk_size=4,mode='adaptive',local_window=0)
    q,k,v=torch.randn(1,4,14,8),torch.randn(1,2,14,8),torch.randn(1,2,14,8)
    layer(q,k,v).square().mean().backward()
    assert layer.cell_router.weight.grad is not None
    assert layer.cell_router.weight.grad.abs().sum().item()>0
    for p in layer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_decode_step_and_state_memory_bytes():
    layer=CeNNDeltaCellV3(4,2,8,feature_dim=6,chunk_size=4,mode='adaptive',local_window=3)
    q,k,v=torch.randn(1,4,8,8),torch.randn(1,2,8,8),torch.randn(1,2,8,8)
    result=layer(q,k,v)
    state=None;parts=[]
    for i in range(8):
        o,state=layer(q[:,:,i:i+1],k[:,:,i:i+1],v[:,:,i:i+1],state,True)
        parts.append(o)
    torch.testing.assert_close(result,torch.cat(parts,dim=2),atol=1e-5,rtol=1e-5)
    assert state.nbytes==layer.recurrent_state_bytes(1)


def test_reject_invalid_config_and_state():
    with pytest.raises(ValueError): CeNNDeltaCellV3(3,2,8)
    with pytest.raises(ValueError): CeNNDeltaCellV3(4,2,8,chunk_size=64)
    with pytest.raises(ValueError): CeNNDeltaCellV3(4,2,8,mode='invalid')
    layer=CeNNDeltaCellV3(4,2,8)
    with pytest.raises(ValueError): layer(torch.randn(1,4,3,8),torch.randn(1,3,3,8),torch.randn(1,3,3,8))


def test_qwen_wrapper_contract_and_frozen_pretrained_projections(monkeypatch):
    # Stub rotary application only; this tests the wrapper's tensor contract.
    path='transformers.models.qwen3_5.modeling_qwen3_5'
    names=['transformers','transformers.models','transformers.models.qwen3_5',path]
    for name in names: monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    sys.modules[path].apply_rotary_pos_emb=lambda q,k,cos,sin:(q,k)
    class Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim=8;self.num_key_value_groups=2;self.layer_idx=11
            self.config=types.SimpleNamespace();self.scaling=8**(-.5)
            self.q_proj=nn.Linear(16,4*8*2,bias=False)
            self.k_proj=nn.Linear(16,2*8,bias=False)
            self.v_proj=nn.Linear(16,2*8,bias=False)
            self.o_proj=nn.Linear(4*8,16,bias=False)
            self.q_norm=nn.Identity();self.k_norm=nn.Identity()
    native=Native().eval().requires_grad_(False)
    core=CeNNDeltaCellV3(4,2,8,feature_dim=5,chunk_size=4,mode='adaptive')
    wrap=Qwen35DeltaCellAttention(native,core)
    x=torch.randn(2,9,16)
    emb=(torch.ones(2,9,8),torch.zeros(2,9,8))
    mask=torch.zeros(2,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),float('-inf'))
    y,weights=wrap(x,position_embeddings=emb,attention_mask=mask)
    assert y.shape==x.shape and weights is None
    (y.square().mean()).backward()
    assert core.log_gain.grad is not None
    assert all(p.grad is None for p in native.parameters())
    with pytest.raises(RuntimeError):
        wrap(x,position_embeddings=emb,past_key_values=object())
    with pytest.raises(ValueError):
        wrap(x,position_embeddings=emb,attention_mask=torch.zeros(2,1,9,10))
