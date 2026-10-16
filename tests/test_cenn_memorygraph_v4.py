"""CeNN-MemoryGraph V4 CPU numerical and contract tests (no model download)."""
from __future__ import annotations

import math
import sys
import types

import pytest
import torch
from torch import nn

from tinycenn_lm.cenn_deltacell_v3 import CeNNDeltaCellV3, Qwen35DeltaCellAttention
from tinycenn_lm.cenn_memorygraph_v4 import CeNNMemoryGraphV4


def toy(mode='learned', adaptive=True, router=True, window=5):
    return CeNNMemoryGraphV4(
        4, 2, 8, feature_dim=7, chunk_size=5, local_window=window,
        graph_mode=mode, top_k=3, edge_dim=5,
        content_adaptive=adaptive, query_router=router if window else False,
    )


def inputs(n=17):
    torch.manual_seed(31)
    return (torch.randn(2, 4, n, 8), torch.randn(2, 2, n, 8),
            torch.randn(2, 2, n, 8))


@pytest.mark.parametrize('graph_mode', ['none', 'learned', 'shuffled'])
@pytest.mark.parametrize('router', [False, True])
def test_chunk_streaming_equivalence(graph_mode, router):
    torch.manual_seed(1)
    layer = toy(graph_mode, adaptive=True, router=router)
    q,k,v=inputs()
    full, final=layer(q,k,v,return_state=True)
    pieces=[];state=None
    for lo,hi in ((0,1),(1,5),(5,6),(6,12),(12,13),(13,17)):
        y,state=layer(q[:,:,lo:hi],k[:,:,lo:hi],v[:,:,lo:hi],state,True)
        pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces,2),full,rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(state.memory,final.memory,rtol=2e-5,atol=2e-5)
    assert state.position == 17
    assert final.nbytes == layer.recurrent_state_bytes(batch_size=2)


@pytest.mark.parametrize('mode', ['none','learned','shuffled'])
def test_strict_causality(mode):
    layer=toy(mode)
    q,k,v=inputs(n=21)
    old=layer(q,k,v)
    q[:,:,11:]+=100.;k[:,:,11:]+=100.;v[:,:,11:]+=100.
    new=layer(q,k,v)
    torch.testing.assert_close(old[:,:,:11],new[:,:,:11],atol=1e-5,rtol=1e-5)


def test_v3_delta_local32_exact_control():
    torch.manual_seed(1)
    ref=CeNNDeltaCellV3(4,2,8,feature_dim=7,chunk_size=5,mode='none',local_window=5)
    candidate=toy('none',adaptive=False,router=False)
    missing, unexpected=candidate.load_state_dict(ref.state_dict(),strict=False)
    assert all(x.startswith(('edge_','log_basis','log_content','graph_strength','cell_rate','query_local_router')) or x=='shuffle_perm' for x in missing),missing
    assert not unexpected
    q,k,v=inputs()
    torch.testing.assert_close(ref(q,k,v),candidate(q,k,v),atol=1e-6,rtol=1e-6)


def test_shuffling_edges_changes_transition():
    torch.manual_seed(3)
    a=toy('learned')
    b=toy('shuffled')
    b.load_state_dict(a.state_dict())
    m=torch.randn(2,2,7,8)
    with torch.no_grad():
        a.graph_strength_logits.fill_(2.)
        b.graph_strength_logits.fill_(2.)
    assert not torch.allclose(a.diffuse(m),b.diffuse(m),atol=1e-5)


def test_neighbor_validity_and_sparsity():
    x=toy()
    idx=x.selected_neighbors()
    assert idx.shape==(2,7,3)
    assert bool(((idx>=0)&(idx<7)).all())
    assert bool((idx != torch.arange(7)[None,:,None]).all())
    for h in range(2):
        for f in range(7):
            assert idx[h,f].unique().numel()==3


def test_gradients_router_graph_cells():
    torch.manual_seed(11)
    x=toy('learned',adaptive=True,router=True)
    q,k,v=inputs(n=12)
    result=x(q,k,v)
    result.square().mean().backward()
    for name in ['edge_q','edge_k','graph_strength_logits',
                 'cell_rate.weight','query_local_router','wq','wk']:
        p=dict(x.named_parameters())[name]
        assert p.grad is not None, name
        assert bool(torch.isfinite(p.grad).all()), name
        assert p.grad.abs().sum().item()>0,name


def test_per_token_router_differs_across_tokens():
    x=toy('learned',adaptive=True,router=True)
    with torch.no_grad():
        x.query_local_router.normal_(std=3.)
    q,k,v=inputs(n=9)
    y=x(q,k,v)
    y_fixed=toy('learned',adaptive=True,router=False)
    y_fixed.load_state_dict(x.state_dict())
    assert y.shape==y_fixed(q,k,v).shape
    assert not torch.allclose(y,y_fixed(q,k,v),atol=1e-6)


def test_graph_memory_is_convex_transition_for_zero_rate():
    a=toy('learned')
    b=torch.randn(1,2,7,8)
    with torch.no_grad():
        a.graph_strength_logits.fill_(-25.)
        a.cell_rate.weight.zero_()
    torch.testing.assert_close(a.diffuse(b),b,rtol=1e-5,atol=1e-5)


def test_mask_and_cache_contract(monkeypatch):
    modname='transformers.models.qwen3_5.modeling_qwen3_5'
    for name in ['transformers','transformers.models','transformers.models.qwen3_5',modname]:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    sys.modules[modname].apply_rotary_pos_emb=lambda q,k,cos,sin:(q,k)
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
    native=Native().requires_grad_(False)
    a=Qwen35DeltaCellAttention(native,toy())
    z=torch.randn(2,9,16)
    pos=(torch.ones(2,9,8),torch.zeros(2,9,8))
    mask=torch.zeros(2,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),float('-inf'))
    o,_=a(z,position_embeddings=pos,attention_mask=mask)
    assert o.shape==z.shape
    o.square().mean().backward()
    assert a.core.query_local_router.grad is not None
    assert all(p.grad is None for p in native.parameters())
    with pytest.raises(RuntimeError): a(z,position_embeddings=pos,past_key_values=object())
    with pytest.raises(ValueError): a(z,position_embeddings=pos,attention_mask=torch.zeros(2,1,9,8))


def test_invalid_configs():
    for kwargs in [{'top_k':0},{'top_k':7},{'edge_dim':0},{'graph_mode':'invalid'},
                   {'local_window':0,'query_router':True}]:
        with pytest.raises(ValueError):
            CeNNMemoryGraphV4(4,2,8,feature_dim=7,chunk_size=5,**kwargs)