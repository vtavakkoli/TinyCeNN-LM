"""No transformers dependency required for CPU parity of the V5 adapter contract."""
import sys
import types
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# In this local CPU test environment Hugging Face may be absent. The reference
# RoPE below reproduces the function called by Qwen3.5 (zero rotation case).
mod=types.ModuleType('transformers.models.qwen3_5.modeling_qwen3_5')
def fake_rope(q,k,cos,sin):
    def rot(x):
        h=x.shape[-1]//2
        return torch.cat((-x[...,h:],x[...,:h]),dim=-1)
    c,s=cos.unsqueeze(1),sin.unsqueeze(1)
    return q*c+rot(q)*s, k*c+rot(k)*s
mod.apply_rotary_pos_emb=fake_rope

@pytest.fixture(autouse=True)
def isolated_qwen_rope(monkeypatch):
    # Scope the lightweight reference to these tests. A collection-time stub
    # hides the real Qwen model classes from unrelated integration tests.
    monkeypatch.setitem(sys.modules, 'transformers.models.qwen3_5.modeling_qwen3_5', mod)

from tinycenn_lm.cenn_deltacell_v3 import CeNNDeltaCellV3
from tinycenn_lm.cenn_memorygraph_v4 import CeNNMemoryGraphV4
from tinycenn_lm.cenn_head_rescue_v5 import Qwen35HeadRescueV5

class FakeNative(nn.Module):
    def __init__(self,heads=4,kvheads=2,hd=8):
        super().__init__()
        dim=heads*hd
        self.head_dim=hd
        self.num_key_value_groups=heads//kvheads
        self.q_proj=nn.Linear(dim,dim*2,bias=False)
        self.k_proj=nn.Linear(dim,kvheads*hd,bias=False)
        self.v_proj=nn.Linear(dim,kvheads*hd,bias=False)
        self.o_proj=nn.Linear(dim,dim,bias=False)
        self.q_norm=nn.Identity()
        self.k_norm=nn.Identity()
        self.scaling=hd**-0.5
        self.config={'test':True}
        self.layer_idx=11

def rope(b,t,d):
    return torch.ones(b,t,d),torch.zeros(b,t,d)

def manual_native(o,x,pos):
    b,t,_=x.shape;hd=o.head_dim
    q,gate=torch.chunk(o.q_proj(x).reshape(b,t,-1,hd*2),2,dim=-1)
    gate=gate.reshape(b,t,-1)
    q=o.q_norm(q).transpose(1,2)
    k=o.k_norm(o.k_proj(x).reshape(b,t,-1,hd)).transpose(1,2)
    v=o.v_proj(x).reshape(b,t,-1,hd).transpose(1,2)
    q,k=fake_rope(q,k,*pos)
    y=F.scaled_dot_product_attention(q,k.repeat_interleave(o.num_key_value_groups,1),
                                      v.repeat_interleave(o.num_key_value_groups,1),is_causal=True,scale=o.scaling)
    return o.o_proj(y.transpose(1,2).reshape(b,t,-1)*gate.sigmoid())

def make(corekind='delta',exact=(),window=4):
    native=FakeNative()
    remaining=2-len(exact)
    if remaining:
        opts=dict(num_heads=remaining*2,num_kv_heads=remaining,head_dim=8,
                  feature_dim=7,chunk_size=4,local_window=window)
        if corekind=='delta':core=CeNNDeltaCellV3(**opts,mode='none')
        else: core=CeNNMemoryGraphV4(**opts,graph_mode='learned',top_k=2,edge_dim=4)
    else:core=None
    return Qwen35HeadRescueV5(native,core,exact)

@pytest.mark.parametrize('length',[1,2,8,17])
def test_exact_parity(length):
    torch.manual_seed(21)
    adapter=make(exact=(0,1))
    x=torch.randn(2,length,32)
    p=rope(2,length,8)
    actual=adapter(x,position_embeddings=p)[0]
    expected=manual_native(adapter.original,x,p)
    assert torch.allclose(actual,expected,rtol=1e-5,atol=1e-6),float((actual-expected).abs().max())

@pytest.mark.parametrize('exact',[(),(0,),(1,),(0,1)])
def test_output_shapes_finite(exact):
    adapter=make(exact=exact)
    x=torch.randn(2,11,32)
    y,w=adapter(x,position_embeddings=rope(2,11,8))
    assert y.shape==x.shape and w is None and torch.isfinite(y).all()
    assert adapter.num_exact_heads==2*len(exact)

@pytest.mark.parametrize('family',['delta','graph'])
@pytest.mark.parametrize('exact',[(0,),(1,)])
def test_core_gradients(family,exact):
    torch.manual_seed(8)
    adapter=make(corekind=family,exact=exact)
    # freeze pretrained Q/K/V as in Colab
    adapter.original.requires_grad_(False)
    x=torch.randn(2,8,32)
    y=adapter(x,position_embeddings=rope(2,8,8))[0]
    y.square().mean().backward()
    assert adapter.core.wq.grad is not None
    assert torch.isfinite(adapter.core.wq.grad).all()

@pytest.mark.parametrize('exact',[(),(0,),(1,),(0,1)])
def test_causality(exact):
    torch.manual_seed(8)
    adapter=make(exact=exact).eval()
    x=torch.randn(1,11,32)
    second=x.clone(); second[:,8:]+=42
    a=adapter(x,position_embeddings=rope(1,11,8))[0]
    b=adapter(second,position_embeddings=rope(1,11,8))[0]
    assert torch.allclose(a[:,:8],b[:,:8],rtol=1e-5,atol=1e-6)

def test_cache_rejected():
    adapter=make(exact=(0,))
    with pytest.raises(RuntimeError,match='prefill-only'):
        adapter(torch.randn(1,4,32),position_embeddings=rope(1,4,8),past_key_values=object())

def test_invalid_group_core():
    with pytest.raises(ValueError):make(exact=(2,))
    with pytest.raises(ValueError):Qwen35HeadRescueV5(FakeNative(),None,exact_groups=(0,))

@pytest.mark.parametrize('family',['delta','graph'])
def test_native_causal_4d_mask(family):
    adapter=make(corekind=family,exact=(0,))
    x=torch.randn(1,7,32)
    triu=torch.full((7,7),float('-inf')).triu(1)
    attnmask=triu[None,None]
    y=adapter(x,position_embeddings=rope(1,7,8),attention_mask=attnmask)[0]
    assert torch.isfinite(y).all()

@pytest.mark.parametrize('exact', [(), (0,), (1,), (0,1)])
def test_additive_causal_mask_matches_implicit_causality(exact):
    torch.manual_seed(99)
    adapter=make(exact=exact).eval()
    x=torch.randn(2,9,32)
    t=x.shape[1]
    causal=torch.full((t,t),float('-inf')).triu(1)[None,None]
    embeddings=rope(2,t,8)
    with torch.no_grad():
        implicit=adapter(x,position_embeddings=embeddings)[0]
        explicit=adapter(x,position_embeddings=embeddings,attention_mask=causal)[0]
    assert torch.allclose(explicit,implicit,rtol=1e-5,atol=1e-6)


def test_rejects_noncausal_mask():
    adapter=make(exact=(0,))
    x=torch.randn(1,8,32)
    invalid=torch.zeros((1,1,8,8))
    with pytest.raises(ValueError,match='future|Future|Noncausal'):
        adapter(x,position_embeddings=rope(1,8,8),attention_mask=invalid)


def test_rejects_masked_past_token():
    adapter=make(exact=(1,))
    x=torch.randn(1,8,32)
    invalid=torch.full((8,8),float('-inf')).triu(1)[None,None]
    invalid[...,4,2]=float('-inf')
    with pytest.raises(ValueError,match='Noncausal|padded'):
        adapter(x,position_embeddings=rope(1,8,8),attention_mask=invalid)
