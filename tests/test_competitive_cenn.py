"""Unit tests for an attention-free *module* with attention-like cellular retrieval."""
import torch
import torch.nn.functional as F
import pytest
from tinycenn_lm.competitive_cenn import CCMConfig,CompetitiveCeNN
from scripts.attention_concentration_smol360 import analyze_probability_value


def test_streaming_parity():
    torch.manual_seed(7)
    m=CompetitiveCeNN(CCMConfig(hidden_size=12,slots=8,tag_width=8,cell_width=9)).eval()
    x=torch.randn(2,23,12)
    full=m(x)
    st=None;pieces=[]
    for a,b in [(0,1),(1,3),(3,7),(7,16),(16,23)]:
        y,st=m(x[:,a:b],st,return_state=True);pieces.append(y)
    torch.testing.assert_close(full,torch.cat(pieces,1),atol=1e-5,rtol=1e-5)
    assert st.position==23


def test_causality_and_gradients():
    torch.manual_seed(2)
    m=CompetitiveCeNN(CCMConfig(hidden_size=12,slots=8,tag_width=8,cell_width=9))
    x=torch.randn(2,12,12,requires_grad=True)
    y=m(x)
    later=x.detach().clone();later[:,6:]+=2.0
    torch.testing.assert_close(y[:,:6],m(later)[:,:6],atol=1e-6,rtol=1e-6)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert not any(isinstance(z,torch.nn.MultiheadAttention) for z in m.modules())


def test_shape_error_and_synthetic_attention_oracle():
    with pytest.raises(ValueError):CCMConfig(slots=15).validate()
    torch.manual_seed(3)
    B,H,T,D=2,3,18,7
    logits=torch.randn(B,H,T,T)
    causal=torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)
    probs=torch.softmax(logits.masked_fill(causal,-1e9),dim=-1)
    values=torch.randn(B,H,T,D)
    z=analyze_probability_value(probs,values,window=5)
    assert z['topk']['1']['mean_probability_mass']<=z['topk']['4']['mean_probability_mass']
    assert z['topk']['4']['output_relative_mse']>=0
    assert 0<z['effective_context_size_mean']<=T
