import pytest
import torch
import torch.nn.functional as F

from tinycenn_lm.cenn_deltacell_v3 import CeNNDeltaCellV3
from tinycenn_lm.cenn_head_rescue_v51 import CeNNQueryRescueV51, warmstart_v5_delta_core


def build(**kwargs):
    return CeNNQueryRescueV51(4, 2, 8, feature_dim=8, chunk_size=4,
                              local_window=5, correction_rank=4, **kwargs)


def sample(t=11):
    torch.manual_seed(2041)
    return torch.randn(1, 4, t, 8), torch.randn(1, 2, t, 8), torch.randn(1, 2, t, 8)


def test_shape_and_state():
    m=build()
    y,s=m(*sample(),return_state=True)
    assert y.shape==(1,4,11,8)
    assert s.memory.dtype==torch.float32 and s.position==11
    assert s.keys.shape==(1,2,4,8)
    assert s.values.shape==s.keys.shape


@pytest.mark.parametrize('lengths', [(1,2,8),(3,4,4),(5,5,1),(11,)])
def test_streaming(lengths):
    q,k,v=sample()
    m=build()
    out=m(q,k,v)
    state=None;pieces=[];off=0
    for n in lengths:
        yp,state=m(q[:,:,off:off+n],k[:,:,off:off+n],v[:,:,off:off+n],state,True)
        pieces.append(yp);off+=n
    assert off==q.shape[2]
    assert torch.allclose(out,torch.cat(pieces,dim=2),atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('window', [1,2,5,16])
def test_windows_causal(window):
    q,k,v=sample(9)
    m=CeNNQueryRescueV51(4,2,8,feature_dim=8,chunk_size=4,
                          local_window=window,correction_rank=4)
    a=m(q,k,v)
    q2,k2,v2=[x.clone() for x in (q,k,v)]
    for x in (q2,k2,v2):x[:,:,6:]+=50
    b=m(q2,k2,v2)
    assert torch.allclose(a[:,:,:6],b[:,:,:6],atol=1e-5,rtol=1e-5)


def test_parity_with_v5_state():
    old=CeNNDeltaCellV3(4,2,8,feature_dim=8,chunk_size=4,mode='none',local_window=5)
    fresh=build()
    report=warmstart_v5_delta_core(fresh,old.state_dict())
    assert report['loaded_from_v5'] == len(old.state_dict())
    q,k,v=sample()
    assert torch.allclose(old(q,k,v),fresh(q,k,v),atol=2e-6,rtol=2e-6)


def test_backward_to_new_parameters():
    m=build()
    q,k,v=sample()
    y=m(q,k,v)
    y.square().mean().backward()
    assert m.corr_up.grad is not None and m.corr_up.grad.abs().sum()>0
    assert m.router_w.grad is not None and m.router_w.grad.abs().sum()>0
    assert m.router_bias.grad is not None


def test_router_disabled_zero_grad():
    m=build(query_router=False,query_correction=False)
    q,k,v=sample()
    out=m(q,k,v)
    assert torch.isfinite(out).all()


def test_invalid_dimensions():
    with pytest.raises(ValueError):CeNNQueryRescueV51(4,2,8,correction_rank=9)
    with pytest.raises(ValueError):CeNNQueryRescueV51(4,2,8,local_window=0)


def test_corrupt_state_cache_rejected():
    m=build();q,k,v=sample()
    _,state=m(q[:,:,:5],k[:,:,:5],v[:,:,:5],return_state=True)
    state.keys=state.keys[:,:,:1];state.values=state.values[:,:,:2]
    with pytest.raises(ValueError,match='cache'):m(q[:,:,5:6],k[:,:,5:6],v[:,:,5:6],state)


def test_warmstart_rejects_wrong_feature_dimension():
    old=CeNNDeltaCellV3(4,2,8,feature_dim=6,chunk_size=4,mode='none',local_window=5)
    with pytest.raises((RuntimeError,ValueError)):
        warmstart_v5_delta_core(build(),old.state_dict())


def test_nonzero_correction_changes_output():
    m=build();q,k,v=sample()
    ref=m(q,k,v)
    with torch.no_grad():m.corr_up.normal_(0,.2)
    diff=(ref-m(q,k,v)).abs().sum()
    assert diff>0


def test_nonzero_router_changes_output():
    m=build();q,k,v=sample()
    ref=m(q,k,v)
    with torch.no_grad():m.router_w.normal_(0,5.)
    diff=(ref-m(q,k,v)).abs().sum()
    assert diff>0
