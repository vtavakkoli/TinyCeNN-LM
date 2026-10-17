"""Deterministic CPU contract checks for the V5.2 extension."""
import pytest
import torch

from tinycenn_lm.cenn_head_rescue_v51 import CeNNQueryRescueV51
from tinycenn_lm.cenn_head_rescue_v52 import CeNNHeadRescueV52, warmstart_v51_to_v52


def inputs(t=13):
    torch.manual_seed(52)
    return (torch.randn(1, 4, t, 8), torch.randn(1, 2, t, 8),
            torch.randn(1, 2, t, 8))


def make(cls=CeNNHeadRescueV52, **kwargs):
    if cls is CeNNHeadRescueV52:
        kwargs.setdefault("post_rank", 4)
    return cls(4, 2, 8, feature_dim=7, chunk_size=4, local_window=6,
               correction_rank=4, **kwargs)


@pytest.mark.parametrize('flags', [
    dict(query_router=False, query_correction=False),
    dict(query_router=True, query_correction=False),
    dict(query_router=True, query_correction=True),
])
def test_parity_at_initialization(flags):
    old=make(CeNNQueryRescueV51,**flags)
    new=make(**flags)
    report=warmstart_v51_to_v52(new,old.state_dict())
    assert report['loaded_parameters'] == len(old.state_dict())
    assert torch.allclose(old(*inputs()),new(*inputs()),rtol=0,atol=1e-6)


@pytest.mark.parametrize('lengths', [(1,3,9),(4,4,5),(13,)])
def test_streaming_parity(lengths):
    m=make();q,k,v=inputs()
    whole=m(q,k,v)
    i=0;state=None;pieces=[]
    for n in lengths:
        x,state=m(q[:,:,i:i+n],k[:,:,i:i+n],v[:,:,i:i+n],state,True)
        pieces.append(x);i+=n
    assert i==13
    assert torch.allclose(whole,torch.cat(pieces,dim=2),rtol=2e-5,atol=2e-5)


def test_causality():
    m=make();q,k,v=inputs(13)
    before=m(q,k,v)
    q2,k2,v2=[x.clone() for x in (q,k,v)]
    for tensor in (q2,k2,v2):tensor[:,:,7:]+=30
    after=m(q2,k2,v2)
    assert torch.allclose(before[:,:,:7],after[:,:,:7],atol=1e-5,rtol=1e-5)


def test_post_gradients_and_disabled_mode():
    m=make()
    y=m(*inputs())
    y.square().mean().backward()
    assert m.post_up.grad is not None and m.post_up.grad.abs().sum().item()>0
    m_disabled=make(post_correction=False)
    old=make(CeNNQueryRescueV51)
    warmstart_v51_to_v52(m_disabled,old.state_dict())
    assert torch.allclose(m_disabled(*inputs()),old(*inputs()),atol=1e-6)


def test_strict_warmstart_rejects_bad_state():
    m=make();old=make(CeNNQueryRescueV51).state_dict()
    with pytest.raises(ValueError): warmstart_v51_to_v52(m,{k:v for k,v in old.items() if k!='wq'})


def test_invalid_rank():
    with pytest.raises(ValueError, match='post_rank'):
        make(post_rank=9)