import math
import torch
import pytest
import torch.nn.functional as F
from tinycenn_lm.competitive_consensus_v2 import ConsensusConfig, CompetitiveConsensusCeNN


def model(slots=8):
    return CompetitiveConsensusCeNN(ConsensusConfig(hidden_size=24,slots=slots,tag_width=8,cell_width=12,score_scale=2.0))


def test_consensus_matches_exact_bounded_softmax_reference():
    torch.manual_seed(2)
    m=model().eval()
    x=torch.randn(2,19,24)
    with torch.no_grad():
        actual=m(x)
        q,k,v=m._project(x)
        outputs=[]
        for t in range(x.shape[1]):
            start=max(0,t-m.config.slots+1)
            keys=F.normalize(k[:,start:t+1],dim=-1)
            qt=F.normalize(q[:,t:t+1],dim=-1)
            a=(qt*keys).sum(-1)*m.log_scale.exp()
            w=a.softmax(-1)
            outputs.append(m.out_proj((w[...,None]*v[:,start:t+1]).sum(1))*m.log_gain.exp())
        expected=torch.stack(outputs,dim=1)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-6)


def test_stream_chunks_equivalent():
    torch.manual_seed(3)
    m=model();x=torch.randn(2,27,24)
    single=m(x)
    state=None;chunks=[]
    for a,b in [(0,1),(1,5),(5,8),(8,17),(17,27)]:
        got,state=m(x[:,a:b],state=state,return_state=True);chunks.append(got)
    torch.testing.assert_close(single,torch.cat(chunks,dim=1),rtol=1e-5,atol=2e-6)
    assert state.tags.shape==(2,8,8)


def test_strict_causality():
    torch.manual_seed(4)
    m=model();a=torch.randn(1,20,24);b=a.clone();b[:,11:]+=10
    torch.testing.assert_close(m(a)[:,:11],m(b)[:,:11],rtol=1e-5,atol=1e-6)


def test_gradients_finite():
    m=model();x=torch.randn(2,8,24,requires_grad=True)
    loss=m(x).square().sum();loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_reject_bad_configuration():
    with pytest.raises(ValueError): ConsensusConfig(slots=7).validate()
    with pytest.raises(ValueError): model()(torch.randn(3,7,16))


def test_dtype_preserved_with_float32_parameters():
    m=model()
    x=torch.randn(1,8,24,dtype=torch.bfloat16)
    assert m(x).dtype==torch.bfloat16
