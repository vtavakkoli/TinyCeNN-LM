"""CPU tests of V5.3's functional numerical contract."""
import pytest
import torch

from tinycenn_lm.cenn_head_rescue_v52 import CeNNHeadRescueV52
from tinycenn_lm.cenn_head_rescue_v53 import CeNNHeadRescueV53, warmstart_v52_to_v53


def cfg():
    return dict(num_heads=4,num_kv_heads=2,head_dim=8,feature_dim=7,
                chunk_size=4,local_window=6,correction_rank=4,post_rank=4)


def inputs(t=13):
    torch.manual_seed(53)
    return (torch.randn(1,4,t,8),torch.randn(1,2,t,8),torch.randn(1,2,t,8))


@pytest.mark.parametrize('cross,calibration',[(True,True),(False,True),(True,False),(False,False)])
def test_exact_warmstart_parity(cross,calibration):
    torch.manual_seed(11)
    base=CeNNHeadRescueV52(**cfg())
    upgraded=CeNNHeadRescueV53(**cfg(),cross_rank=4,cross_correction=cross,
                               output_calibration=calibration)
    report=warmstart_v52_to_v53(upgraded,base.state_dict())
    assert report['loaded_keys'] == len(base.state_dict())
    x=inputs()
    assert torch.allclose(base(*x),upgraded(*x),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('partitions',[[1,2,10],[4,4,5],[13]])
def test_streaming_equivalence(partitions):
    core=CeNNHeadRescueV53(**cfg(),cross_rank=4)
    q,k,v=inputs()
    y=core(q,k,v)
    state=None;start=0;pieces=[]
    for t in partitions:
        p,state=core(q[:,:,start:start+t],k[:,:,start:start+t],v[:,:,start:start+t],state,True)
        pieces.append(p);start+=t
    assert start==13
    assert torch.allclose(y,torch.cat(pieces,dim=2),atol=2e-5,rtol=2e-5)


def test_causal_prefix():
    core=CeNNHeadRescueV53(**cfg(),cross_rank=4)
    q,k,v=inputs()
    before=core(q,k,v)
    modified=[x.clone() for x in (q,k,v)]
    for t in modified:t[:,:,7:]+=9.0
    after=core(*modified)
    assert torch.allclose(before[:,:,:7],after[:,:,:7],atol=2e-5,rtol=2e-5)


def test_new_gradients():
    core=CeNNHeadRescueV53(**cfg(),cross_rank=4)
    out=core(*inputs())
    out.square().mean().backward()
    assert core.cross_up.grad is not None and core.cross_up.grad.abs().sum().item()>0
    assert core.output_log_scale.grad is not None
    assert core.output_bias.grad is not None


def test_warmstart_incompatible_checkpoint():
    core=CeNNHeadRescueV53(**cfg(),cross_rank=4)
    bad=CeNNHeadRescueV52(**cfg()).state_dict()
    del bad['wk']
    with pytest.raises(ValueError):warmstart_v52_to_v53(core,bad)


def test_bad_rank():
    with pytest.raises(ValueError,match='cross_rank'):
        CeNNHeadRescueV53(**cfg(),cross_rank=9)


def test_nonzero_cross_changes_output():
    core=CeNNHeadRescueV53(**cfg(),cross_rank=4)
    x=inputs()
    old=core(*x)
    with torch.no_grad():
        core.cross_up.fill_(.1)
        core.output_log_scale.fill_(.01)
    after=core(*x)
    assert not torch.allclose(old,after)
