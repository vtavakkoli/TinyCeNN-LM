"""CPU tests for V5.5 chunk-state CeNN; monkeypatch backend so no HF download."""
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
# The test runner in Colab resolves the existing repo module; local prototype
# can also find previous published V5.4 reference under /mnt/data.
from tinycenn_lm.delta_cenn_qwen35_v55 import ChunkCellDeltaCeNNRule
import tinycenn_lm.delta_cenn_qwen35_v55 as v55
from tinycenn_lm.delta_cenn_qwen35 import qwen_l2norm


def reference_chunk(q,k,v,g,beta,initial_state,chunk_size):
    b,t,h,d=q.shape
    state=torch.zeros(b,h,d,v.shape[-1],dtype=torch.float32) if initial_state is None else initial_state
    q=qwen_l2norm(q.float())/(d**.5);k=qwen_l2norm(k.float());v=v.float()
    result=[]
    for t0 in range(t):
        state=state*g[:,t0].float().exp()[...,None,None]
        pred=torch.einsum('bhk,bhkv->bhv',k[:,t0],state)
        delta=(v[:,t0]-pred)*beta[:,t0].float()[...,None]
        state=state+k[:,t0][...,None]*delta[...,None,:]
        result.append(torch.einsum('bhk,bhkv->bhv',q[:,t0],state))
    return torch.stack(result,dim=1),state


@pytest.fixture(autouse=True)
def backend(monkeypatch):
    monkeypatch.setattr(v55,'qwen_chunk_backend',reference_chunk)


def tensors(T=13):
    torch.manual_seed(225)
    return (torch.randn(1,T,2,4),torch.randn(1,T,2,4),
            torch.randn(1,T,2,3),-torch.rand(1,T,2),
            torch.sigmoid(torch.randn(1,T,2)))


def test_no_cells_equivalent_to_reference():
    q,k,v,g,b=tensors()
    core=ChunkCellDeltaCeNNRule(2,4,3,cells=False)
    expected,_=reference_chunk(q,k,v,g,b,None,64)
    torch.testing.assert_close(core(q,k,v,g,b),expected,atol=2e-6,rtol=1e-6)
    assert core.config['engine']=='chunk_cell_v55'


@pytest.mark.parametrize('interval',[1,4,16])
def test_streaming_arbitrary_boundaries(interval):
    q,k,v,g,b=tensors(T=13)
    core=ChunkCellDeltaCeNNRule(2,4,3,cells=True,cell_interval=interval)
    whole,last=core(q,k,v,g,b,return_state=True)
    state=None;pieces=[]
    for a,z in [(0,3),(3,8),(8,13)]:
        p,state=core(q[:,a:z],k[:,a:z],v[:,a:z],g[:,a:z],b[:,a:z],state,True)
        pieces.append(p)
    torch.testing.assert_close(whole,torch.cat(pieces,dim=1),atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(state.memory,last.memory)
    assert state.position==13


def test_gradients_and_future_causality():
    q,k,v,g,b=tensors(T=9)
    core=ChunkCellDeltaCeNNRule(2,4,3,cells=True,cell_interval=3)
    core.cell_strength_logits.data.fill_(-1)
    original=core(q,k,v,g,b)
    v2=v.clone();v2[:,5:]+=4
    modified=core(q,k,v2,g,b)
    torch.testing.assert_close(original[:,:5],modified[:,:5],atol=1e-6,rtol=1e-6)
    original.square().sum().backward()
    assert core.cell_strength_logits.grad is not None
    assert torch.isfinite(core.cell_strength_logits.grad).all()


def test_reject_compressed_and_bad_chunk():
    with pytest.raises(ValueError):ChunkCellDeltaCeNNRule(2,4,3,feature_dim=3)
    with pytest.raises(ValueError):ChunkCellDeltaCeNNRule(2,4,3,backend_chunk_size=24)


def test_model_state_and_fp32():
    q,k,v,g,b=tensors(T=9)
    core=ChunkCellDeltaCeNNRule(2,4,3)
    y,state=core(q,k,v,g,b,return_state=True)
    assert y.shape==(1,9,2,3)
    assert state.memory.dtype==torch.float32
