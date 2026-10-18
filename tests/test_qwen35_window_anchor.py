"""Small deterministic unit tests for the causal No-CeNN Qwen3.5 mixer."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tinycenn_lm.qwen35_window_anchor import causal_window_anchor
def test_no_future_leakage_and_shapes():
    torch.manual_seed(7)
    q = torch.randn(1, 2, 12, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    a = causal_window_anchor(q,k,v,None,window=3,anchors=1,chunk_size=4)
    v[:, :, 10:, :] += 1000.
    b = causal_window_anchor(q,k,v,None,window=3,anchors=1,chunk_size=4)
    assert a.shape == (1,2,12,8)
    torch.testing.assert_close(a[:,:,:10],b[:,:,:10],atol=1e-5,rtol=1e-5)

def test_anchors_preserve_prefix_tokens_outside_window():
    q=torch.ones((1,1,8,4))
    k=torch.zeros((1,1,8,4))
    v=torch.zeros((1,1,8,4))
    v[:,:,0,:]=100.
    with_anchor=causal_window_anchor(q,k,v,None,window=2,anchors=1,chunk_size=3)
    no_anchor=causal_window_anchor(q,k,v,None,window=2,anchors=0,chunk_size=3)
    assert with_anchor[0,0,7].mean()>20
    assert no_anchor[0,0,7].abs().max()<1e-6

def test_padding_is_ignored():
    q=torch.ones((1,1,5,4))
    k=torch.zeros((1,1,5,4))
    v=torch.ones((1,1,5,4))
    v[:,:,4,:]=9999
    valid=torch.tensor([[1,1,1,1,0]])
    result=causal_window_anchor(q,k,v,valid,window=5,anchors=1,chunk_size=2)
    assert torch.all(result[:,:,:4]<10)
