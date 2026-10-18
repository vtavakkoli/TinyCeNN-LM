"""Targeted Qwen3.5 V2 tests; no model downloads or GPU required."""
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
from torch import nn
from tinycenn_lm.qwen35_window_anchor_v2 import (
    causal_window_anchor_biased, WindowAnchorDeltaV2,
)

def test_biased_window_anchor_no_future_leak():
    torch.manual_seed(103)
    q=torch.randn(1,2,16,8)
    k=torch.randn(1,2,16,8)
    v=torch.randn(1,2,16,8)
    score=torch.tensor([6.0,7.0])
    decay=torch.tensor([0.01,0.025])
    original=causal_window_anchor_biased(q,k,v,None,6,2,4,score,decay)
    v[:,:,12:]+=10000
    changed=causal_window_anchor_biased(q,k,v,None,6,2,4,score,decay)
    assert original.shape==(1,2,16,8)
    torch.testing.assert_close(original[:,:,:12],changed[:,:,:12],
                               rtol=1e-5,atol=1e-5)

def test_biased_window_anchor_gradient_to_temperature_and_decay():
    torch.manual_seed(104)
    q=torch.randn(1,2,13,8,requires_grad=True)
    k=torch.randn(1,2,13,8)
    v=torch.randn(1,2,13,8)
    logscore=nn.Parameter(torch.tensor([1.0,1.5]))
    logdecay=nn.Parameter(torch.tensor([-4.0,-3.5]))
    output=causal_window_anchor_biased(
        q,k,v,None,5,2,4,logscore.exp(),logdecay.exp())
    output.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert logscore.grad is not None and torch.isfinite(logscore.grad).all()
    assert logdecay.grad is not None and torch.isfinite(logdecay.grad).all()

class DummyGateNorm(nn.Module):
    def forward(self,x,z):
        return x*torch.sigmoid(z)

class FakeGatedDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx=0
        self.num_k_heads=2;self.num_v_heads=2
        self.head_k_dim=8;self.head_v_dim=8
        self.key_dim=16;self.value_dim=16
        self.conv1d=nn.Conv1d(48,48,kernel_size=3,groups=48,
                              padding=2,bias=False)
        self.in_proj_qkv=nn.Linear(16,48,bias=False)
        self.in_proj_z=nn.Linear(16,16,bias=False)
        self.in_proj_b=nn.Linear(16,2,bias=False)
        self.norm=DummyGateNorm()
        self.out_proj=nn.Linear(16,16,bias=False)
        self.activation="silu"

def test_delta_substitute_no_recurrent_cache_and_backward():
    torch.manual_seed(105)
    module=WindowAnchorDeltaV2(FakeGatedDelta(),window=8,
                               anchors=2,chunk_size=4)
    x=torch.randn(1,12,16,requires_grad=True)
    out=module(x,attention_mask=torch.ones(1,12,dtype=torch.long))
    assert out.shape==(1,12,16)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for param in (module.log_temperature,module.log_decay,
                  module.direct_mix_logit):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    with pytest.raises(RuntimeError,match="use_cache=False"):
        module(x,cache_params=object())
