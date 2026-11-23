"""Window-anchor V3 CPU regression suite. No HF model download or GPU."""
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from torch import nn
from tinycenn_lm.qwen35_window_anchor_v3 import (
    causal_window_anchor_biased,WindowAnchorDeltaV2
)

@pytest.mark.parametrize("length,window,anchors",[
    (16,8,2),(64,32,6),(128,64,16),(192,128,16)])
def test_fused_matches_chunk_reference(length,window,anchors):
    torch.manual_seed(613+length)
    q=torch.randn(1,4,length,16)
    k=torch.randn(1,4,length,16)
    v=torch.randn(1,4,length,16)
    score=torch.tensor([4.,5.,8.,11.])
    decay=torch.tensor([.01,.02,.04,.005])
    a=causal_window_anchor_biased(
        q,k,v,None,window,anchors,24,score,decay,dense_threshold=512)
    b=causal_window_anchor_biased(
        q,k,v,None,window,anchors,24,score,decay,dense_threshold=0)
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)

def test_fused_no_future_leak_and_trains_bias_parameters():
    torch.manual_seed(671)
    q=torch.randn(1,2,64,8,requires_grad=True)
    k=torch.randn(1,2,64,8)
    v=torch.randn(1,2,64,8)
    log_temp=nn.Parameter(torch.tensor([1.5,2.0]))
    log_decay=nn.Parameter(torch.tensor([-4.5,-3.3]))
    out=causal_window_anchor_biased(
        q,k,v,None,32,4,16,log_temp.exp(),log_decay.exp())
    perturbed=v.detach().clone()
    perturbed[:,:,40:]+=1000.
    alt=causal_window_anchor_biased(
        q,k,perturbed,None,32,4,16,log_temp.exp(),log_decay.exp())
    torch.testing.assert_close(out[:,:,:40],alt[:,:,:40],rtol=1e-5,atol=1e-5)
    out.square().mean().backward()
    for p in (q,log_temp,log_decay):
        assert p.grad is not None and torch.isfinite(p.grad).all()

class DummyNorm(nn.Module):
    def forward(self,x,z):
        return x*torch.sigmoid(z)

class FakeDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx=0
        self.num_k_heads=2;self.num_v_heads=2
        self.head_k_dim=8;self.head_v_dim=8
        self.key_dim=16;self.value_dim=16
        self.conv1d=nn.Conv1d(48,48,3,padding=2,groups=48,bias=False)
        self.in_proj_qkv=nn.Linear(16,48,bias=False)
        self.in_proj_z=nn.Linear(16,16,bias=False)
        self.in_proj_b=nn.Linear(16,2,bias=False)
        self.norm=DummyNorm()
        self.out_proj=nn.Linear(16,16,bias=False)
        self.activation="silu"

def test_v3_delta_output_calibration_has_grad_and_no_cache():
    torch.manual_seed(697)
    module=WindowAnchorDeltaV2(FakeDelta(),window=8,anchors=2,chunk_size=4)
    x=torch.randn(1,16,16,requires_grad=True)
    y=module(x,attention_mask=torch.ones((1,16),dtype=torch.long))
    assert y.shape==(1,16,16)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    for p in (module.log_temperature,module.log_decay,module.direct_mix_logit,
              module.output_gain,module.output_bias):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    with pytest.raises(RuntimeError,match="use_cache=False"):
        module(x,cache_params=object())
