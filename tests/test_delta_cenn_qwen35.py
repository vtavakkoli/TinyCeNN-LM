"""CPU contract checks for Delta-CeNN and the Qwen3.5 linear-attention adapter."""
import sys
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tinycenn_lm.delta_cenn_qwen35 import DeltaCeNNRule,Qwen35DeltaCeNNLinear,qwen_l2norm


def tensors(t=8,heads=2,dk=4,dv=3):
    torch.manual_seed(4)
    return (torch.randn(1,t,heads,dk),torch.randn(1,t,heads,dk),
            torch.randn(1,t,heads,dv),-torch.rand(1,t,heads),
            torch.sigmoid(torch.randn(1,t,heads)))


def oracle(q,k,v,g,beta):
    q=qwen_l2norm(q)/q.shape[-1]**.5
    k=qwen_l2norm(k)
    mem=torch.zeros(q.shape[0],q.shape[2],q.shape[-1],v.shape[-1])
    y=[]
    for t in range(q.shape[1]):
        mem=mem*g[:,t].exp()[:,:,None,None]
        error=(v[:,t]-torch.einsum('bhf,bhfv->bhv',k[:,t],mem))*beta[:,t,: ,None]
        mem=mem+k[:,t,:,:,None]*error[:,:,None,:]
        y.append(torch.einsum('bhf,bhfv->bhv',q[:,t],mem))
    return torch.stack(y,1)


def test_exact_native_token_reference():
    q,k,v,g,b=tensors()
    core=DeltaCeNNRule(2,4,3,cells=False)
    assert torch.allclose(core(q,k,v,g,b),oracle(q,k,v,g,b),atol=2e-7)

@pytest.mark.parametrize('cells',[False,True])
def test_streaming_aligned_chunks(cells):
    q,k,v,g,b=tensors(t=13)
    core=DeltaCeNNRule(2,4,3,cells=cells,cell_interval=4)
    y,end=core(q,k,v,g,b,return_state=True)
    parts=[];state=None
    for start,stop in [(0,3),(3,8),(8,13)]:
        z,state=core(q[:,start:stop],k[:,start:stop],v[:,start:stop],
                     g[:,start:stop],b[:,start:stop],state,True)
        parts.append(z)
    assert torch.allclose(y,torch.cat(parts,1),atol=3e-7)
    assert state.position==end.position==13


def test_causality():
    q,k,v,g,b=tensors(t=9)
    core=DeltaCeNNRule(2,4,3,cells=True,cell_interval=2)
    y=core(q,k,v,g,b)
    v[:,6:]+=9
    z=core(q,k,v,g,b)
    assert torch.allclose(y[:,:6],z[:,:6],atol=1e-7)


def test_gradients_and_cell_rate():
    q,k,v,g,b=tensors(t=6)
    core=DeltaCeNNRule(2,4,3,cells=True,cell_interval=2)
    core(q,k,v,g,b).square().mean().backward()
    assert core.cell_strength_logits.grad is not None
    assert torch.isfinite(core.cell_strength_logits.grad).all()
    assert core.output_log_gain.grad.abs().sum()>0


def test_bad_inputs():
    q,k,v,g,b=tensors()
    core=DeltaCeNNRule(2,4,3)
    with pytest.raises(ValueError):core(q,k,v,g[:,:-1],b)
    with pytest.raises(ValueError):DeltaCeNNRule(2,4,3,feature_dim=5)
    with pytest.raises(ValueError):core(q,k,v,g,b,state=core.initial_state(2,q.device))


def test_compressed_features():
    q,k,v,g,b=tensors()
    core=DeltaCeNNRule(2,4,3,feature_dim=2,cells=True)
    assert core(q,k,v,g,b).shape==(1,8,2,3)


class FakeGatedNorm(nn.Module):
    def forward(self,x,z):
        return x*torch.sigmoid(z)


class FakeQwenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size=8;self.num_v_heads=2;self.num_k_heads=1
        self.head_k_dim=4;self.head_v_dim=4;self.key_dim=4;self.value_dim=8
        self.conv_dim=16;self.conv_kernel_size=3;self.activation='silu'
        self.layer_idx=1;self.layer_type='linear_attention'
        self.in_proj_qkv=nn.Linear(8,16,bias=False)
        self.in_proj_z=nn.Linear(8,8,bias=False)
        self.in_proj_b=nn.Linear(8,2,bias=False)
        self.in_proj_a=nn.Linear(8,2,bias=False)
        self.conv1d=nn.Conv1d(16,16,3,groups=16,bias=False,padding=2)
        self.A_log=nn.Parameter(torch.zeros(2))
        self.dt_bias=nn.Parameter(torch.ones(2))
        self.norm=FakeGatedNorm()
        self.out_proj=nn.Linear(8,8,bias=False)


def test_wrapper_forward_and_grad():
    m=FakeQwenLinear().requires_grad_(False)
    adapter=Qwen35DeltaCeNNLinear(m,cells=True,cell_interval=2)
    x=torch.randn(1,7,8)
    y=adapter(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert adapter.core.output_bias.grad is not None
    assert not any(p.grad is not None for p in m.parameters())


def test_wrapper_rejects_cache_or_padding():
    m=Qwen35DeltaCeNNLinear(FakeQwenLinear())
    with pytest.raises(RuntimeError):m(torch.randn(1,4,8),cache_params=object())
    with pytest.raises(ValueError):m(torch.randn(1,4,8),attention_mask=torch.tensor([[1,1,0,1]]))


def test_backend_oracle_rejected_for_cellular_or_compressed():
    native=FakeQwenLinear()
    with pytest.raises(ValueError,match='full-rank'):
        Qwen35DeltaCeNNLinear(native,cells=True,native_backend_oracle=True)
    with pytest.raises(ValueError,match='full-rank'):
        Qwen35DeltaCeNNLinear(native,feature_dim=2,cells=False,native_backend_oracle=True)


def test_native_backend_oracle_is_diagnostic_only(monkeypatch):
    # A stub Qwen backend lets this contract test run without Transformers or GPU.
    import types
    full='transformers.models.qwen3_5.modeling_qwen3_5'
    for package in ('transformers','transformers.models','transformers.models.qwen3_5'):
        monkeypatch.setitem(sys.modules,package,types.ModuleType(package))
    stub=types.ModuleType(full)
    observed=[]
    def chunk_backend(q,k,v,g,beta,output_final_state,use_qk_l2norm_in_kernel):
        observed.append((q.shape,output_final_state,use_qk_l2norm_in_kernel))
        return oracle(q.float(),k.float(),v.float(),g.float(),beta.float()).to(q.dtype),None
    stub.torch_chunk_gated_delta_rule=chunk_backend
    monkeypatch.setitem(sys.modules,full,stub)
    native=FakeQwenLinear().eval().requires_grad_(False)
    wrapper=Qwen35DeltaCeNNLinear(native,cells=False,native_backend_oracle=True).eval()
    student=Qwen35DeltaCeNNLinear(native,cells=False,native_backend_oracle=False).eval()
    x=torch.randn(1,5,8)
    with torch.no_grad():
        a=wrapper(x); b=student(x)
    assert len(observed)==1
    assert observed[0][1:] == (False,True)
    assert torch.allclose(a,b,atol=5e-6)
    assert wrapper.native_backend_oracle is True
    assert student.native_backend_oracle is False
