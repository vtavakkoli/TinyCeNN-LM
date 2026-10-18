import pytest
import torch
import torch.nn.functional as F
from torch import nn
from tinycenn_lm.ccm_v3_rotary_consensus import RotaryConsensusConfig,RotaryCellularConsensus,rope

class NativeLike(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q_proj=nn.Linear(cfg.hidden_size,cfg.query_heads*cfg.head_dim,bias=False)
        self.k_proj=nn.Linear(cfg.hidden_size,cfg.key_value_heads*cfg.head_dim,bias=False)
        self.v_proj=nn.Linear(cfg.hidden_size,cfg.key_value_heads*cfg.head_dim,bias=False)
        self.o_proj=nn.Linear(cfg.query_heads*cfg.head_dim,cfg.hidden_size,bias=False)
        self.cfg=cfg
    def forward(self,x,rope_vals):
        c=self.cfg;b,t,d=x.shape
        q=self.q_proj(x).reshape(b,t,c.query_heads,c.head_dim).transpose(1,2)
        k=self.k_proj(x).reshape(b,t,c.key_value_heads,c.head_dim).transpose(1,2)
        v=self.v_proj(x).reshape(b,t,c.key_value_heads,c.head_dim).transpose(1,2)
        q,k=rope(q,k,*rope_vals)
        k=k.repeat_interleave(c.query_heads//c.key_value_heads,dim=1)
        v=v.repeat_interleave(c.query_heads//c.key_value_heads,dim=1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o_proj(y.transpose(1,2).reshape(b,t,c.query_heads*c.head_dim))

def cos_sin(b,t,head_dim):
    freq=torch.arange(head_dim//2).float()/head_dim
    a=torch.arange(t).float()[:,None]*freq[None,:]*3.11
    angles=torch.cat([a,a],-1).unsqueeze(0).expand(b,-1,-1)
    return angles.cos(),angles.sin()

@pytest.mark.parametrize('context,slots',[ (1,2),(7,8),(15,16),(24,32)])
def test_matches_native_causal_attention(context,slots):
    torch.manual_seed(3)
    cfg=RotaryConsensusConfig(hidden_size=24,query_heads=6,key_value_heads=2,head_dim=4,slots=slots,query_block=5,correction_rank=4)
    native=NativeLike(cfg)
    new=RotaryCellularConsensus.from_pretrained_attention(native,cfg)
    x=torch.randn(2,context,24)
    pe=cos_sin(2,context,4)
    y=new(x,position_embeddings=pe)[0]
    want=native(x,pe)
    torch.testing.assert_close(y,want,atol=4e-6,rtol=2e-5)


def test_old_random_weights_are_not_compatible():
    cfg=RotaryConsensusConfig(hidden_size=16,query_heads=4,key_value_heads=2,head_dim=4,slots=8)
    new=RotaryCellularConsensus(cfg)
    with pytest.raises(RuntimeError,match='initialized'):
        new(torch.randn(1,3,16),position_embeddings=cos_sin(1,3,4))


def test_window_changes_after_capacity():
    cfg=RotaryConsensusConfig(hidden_size=16,query_heads=4,key_value_heads=2,head_dim=4,slots=4)
    n=NativeLike(cfg);cell=RotaryCellularConsensus.from_pretrained_attention(n,cfg)
    x=torch.randn(1,7,16);pe=cos_sin(1,7,4)
    out=cell(x,position_embeddings=pe)[0]
    torch.testing.assert_close(out[:,:4],n(x[:,:4],(pe[0][:,:4],pe[1][:,:4])),atol=1e-5,rtol=1e-5)
    assert out.shape==(1,7,16)
    assert torch.isfinite(out).all()


def test_causality_and_grad_only_correction():
    cfg=RotaryConsensusConfig(hidden_size=16,query_heads=4,key_value_heads=2,head_dim=4,slots=8,correction_rank=4)
    n=NativeLike(cfg);cell=RotaryCellularConsensus.from_pretrained_attention(n,cfg)
    pe=cos_sin(1,6,4);x=torch.randn(1,6,16)
    y=cell(x,position_embeddings=pe)[0]
    z=x.clone();z[:,4:]=torch.randn_like(z[:,4:])*10
    y2=cell(z,position_embeddings=pe)[0]
    torch.testing.assert_close(y[:,:4],y2[:,:4],atol=1e-5,rtol=1e-5)
    (y.square().mean()).backward()
    assert cell.correct_up.weight.grad is not None
    assert all(p.grad is None for name,p in cell.named_parameters() if name.split('.')[0] in ('q_proj','k_proj','v_proj','o_proj'))


def test_require_position_embeddings_and_reject_cache():
    cfg=RotaryConsensusConfig(hidden_size=16,query_heads=4,key_value_heads=2,head_dim=4,slots=8)
    cell=RotaryCellularConsensus.from_pretrained_attention(NativeLike(cfg),cfg)
    x=torch.randn(1,3,16)
    with pytest.raises(ValueError,match='RoPE'): cell(x)
    with pytest.raises(NotImplementedError): cell(x,position_embeddings=cos_sin(1,3,4),use_cache=True)


def test_additive_mask_excludes_extra_positions():
    cfg=RotaryConsensusConfig(hidden_size=16,query_heads=4,key_value_heads=2,head_dim=4,slots=8,correction_rank=0)
    n=NativeLike(cfg);cell=RotaryCellularConsensus.from_pretrained_attention(n,cfg)
    x=torch.randn(1,6,16);pe=cos_sin(1,6,4)
    # Completely hide token 1 for later tokens, model mask [B,1,T,T].
    mask=torch.zeros(1,1,6,6);mask[:,:,:,1]=-1e9
    y=cell(x,position_embeddings=pe,attention_mask=mask)[0]
    q=n.q_proj(x).reshape(1,6,4,4).transpose(1,2)
    k=n.k_proj(x).reshape(1,6,2,4).transpose(1,2)
    v=n.v_proj(x).reshape(1,6,2,4).transpose(1,2)
    q,k=rope(q,k,*pe)
    want=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask,is_causal=True)
    want=n.o_proj(want.transpose(1,2).reshape(1,6,16))
    # Token 1 has its own self masked, so its output is arbitrary. Compare other steps.
    torch.testing.assert_close(y[:,[0,2,3,4,5]],want[:,[0,2,3,4,5]],atol=5e-6,rtol=5e-5)
