"""CPU regression tests for modernbert SlidingWindowFusion."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.sliding_window_fusion import (
    SlidingWindowFusion, local_sliding_sdpa
)


@pytest.mark.parametrize("length", [1, 7, 13, 37, 64, 65, 130])
@pytest.mark.parametrize("radius", [0, 2, 4, 16])
def test_block_sliding_attention_matches_dense_reference(length, radius):
    torch.manual_seed(42)
    b, h, d = 2, 3, 8
    q, k, v = (torch.randn(b, h, length, d) for _ in range(3))
    valid = torch.ones(b, length, dtype=torch.bool)
    if length > 2:
        valid[1, -2:] = False
    qi = torch.arange(length)[:, None]
    ki = torch.arange(length)[None, :]
    allowed = ((qi-ki).abs() <= radius)[None,None] & valid[:,None,None,:]
    dense = F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed, dropout_p=0.0
    ) * valid[:,None,:,None]
    sparse = local_sliding_sdpa(q, k, v, valid, radius=radius, block_size=11)
    torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=1e-5)


class _IdentityRotary(nn.Module):
    def forward(self, qkv, position_ids):
        b,t,_,_,d = qkv.shape
        return (torch.ones(b,t,d,device=qkv.device,dtype=qkv.dtype),
                torch.zeros(b,t,d,device=qkv.device,dtype=qkv.dtype))


class _ReferenceSliding(nn.Module):
    def __init__(self, radius=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, num_attention_heads=2)
        self.layer_id = 1
        self.num_heads = 2
        self.head_dim = 8
        self.local_attention = (radius,radius)
        self.Wqkv = nn.Linear(16,48,bias=False)
        self.Wo = nn.Linear(16,16,bias=False)
        self.out_drop = nn.Identity()
        self.rotary_emb = _IdentityRotary()

    def forward(self, x, attention_mask=None, sliding_window_mask=None,
                position_ids=None, **kwargs):
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
        b,t,_ = x.shape
        qkv = self.Wqkv(x).reshape(b,t,3,2,8)
        q,k,v = qkv.transpose(3,1).unbind(2)
        cos,sin = self.rotary_emb(qkv,position_ids)
        q,k = apply_rotary_pos_emb(q,k,cos,sin)
        z = F.scaled_dot_product_attention(q,k,v,attn_mask=sliding_window_mask)
        return (self.out_drop(self.Wo(z.transpose(1,2).reshape(b,t,16))),)


def test_teacher_radius_reproduces_native_sliding():
    torch.manual_seed(123)
    teacher = _ReferenceSliding(radius=4).eval()
    replacement = SlidingWindowFusion(teacher,radius=4,block_size=8).eval()
    b,t=2,19
    valid=torch.ones(b,t,dtype=torch.bool)
    valid[1,-3:]=False
    qi=torch.arange(t)[:,None]
    ki=torch.arange(t)[None,:]
    allowed = (qi-ki).abs()<=4
    mask = torch.zeros(b,1,t,t)
    mask.masked_fill_(~allowed[None,None],torch.finfo(torch.float32).min)
    mask.masked_fill_(~valid[:,None,None,:],torch.finfo(torch.float32).min)
    x=torch.randn(b,t,16)
    pos=torch.arange(t)[None].expand(b,-1)
    with torch.no_grad():
        expected=teacher(x,attention_mask=mask,sliding_window_mask=mask,position_ids=pos)[0]
        actual=replacement(x,attention_mask=mask,sliding_window_mask=mask,position_ids=pos)[0]
    cosine=F.cosine_similarity(expected[valid],actual[valid],dim=-1).mean().item()
    assert cosine >= 0.99, cosine
    assert replacement.config_dict()["window_positions"]==9
    assert replacement.config_dict()["attention_free"] is False


def test_reject_global_or_excessive_radius():
    teacher = _ReferenceSliding(radius=4)
    with pytest.raises(ValueError,match="exceeds"):
        SlidingWindowFusion(teacher,radius=6)
    teacher.local_attention=(-1,-1)
    with pytest.raises(ValueError,match="sliding attention only"):
        SlidingWindowFusion(teacher,radius=4)
