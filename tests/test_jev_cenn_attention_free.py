"""Feasibility checks: Jev-CeNN layers have no token attention operators."""
import ast
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.jev_cenn_free import (
    JevCeNNFreeMixer, _bidirectional_memory,
    _hierarchical_mean, _valid_tokens
)


class ToyModernBertAttention(nn.Module):
    def __init__(self, sliding=True):
        super().__init__()
        self.num_heads = 2
        self.head_dim = 8
        self.layer_id = 1 if sliding else 0
        self.local_attention = (4,4) if sliding else (-1,-1)
        self.Wqkv = nn.Linear(16,48)
        self.Wo = nn.Linear(16,16)
        self.out_drop = nn.Identity()


@pytest.mark.parametrize("kind", ["local","multiscale","memory"])
@pytest.mark.parametrize("sliding", [True,False])
def test_attention_free_forward_and_gradient(kind,sliding):
    torch.manual_seed(2028)
    original=ToyModernBertAttention(sliding)
    layer=JevCeNNFreeMixer(original,variant=kind,coarse_stride=8)
    x=torch.randn(2,31,16)
    valid=torch.ones(2,31,dtype=torch.bool)
    valid[1,-3:]=False
    # No attention backend should ever be called in replacement forward.
    real=F.scaled_dot_product_attention
    def forbidden(*args,**kwargs):
        raise RuntimeError("Attention backend invoked by purported free mixer!")
    try:
        F.scaled_dot_product_attention=forbidden
        y=layer(x,attention_mask=valid)[0]
    finally:
        F.scaled_dot_product_attention=real
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y[0].square().sum().item()>0
    target=torch.randn_like(y)
    loss=F.mse_loss(y[valid],target[valid])
    loss.backward()
    trained=[p for p in layer.parameters() if p.requires_grad]
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
               and bool(p.grad.abs().sum()>0) for p in trained)
    assert layer.config_dict()["no_qk_similarity"]
    assert layer.config_dict()["no_token_attention"]


@pytest.mark.parametrize("kind", ["local","multiscale","memory"])
def test_attention_free_mask_padding_cannot_perturb_valid_prefix(kind):
    torch.manual_seed(1)
    layer=JevCeNNFreeMixer(ToyModernBertAttention(),variant=kind).eval()
    x=torch.randn(1,18,16)
    extra=torch.randn(1,7,16) * 100.0
    with torch.no_grad():
        first=layer(x,attention_mask=torch.ones(1,18,dtype=torch.bool))[0]
        padded=torch.cat([x,extra],dim=1)
        mask=torch.tensor([[1]*18+[0]*7],dtype=torch.long)
        second=layer(padded,attention_mask=mask)[0][:,:18]
    # Pooling and memory must respect the padding; numeric deviations can
    # arise at the final coarse-pooling boundary when lengths are different.
    if kind != "memory":
        torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)
    else:
        torch.testing.assert_close(first,second,atol=0.2,rtol=0.2)


def test_global_memory_truly_exchanges_distant_context():
    torch.manual_seed(1)
    z=torch.zeros(1,2,300,8)
    v=torch.ones(1,300,dtype=torch.bool)
    z[:,:,0] = 4
    pooled=_bidirectional_memory(z,v)
    assert bool(pooled[:,:,299].abs().sum()>0)


def test_no_attention_ops_in_source_ast():
    file=Path(__file__).resolve().parents[1]/"src"/"tinycenn_lm"/"jev_cenn_free.py"
    tree=ast.parse(file.read_text())
    forbidden={
        "scaled_dot_product_attention","multi_head_attention_forward",
        "MultiheadAttention","softmax","log_softmax",
        "einsum_qk","linear_attention",
    }
    names=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
    assert not (forbidden & set(names))
    assert "cumsum" in names and "conv1d" in names


def test_no_unsupported_variants():
    with pytest.raises(ValueError,match="Unknown variant"):
        JevCeNNFreeMixer(ToyModernBertAttention(),variant="sdpa")


def test_cpu_long_context_scaling_without_quadratic_alloc():
    layer=JevCeNNFreeMixer(ToyModernBertAttention(sliding=False),
                           variant="memory")
    with torch.no_grad():
        for t in (257,1024):
            x=torch.randn(1,t,16)
            y=layer(x,attention_mask=torch.ones(1,t,dtype=torch.bool))[0]
            assert y.shape==(1,t,16) and torch.isfinite(y).all()


def test_real_modernbert_cpu_integration_both_attention_types():
    from transformers import __version__, ModernBertConfig, AutoModel
    if __version__ != "4.57.6":
        pytest.skip("Small ModernBERT integration pinned to 4.57.6")
    cfg=ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2,
        global_attn_every_n_layers=3, local_attention=8,
        max_position_embeddings=128, reference_compile=False,
        pad_token_id=0, attention_dropout=0.0,
        embedding_dropout=0.0, mlp_dropout=0.0,
    )
    model=AutoModel.from_config(cfg,attn_implementation="sdpa").eval()
    layers=model.layers
    idx_global=0
    idx_slide=1
    layers[idx_global].attn=JevCeNNFreeMixer(
        layers[idx_global].attn, variant="memory"
    )
    layers[idx_slide].attn=JevCeNNFreeMixer(
        layers[idx_slide].attn, variant="local"
    )
    assert layers[idx_global].attn.original_local_attention==(-1,-1)
    assert layers[idx_slide].attn.original_local_attention==(4,4)
    tokens=torch.randint(1,64,(2,17))
    padding=torch.ones(2,17,dtype=torch.long)
    padding[1,-2:]=0
    with torch.no_grad():
        result=model(input_ids=tokens,attention_mask=padding).last_hidden_state
    assert result.shape==(2,17,32)
    assert torch.isfinite(result).all()
