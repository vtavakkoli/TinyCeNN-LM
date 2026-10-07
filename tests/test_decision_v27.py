import torch
from transformers import AutoModel, ModernBertConfig

from tinycenn_lm.decision_v27 import (
    SlidingAttentionCeNN,
    make_v27_replacement,
)
from tinycenn_lm.laya_lab.pdelta import PDelta3GDN2CLVRAttention


torch.set_num_threads(1)


def encoder():
    cfg = ModernBertConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        global_attn_every_n_layers=3,
        local_attention=8,
        max_position_embeddings=128,
        reference_compile=False,
        pad_token_id=0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
    )
    return AutoModel.from_config(cfg, attn_implementation="sdpa").eval()


def test_attention_cenn_padding_invariance_and_gradients():
    enc = encoder()
    block = SlidingAttentionCeNN(enc.layers[1].attn, window=8, steps=2)

    x = torch.randn(2, 13, 16, requires_grad=True)
    mask = torch.ones(2, 13, dtype=torch.long)
    y = block(x, attention_mask=mask)[0]

    padded = torch.cat([x.detach(), torch.randn(2, 9, 16) * 100], dim=1)
    padded_mask = torch.nn.functional.pad(mask, (0, 9))
    yp = block(padded, attention_mask=padded_mask)[0]

    torch.testing.assert_close(y, yp[:, :13], atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(yp[:, 13:]) == 0

    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in block.core_parameters()
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in block.output_parameters()
    )


def test_attention_cenn_is_local_bidirectional_and_has_no_pairwise_sdpa(monkeypatch):
    enc = encoder()
    block = SlidingAttentionCeNN(enc.layers[1].attn, window=8, steps=2).eval()

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("AttentionCeNN must not call pairwise SDPA")
        ),
    )

    x = torch.randn(1, 32, 16)
    mask = torch.ones(1, 32, dtype=torch.long)
    y = block(x, attention_mask=mask)[0]

    near = x.clone()
    near[:, 4] += 5.0
    far = x.clone()
    far[:, 24] += 5.0

    yn = block(near, attention_mask=mask)[0]
    yf = block(far, attention_mask=mask)[0]

    assert (y[:, 1] - yn[:, 1]).abs().max() > 1e-7
    torch.testing.assert_close(y[:, 1], yf[:, 1], atol=1e-6, rtol=1e-6)


def test_v27_factory_selects_pdelta_local32_and_attention_cenn():
    enc = encoder()

    full = make_v27_replacement(
        enc.layers[0].attn,
        "full_attention",
        full_feature_dim=8,
        full_local_window=4,
        full_chunk_size=4,
        sliding_window=8,
        sliding_steps=2,
    )
    sliding = make_v27_replacement(
        enc.layers[1].attn,
        "sliding_attention",
        full_feature_dim=8,
        full_local_window=4,
        full_chunk_size=4,
        sliding_window=8,
        sliding_steps=2,
    )

    assert isinstance(full, PDelta3GDN2CLVRAttention)
    assert full.local_window == 4
    assert full.feature_dim == 8
    assert all(p.requires_grad for p in full.Wo.parameters())

    assert isinstance(sliding, SlidingAttentionCeNN)
    assert sliding.window == 8
    assert sliding.steps == 2
    assert sliding.config_dict()["exact_pairwise_attention"] is False
