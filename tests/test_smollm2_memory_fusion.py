import torch

from tinycenn_lm.smollm2_memory_fusion import (
    MemoryFusionLlamaAttention,
    SmolMemoryFusionConfig,
    freeze_for_global_training,
    replace_all_attention,
    structural_summary,
)


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    return LlamaForCausalLM(cfg)


def test_replace_all_attention_and_forward():
    torch.manual_seed(11)
    model = tiny_llama()
    config = SmolMemoryFusionConfig(
        feature_dim=8,
        memory_rank=8,
        dilations=(1, 2, 4),
        shifted_window=4,
    )
    replace_all_attention(model, config)
    summary = structural_summary(model)
    assert summary == {"memory_fusion_layers": 2, "transformer_attention_layers": 0}
    assert all(isinstance(layer.self_attn, MemoryFusionLlamaAttention) for layer in model.model.layers)

    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    assert out.logits.shape == (1, 12, model.config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_full_replacement_is_causal():
    torch.manual_seed(19)
    model = tiny_llama().eval()
    config = SmolMemoryFusionConfig(
        feature_dim=8,
        memory_rank=8,
        dilations=(1, 2, 4),
        shifted_window=4,
    )
    replace_all_attention(model, config)
    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    changed = ids.clone()
    changed[:, 8:] = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        a = model(input_ids=ids, use_cache=False).logits
        b = model(input_ids=changed, use_cache=False).logits
    torch.testing.assert_close(a[:, :8], b[:, :8], atol=2e-5, rtol=2e-5)


def test_global_training_freezes_pretrained_qkv():
    model = tiny_llama()
    config = SmolMemoryFusionConfig(
        feature_dim=8,
        memory_rank=8,
        dilations=(1, 2),
        shifted_window=4,
        train_output_projection=True,
    )
    replace_all_attention(model, config)
    trainable = freeze_for_global_training(model, train_output_projection=True)
    assert trainable
    for layer in model.model.layers:
        attn = layer.self_attn
        assert not attn.q_proj.weight.requires_grad
        assert not attn.k_proj.weight.requires_grad
        assert not attn.v_proj.weight.requires_grad
        assert attn.o_proj.weight.requires_grad
        assert any(p.requires_grad for p in attn.core.parameters())
