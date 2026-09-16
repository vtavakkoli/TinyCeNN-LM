import tempfile

import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

from tinycenn_lm.gemma4_checkpoint import load_gemma4_text_causal


def tiny_text_config():
    layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    per_layer_config = {
        i: {"head_dim": 32}
        for i, kind in enumerate(layer_types)
        if kind == "full_attention"
    }
    cfg = Gemma4TextConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=layer_types,
        per_layer_config=per_layer_config,
        num_kv_shared_layers=0,
        sliding_window=16,
        max_position_embeddings=128,
        vocab_size_per_layer_input=97,
        hidden_size_per_layer_input=8,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        final_logit_softcapping=None,
        use_cache=True,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def test_multimodal_checkpoint_remaps_real_text_weights_into_causal_lm():
    torch.manual_seed(123)
    text_cfg = tiny_text_config()
    full_cfg = Gemma4Config(text_config=text_cfg, vision_config=None, audio_config=None)
    full = Gemma4ForConditionalGeneration(full_cfg).eval()

    with tempfile.TemporaryDirectory() as tmp:
        full.save_pretrained(tmp)
        loaded, info = load_gemma4_text_causal(tmp, dtype=torch.float32)

    assert info["core_missing_keys"] == []
    assert info["unexpected_text_keys"] == []

    source = full.model.language_model.state_dict()
    target = loaded.model.state_dict()
    for key in (
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.1.self_attn.q_proj.weight",
        "norm.weight",
    ):
        torch.testing.assert_close(target[key], source[key], atol=0, rtol=0)
