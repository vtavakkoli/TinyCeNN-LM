from __future__ import annotations

from contextlib import nullcontext

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from scripts import train_qwen35_memory_fusion_sequential_v2 as v2


def tiny_qwen35_full_attention_model():
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=79,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5ForCausalLM(config).eval()


def test_qwen35_capture_keeps_required_positional_attention_mask():
    model = tiny_qwen35_full_attention_model()
    ids = torch.randint(0, 79, (1, 19))
    capture, _ = v2.capture_attention_input_qwen(
        model,
        ids,
        0,
        lambda: nullcontext(),
        with_output=False,
    )

    assert "position_embeddings" in capture
    assert "attention_mask" in capture

    kwargs = v2.seq.attention_kwargs_from_capture(capture)
    with torch.no_grad():
        out = v2.seq.call_attention(
            model.model.layers[0].self_attn,
            capture["hidden"],
            kwargs,
        )
    assert out.shape == capture["hidden"].shape
    assert torch.isfinite(out).all()


def test_qwen35_v2_installs_capture_fix():
    v2.install_qwen35_capture_fix()
    assert v2.seq.capture_attention_input is v2.capture_attention_input_qwen
    assert v2.seq.real_hidden_function_metrics is v2.real_hidden_function_metrics_qwen
