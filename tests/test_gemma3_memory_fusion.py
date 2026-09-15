from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from tinycenn_lm.gemma3_memory_fusion import (
    Gemma3MemoryFusionConfig,
    MemoryFusionGemma3Attention,
    full_attention_layers,
    replace_attention_layers,
    structural_summary,
)


def tiny_model():
    torch.manual_seed(91)
    config = Gemma3TextConfig(
        vocab_size=79,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        # Deliberately not hidden_size / num_heads. FunctionGemma also uses an
        # attention projection width different from hidden_size.
        head_dim=20,
        max_position_embeddings=128,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        query_pre_attn_scalar=20,
        rope_theta=1000000.0,
        rope_local_base_freq=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return Gemma3ForCausalLM(config).eval()


def test_memory_fusion_replaces_only_full_attention_and_runs():
    model = tiny_model()
    assert full_attention_layers(model) == [1]
    cfg = Gemma3MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    replace_attention_layers(model, cfg, [1])
    wrapped = model.model.layers[1].self_attn
    assert isinstance(wrapped, MemoryFusionGemma3Attention)
    assert wrapped.attention_width == 80
    assert wrapped.hidden_size == 64
    summary = structural_summary(model)
    assert summary["memory_fusion_layers"] == [1]
    assert summary["remaining_full_attention_layers"] == []
    assert summary["sliding_attention_layers"] == [0, 2]
    ids = torch.randint(0, 79, (1, 17))
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    assert logits.shape == (1, 17, 79)
    assert torch.isfinite(logits).all()


def test_memory_fusion_rejects_sliding_attention():
    model = tiny_model()
    cfg = Gemma3MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    try:
        replace_attention_layers(model, cfg, [0])
    except ValueError as error:
        assert "sliding" in str(error).lower()
    else:
        raise AssertionError("expected sliding-attention replacement to be rejected")


def test_functiongemma_sequential_cli_help():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_functiongemma_memory_fusion_sequential.py"), "--help"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "FunctionGemma Memory Fusion" in completed.stdout
    assert "--target-layers" in completed.stdout
