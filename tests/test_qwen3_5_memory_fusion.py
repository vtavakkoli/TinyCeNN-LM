from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tinycenn_lm.qwen3_5_memory_fusion import (
    MemoryFusionQwen35Attention,
    Qwen35MemoryFusionConfig,
    full_attention_layers,
    load_selected_attention_state,
    replace_attention_layers,
    selected_attention_state,
    structural_summary,
)


def tiny_full_model():
    torch.manual_seed(123)
    config = Qwen3_5TextConfig(
        vocab_size=71,
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


def test_qwen35_full_attention_replacement_runs_and_preserves_gate_contract():
    model = tiny_full_model()
    assert full_attention_layers(model) == [0]
    original = model.model.layers[0].self_attn
    assert original.q_proj.out_features == 2 * 4 * 16

    cfg = Qwen35MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    replace_attention_layers(model, cfg, [0])
    wrapped = model.model.layers[0].self_attn
    assert isinstance(wrapped, MemoryFusionQwen35Attention)
    assert wrapped.attention_width == 64
    assert wrapped.q_proj.out_features == 128

    ids = torch.randint(0, 71, (1, 17))
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    assert logits.shape == (1, 17, 71)
    assert torch.isfinite(logits).all()


def test_qwen35_hybrid_structure_targets_only_full_anchors():
    model = tiny_full_model()
    model.config.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    # full_attention_layers reads the config layout and should identify only the anchor.
    assert full_attention_layers(model) == [3]


def test_qwen35_state_roundtrip():
    model = tiny_full_model()
    cfg = Qwen35MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    replace_attention_layers(model, cfg, [0])
    with torch.no_grad():
        for p in model.model.layers[0].self_attn.core.parameters():
            p.add_(0.01 * torch.randn_like(p))
    state = selected_attention_state(model, [0])

    recovered = tiny_full_model()
    replace_attention_layers(recovered, cfg, [0])
    load_selected_attention_state(recovered, state, [0])
    for key, value in selected_attention_state(recovered, [0]).items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)

    summary = structural_summary(recovered)
    assert summary["memory_fusion_layers"] == [0]
    assert summary["remaining_full_attention_layers"] == []


def test_qwen35_sequential_cli_help():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["TINYCENN_PARENT_BACKUP_ACTIVE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_qwen35_memory_fusion_sequential.py"), "--help"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Sequential Qwen3.5 Memory Fusion" in completed.stdout
    assert "--target-layers" in completed.stdout
