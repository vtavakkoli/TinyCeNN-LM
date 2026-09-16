from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from tinycenn_lm.gemma4_memory_fusion import (
    DualGemma4Attention,
    Gemma4MemoryFusionConfig,
    all_attention_layers,
    ensure_dual_layer,
    ensure_dual_layers,
    load_selected_memory_state,
    selected_memory_state,
    set_attention_mode,
    structural_summary,
)


def tiny_model(*, shared: bool = False):
    torch.manual_seed(123)
    layers = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    cfg = Gemma4TextConfig(
        vocab_size=71,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=128,
        layer_types=layers,
        sliding_window=16,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=2 if shared else 0,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        attention_k_eq_v=False,
        final_logit_softcapping=30.0,
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
        },
        use_cache=False,
    )
    cfg._attn_implementation = "sdpa"
    return Gemma4ForCausalLM(cfg).eval()


def test_gemma4_all_attention_can_be_dual_wrapped_and_run():
    model = tiny_model(shared=False)
    mf = Gemma4MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    assert all_attention_layers(model) == [0, 1, 2, 3]
    ensure_dual_layers(model, mf, all_attention_layers(model))
    assert all(isinstance(layer.self_attn, DualGemma4Attention) for layer in model.model.layers)

    ids = torch.randint(0, 71, (1, 17))
    with torch.no_grad():
        set_attention_mode(model, "original")
        original = model(input_ids=ids, use_cache=False).logits
        set_attention_mode(model, "memory")
        adapted = model(input_ids=ids, use_cache=False).logits
    assert original.shape == adapted.shape == (1, 17, 71)
    assert torch.isfinite(adapted).all()
    summary = structural_summary(model)
    assert summary["memory_fusion_layers"] == [0, 1, 2, 3]
    assert summary["remaining_attention_layers"] == []


def test_gemma4_shared_kv_layers_get_closest_local_donor_and_full_memory_forward_runs():
    model = tiny_model(shared=True)
    # With 4 layers and num_kv_shared_layers=2, layers 2/3 have no native K/V projections.
    assert not hasattr(model.model.layers[2].self_attn, "k_proj")
    assert not hasattr(model.model.layers[3].self_attn, "k_proj")
    mf = Gemma4MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    ensure_dual_layers(model, mf, [0, 1, 2, 3])
    assert hasattr(model.model.layers[2].self_attn.memory, "k_proj")
    assert hasattr(model.model.layers[3].self_attn.memory, "k_proj")

    ids = torch.randint(0, 71, (1, 19))
    set_attention_mode(model, "memory")
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    assert logits.shape == (1, 19, 71)
    assert torch.isfinite(logits).all()


def test_gemma4_memory_state_roundtrip():
    model = tiny_model(shared=True)
    mf = Gemma4MemoryFusionConfig(feature_dim=8, memory_rank=8, dilations=(1, 2), shifted_window=4)
    ensure_dual_layers(model, mf, [0, 1, 2, 3])
    with torch.no_grad():
        for p in model.model.layers[2].self_attn.memory.core.parameters():
            p.add_(0.01 * torch.randn_like(p))
    state = selected_memory_state(model, [0, 1, 2, 3])

    recovered = tiny_model(shared=True)
    ensure_dual_layers(recovered, mf, [0, 1, 2, 3])
    load_selected_memory_state(recovered, state, [0, 1, 2, 3])
    for key, value in selected_memory_state(recovered, [0, 1, 2, 3]).items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)


def test_gemma4_all_attention_cli_help():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["TINYCENN_PARENT_BACKUP_ACTIVE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "train_gemma4_memory_fusion_all_attention.py"), "--help"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "replace every Gemma 4 E2B" in completed.stdout
    assert "--force-after-rounds" in completed.stdout
