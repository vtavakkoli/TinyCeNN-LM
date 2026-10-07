import json
from pathlib import Path

from transformers import AutoModel, ModernBertConfig

from tinycenn_lm.decision_v28 import (
    make_v28_full_replacement,
    split_v28_parameters,
)
from tinycenn_lm.laya_lab.pdelta import PDelta3GDN2CLVRAttention


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


def test_v28_full_factory_uses_pdelta_local_attention_and_frozen_qkv():
    enc = encoder()
    block = make_v28_full_replacement(
        enc.layers[0].attn,
        feature_dim=8,
        local_window=4,
        chunk_size=4,
    )
    assert isinstance(block, PDelta3GDN2CLVRAttention)
    assert block.feature_dim == 8
    assert block.local_window == 4
    assert block.chunk_size == 4
    assert not any(p.requires_grad for p in block.Wqkv.parameters())
    assert all(p.requires_grad for p in block.Wo.parameters())

    core, output = split_v28_parameters(block)
    assert core
    assert output
    assert not ({id(p) for p in core} & {id(p) for p in output})

    # Accepted local replacements are frozen before the one-shot joint phase;
    # discovery must still find the intended parameters so they can be re-enabled.
    block.requires_grad_(False)
    frozen_core, frozen_output = split_v28_parameters(block)
    assert {id(p) for p in frozen_core} == {id(p) for p in core}
    assert {id(p) for p in frozen_output} == {id(p) for p in output}


def test_v28_notebook_is_full_attention_first_and_fast():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "Laya_V28_Fast_FullAttention_First_Colab.ipynb"
    nb = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
    )

    assert "SCREEN_STEPS = 200" in source
    assert "POLISH_STEPS = 100" in source
    assert "HARD_MAX_STEPS = 600" in source
    assert "cache_all_full_teacher_io" in source
    assert "One teacher forward captures all ten full-attention targets" in source
    assert "ALL full-attention layers installed" in source
    assert "single_joint_recovery" in source
    assert '"sliding_attention": "native ModernBERT sliding attention"' in source

    # V2.8 must not convert sliding attention yet.
    assert "SlidingAttentionCeNN" not in source
    assert "make_v27_replacement" not in source

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
