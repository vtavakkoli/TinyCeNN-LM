from types import SimpleNamespace

import torch
from torch import nn

from tinycenn_lm.standalone_decision import (
    IntegratedMemoryV23Attention,
    collate_items,
    converted_full_attention_indices,
    full_attention_indices,
    install_integrated_memory,
    normalize_question,
    render_options,
)


class FakeAttention(nn.Module):
    def __init__(self, hidden=16, heads=4, layer_idx=0):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_attention_heads=heads)
        self.layer_idx = layer_idx
        self.head_dim = hidden // heads
        self.Wqkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.Wo = nn.Linear(hidden, hidden, bias=False)
        self.out_drop = nn.Identity()


class FakeLayer(nn.Module):
    def __init__(self, attention_type, idx):
        super().__init__()
        self.attention_type = attention_type
        self.attn = FakeAttention(layer_idx=idx)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            FakeLayer("sliding_attention", 0),
            FakeLayer("full_attention", 1),
            FakeLayer("sliding_attention", 2),
            FakeLayer("full_attention", 3),
        ])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()


def test_choice_normalization():
    q = normalize_question({"type": "choice", "instructions": "Pick", "criteria": ["left", "right"]})
    assert q["t"] == "choice"
    assert render_options(q) == ["left", "right"]


def test_collate_shapes():
    b = collate_items([
        {"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0},
        {"ids": [1, 4], "markers": [1], "qtype": 2},
    ], pad_id=0)
    assert tuple(b["input_ids"].shape) == (2, 3)
    assert tuple(b["marker_pos"].shape) == (2, 2)
    assert b["marker_mask"].sum().item() == 3


def test_v23_forward_shape():
    attn = IntegratedMemoryV23Attention(FakeAttention(), feature_dim=8, local_kernel=3)
    x = torch.randn(2, 7, 16)
    y, aux = attn(x, attention_mask=torch.ones(2, 7, dtype=torch.long))
    assert aux is None
    assert y.shape == x.shape
    assert tuple(attn.last_core_output.shape) == (2, 4, 7, 4)


def test_install_replaces_every_full_attention_only():
    model = FakeModel()
    assert full_attention_indices(model) == [1, 3]
    installed = install_integrated_memory(model, full_attention_indices(model), feature_dim=8, local_kernel=3)
    assert installed == [1, 3]
    assert converted_full_attention_indices(model) == [1, 3]
    assert not isinstance(model.encoder.layers[0].attn, IntegratedMemoryV23Attention)
    assert not isinstance(model.encoder.layers[2].attn, IntegratedMemoryV23Attention)
