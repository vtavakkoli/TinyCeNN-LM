import json

import torch
from torch import nn

from tinycenn_lm import (
    CeNNConfig,
    CeNNReplacementLayer,
    freeze_student_interfaces,
    load_cenn_student_weights,
    replace_transformer_with_cenn,
    save_cenn_student,
)


class DummyConfig:
    hidden_size = 8
    use_cache = True


class DummyGenerationConfig:
    use_cache = True


class DummyTransformerLayer(nn.Module):
    def __init__(self, hidden_size=8, dtype=torch.float32):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class DummyBackbone(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList([DummyTransformerLayer(dtype=dtype)])


class DummyModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.config = DummyConfig()
        self.generation_config = DummyGenerationConfig()
        self.model = DummyBackbone(dtype=dtype)
        self.embed = nn.Embedding(16, 8, dtype=dtype)
        self.lm_head = nn.Linear(8, 16, bias=False, dtype=dtype)


def test_transformer_layer_is_replaced_and_removed():
    model = DummyModel()
    old_layer = model.model.layers[0]
    old_param_ids = {id(p) for p in old_layer.parameters()}
    config = CeNNConfig(hidden_size=8, steps=3, expansion=2, dilations=(1, 2, 4))

    replace_transformer_with_cenn(model, config)

    assert isinstance(model.model.layers[0], CeNNReplacementLayer)
    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False
    assert not old_param_ids.intersection({id(p) for p in model.parameters()})


def test_replacement_is_causal_and_shape_preserving():
    torch.manual_seed(0)
    layer = CeNNReplacementLayer(
        CeNNConfig(hidden_size=8, steps=3, expansion=2, dilations=(1, 2, 4))
    ).eval()
    nn.init.normal_(layer.cenn.cell.out_proj.weight, std=0.02)
    x1 = torch.randn(1, 12, 8)
    x2 = x1.clone()
    x2[:, 8:, :] = torch.randn_like(x2[:, 8:, :]) * 20
    y1 = layer(x1)
    y2 = layer(x2)
    assert y1.shape == x1.shape
    torch.testing.assert_close(y1[:, :8], y2[:, :8], atol=1e-5, rtol=1e-5)


def test_replacement_inherits_bfloat16():
    model = DummyModel(dtype=torch.bfloat16)
    config = CeNNConfig(hidden_size=8, steps=2, expansion=2)
    replace_transformer_with_cenn(model, config)
    layer = model.model.layers[0]
    assert isinstance(layer, CeNNReplacementLayer)
    assert all(p.dtype == torch.bfloat16 for p in layer.parameters())


def test_freeze_student_interfaces_trains_only_cenn():
    model = DummyModel()
    replace_transformer_with_cenn(model, CeNNConfig(hidden_size=8, steps=2, expansion=2))
    freeze_student_interfaces(model)
    assert any(p.requires_grad for p in model.model.layers[0].parameters())
    assert not any(p.requires_grad for p in model.embed.parameters())
    assert not any(p.requires_grad for p in model.lm_head.parameters())


def test_student_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    config = CeNNConfig(hidden_size=8, steps=2, expansion=2)
    model = DummyModel()
    replace_transformer_with_cenn(model, config)
    nn.init.normal_(model.model.layers[0].cenn.cell.out_proj.weight, std=0.03)
    save_cenn_student(model, tmp_path, config=config)

    metadata = json.loads((tmp_path / "student_config.json").read_text())
    assert metadata["architecture"] == "cenn-only-replacement"
    assert (tmp_path / "cenn_student.pt").exists()

    restored = DummyModel()
    replace_transformer_with_cenn(restored, config)
    load_cenn_student_weights(restored, tmp_path)

    for p1, p2 in zip(model.model.layers[0].parameters(), restored.model.layers[0].parameters()):
        torch.testing.assert_close(p1, p2)
