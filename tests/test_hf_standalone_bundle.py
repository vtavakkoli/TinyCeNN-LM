"""CPU-only smoke/contract tests for portable, no-project HF model bundles."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "tinycenn_lm"


def test_standalone_modules_parse_and_do_not_require_project_runtime():
    runtime = SOURCE / "standalone_fastwindow_runtime.py"
    loader = SOURCE / "standalone_hf_loader.py"
    exporter = SOURCE / "hf_standalone.py"
    for module in (runtime, loader, exporter):
        ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    for module in (runtime, loader):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(x.startswith("tinycenn_lm") or x.startswith("laya")
                       for x in imported), imported


def test_v29_notebook_uses_joint_supervised_training_and_optional_hf_export():
    nb = json.loads((ROOT / "notebooks" /
        "Laya_V29_AllGlobal_FastWindowMemory_Colab.ipynb").read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    source = "\n".join("".join(c["source"]) for c in cells)
    assert "for i in GLOBAL_LAYERS:" in source
    assert "F.cross_entropy(slog.float(), gold)" in source
    assert "snapshot_replacements()" in source
    assert "restore_replacements(best_replacements)" in source
    assert "export_v29_fastwindow(" in source
    assert "verify_standalone_bundle(" in source
    assert "upload_standalone_to_hub(" in source
    assert "HF_PRIVATE = True" in source
    for c in cells:
        assert not c.get("outputs", [])
        assert c.get("execution_count") is None


def test_cpu_standalone_strict_weight_and_forward_parity(tmp_path, monkeypatch):
    import transformers
    if transformers.__version__ != "4.57.6":
        pytest.skip("Standalone runtime is intentionally pinned to transformers==4.57.6")
    from transformers import AutoModel, ModernBertConfig
    from tinycenn_lm.hf_standalone import export_standalone_bundle
    from tinycenn_lm import standalone_fastwindow_runtime as runtime

    torch.manual_seed(42)
    config = ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2,
        global_attn_every_n_layers=3, local_attention=8,
        max_position_embeddings=128, reference_compile=False,
        pad_token_id=0, attention_dropout=0.0,
        embedding_dropout=0.0, mlp_dropout=0.0
    )
    encoder = AutoModel.from_config(config, attn_implementation="sdpa").eval()
    model = runtime.StandaloneDecisionModel(encoder, head_layers=0, n_act=2).eval()
    global_layers = [i for i, block in enumerate(model.encoder.layers)
                     if tuple(block.attn.local_attention) == (-1, -1)]
    assert global_layers, "Expected ModernBERT global attention in CPU smoke model"
    for i in global_layers:
        original = model.encoder.layers[i].attn
        model.encoder.layers[i].attn = runtime.FastWindowAttention(
            original, local_window=64, feature_dim=8,
            landmark_stride=4, max_landmarks=8
        )
    manifest = {
        "architecture": "laya-v29-fastwindow", "runtime_format": 1,
        "head_layers": 0, "n_act": 2, "global_layers": global_layers,
        "window": 64, "max_len": 32,
        "feature_dim": 8, "landmark_stride": 4, "max_landmarks": 8,
    }

    class DummyTokenizer:
        def save_pretrained(self, folder):
            (Path(folder) / "tokenizer_config.json").write_text("{}")

    folder = export_standalone_bundle(
        model=model, tokenizer=DummyTokenizer(), output_dir=tmp_path / "bundle",
        runtime_source=SOURCE / "standalone_fastwindow_runtime.py",
        loader_source=SOURCE / "standalone_hf_loader.py",
        encoder_config=model.encoder.config, manifest=manifest
    )
    assert (folder / "model.safetensors").is_file()
    assert (folder / "runtime.py").is_file()
    assert (folder / "load_model.py").is_file()

    spec = importlib.util.spec_from_file_location("standalone_loader_test",
                                                  folder / "load_model.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: DummyTokenizer())
    recovered, _ = loader.load_model(folder)
    recovered.eval()
    batch = {
        "input_ids": torch.randint(1, 64, (1, 20)),
        "attention_mask": torch.ones(1, 20, dtype=torch.long),
        "marker_pos": torch.tensor([[2, 5]], dtype=torch.long),
        "marker_mask": torch.ones(1, 2, dtype=torch.bool),
        "qtype": torch.tensor([0], dtype=torch.long),
    }
    with torch.no_grad():
        expected = model(**batch)
        actual = recovered(**batch)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert not any(x.startswith("tinycenn_lm")
                   for x in recovered.__class__.__module__.split("."))


def test_export_fails_closed_on_missing_architecture(tmp_path):
    from tinycenn_lm.hf_standalone import export_standalone_bundle
    with pytest.raises(ValueError, match="runtime_format"):
        export_standalone_bundle(
            model=torch.nn.Linear(2, 2), tokenizer=None,
            output_dir=tmp_path, runtime_source=__file__,
            encoder_config=None, manifest={"architecture": "test"}
        )
