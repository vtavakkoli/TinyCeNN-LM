"""V3.1 fully converted, no-project-import Hugging Face runtime contracts."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tinycenn_lm"
RUNTIME = SRC / "standalone_all_attention_runtime.py"
NOTEBOOK = ROOT / "notebooks" / "Laya_V31_AllAttention_FineTune_HF_Colab.ipynb"


def test_mixed_runtime_has_no_research_package_imports():
    tree = ast.parse(RUNTIME.read_text(encoding="utf8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(x.name for x in node.names)
    assert not any(x.startswith("tinycenn_lm") or x.startswith("laya")
                   for x in imported)


def test_colab_replaces_both_kinds_and_leaves_test_blind():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 13
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    code = "\n".join("".join(x["source"]) for x in code_cells)
    for fragment in [
        "GLOBAL_LAYERS", "SLIDING_LAYERS", "build_v28(FULL_WINDOW)",
        "SlidingWindowFusion(", "before_joint", "JOINT_STEPS",
        "decision_distill_loss(", "F.cross_entropy(",
        'split="test"', "quality_gate_pass",
        "export_v31_all_attention(", "verify_standalone_bundle(",
        "upload_standalone_to_hub(", "PUBLISH_ONLY_IF_PASSES",
        "HF_PRIVATE = True", "student_state",
    ]:
        assert fragment in code, fragment
    assert code.index("final_checkpoint=OUTPUT_DIR") < code.index('split="test"')
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for c in code_cells)


def test_complete_rebuild_state_dict_and_cpu_forward_parity(tmp_path):
    from transformers import AutoModel, ModernBertConfig, __version__
    if __version__ != "4.57.6":
        pytest.skip("Portable V3.1 mixed runtime is pinned to Transformers 4.57.6")
    from tinycenn_lm import standalone_all_attention_runtime as mixed
    from tinycenn_lm.hf_standalone import export_standalone_bundle

    torch.manual_seed(123)
    config = ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2,
        global_attn_every_n_layers=3, local_attention=8,
        max_position_embeddings=128, reference_compile=False,
        pad_token_id=0, attention_dropout=0.0,
        embedding_dropout=0.0, mlp_dropout=0.0,
    )
    encoder = AutoModel.from_config(config, attn_implementation="sdpa")
    model = mixed.StandaloneDecisionModel(
        encoder, head_layers=0, n_act=2).eval()
    global_layers = [
        i for i, layer in enumerate(model.encoder.layers)
        if tuple(layer.attn.local_attention) == (-1, -1)
    ]
    slide_layers = [
        i for i in range(len(model.encoder.layers)) if i not in global_layers
    ]
    assert global_layers and slide_layers
    for i in global_layers:
        orig = model.encoder.layers[i].attn
        model.encoder.layers[i].attn = mixed.FastWindowAttention(
            orig, local_window=17, landmark_stride=4,
            feature_dim=8, max_landmarks=8
        )
    for i in slide_layers:
        orig = model.encoder.layers[i].attn
        model.encoder.layers[i].attn = mixed.SlidingWindowFusion(
            orig, radius=4, block_size=8
        )

    manifest = {
        "architecture": "laya-v31-all-attention", "runtime_format": 1,
        "head_layers": 0, "n_act": 2,
        "global_layers": global_layers, "sliding_layers": slide_layers,
        "window": 17, "max_len": 32,
        "feature_dim": 8, "landmark_stride": 4,
        "max_landmarks": 8, "sliding_radius": 4,
        "sliding_block_size": 8,
    }
    root = export_standalone_bundle(
        model=model, output_dir=tmp_path/"bundle",
        runtime_source=RUNTIME, encoder_config=config,
        manifest=manifest,
    )
    assert (root/"model.safetensors").exists()
    assert (root/"runtime.py").exists()
    assert (root/"load_model.py").exists()
    spec = importlib.util.spec_from_file_location("v31_loader",root/"load_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recovered, tokenizer = module.load_model(root, device="cpu")
    assert tokenizer is None  # optional for generic model export
    assert len(recovered.encoder.layers) == 4
    for i in global_layers:
        assert type(recovered.encoder.layers[i].attn).__name__ == 'FastWindowAttention'
    for i in slide_layers:
        assert type(recovered.encoder.layers[i].attn).__name__ == 'SlidingWindowFusion'
    batch = {
        "input_ids": torch.randint(1,64,(1,20)),
        "attention_mask": torch.ones(1,20,dtype=torch.long),
        "marker_pos": torch.tensor([[2,5]],dtype=torch.long),
        "marker_mask": torch.ones(1,2,dtype=torch.bool),
        "qtype": torch.tensor([0]),
    }
    with torch.inference_mode():
        expected = model(**batch)
        actual = recovered(**batch)
    for left,right in zip(expected,actual):
        torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5)
    assert sorted(model.state_dict()) == sorted(recovered.state_dict())


def test_manifest_enforces_full_coverage(tmp_path):
    from transformers import __version__
    if __version__ != "4.57.6":
        pytest.skip("Pinned runtime contract")
    from tinycenn_lm.standalone_all_attention_runtime import build_model
    with pytest.raises(ValueError,match="Overlap"):
        build_model(tmp_path, {
            "architecture": "laya-v31-all-attention",
            "runtime_format": 1,
            "global_layers": [0], "sliding_layers": [0]
        })
