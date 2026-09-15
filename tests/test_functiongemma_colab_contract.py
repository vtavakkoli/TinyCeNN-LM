import json
from pathlib import Path


def _cells():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "notebooks" / "FunctionGemma_270M_MemoryFusion_Sequential_Acceptance_Colab.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]


def _cell_containing(cells, needle):
    matches = [cell for cell in cells if needle in cell]
    assert matches, f"missing notebook cell containing {needle!r}"
    return matches[0]


def test_functiongemma_colab_uses_direct_trainer_and_hf_login():
    cells = _cells()
    auth = _cell_containing(cells, 'userdata.get("HF_TOKEN")')
    train = _cell_containing(cells, "train_functiongemma_memory_fusion_sequential.py")
    assert 'userdata.get("HF_TOKEN")' in auth
    assert "subprocess.run(cmd" in train
    assert "bash" not in train
    assert "tee -a" not in train


def test_functiongemma_colab_publishes_model_and_resume_checkpoint():
    cells = _cells()
    config = _cell_containing(cells, "HF_MODEL_REPO")
    publish = _cell_containing(cells, "api.upload_folder")
    assert "HF_MODEL_REPO" in config
    assert "api.upload_folder" in publish
    assert "save_adapter" in publish
    assert "sequential_in_progress.pt" in publish
    assert "README.md" in publish
