import json
from pathlib import Path


def _cells():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "notebooks" / "FunctionGemma_270M_MemoryFusion_Sequential_Acceptance_Colab.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    return {c.get("metadata", {}).get("id"): "".join(c.get("source", [])) for c in nb["cells"]}


def test_functiongemma_colab_uses_direct_trainer_and_hf_login():
    cells = _cells()
    assert "userdata.get(\"HF_TOKEN\")" in cells["auth"]
    assert "subprocess.run(cmd" in cells["train"]
    assert "bash" not in cells["train"]
    assert "tee -a" not in cells["train"]


def test_functiongemma_colab_publishes_model_and_resume_checkpoint():
    cells = _cells()
    assert "HF_MODEL_REPO" in cells["config"]
    assert "api.upload_folder" in cells["publish"]
    assert "save_adapter" in cells["publish"]
    assert "sequential_in_progress.pt" in cells["publish"]
    assert "README.md" in cells["publish"]
