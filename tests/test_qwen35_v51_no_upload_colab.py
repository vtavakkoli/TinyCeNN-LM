"""Fast, CPU-only contract checks for the V5.1 no-upload research notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (Path(__file__).resolve().parents[1] / "notebooks" /
    "Qwen35_08B_CeNN_HeadRescue_V51_Warmstart_Colab.ipynb")


def _cells():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    return data["cells"]


def _source():
    return "\n".join("".join(c.get("source", [])) for c in _cells())


def test_all_python_cells_compile():
    for idx, cell in enumerate(_cells()):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(("%pip", "!")):
            continue
        ast.parse(source, filename=f"V5.1 cell {idx}")


def test_no_interactive_checkpoint_upload():
    source = _source()
    for forbidden in ("files.upload(", "google.colab import files", "ZIP_PATH",
                      "source_zip_name=", "V5_PAYLOAD"):
        assert forbidden not in source, forbidden


def test_bootstrap_precedes_v51_recovery_and_is_auditable():
    source = _source()
    assert source.index("def new_v5_baseline()") < source.index("def build_candidate(name):")
    for required in ("BASELINE_STEPS", "BOOTSTRAP_FINGERPRINT", "V5_CORE_STATE",
                     "baseline_checkpoint_sha256", "warmstart_v5_delta_core",
                     "model_info(MODEL_ID).sha"):
        assert required in source, required


def test_optional_kl_candidate_can_be_restored():
    source = _source()
    assert "'v51_joint' if name=='v51_joint_plus_kl' else name" in source
    assert "'v51_joint' if variant=='v51_joint_plus_kl' else variant" in source