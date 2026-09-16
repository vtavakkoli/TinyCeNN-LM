from __future__ import annotations

import json
from pathlib import Path


def _cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_qwen35_memory_fusion_colab_uses_v2_trainer_and_status():
    repo = Path(__file__).resolve().parents[1]
    nb = json.loads((repo / "notebooks" / "Qwen3_5_0_8B_MemoryFusion_Sequential_Acceptance_Colab.ipynb").read_text(encoding="utf-8"))
    cells = {c.get("metadata", {}).get("id"): _cell_text(c) for c in nb["cells"]}
    assert "train_qwen35_memory_fusion_sequential_v2.py" in cells["train"]
    assert "test_qwen35_memory_fusion_sequential_v2.py" in cells["preflight"]
    assert "colab_run_status.json" in cells["status"]
