from __future__ import annotations

import json
from pathlib import Path

from tinycenn_lm.qwen35_memory_fusion_colab import dashboard_html, progress_info


def test_progress_info_surfaces_trainer_failure(tmp_path: Path):
    (tmp_path / "colab_run_status.json").write_text(
        json.dumps({
            "status": "trainer_failed",
            "return_code": 1,
            "current_layer": 3,
            "error": "missing attention mask",
        }),
        encoding="utf-8",
    )
    info = progress_info(tmp_path, [3, 7, 11, 15, 19, 23])
    assert info["status"] == "trainer_failed"
    assert info["current"] == 3
    assert info["error"] == "missing attention mask"
    html = dashboard_html(info)
    assert "trainer_failed" in html
    assert "missing attention mask" in html
