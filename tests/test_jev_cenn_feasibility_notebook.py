"""Regression: the attention-free Colab defines independent checkpoint snapshots.

This caught a missing state_cpu helper when the real global-layer tournament
called train_400() after a successful attention-free preflight.
"""
import ast
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Jev_CeNN_Attention_Free_Global_Sliding_Feasibility_Colab.ipynb"


def test_jev_cenn_checkpoint_helper_exists_before_train_and_clones_weights():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][5]["source"])
    tree = ast.parse(code)
    names = [item.name for item in tree.body if isinstance(item, ast.FunctionDef)]
    assert "state_cpu" in names
    assert "train_400" in names
    assert names.index("state_cpu") < names.index("train_400")
    snippet = next(item for item in tree.body
                   if isinstance(item, ast.FunctionDef) and item.name == "state_cpu")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[snippet], type_ignores=[]),
                 "<jev_checkpoint_helper>", "exec"), namespace)
    torch.manual_seed(2028)
    module = torch.nn.Linear(4, 4)
    frozen = namespace["state_cpu"](module)
    assert set(frozen) == set(module.state_dict())
    assert all(t.device.type == "cpu" and not t.requires_grad for t in frozen.values())
    initial = frozen["weight"].clone()
    with torch.no_grad():
        module.weight.add_(10)
    torch.testing.assert_close(frozen["weight"], initial)
    # Restored snapshots must be usable as strict state_dicts.
    module.load_state_dict(frozen, strict=True)
    torch.testing.assert_close(module.weight, initial)


def test_tournament_checkpoint_paths_refer_to_defined_snapshot_helper():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_code = "\n".join("".join(c["source"]) for c in notebook["cells"]
                         if c["cell_type"] == "code")
    assert "def state_cpu(" in all_code
    assert "best_state = state_cpu(model)" in all_code
    assert '"state_dict":state_cpu(candidate)' in all_code
    assert "model.load_state_dict(best_state, strict=True)" in all_code
