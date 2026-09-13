import json
from pathlib import Path

from tinycenn_lm.colab_live_backup import (
    is_tinycenn_training_command,
    output_dir_from_command,
    training_script_name,
)
from tinycenn_lm.direct_colab_backup import _parent_backup_is_active


def test_training_command_detection():
    cmd = ["python", "/content/TinyCeNN-LM/scripts/train_smollm2_amcenn_v2.py", "--max-tokens", "1000"]
    assert is_tinycenn_training_command(cmd)
    assert training_script_name(cmd) == "train_smollm2_amcenn_v2"
    assert not is_tinycenn_training_command(["python", "scripts/generate.py"])
    assert not is_tinycenn_training_command(["nvidia-smi"])


def test_output_dir_parsing_absolute_and_relative(tmp_path: Path):
    assert output_dir_from_command(["python", "train_x.py", "--output-dir", "/tmp/x"]) == Path("/tmp/x")
    assert output_dir_from_command(["python", "train_x.py", "--output-dir=checkpoints/x"], cwd=tmp_path) == tmp_path / "checkpoints/x"
    assert output_dir_from_command(["python", "train_x.py"], cwd=tmp_path) is None


def test_direct_backup_detects_active_parent_run(tmp_path: Path):
    run_dir = tmp_path / ".colab_live_backup" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "command": ["python", "scripts/train_smollm2_amcenn_v2.py", "--max-tokens", "1000"],
            }
        ),
        encoding="utf-8",
    )

    assert _parent_backup_is_active("train_smollm2_amcenn_v2", tmp_path)
    assert not _parent_backup_is_active("train_story_v2", tmp_path)
