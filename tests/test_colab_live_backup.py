from pathlib import Path

from tinycenn_lm.colab_live_backup import (
    is_tinycenn_training_command,
    output_dir_from_command,
    training_script_name,
)


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
