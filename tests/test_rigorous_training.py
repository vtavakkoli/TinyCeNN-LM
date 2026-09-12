"""Exercise real Transformers forward/backward/checkpoint paths without downloads."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from tinycenn_lm import build_cenn_student  # noqa: E402
from tinycenn_lm.distill_utils import evaluate_distillation  # noqa: E402


class LocalTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def get_vocab(self):
        return {str(i): i for i in range(32)}

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 15 + 1 for c in text]}

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer_config.json").write_text("{}")


def test_real_trainer_learns_and_resumes_saved_weights_and_stream(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/train_distill_rigorous.py"
    spec = importlib.util.spec_from_file_location("rigorous", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, use_cache=False,
    )
    def load_model(*args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(7)
            return transformers.LlamaForCausalLM(config).to(dtype=kwargs.get("dtype", torch.float32))

    monkeypatch.setattr(trainer.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(trainer.AutoTokenizer, "from_pretrained", lambda *a, **k: LocalTokenizer())
    monkeypatch.setattr(trainer, "load_dataset", lambda *a, **k: (
        {"text": f"abc abc abc abc {i}"} for i in range(4000)
    ))
    original_collect = trainer.collect_eval_batches
    captured_batches = []
    def collect(*args, **kwargs):
        batches = original_collect(*args, **kwargs)
        captured_batches[:] = batches
        return batches
    monkeypatch.setattr(trainer, "collect_eval_batches", collect)
    common = ["--base-model", "offline-test", "--dataset", "offline-corpus",
              "--context-length", "8", "--batch-size", "2", "--grad-accum", "2",
              "--max-tokens", "512", "--steps", "2", "--dilations", "1,2",
              "--expansion", "2", "--eval-batches", "2", "--eval-batch-size", "2",
              "--eval-every", "4", "--log-every", "4", "--no-compile",
              "--learning-rate", "0.02", "--interface-lr-scale", "0.5",
              "--lr-schedule", "wsd", "--kl-final-weight", "0.25",
              "--hidden-final-weight", "0", "--shuffle-buffer", "16"]
    first = tmp_path / "first"
    monkeypatch.setattr(sys, "argv", [str(path), *common, "--train-interfaces", "all",
                                     "--output-dir", str(first)])
    trainer.main()
    report = json.loads((first / "distillation_report.json").read_text())
    assert report["final"]["student_ce"] < report["run_start"]["student_ce"]
    assert report["seen_tokens_this_run"] == 512
    assert report["train_history"][-1]["hidden_weight"] == 0
    assert report["train_history"][-1]["kl_weight"] == 0.25

    # Validate the final artifact against its own metrics, not the best model.
    restored = build_cenn_student(first, dtype=torch.float32).eval()
    teacher = load_model().eval()
    metrics = evaluate_distillation(
        teacher, restored, captured_batches, device=torch.device("cpu"), dtype=torch.float32,
        temperature=2, kl_chunk_rows=256, ce_weight=1, kl_weight=1, hidden_weight=0.25,
    )
    assert metrics["student_ce"] == pytest.approx(report["checkpoint"]["metrics"]["student_ce"], abs=1e-6)

    best_path = Path(str(first) + "-best")
    metadata = json.loads((best_path / "student_config.json").read_text())
    state = torch.load(best_path / "cenn_student.pt", weights_only=True)
    assert all(p.dtype == torch.float32 for p in state.values())
    assert "model.embed_tokens.weight" in state and "lm_head.weight" in state
    second = tmp_path / "second"
    monkeypatch.setattr(sys, "argv", [str(path), *common, "--train-interfaces", "none",
                                     "--resume-student-dir", str(best_path), "--output-dir", str(second)])
    trainer.main()
    continued = json.loads((second / "distillation_report.json").read_text())
    assert continued["previous_training_tokens"] == metadata["training"]["cumulative_tokens"]
    assert continued["training_stream"]["resume_mode"] == "exact_token_offset"
    assert continued["training_stream"]["start_token_offset"] == metadata["training"]["data_stream"]["next_token_offset"]
    assert continued["run_start"]["student_ce"] == pytest.approx(report["best"]["student_ce"], abs=1e-6)
    new_state = torch.load(second / "cenn_student.pt", weights_only=True)
    torch.testing.assert_close(state["lm_head.weight"], new_state["lm_head.weight"], rtol=0, atol=0)
