import json
from pathlib import Path

from tinycenn_lm.hf_persistence import build_model_card, collect_reports, redact_secrets


def test_model_card_uses_report_metrics_and_redacts_tokens(tmp_path: Path):
    report = {
        "architecture": "smollm2-amcenn-top2-v2",
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "seen_tokens": 123456,
        "last_training_ce": 4.25,
        "last_distillation_kl": 0.75,
        "feature_dim": 128,
        "num_shards": 8,
        "top_k": 2,
        "evaluation_performed": False,
    }
    (tmp_path / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("secret hf_ABCDEFGHIJKLMNOPQRSTUV123456", encoding="utf-8")
    card = build_model_card(tmp_path, title="Test AM-CeNN")
    assert "SmolLM2-135M" in card
    assert "123,456" in card
    assert "4.25" in card
    assert "fineweb-edu" in card
    assert "attention-free" in card
    assert "mixture-of-experts" in card
    assert "hf_ABCDEFGHIJKLMNOPQRSTUV123456" not in redact_secrets(card)


def test_collect_reports_ignores_unrelated_json(tmp_path: Path):
    (tmp_path / "training_report.json").write_text('{"seen_tokens": 42}', encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text('{"not": "a report"}', encoding="utf-8")
    reports = collect_reports(tmp_path)
    assert "training_report.json" in reports
    assert "tokenizer.json" not in reports


def test_redact_secrets():
    assert redact_secrets("token=hf_123456789012345678901234") == "token=hf_REDACTED"
