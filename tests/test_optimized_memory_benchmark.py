"""Offline end-to-end V2 run with actual Llama layers and mocked external I/O."""
import json
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from scripts import benchmark_cenn_optimized_memory as benchmark


def test_complete_v2_run_selects_before_joint_test_and_preserves_base(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub
    import transformers
    torch.manual_seed(101)
    config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    class Tokenizer:
        def __call__(self, text, **kwargs):
            offset = sum(text.encode()) % 41
            return {"input_ids": [(offset + i) % 41 for i in range(kwargs["max_length"])]}
    class Stream:
        def shuffle(self, **kwargs):
            return self
        def __iter__(self):
            return iter({"text": f"unique scientific document number {i}"} for i in range(1000))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: Stream())
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40),
        dataset_info=lambda *a, **kw: SimpleNamespace(sha="b" * 40)))
    args = SimpleNamespace(
        base_model="offline-llama", model_revision="main", dataset="offline-text",
        dataset_config="default", dataset_revision="main", layers=[0, 1], feature_dims=[8],
        variants=["transformer_readout", "sink_window", "cenn_linear", "cenn_partition"],
        context=8, test_contexts=[8, 12], train_documents=2, validation_documents=2,
        test_documents=2, steps=1, lm_steps=1, eval_every=1, lr=0.002, ridge=0.01,
        block_size=2, sink_tokens=1, seed=101, dataset_seed=9208, nll_margin=0.02,
        compute_dtype="float32", compile_kernels=False, exclude_manifest=[],
        output_dir=str(tmp_path / "run"))
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    root = tmp_path / "run"
    report = json.loads((root / "optimized_memory_report.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    assert report["status"] == "completed"
    assert selection["test_not_used_for_selection"]
    assert len(report["candidates"]) == 8
    joint = [r for r in report["rows"] if r.get("scope") == "joint"]
    assert len(joint) == 4
    assert all(r["quality"] == "insufficient_test_documents" for r in joint)
    assert all(r.get("beats_both_quality") is not True for r in report["rows"])
    assert all("transformer_readout" not in name for name in selection["winners"].values())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
    hashes = report["document_hashes"]
    assert len(set(sum(hashes.values(), []))) == 6
    for digest in hashes["test"]:
        assert 20 <= int(digest[:8], 16) % 100 < 30
    for candidate in report["candidates"]:
        assert (root / candidate["checkpoint"]).is_file()
    assert (root / "optimized_memory_summary.csv").is_file()
    assert (root / "test_document_nll.csv").is_file()
    # The pinned Colab CI profile includes plotting dependencies. Execute the
    # actual notebook results cell on this complete offline run, including joint rows.
    import importlib.util
    from pathlib import Path
    if all(importlib.util.find_spec(name) for name in ("pandas", "matplotlib")):
        import matplotlib
        matplotlib.use("Agg")
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/CeNN_Optimized_Memory_V2_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text())
        cells = [c for c in notebook["cells"] if "results" in c.get("metadata", {}).get("tags", [])]
        assert len(cells) == 1
        for cell in cells:
            exec(compile("".join(cell["source"]), str(notebook_path), "exec"),
                 {"OUT": root, "json": json, "display": lambda *args: None})
        assert (root / "selected_decisions.csv").is_file()
        assert (root / "comparison-layer-1.png").stat().st_size > 1000


def test_previous_manifest_hashes_are_excluded():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [0] * kwargs["max_length"]}
    rows = [{"text": f"document {i}"} for i in range(1000)]
    counts = {"train": 2, "validation": 2, "test": 2}
    _, first = benchmark.collect_documents(rows, Tokenizer(), counts, 8)
    excluded = set(sum(first.values(), []))
    _, second = benchmark.collect_documents(rows, Tokenizer(), counts, 8, excluded=excluded)
    assert not excluded.intersection(sum(second.values(), []))


def test_folding_readout_preserves_full_model_logits():
    torch.manual_seed(2027)
    config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    core = benchmark.OptimizedMemory(4, 2, 8, 8, block_size=4)
    with torch.no_grad():
        core.readout.add_(0.05 * torch.randn_like(core.readout))
        ids = torch.randint(0, 41, (1, 17))
        with benchmark.unfused_replace_attention(model, 1, core):
            ordinary = model(input_ids=ids, use_cache=False).logits
        with benchmark.replace_attention(model, 1, core):
            folded = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(folded, ordinary, atol=3e-6, rtol=3e-5)


def test_cli_works_outside_repository(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = Path(benchmark.__file__).resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--exclude-manifest" in result.stdout
