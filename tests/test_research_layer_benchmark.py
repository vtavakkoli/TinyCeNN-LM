"""Offline integration tests: actual random Llama, no downloaded weights/data."""
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.benchmark_cenn_research_layers import (
    benchmark_kernels, capture_samples, collect_documents, document_split,
    evaluate_nll, fit_language_loss, fit_transfer, paired_interval,
    quality_label, replace_attention,
)
from tinycenn_lm.research_layers import ResearchCeNNLayer


def small_llama():
    torch.manual_seed(101)
    config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).eval().requires_grad_(False)


def test_exact_attention_replacement_preserves_real_llama_logits_and_restores_module():
    model = small_llama()
    ids = torch.randint(0, 41, (1, 15))
    original = model.model.layers[1].self_attn
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        with replace_attention(model, 1, None):
            actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert model.model.layers[1].self_attn is original
    with pytest.raises(RuntimeError):
        with replace_attention(model, 1, None):
            raise RuntimeError("interrupted experiment")
    assert model.model.layers[1].self_attn is original


@pytest.mark.parametrize("variant", ["cenn_kda", "cenn_delta2", "cenn_delta2_window"])
def test_layer_training_checkpoint_reload_and_perplexity_smoke(tmp_path, variant):
    device = torch.device("cpu")
    model = small_llama()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    blocks = [torch.randint(0, 41, (17,)) for _ in range(6)]
    train, validation, test = blocks[:2], blocks[2:4], blocks[4:]
    cache = capture_samples(model, train + validation, [1], 8, device)[1]
    core = ResearchCeNNLayer(4, 2, 8, 8, variant, window=3, chunk_size=4)
    args = SimpleNamespace(seed=5, lr=0.003, steps=3, lm_steps=2, eval_every=2,
                           context=8, device=device)
    checkpoint = tmp_path / "core.pt"
    fit_transfer(core, cache[:2], cache[2:], args, checkpoint, variant)
    best, history = fit_language_loss(
        model, 1, core, train, cache[:2], validation, args, checkpoint, variant
    )
    assert torch.isfinite(torch.tensor(best)) and history
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
    baseline = evaluate_nll(model, test, 16, device)
    with replace_attention(model, 1, core):
        candidate = evaluate_nll(model, test, 16, device)
    payload = torch.load(checkpoint, weights_only=True)
    loaded = ResearchCeNNLayer(**payload["config"])
    loaded.load_state_dict(payload["state_dict"])
    with replace_attention(model, 1, loaded):
        reloaded = evaluate_nll(model, test, 16, device)
    assert candidate == reloaded
    delta, low, high = paired_interval(candidate, baseline, repeats=100)
    assert all(torch.isfinite(torch.tensor(x)) for x in (delta, low, high))
    metrics = benchmark_kernels(core, cache[0], device, repeats=2)
    assert metrics["prefill_ms"] > 0 and metrics["decode_step_ms"] > 0


def test_document_splits_are_disjoint_and_duplicates_do_not_cross():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(kwargs["max_length"]))}
    rows = [{"text": f"document number {i} contents"} for i in range(1000)]
    rows = rows + rows
    blocks, hashes = collect_documents(rows, Tokenizer(),
                                       {"train": 3, "validation": 3, "test": 3}, 8)
    assert all(len(x) == 3 for x in blocks.values())
    assert len(set(sum(hashes.values(), []))) == 9
    assert document_split(" a   b \n c ") == document_split("a b c")
    with pytest.raises(RuntimeError):
        collect_documents(rows, Tokenizer(), {"train": 300, "validation": 300, "test": 300},
                          8, max_documents=5)


def test_quality_labels_use_declared_margin_and_adequate_sample_size():
    assert paired_interval([1] * 10, [1] * 10, repeats=100) == (0, 0, 0)
    assert quality_label(0, -0.01, 0.01, 24) == "within_declared_nll_margin"
    assert quality_label(-0.05, -0.06, -0.04, 24) == "lower_nll_on_this_test"
    assert quality_label(0.04, 0.03, 0.05, 24) == "worse_than_declared_margin"
    assert quality_label(0, -0.01, 0.01, 2) == "insufficient_test_documents"
    assert quality_label(0, -0.04, 0.04, 24) == "inconclusive"
