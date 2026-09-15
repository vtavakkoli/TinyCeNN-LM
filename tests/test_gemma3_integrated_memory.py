import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from scripts.benchmark_functiongemma_integrated_memory import functiongemma_cache_equivalence
from tinycenn_lm.gemma3_integrated_memory import (
    adapter_payload,
    build_student,
    full_attention_layers,
    greedy_generate,
    inference_mode,
    new_cache,
    restore_student,
)


def tiny_model():
    torch.manual_seed(77)
    config = Gemma3TextConfig(
        vocab_size=73,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        query_pre_attn_scalar=16,
        rope_theta=1000000.0,
        rope_local_base_freq=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "sdpa"
    return Gemma3ForCausalLM(config).eval()


def test_full_attention_discovery_and_reject_sliding():
    teacher = tiny_model()
    assert full_attention_layers(teacher) == [1]
    build_student(teacher, [1], "cenn_partition", features=8, block_size=4, sinks=1)
    try:
        build_student(teacher, [0], "cenn_partition", features=8, block_size=4, sinks=1)
    except ValueError as error:
        assert "sliding" in str(error)
    else:
        raise AssertionError("Expected sliding layer replacement to be rejected")


def test_wrapper_preserves_gemma3_attention_contract():
    teacher = tiny_model()
    original = teacher.model.layers[1].self_attn
    student = build_student(teacher, [1], "cenn_partition", features=8, block_size=4, sinks=1)
    wrapped = student.model.layers[1].self_attn
    assert wrapped.is_sliding is False
    assert wrapped.layer_idx == original.layer_idx == 1
    assert wrapped.head_dim == original.head_dim
    assert wrapped.num_key_value_groups == original.num_key_value_groups
    assert wrapped.scaling == original.scaling
    assert wrapped.attention_dropout == original.attention_dropout
    assert wrapped.is_causal == original.is_causal
    assert wrapped.attn_logit_softcapping == original.attn_logit_softcapping
    assert wrapped.sliding_window == original.sliding_window is None


def test_transformer_readout_starts_close_to_original():
    teacher = tiny_model()
    student = build_student(teacher, [1], "transformer_readout", features=8, block_size=4, sinks=1)
    ids = torch.randint(0, 73, (1, 19))
    with torch.no_grad():
        reference = teacher(input_ids=ids, use_cache=False).logits
        candidate = student(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(candidate, reference, atol=2e-5, rtol=2e-4)


def test_cenn_cache_matches_full_and_checkpoint_reload():
    teacher = tiny_model()
    student = build_student(teacher, [1], "cenn_partition", features=8, block_size=4, sinks=1)
    ids = torch.randint(0, 73, (1, 23))
    with inference_mode(student):
        full = student(input_ids=ids, use_cache=False).logits
        cache = new_cache(student)
        first = student(input_ids=ids[:, :7], past_key_values=cache, use_cache=True).logits
        pieces = [first]
        for i in range(7, ids.shape[1]):
            pieces.append(student(input_ids=ids[:, i : i + 1], past_key_values=cache, use_cache=True).logits)
        cached = torch.cat(pieces, dim=1)
        torch.testing.assert_close(cached, full, atol=6e-5, rtol=6e-4)
        assert cache.get_seq_length(1) == ids.shape[1]
        assert cache.nbytes > 0

        generated, _ = greedy_generate(student, ids[:, :5], tokens=4)
        assert generated.shape == (1, 4)

    payload = adapter_payload(student, {"test": True})
    recovered = restore_student(teacher, payload)
    with inference_mode(student), inference_mode(recovered):
        torch.testing.assert_close(
            student(input_ids=ids, use_cache=False).logits,
            recovered(input_ids=ids, use_cache=False).logits,
            atol=0,
            rtol=0,
        )


def test_functiongemma_cache_gate_reports_native_baseline_and_passes_clean_model():
    teacher = tiny_model()
    student = build_student(teacher, [1], "cenn_partition", features=8, block_size=4, sinks=1)
    block = torch.randint(0, 73, (32,))
    metrics = functiongemma_cache_equivalence(
        teacher, student, block, torch.device("cpu"), "float32", block_size=4
    )
    assert metrics["cached_logits_nmse"] <= metrics["cache_equivalence_nmse_limit"]
    assert metrics["cached_top1_agreement"] >= metrics["cache_equivalence_top1_limit"]
    assert metrics["teacher_cached_logits_nmse"] >= 0
    assert 0 <= metrics["teacher_cached_top1_agreement"] <= 1
