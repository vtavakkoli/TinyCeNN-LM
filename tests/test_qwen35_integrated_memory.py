import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tinycenn_lm.qwen35_integrated_memory import (
    adapter_payload, build_student, full_attention_layers, inference_mode,
    new_cache, restore_student,
)


def tiny_model():
    torch.manual_seed(91)
    config = Qwen3_5TextConfig(
        vocab_size=73,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        hidden_act="silu",
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        attention_bias=False,
        attention_dropout=0.0,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        rope_parameters={
            "rope_type":"default", "rope_theta":10000,
            "mrope_section":[2,1,1], "mrope_interleaved":True,
            "partial_rotary_factor":0.25,
        },
        use_cache=True,
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5ForCausalLM(config).eval()


def test_full_attention_discovery_and_reject_linear():
    teacher = tiny_model()
    assert full_attention_layers(teacher) == [3]
    build_student(teacher, [3], "cenn_partition", features=8, block_size=4, sinks=1)
    try:
        build_student(teacher, [0], "cenn_partition", features=8, block_size=4, sinks=1)
    except ValueError as error:
        assert "full_attention" in str(error)
    else:
        raise AssertionError("Expected linear-attention replacement to be rejected")


def test_transformer_readout_preserves_qwen_output_gate():
    teacher = tiny_model()
    student = build_student(teacher, [3], "transformer_readout", features=8, block_size=4, sinks=1)
    ids = torch.randint(0, 73, (1, 17))
    with torch.no_grad():
        reference = teacher(input_ids=ids, use_cache=False).logits
        candidate = student(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(candidate, reference, atol=3e-5, rtol=3e-4)


def test_cenn_cache_forward_and_checkpoint_reload():
    teacher = tiny_model()
    student = build_student(teacher, [3], "cenn_partition", features=8, block_size=4, sinks=1)
    ids = torch.randint(0, 73, (1, 19))
    with inference_mode(student):
        full = student(input_ids=ids, use_cache=False).logits
        cache = new_cache(student)
        first = student(input_ids=ids[:, :7], past_key_values=cache, use_cache=True).logits
        pieces = [first]
        for i in range(7, ids.shape[1]):
            pieces.append(student(input_ids=ids[:, i:i+1], past_key_values=cache, use_cache=True).logits)
        cached = torch.cat(pieces, 1)
        assert cached.shape == full.shape
        assert torch.isfinite(cached).all()
        assert cache.nbytes > 0
        agreement = (cached.argmax(-1) == full.argmax(-1)).float().mean()
        assert float(agreement) >= 0.70

    payload = adapter_payload(student, {"test": True})
    recovered = restore_student(teacher, payload)
    with inference_mode(student), inference_mode(recovered):
        torch.testing.assert_close(
            student(input_ids=ids, use_cache=False).logits,
            recovered(input_ids=ids, use_cache=False).logits,
            atol=0, rtol=0,
        )
