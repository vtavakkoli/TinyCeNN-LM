import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from tinycenn_lm.gemma4_integrated_memory import (
    adapter_payload,
    build_student,
    full_attention_layers,
    inference_mode,
    new_cache,
    replaceable_full_attention_layers,
    restore_student,
)


def tiny_model():
    torch.manual_seed(101)
    layer_types = [
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
        "sliding_attention", "full_attention",
    ]
    per_layer_config = {
        i: {"head_dim": 32}
        for i, kind in enumerate(layer_types)
        if kind == "full_attention"
    }
    cfg = Gemma4TextConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=layer_types,
        per_layer_config=per_layer_config,
        num_kv_shared_layers=2,
        sliding_window=16,
        max_position_embeddings=128,
        vocab_size_per_layer_input=97,
        hidden_size_per_layer_input=8,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        final_logit_softcapping=None,
        use_cache=True,
    )
    cfg._attn_implementation = "sdpa"
    return Gemma4ForCausalLM(cfg).eval()


def test_gemma4_discovers_only_independent_global_layers():
    teacher = tiny_model()
    assert full_attention_layers(teacher) == [1, 3, 5, 7]
    assert replaceable_full_attention_layers(teacher) == [1, 3]
    build_student(teacher, [1, 3], "cenn_partition", features=8, block_size=4, sinks=1)
    for unsafe in (0, 5, 7):
        try:
            build_student(teacher, [unsafe], "cenn_partition", features=8, block_size=4, sinks=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected Gemma 4 layer {unsafe} to be rejected")


def test_transformer_readout_preserves_gemma4_attention_contract():
    teacher = tiny_model()
    student = build_student(teacher, [1, 3], "transformer_readout", features=8, block_size=4, sinks=1)
    ids = torch.randint(0, 97, (1, 19))
    with torch.no_grad():
        reference = teacher(input_ids=ids, use_cache=False).logits
        candidate = student(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(candidate, reference, atol=5e-5, rtol=5e-4)


def test_cenn_cache_advances_and_checkpoint_roundtrip():
    teacher = tiny_model()
    student = build_student(teacher, [1, 3], "cenn_partition", features=8, block_size=4, sinks=1)
    ids = torch.randint(0, 97, (1, 19))
    with inference_mode(student):
        full = student(input_ids=ids, use_cache=False).logits
        cache = new_cache(student)
        first = student(input_ids=ids[:, :7], past_key_values=cache, use_cache=True).logits
        assert cache.get_seq_length(1) == 7
        pieces = [first]
        for i in range(7, ids.shape[1]):
            pieces.append(student(input_ids=ids[:, i:i+1], past_key_values=cache, use_cache=True).logits)
            assert cache.get_seq_length(1) == i + 1
        cached = torch.cat(pieces, 1)
        assert cached.shape == full.shape
        assert torch.isfinite(cached).all()
        assert cache.nbytes > 0
        nmse = float((cached.float()-full.float()).square().mean() / full.float().square().mean().clamp_min(1e-8))
        assert nmse < 0.05

    payload = adapter_payload(student, {"test": True})
    recovered = restore_student(teacher, payload)
    with inference_mode(student), inference_mode(recovered):
        torch.testing.assert_close(
            student(input_ids=ids, use_cache=False).logits,
            recovered(input_ids=ids, use_cache=False).logits,
            atol=0,
            rtol=0,
        )
