"""Independent dense oracles, causal gradients and bounded decode-state checks."""
import copy
import math
import pytest
import torch

from tinycenn_lm.optimized_memory import OptimizedMemory, VARIANTS, ridge_calibrate


def dense_reference(core, q, k, v):
    kh, vh = k.repeat_interleave(core.groups, 1), v.repeat_interleave(core.groups, 1)
    t = q.shape[2]
    i, j = torch.arange(t)[:, None], torch.arange(t)[None, :]
    causal = j <= i
    exact_mask = causal & ((j < core.sink_tokens) | (j // core.block_size >= i // core.block_size - 1))
    logits = q @ kh.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if core.variant == "transformer_readout":
        weights = logits.masked_fill(~causal, float("-inf")).softmax(-1)
    elif core.variant == "sink_window":
        weights = logits.masked_fill(~exact_mask, float("-inf")).softmax(-1)
    else:
        approximate = core.features(q, True) @ core.features(k).repeat_interleave(core.groups, 1).transpose(-1, -2)
        if core.variant == "cenn_partition":
            raw_q = torch.nn.functional.normalize(q, dim=-1)
            mass = (core.log_mass[None, :, None] + (
                raw_q * core.mass_w[None, :, None]).sum(-1)).clamp(-12, 12).exp()
            approximate = approximate * mass.unsqueeze(-1)
            weights = torch.where(exact_mask, logits.exp(), approximate) * causal
        else:
            weights = approximate * causal
        weights = weights / weights.sum(-1, keepdim=True)
    return (weights @ vh) @ core.readout[None]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("t", [1, 8, 19])
def test_parallel_prefill_matches_dense_oracle_and_gradients(variant, t):
    torch.manual_seed(32)
    core = OptimizedMemory(6, 2, 8, 12, variant, block_size=4, sink_tokens=2)
    clone = copy.deepcopy(core)
    inputs = [torch.randn(2, h, t, 8).requires_grad_(True) for h in (6, 2, 2)]
    other = [x.detach().clone().requires_grad_(True) for x in inputs]
    out, reference = core(*inputs), dense_reference(clone, *other)
    torch.testing.assert_close(out, reference, atol=3e-5, rtol=3e-4)
    probe = torch.randn_like(out)
    gradients = torch.autograd.grad((out * probe).sum(), inputs)
    expected = torch.autograd.grad((reference * probe).sum(), other)
    for g, e in zip(gradients, expected):
        torch.testing.assert_close(g, e, atol=4e-5, rtol=4e-4)


@pytest.mark.parametrize("variant", VARIANTS)
def test_streaming_agrees_with_prefill_without_future_information(variant):
    torch.manual_seed(8)
    core = OptimizedMemory(4, 2, 8, 8, variant, block_size=4, sink_tokens=3)
    q, k, v = [torch.randn(1, h, 23, 8) for h in (4, 2, 2)]
    with torch.no_grad():
        out, final = core(q, k, v, return_state=True)
        first, state = core(q[:, :, :3], k[:, :, :3], v[:, :, :3], return_state=True)
        tail, state = core(q[:, :, 3:], k[:, :, 3:], v[:, :, 3:], state, return_state=True)
        torch.testing.assert_close(torch.cat((first, tail), 2), out, atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(final.numerator, state.numerator, atol=3e-5, rtol=3e-4)
        assert final.nbytes == state.nbytes
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        q2[:, :, 10:] += 10
        k2[:, :, 10:] *= 30
        v2[:, :, 10:] -= 40
        torch.testing.assert_close(out[:, :, :10], core(q2, k2, v2)[:, :, :10],
                                   atol=3e-5, rtol=3e-4)
        if variant != "transformer_readout":
            assert state.keys.shape[2] <= 2 * core.block_size
            assert state.sinks_k.shape[2] <= core.sink_tokens
        for cache in (state.keys, state.values, state.sinks_k, state.sinks_v):
            assert cache.untyped_storage().nbytes() == cache.numel() * cache.element_size()


def test_partition_exact_when_no_tokens_have_been_compressed():
    torch.manual_seed(4)
    core = OptimizedMemory(4, 2, 8, 8, block_size=8, sink_tokens=4)
    q, k, v = [torch.randn(1, h, 16, 8) for h in (4, 2, 2)]
    exact = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
    torch.testing.assert_close(core(q, k, v), exact, atol=2e-6, rtol=2e-5)


def test_ridge_solution_improves_training_mse_and_reloads():
    torch.manual_seed(3)
    core = OptimizedMemory(4, 2, 8, 8, "cenn_linear", block_size=4)
    samples = []
    for _ in range(3):
        q, k, v = [torch.randn(1, h, 16, 8) for h in (4, 2, 2)]
        target = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
        samples.append((q, k, v, target))
    before = sum((core(q, k, v) - target).square().sum().item() for q, k, v, target in samples)
    result = ridge_calibrate(core, samples, torch.device("cpu"))
    after = sum((core(q, k, v) - target).square().sum().item() for q, k, v, target in samples)
    assert after <= before + 1e-5
    assert result["ridge_relative_residual"] < 1e-10
    clone = OptimizedMemory(**core.config)
    clone.load_state_dict(core.state_dict())
    for q, k, v, _ in samples:
        torch.testing.assert_close(core(q, k, v), clone(q, k, v))


def test_zero_sink_and_short_final_block():
    core = OptimizedMemory(2, 1, 8, 8, block_size=4, sink_tokens=0)
    q, k, v = [torch.randn(1, h, 13, 8) for h in (2, 1, 1)]
    torch.testing.assert_close(core(q, k, v), dense_reference(core, q, k, v), atol=2e-5, rtol=2e-4)
