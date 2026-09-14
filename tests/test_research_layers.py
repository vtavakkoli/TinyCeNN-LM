"""Numerical gates for the research prototypes; these are not LM quality results."""
import copy

import pytest
import torch
import torch.nn.functional as F

from tinycenn_lm.research_layers import (
    VARIANTS, ResearchCeNNLayer, delta_recurrence, delta_token_reference,
    local_window_attention, softmax_reference,
)


@pytest.mark.parametrize("chunk", [1, 7, 32])
@pytest.mark.parametrize("groups", [1, 3])
def test_asymmetric_chunk_matches_independent_recurrence_and_gradients(chunk, groups):
    torch.manual_seed(5)
    b, h, t, f, d = 2, 2, 37, 5, 4
    q = torch.randn(b, h * groups, t, f, dtype=torch.double)
    k = F.normalize(torch.randn(b, h, t, f, dtype=torch.double), dim=-1)
    z = torch.randn(b, h, t, d, dtype=torch.double)
    erase = k * torch.rand_like(k)
    decay = -0.25 * torch.rand_like(k)
    state = torch.randn(b, h, f, d, dtype=torch.double)
    args1 = [x.requires_grad_(True) for x in (q, k, z, erase, decay, state)]
    args2 = [x.detach().clone().requires_grad_(True) for x in args1]
    out, end = delta_recurrence(*args1, groups, chunk)
    expected, end_expected = delta_token_reference(*args2, groups)
    torch.testing.assert_close(out, expected, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(end, end_expected, atol=1e-9, rtol=1e-9)
    probe = torch.randn_like(out)
    grads = torch.autograd.grad((out * probe).sum() + end.square().sum(), args1)
    refs = torch.autograd.grad((expected * probe).sum() + end_expected.square().sum(), args2)
    for grad, ref in zip(grads, refs):
        torch.testing.assert_close(grad, ref, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gqa_streaming_causality_state_and_gradients(variant):
    torch.manual_seed(20)
    module = ResearchCeNNLayer(6, 2, 8, 12, variant, window=5, chunk_size=7)
    q, k, v = torch.randn(2, 6, 23, 8), torch.randn(2, 2, 23, 8), torch.randn(2, 2, 23, 8)
    out, end = module(q, k, v, return_state=True)
    parts, state = [], None
    for a, b in [(0, 3), (3, 8), (8, 9), (9, 23)]:
        value, state = module(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b],
                              state=state, return_state=True)
        parts.append(value)
    torch.testing.assert_close(out, torch.cat(parts, dim=2), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(end.memory, state.memory, atol=2e-5, rtol=2e-4)
    assert state.nbytes == module.recurrent_state_bytes(batch_size=2)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, :, 11:] += 5
    k2[:, :, 11:] -= 4
    v2[:, :, 11:] *= 20
    torch.testing.assert_close(out[:, :, :11], module(q2, k2, v2)[:, :, :11],
                               atol=2e-5, rtol=2e-4)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())
    clone = ResearchCeNNLayer(**module.config)
    clone.load_state_dict(module.state_dict())
    torch.testing.assert_close(out, clone(q, k, v))


@pytest.mark.parametrize("window", [1, 4, 16])
def test_bounded_window_matches_masked_softmax(window):
    torch.manual_seed(22)
    q, k, v = torch.randn(2, 6, 9, 8), torch.randn(2, 2, 9, 8), torch.randn(2, 2, 9, 8)
    t = q.shape[2]
    i = torch.arange(t)
    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=mask
    )
    torch.testing.assert_close(local_window_attention(q, k, v, window, 3),
                               expected, atol=1e-6, rtol=1e-5)


def test_delta2_reduces_to_kda_with_tied_gates():
    torch.manual_seed(7)
    kda = ResearchCeNNLayer(4, 2, 8, 8, "cenn_kda")
    untied = ResearchCeNNLayer(4, 2, 8, 8, "cenn_delta2")
    q, k, v = torch.randn(1, 4, 17, 8), torch.randn(1, 2, 17, 8), torch.randn(1, 2, 17, 8)
    # Gate weights start at zero with the same scalar bias, so untied gates
    # collapse to the tied rule; all feature maps are identity at F=D.
    torch.testing.assert_close(kda(q, k, v), untied(q, k, v), atol=1e-6, rtol=1e-5)


def test_optimizer_can_improve_attention_transfer():
    torch.manual_seed(41)
    model = ResearchCeNNLayer(2, 1, 8, 8, "cenn_delta2_window", window=4)
    q, k, v = torch.randn(1, 2, 8, 8), torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)
    target = softmax_reference(q, k, v, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = (model(q, k, v) - target).square().mean().item()
    for _ in range(25):
        optimizer.zero_grad()
        loss = (model(q, k, v) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert (model(q, k, v) - target).square().mean().item() < initial * 0.85


def test_reject_invalid_configuration():
    with pytest.raises(ValueError):
        ResearchCeNNLayer(5, 2, 8)
    with pytest.raises(ValueError):
        ResearchCeNNLayer(4, 2, 8, chunk_size=64)
