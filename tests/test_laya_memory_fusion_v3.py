from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tinycenn_lm.laya_lab.memory_fusion_v3 import (
    MemoryFusionV3Attention, LayaMemoryFusionV3Config, _candidate_layers, _state_cpu,
)


class Original(nn.Module):
    def __init__(self, radius=None):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, num_attention_heads=2)
        self.layer_idx = 0
        self.head_dim = 8
        self.sliding_window = radius
        self.Wqkv = nn.Linear(16, 48)
        self.Wo = nn.Linear(16, 16)
        self.out_drop = nn.Dropout(0)


def replacement(radius=None):
    return MemoryFusionV3Attention(Original(radius), 4, 4, (1, 2, 4))


@pytest.mark.parametrize('radius', [None, 0, 2, 40])
def test_finite_forward_backward_and_padding(radius):
    m = replacement(radius)
    x = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1] = False
    y, _ = m(x, attention_mask=mask)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert m.hedge_q.grad is not None


def test_sliding_no_outside_window_or_padding_leakage():
    torch.manual_seed(3)
    m = replacement(2).eval()
    x = torch.randn(1, 12, 16)
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[:, 6] = False
    y = m(x, attention_mask=mask)[0]
    modified = x.clone()
    modified[:, 7:] += 100
    modified[:, 6] -= 100
    z = m(modified, attention_mask=mask)[0]
    torch.testing.assert_close(y[:, 4], z[:, 4])
    with pytest.raises(ValueError):
        m.memory_enabled = True


@pytest.mark.parametrize('radius', [0, 2, 70])
def test_window_feature_branch_matches_dense_reference_across_chunks(radius):
    m = replacement(radius)
    q, k, v = [torch.randn(1, 2, 131, 8) for _ in range(3)]
    valid = torch.rand(1, 131) > .2
    out = m._hedgehog_full(q, k, v, valid)
    sharp = m.hedge_log_sharpness.clamp(-1.4, 2.1).exp()[None, :, None, None]
    qf = ((torch.einsum('bhtd,hrd->bhtr', q, m.hedge_q) + m.hedge_q_bias[None, :, None]) * sharp).softmax(-1) * 2
    kf = ((torch.einsum('bhtd,hrd->bhtr', k, m.hedge_k) + m.hedge_k_bias[None, :, None]) * sharp).softmax(-1) * 2
    weights = qf @ kf.transpose(-1, -2)
    pos = torch.arange(131)
    weights = weights * ((pos[:, None] - pos[None, :]).abs() <= radius)[None, None] * valid[:, None, None]
    expected = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)) @ v
    expected *= valid[:, None, :, None]
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_checkpoint_restores_branch_topology_and_exact_output():
    m = replacement().eval()
    x = torch.randn(1, 7, 16)
    before = m(x)[0].detach()
    saved = _state_cpu(m)
    m.enable_memory_refinement()
    assert m.memory_enabled
    m.load_state_dict(saved)
    assert not m.memory_enabled
    torch.testing.assert_close(m(x)[0], before)
    m.enable_memory_refinement()
    active = _state_cpu(m)
    other = replacement().eval()
    other.load_state_dict(active)
    assert other.memory_enabled
    torch.testing.assert_close(other(x)[0], m(x)[0])


def test_all_layers_and_explicit_candidates():
    model = SimpleNamespace(encoder=SimpleNamespace(layers=[
        SimpleNamespace(attention_type=t) for t in ['full_attention', 'sliding_attention', 'sliding_attention']
    ]))
    assert _candidate_layers(model, LayaMemoryFusionV3Config(target_all_attention=True)) == [0, 1, 2]
    assert _candidate_layers(model, LayaMemoryFusionV3Config(target_layers=(2, 1, 2))) == [2, 1]
    with pytest.raises(ValueError):
        _candidate_layers(model, LayaMemoryFusionV3Config(target_all_attention=True, target_layers=(0,)))
    with pytest.raises(ValueError):
        _candidate_layers(model, LayaMemoryFusionV3Config(target_layers=(3,)))


def test_real_modernbert_mixed_layers_forward_backward():
    from transformers import ModernBertConfig, ModernBertModel
    config = ModernBertConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, max_position_embeddings=32,
        local_attention=4, global_attn_every_n_layers=3,
        pad_token_id=0, reference_compile=False,
    )
    model = ModernBertModel(config)
    radii = []
    for layer in model.layers:
        layer.attn = MemoryFusionV3Attention(layer.attn, 4, 4, (1, 2))
        radii.append(layer.attn.sliding_window)
    assert radii == [None, 2, 2]
    ids = torch.randint(1, 32, (2, 12))
    mask = torch.ones_like(ids)
    mask[1, -3:] = 0
    out = model(input_ids=ids, attention_mask=mask).last_hidden_state
    assert out.shape == (2, 12, 16) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(layer.attn.hedge_q.grad is not None for layer in model.layers)


def test_runner_cumulative_acceptance_and_rejection(monkeypatch, tmp_path):
    import sys
    import tinycenn_lm.laya_lab.memory_fusion_v3 as lab
    class Layer(nn.Module):
        def __init__(self, kind, radius):
            super().__init__()
            self.attention_type = kind
            self.attn = Original(radius)
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.config = SimpleNamespace()
    model.encoder.layers = nn.ModuleList([
        Layer('full_attention', None), Layer('sliding_attention', 2),
        Layer('sliding_attention', 2),
    ])
    agent = SimpleNamespace(model=model, device=torch.device('cpu'), dtype=torch.float32,
                            cfg={}, tok=SimpleNamespace(pad_token_id=0))
    monkeypatch.setitem(sys.modules, 'laya', SimpleNamespace(load=lambda *a, **kw: agent))
    monkeypatch.setattr(lab, '_load_typed_split', lambda split: [])
    monkeypatch.setattr(lab, '_split_train_gate_rows', lambda *a: ([], []))
    monkeypatch.setattr(lab, 'dataset_cases', lambda *a: [])
    monkeypatch.setattr(lab, '_stratified_test_cases', lambda *a: [])
    monkeypatch.setattr(lab, 'build_training_items', lambda *a: [{}] * 32)
    monkeypatch.setattr(lab, '_build_teacher_cache', lambda *a, **kw: [])
    fitted = []
    def fit(repl, *a):
        fitted.append(repl.sliding_window)
        return {'nmse': .1, 'cosine': .95}, True
    monkeypatch.setattr(lab, '_train_functional_round', fit)
    evaluated = []
    def evaluate(a, *args, **kw):
        if kw.get('label') == 'student':
            evaluated.append([isinstance(l.attn, MemoryFusionV3Attention) for l in a.model.encoder.layers])
            # Earlier accepted layers must be frozen before fitting the next.
            if len(evaluated) > 1:
                assert not any(p.requires_grad for p in a.model.encoder.layers[0].attn.parameters())
        return {'accuracy': 1., 'by_workflow': {}}
    monkeypatch.setattr(lab, 'evaluate_agent', evaluate)
    decisions = iter([True, False, True])
    monkeypatch.setattr(lab, '_accept', lambda *a: (next(decisions), {}, 0.))
    monkeypatch.setattr(lab, '_demo_and_latency', lambda *a: ({}, {'teacher': {'median_ms': 2}, 'student': {'median_ms': 1}}))
    cfg = LayaMemoryFusionV3Config(target_all_attention=True, output_dir=str(tmp_path),
        max_rounds=1, decision_refine_steps=0, train_cache_batches=1, probe_cache_batches=1,
        batch_size=1, feature_dim=4, memory_rank=4, dilations=(1,))
    _, student, report = lab.run_memory_fusion_v3(cfg)
    assert fitted == [None, 2, 2]
    assert evaluated == [[True, False, False], [True, True, False], [True, False, True]]
    assert report['accepted_layers'] == [0, 2]
    assert report['remaining_attention_layers'] == [1]
    assert report['status'] == 'partial' and not report['all_attention_replaced']
    assert isinstance(student.model.encoder.layers[1].attn, Original)
    assert (tmp_path / 'memory_fusion_v3' / 'progress.pt').exists()
