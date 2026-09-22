from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tinycenn_lm.laya_lab.core import LayaLabConfig
from tinycenn_lm.laya_lab.factory import make_replacement
from tinycenn_lm.laya_lab.pdelta import PDelta3GDN2CLVRAttention
from tinycenn_lm.laya_lab.pdelta_optimized import _checkpoint_rank, _split_transfer_rows


def original():
    module = nn.Module()
    module.config = SimpleNamespace(hidden_size=16, num_attention_heads=2)
    module.head_dim = 8
    module.Wqkv = nn.Linear(16, 48)
    module.Wo = nn.Linear(16, 16)
    module.out_drop = nn.Identity()
    return module


@pytest.mark.parametrize('length,window', [(1,32), (19,32), (65,32), (17,5), (7,1)])
def test_local_matches_dense_bidirectional_reference_and_gradients(length, window):
    torch.manual_seed(10)
    module = PDelta3GDN2CLVRAttention(original(), feature_dim=4, local_window=window)
    q, k, v = [torch.randn(2, 2, length, 8, requires_grad=True) for _ in range(3)]
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[1, length//2:] = False
    result = module._local_attention(q, k, v, valid)
    positions = torch.arange(length)
    distance = positions[None, :] - positions[:, None]
    allowed = (distance >= -((window-1)//2)) & (distance <= window//2)
    allowed = allowed[None, None] & valid[:, None, None, :]
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    expected = expected * valid[:, None, :, None]
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    actual_grads = torch.autograd.grad(result.square().sum(), (q,k,v), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), (q,k,v))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_padding_cannot_leak_through_trained_convolutions():
    torch.manual_seed(1)
    module = PDelta3GDN2CLVRAttention(original(), feature_dim=4, local_window=32).eval()
    with torch.no_grad():
        for p in (module.q_conv, module.k_conv, module.v_conv, module.local_weight):
            p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(2, 12, 16)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[0, :3] = False
    valid[1, 8:] = False
    changed = x.clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 100
    a, _ = module(x, attention_mask=valid)
    b, _ = module(changed, attention_mask=valid)
    torch.testing.assert_close(a[valid], b[valid])
    out, _ = module(x, attention_mask=torch.zeros_like(valid))
    assert torch.isfinite(out).all()
    assert module.last_core_output.count_nonzero() == 0


def test_probe_split_keeps_duplicate_states_and_questions_together():
    rows = [{'state': {'id': i}, 'questions': {'a': {}, 'b': {}}} for i in range(30)]
    rows += rows[:10]
    fit, probe = _split_transfer_rows(rows, 2026)
    assert {r['state']['id'] for r in fit}.isdisjoint({r['state']['id'] for r in probe})
    assert len(fit) + len(probe) == len(rows)
    assert _split_transfer_rows(rows, 2026) == (fit, probe)


def test_checkpoint_rank_prefers_probe_passing_over_smaller_aggregate_error():
    cfg = LayaLabConfig()
    fast = {'mean_teacher_kl': 0.01, 'teacher_student_top1_agreement': 0.98}
    passed = {'nmse': 0.29, 'cosine': 0.89}
    failed = {'nmse': 0.10, 'cosine': 0.87}
    assert _checkpoint_rank(passed, fast, cfg) < _checkpoint_rank(failed, fast, cfg)


def test_factory_roundtrip_and_hybrid_gradients():
    cfg = LayaLabConfig(architecture='pdelta3_gdn2_clvr', feature_dim=4, pdelta_local_window=32)
    module = make_replacement(original(), cfg)
    x = torch.randn(2, 9, 16)
    y, _ = module(x)
    y.square().mean().backward()
    for name in ('local_gate_w','local_gate_b','global_wq','wq'):
        grad = getattr(module, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    restored = make_replacement(original(), cfg)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(x)[0], y)


def test_training_saves_rejected_best_adapter_before_rollback(monkeypatch, tmp_path):
    import copy
    import tinycenn_lm.laya_lab.pdelta_optimized as lab

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            layer = nn.Module()
            layer.attn = original()
            layer.attention_type = 'full_attention'
            self.encoder.layers = nn.ModuleList([layer])

    teacher = SimpleNamespace(model=Model(), device=torch.device('cpu'), dtype=torch.float32,
                              tok=SimpleNamespace(pad_token_id=0))
    student = copy.deepcopy(teacher)
    old = student.model.encoder.layers[0].attn
    x = torch.randn(2, 7, 16)
    batch = {'attention_mask': torch.ones(2,7,dtype=torch.bool),
             'marker_mask': torch.ones(2,3,dtype=torch.bool)}

    def teacher_capture(agent, layer_idx, batch):
        with torch.no_grad():
            attn = agent.model.encoder.layers[layer_idx].attn
            q,k,v = attn.Wqkv(x).reshape(2,7,3,2,8).unbind(2)
            core = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
            y = attn.Wo(core.transpose(1,2).reshape(2,7,16))
            return {'x': x, 'pos': None, 'mask': batch['attention_mask'], 'y': y, 'core': core}, (y[:,0,:3],y[:,0,3:6])

    def student_capture(agent, layer_idx, batch, grad):
        attn = agent.model.encoder.layers[layer_idx].attn
        y,_ = attn(x, attention_mask=batch['attention_mask'])
        return {'y': y, 'core': attn.last_core_output}, (y[:,0,:3],y[:,0,3:6])

    monkeypatch.setattr(lab, '_batch_from_items', lambda *args: batch)
    monkeypatch.setattr(lab, '_teacher_forward_capture', teacher_capture)
    monkeypatch.setattr(lab, '_student_forward_capture', student_capture)
    monkeypatch.setattr(lab, 'fast_teacher_student_eval', lambda *a,**kw: {
        'teacher_student_top1_agreement': .8, 'mean_teacher_kl': .02,
        'mean_probability_l1': .04, 'forward_speedup_vs_teacher': .5})
    monkeypatch.setattr(lab, 'evaluate_agent', lambda *a,**kw: {'accuracy': .3})
    monkeypatch.setattr(lab, '_accept', lambda *args: (False, {'test_rejection': False}, .1))
    cfg = LayaLabConfig(architecture='pdelta3_gdn2_clvr', feature_dim=4,
                        pdelta_local_window=32, pdelta_train_qkv=True,
                        learning_rate=.01, final_learning_rate=.001, output_dir=str(tmp_path))
    accepted, rec = lab._train_candidate(teacher, student, 0, cfg, [batch], [batch], [batch], [], {}, 3, 2)
    assert not accepted
    assert student.model.encoder.layers[0].attn is old
    saved = torch.load(tmp_path / cfg.architecture / 'candidate_layer_0.pt', weights_only=False)
    assert saved['state_dict'] is not None
    assert saved['metrics']['best_step'] == rec['best_step']
    assert 'local_gate_w' in saved['state_dict']
    assert not torch.equal(saved['state_dict']['Wqkv.weight'], old.Wqkv.weight)
    assert rec['training']['core_learning_rate_start'] == .01
    assert rec['training']['core_learning_rate_end'] == .001


def test_local_probe_metrics_do_not_depend_on_microbatch_size():
    from tinycenn_lm.laya_lab.pdelta_optimized import _probe_local_cached
    torch.manual_seed(7)
    module = PDelta3GDN2CLVRAttention(original(), feature_dim=4, local_window=32).eval()
    x = torch.randn(2,9,16)
    valid = torch.ones(2,9,dtype=torch.bool)
    valid[1,5:] = False
    probe = {'x': x, 'pos': None, 'mask': valid, 'valid': valid,
             'y': torch.randn(2,9,16), 'core': torch.randn(2,2,9,8)}
    probe['y'][1] *= 10
    probe['core'][1] *= 5
    agent = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float32)
    whole = _probe_local_cached(module, probe, agent)
    split = [{k: (v[i:i+1] if torch.is_tensor(v) else v) for k,v in probe.items()} for i in range(2)]
    parts = _probe_local_cached(module, split, agent)
    for key in whole:
        assert parts[key] == pytest.approx(whole[key], rel=1e-5, abs=1e-6)


def test_real_modernbert_rope_and_mask_contract():
    import copy
    from transformers import ModernBertConfig, ModernBertModel
    cfg = ModernBertConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
                           num_hidden_layers=2, num_attention_heads=4,
                           max_position_embeddings=64, local_attention=16,
                           global_attn_every_n_layers=2, reference_compile=False,
                           attn_implementation='sdpa')
    teacher = ModernBertModel(cfg).eval()
    student = copy.deepcopy(teacher)
    replacement = PDelta3GDN2CLVRAttention(student.layers[0].attn, feature_dim=4, local_window=32).eval()
    student.layers[0].attn = replacement
    # With the full short sequence inside Local32 and gate -> 1, the
    # copied QKV/RoPE/Wo path must reproduce the actual encoder, not a dummy.
    with torch.no_grad():
        replacement.local_gate_b.fill_(25)
        ids = torch.randint(1,100,(2,9))
        mask = torch.ones_like(ids)
        mask[1,6:] = 0
        expected = teacher(ids, attention_mask=mask).last_hidden_state
        actual = student(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=2e-5, rtol=2e-5)
