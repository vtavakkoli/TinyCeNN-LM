import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tinycenn_lm.laya_lab.core import LayaLabConfig
from tinycenn_lm.laya_lab.integrated import IntegratedMemoryV22Attention
from tinycenn_lm.laya_lab.integrated_train import _integrated_checkpoint_rank


class NativeAttention(nn.Module):
    def __init__(self, radius=None):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, num_attention_heads=2, local_attention=2 * (radius if radius is not None else 1))
        self.sliding_window = radius
        self.head_dim = 8
        self.Wqkv = nn.Linear(16, 48)
        self.Wo = nn.Linear(16, 16)
        self.out_drop = nn.Identity()

    def forward(self, x, position_embeddings=None, attention_mask=None):
        q, k, v = self.Wqkv(x).reshape(*x.shape[:2], 3, 2, 8).unbind(2)
        core = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               attn_mask=attention_mask[:, None, None, :].bool())
        return self.Wo(core.transpose(1, 2).reshape_as(x)), None


@pytest.mark.parametrize('radius,length', [(0,7), (2,19), (4,70), (None,17)])
def test_kernel_matches_dense_reference_and_gradients(radius, length):
    torch.manual_seed(41)
    mod = IntegratedMemoryV22Attention(NativeAttention(radius), 4, 3)
    x = torch.randn(2, length, 16)
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[1, -3:] = False
    mod(x, attention_mask=valid)
    actual = mod.last_core_output
    q, k, v = mod.qkv(x)
    qf, kf = mod._project_features(q, mod.wq), mod._project_features(k, mod.wk)
    allowed = valid[:, None, None, :]
    if radius is not None:
        pos = torch.arange(length)
        allowed = allowed & ((pos[:, None] - pos[None, :]).abs() <= radius)[None, None]
    weights = (qf @ kf.transpose(-1, -2)) * allowed
    global_out = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)) @ v.float()
    mix = mod.mix_logits.softmax(-1)
    # Identity convolution at initialization equals direct values.
    expected = (mix[:, 0][None, :, None, None] * global_out
                + (mix[:, 1] + mix[:, 2])[None, :, None, None] * v.float())
    expected = expected * valid[:, None, :, None]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    grads = torch.autograd.grad(actual.square().sum(), (mod.wq, mod.wk), retain_graph=True)
    ref_grads = torch.autograd.grad(expected.square().sum(), (mod.wq, mod.wk))
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('radius', [None, 1, 3])
def test_padding_and_window_do_not_leak(radius):
    mod = IntegratedMemoryV22Attention(NativeAttention(radius), 4, 5)
    with torch.no_grad():
        mod.local_weight.normal_()
    x = torch.randn(1, 12, 16)
    valid = torch.ones(1, 12, dtype=torch.bool)
    valid[:, -3:] = False
    a = mod(x, attention_mask=valid)[0]
    changed = x.clone()
    changed[:, -3:] += 100
    b = mod(changed, attention_mask=valid)[0]
    torch.testing.assert_close(a[valid], b[valid])
    if radius is not None:
        changed = x.clone()
        changed[:, radius+1:] += 100
        b = mod(changed, attention_mask=valid)[0]
        torch.testing.assert_close(a[:, 0], b[:, 0])
    mod(x, attention_mask=torch.zeros_like(valid))
    assert mod.last_core_output.count_nonzero() == 0


def test_checkpoint_prefers_decisions_after_local_gates_pass():
    cfg = LayaLabConfig()
    good = _integrated_checkpoint_rank({'nmse': .2, 'cosine': .92},
        {'teacher_student_top1_agreement': 1., 'mean_teacher_kl': .002}, cfg)
    worse = _integrated_checkpoint_rank({'nmse': .1, 'cosine': .96},
        {'teacher_student_top1_agreement': .96, 'mean_teacher_kl': .001}, cfg)
    assert good < worse


class DecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(24, 16)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.encoder.layers:
            layer.attn = NativeAttention()
        self.head = nn.Linear(16, 3)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        x = self.embed(input_ids)
        for layer in self.encoder.layers:
            x = x + layer.attn(x, attention_mask=attention_mask)[0]
        logits = self.head(x.mean(1))
        return logits, logits[:, :2]


def test_training_updates_only_candidate_and_restores_installation(monkeypatch):
    import tinycenn_lm.laya_lab.integrated_train as lab
    torch.manual_seed(7)
    model = DecisionModel().eval().requires_grad_(False)
    teacher = SimpleNamespace(model=model, device=torch.device('cpu'), dtype=torch.float32,
                              tok=SimpleNamespace(pad_token_id=0))
    student = copy.copy(teacher)
    student.model = copy.deepcopy(model)
    upstream = IntegratedMemoryV22Attention(student.model.encoder.layers[0].attn, 4)
    student.model.encoder.layers[0].attn = upstream
    before = {k: v.clone() for k, v in student.model.state_dict().items()}
    old = student.model.encoder.layers[1].attn
    def batch(items, offset, size, pad):
        ids = torch.tensor([items[(offset+i) % len(items)] for i in range(size)])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    marker_pos=torch.zeros(size, 3, dtype=torch.long),
                    marker_mask=torch.ones(size, 3, dtype=torch.bool), qtype=torch.zeros(size, dtype=torch.long))
    monkeypatch.setattr(lab, '_batch_from_items', batch)
    calls = []
    def fast(t, s, items, batch_size, limit):
        b = batch(items, 0, len(items), 0)
        with torch.no_grad():
            tp = t.model(**b)[0].softmax(-1)
            sp = s.model(**b)[0].softmax(-1)
        calls.append(sp)
        return {'teacher_student_top1_agreement': float((tp.argmax(-1) == sp.argmax(-1)).float().mean()),
                'mean_teacher_kl': float((tp*(tp.log()-sp.log())).sum(-1).mean())}
    monkeypatch.setattr(lab, 'fast_teacher_student_eval', fast)
    cfg = LayaLabConfig(feature_dim=4, learning_rate=.01, integrated_probe_every=2, early_stop_local=False)
    replacement, metrics = lab.train_integrated_replacement(teacher, student, 1, cfg,
        [[1,2,3,4], [2,4,6,8]], [[3,5,7,9], [8,7,4,3]], 4, 2)
    assert len(calls) == 3
    assert metrics['selection'] == 'fixed_local_and_cumulative_decision_probe'
    assert student.model.encoder.layers[1].attn is old
    assert not torch.equal(replacement.wq, replacement.wk)  # learned separate gradients
    for k, v in student.model.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert all(not p.requires_grad for p in student.model.parameters())
    assert all(p.grad is None for p in teacher.model.parameters())
    # Exceptions must also restore the original module and remove hooks.
    def fail(*args, **kwargs):
        raise RuntimeError('probe failure')
    monkeypatch.setattr(lab, 'fast_teacher_student_eval', fail)
    with pytest.raises(RuntimeError, match='probe failure'):
        lab.train_integrated_replacement(teacher, student, 1, cfg,
            [[1,2,3,4]], [[3,5,7,9]], 2, 1)
    assert student.model.encoder.layers[1].attn is old
    assert not teacher.model.encoder.layers[1].attn._forward_hooks


def test_real_modernbert_full_and_sliding_roundtrip():
    from transformers import ModernBertConfig, ModernBertModel
    cfg = ModernBertConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=128,
        local_attention=8, global_attn_every_n_layers=2,
        reference_compile=False, attn_implementation='sdpa')
    model = ModernBertModel(cfg).eval()
    restored = copy.deepcopy(model)
    for i, layer in enumerate(model.layers):
        layer.attn = IntegratedMemoryV22Attention(layer.attn, feature_dim=4).eval()
        other = IntegratedMemoryV22Attention(restored.layers[i].attn, feature_dim=4).eval()
        other.load_state_dict(layer.attn.state_dict())
        restored.layers[i].attn = other
    assert model.layers[0].attn.sliding_window is None
    assert model.layers[1].attn.sliding_window == 4
    ids = torch.randint(1, 32, (2, 70))
    mask = torch.ones_like(ids)
    mask[1, -5:] = 0
    actual = model(ids, attention_mask=mask).last_hidden_state
    expected = restored(ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for layer in model.layers:
        assert layer.attn.wq.grad is not None
        assert torch.isfinite(layer.attn.wq.grad).all()


def test_runner_rolls_back_rejection_and_reports_independent_final_failure(monkeypatch, tmp_path):
    import sys
    import types
    import tinycenn_lm.laya_lab.runner as runner
    import tinycenn_lm.laya_lab.integrated_train as training
    model = DecisionModel()
    for i, layer in enumerate(model.encoder.layers):
        layer.attention_type = 'full_attention' if i == 0 else 'sliding_attention'
    agent = SimpleNamespace(model=model, device=torch.device('cpu'), dtype=torch.float32,
        tok=SimpleNamespace(pad_token_id=0), cfg={'max_len': 32}, predict=lambda *a: {})
    monkeypatch.setitem(sys.modules, 'laya', types.SimpleNamespace(load=lambda *a, **k: agent))
    rows = [{'state': {'id': i}, 'questions': {}, 'gold': {}, 'workflow': str(i % 2)} for i in range(100)]
    monkeypatch.setattr(runner, '_load_typed_split', lambda split: rows)
    monkeypatch.setattr(runner, 'build_training_items', lambda *a: [[1,2,3]])
    monkeypatch.setattr(runner, 'benchmark_latency', lambda *a: {'median_ms': 1.})
    seen = {}
    def evaluate(agent, cases, teacher_agent=None, label=''):
        seen.setdefault(label, []).append({c[0]['id'] for c in cases})
        replaced = sum(isinstance(l.attn, IntegratedMemoryV22Attention) for l in agent.model.encoder.layers)
        agreement = 1. if label == 'student' and replaced == 1 else .9
        return {'accuracy': .8, 'teacher_agreement': agreement, 'mean_teacher_kl': .001}
    monkeypatch.setattr(runner, 'evaluate_agent', evaluate)
    def train(teacher, student, idx, cfg, *args):
        return IntegratedMemoryV22Attention(teacher.model.encoder.layers[idx].attn, 4), {'nmse': .1, 'cosine': .99}
    monkeypatch.setattr(training, 'train_integrated_replacement', train)
    cfg = LayaLabConfig(mode='smoke', target_all_attention=True, integrated_functional_training=True,
                        output_dir=str(tmp_path))
    _, student, report = runner.run_experiment(cfg)
    assert report['candidate_layers'] == [0, 1]
    assert report['accepted_layers'] == [0]
    assert report['remaining_attention_layers'] == [1]
    assert report['target_acceptance_rate'] == .5
    assert not report['all_attention_replaced']
    assert not report['final_quality_passed']
    assert isinstance(student.model.encoder.layers[1].attn, NativeAttention)
    assert seen['teacher'][0].isdisjoint(seen['teacher'][1])
    assert report['replacement_trainable_parameters'] > 0
    assert (tmp_path / cfg.architecture / 'adapter.pt').exists()
