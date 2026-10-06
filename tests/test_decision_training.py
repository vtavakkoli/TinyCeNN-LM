import copy
import json
import math
from pathlib import Path

import pytest
import torch
from tinycenn_lm.decision_training import (
    guarded_step, gold_checkpoint_key, recovery_lr, save_stage, load_stage,
)


def test_nan_loss_never_changes_weights():
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([p], lr=.01)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    before = p.detach().clone()
    with pytest.raises(FloatingPointError, match='loss'):
        guarded_step(p.sum()*float('nan'), opt, scaler, [([p], 1.)], 'Stage B')
    torch.testing.assert_close(p, before)
    assert not opt.state


def test_finite_loss_with_nonfinite_gradient_never_updates():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=.01)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    # sqrt(0) is a finite loss with infinite derivative.
    with pytest.raises(FloatingPointError, match='gradients'):
        guarded_step(p.sqrt().sum(), opt, scaler, [([p], 1.)], 'Stage B')
    assert p.item() == 0


def test_amp_overflow_skips_update_and_backs_off_scale():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=128.)
    assert not guarded_step(p.sqrt().sum(), opt, scaler, [([p], 1.)], 'Stage B')
    assert p.item() == 0 and scaler.get_scale() == 64
    opt.zero_grad(set_to_none=True)
    assert guarded_step((p-1).square().sum(), opt, scaler, [([p], 1.)], 'Stage B')
    assert p.item() > 0


def test_accuracy_first_key_rejects_the_observed_wrong_choice():
    # Original V2.4 selected 64.8% despite having seen 65.6% on dev.
    better = dict(accuracy=.656, kl_from_gold=.2667, brier=.146)
    worse = dict(accuracy=.648, kl_from_gold=.2508, brier=.1376)
    assert gold_checkpoint_key(better) < gold_checkpoint_key(worse)
    with pytest.raises(FloatingPointError):
        gold_checkpoint_key({**better, 'kl_from_gold': float('nan')})


def test_warmup_and_final_rate():
    assert recovery_lr(0, 1600, .001, .0001) == pytest.approx(.00005)
    assert recovery_lr(19, 1600, .001, .0001) == pytest.approx(.001)
    assert recovery_lr(1599, 1600, .001, .0001) == pytest.approx(.0001)


def test_stage_checkpoint_rejects_configuration_mismatch(tmp_path):
    model = torch.nn.Linear(3, 2)
    expected = copy.deepcopy(model.state_dict())
    metadata = {'stage': 'A', 'revision': 'test'}
    save_stage(model, tmp_path, metadata)
    with torch.no_grad(): model.weight.add_(10)
    load_stage(model, tmp_path, metadata)
    for key,value in expected.items(): torch.testing.assert_close(model.state_dict()[key], value)
    with pytest.raises(ValueError, match='differs'):
        load_stage(model, tmp_path, {'stage': 'A', 'revision': 'different'})


def test_v24_layer_detection_on_pinned_modernbert():
    from transformers import ModernBertConfig, AutoModel
    from tinycenn_lm.standalone_decision import StandaloneDecisionModel, full_attention_indices
    cfg = ModernBertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=4,
        num_attention_heads=2, intermediate_size=24, global_attn_every_n_layers=3,
        reference_compile=False, pad_token_id=0)
    m = StandaloneDecisionModel(AutoModel.from_config(cfg, attn_implementation='sdpa'), head_layers=0)
    assert full_attention_indices(m) == [0,3]


def test_distillation_extreme_finite_fp16_actions_remain_finite():
    from tinycenn_lm.decision_training import decision_distill_loss
    # FP16 log-softmax can produce -inf for this finite range, leading to 0*inf.
    student = torch.tensor([[60000., -60000.]], dtype=torch.float16, requires_grad=True)
    teacher = torch.tensor([[-60000., 60000.]], dtype=torch.float16)
    logits = torch.tensor([[1., -1.]], requires_grad=True)
    loss = decision_distill_loss(logits, logits.detach(), torch.ones(1, 2, dtype=torch.bool), student, teacher)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all() and torch.isfinite(logits.grad).all()
    with pytest.raises(FloatingPointError, match='student action'):
        decision_distill_loss(logits, logits.detach(), torch.ones(1, 2, dtype=torch.bool),
                             student*float('inf'), teacher)


@pytest.mark.parametrize('failure', ['loss', 'gradient', 'validation', 'parameter'])
def test_recovery_restores_best_clears_moments_and_reduces_rates(failure):
    from tinycenn_lm.decision_training import run_recovery_phase
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    broken = False
    restored = False
    last_valid = model.weight.detach().clone()
    def evaluate():
        nonlocal broken, last_valid
        if failure == 'validation' and not broken and float(model.weight.detach()) < .985:
            broken = True
            return {'loss': float('nan')}
        last_valid = model.weight.detach().clone()
        return {'loss': float(model.weight.detach().square().sum())}
    def loss_fn(step):
        nonlocal broken, restored
        if broken and not restored:
            torch.testing.assert_close(model.weight, last_valid, rtol=0, atol=0)
            assert not optimizer.state
            assert optimizer.param_groups[0]['lr'] <= .0025
            restored = True
        if step == 1 and not broken and failure != 'validation':
            broken = True
            if failure == 'parameter':
                with torch.no_grad(): model.weight.fill_(float('inf'))
            if failure == 'loss': return model.weight.sum()*float('nan')
            if failure == 'gradient': return (model.weight-model.weight.detach()).sqrt().sum()
        return model.weight.square().sum()
    with pytest.warns(UserWarning, match='restored best checkpoint'):
        result = run_recovery_phase(model, optimizer, [(list(model.parameters()), 1.)],
            4, [(.01, .002)], loss_fn, evaluate, lambda m:m['loss'], 'test',
            eval_every=1, warmup_steps=0)
    assert broken and restored and result['restarts'] == 1
    assert result['lr_factor'] == .25 and result['completed_steps'] == 4
    assert torch.isfinite(model.weight).all()


def test_recovery_exhausted_budget_restores_baseline_and_fails_closed():
    from tinycenn_lm.decision_training import run_recovery_phase
    model = torch.nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    def broken(step):
        with torch.no_grad(): model.weight.add_(1.)
        return model.weight.sum()*float('nan')
    with pytest.warns(UserWarning, match='restored best checkpoint'):
        with pytest.raises(FloatingPointError, match='retry budget exhausted'):
            run_recovery_phase(model, optimizer, [(list(model.parameters()), 1.)],
                2, [(.01, .002)], broken, lambda:{'score': 0.}, lambda m:m['score'],
                'test', max_restarts=1)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert not optimizer.state


def test_amp_skipped_batch_retries_same_step_with_fresh_phase_scale(monkeypatch):
    import tinycenn_lm.decision_training as training
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    seen = []
    real_step = training.guarded_step
    def skip_once(loss, opt, scaler, groups, context):
        if len(seen) == 1:
            assert scaler.get_scale() == 128.
            return False
        return real_step(loss, opt, scaler, groups, context)
    monkeypatch.setattr(training, 'guarded_step', skip_once)
    def loss_fn(step):
        seen.append(step)
        return model.weight.square().sum()
    result = training.run_recovery_phase(model, optimizer, [(list(model.parameters()), 1.)],
        2, [(.01, .002)], loss_fn, lambda:{'score': float(model.weight.detach().square().sum())},
        lambda m:m['score'], 'test', amp_enabled=True)
    assert seen == [0, 0, 1] and result['amp_skips'] == 1
