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
