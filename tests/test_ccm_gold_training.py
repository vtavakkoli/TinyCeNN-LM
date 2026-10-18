"""No pretrained model download needed for gold-label training contract checks."""
import math
from types import SimpleNamespace

import torch
from torch import nn
import pytest

from scripts.ccm_gold_training import gold_training_step, check_gold_metrics
from scripts.benchmark_ccm_consensus_smol360 import ConsensusAttentionAdapter
from tinycenn_lm.competitive_consensus_v2 import ConsensusConfig, CompetitiveConsensusCeNN


class Native(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states), None


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = Native()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y, _ = self.self_attn(hidden_states=self.norm(x), use_cache=False)
        return x + y


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(17, 24)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(24)])
        self.lm_head = nn.Linear(24, 17, bias=False)

    def forward(self, input_ids, use_cache=False, return_dict=True):
        x = self.model.layers[0](self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(x))


def test_gold_only_training_and_restore():
    torch.manual_seed(33)
    model = TinyLM()
    model.requires_grad_(False)
    native = model.model.layers[0].self_attn
    cell = CompetitiveConsensusCeNN(ConsensusConfig(hidden_size=24, slots=8,
                                                     tag_width=8, cell_width=12))
    adapter = ConsensusAttentionAdapter(cell)
    optimizer = torch.optim.AdamW(cell.parameters(), lr=0.002)
    before = [p.detach().clone() for p in cell.parameters()]
    batch = torch.randint(0, 17, (2, 10))
    metrics = gold_training_step(model, native, adapter, batch, optimizer)
    assert math.isfinite(metrics['train_gold_ce'])
    assert 0 <= metrics['train_gold_accuracy'] <= 1
    assert model.model.layers[0].self_attn is native
    assert any(not torch.equal(a, b) for a, b in zip(before, cell.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_invalid_gold_batch_rejected_without_mutation():
    model = TinyLM()
    model.requires_grad_(False)
    native = model.model.layers[0].self_attn
    cell = CompetitiveConsensusCeNN(ConsensusConfig(hidden_size=24, slots=8,
                                                     tag_width=8, cell_width=12))
    adapter = ConsensusAttentionAdapter(cell)
    optimizer = torch.optim.AdamW(cell.parameters(), lr=0.001)
    with pytest.raises(ValueError):
        gold_training_step(model, native, adapter, torch.ones(1, 1, dtype=torch.long), optimizer)
    assert model.model.layers[0].self_attn is native


def test_gold_score_ignores_teacher_agreement():
    metrics = {'teacher_nll': 2.0, 'student_nll': 2.03,
               'teacher_gold_accuracy': .4, 'student_gold_accuracy': .39,
               'teacher_agreement': 0.0}
    x = check_gold_metrics(metrics)
    assert x['gold_nll_delta'] == pytest.approx(.03)
    assert x['gold_accuracy_drop'] == pytest.approx(.01)
