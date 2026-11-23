"""Regression for reduced-precision Colab parity false rejection, with strict FP32 proof."""
from types import SimpleNamespace
import torch
from torch import nn
from scripts.benchmark_ccm_v3_smol360 import parity_decision, fp32_parity_check
from tinycenn_lm.ccm_v3_rotary_consensus import RotaryCellularConsensus, RotaryConsensusConfig
from test_ccm_v3_rotary_consensus import NativeLike, cos_sin


def metrics(mx,mean,rel,agreement=1.0):
    return dict(max_abs_logits=mx, mean_abs_logits=mean,
                native_vs_candidate_relative_mse=rel,
                top1_agreement=agreement, context_fits=True)


def test_real_observed_t4_runtime_gap_allowed_only_with_precise_fp32():
    runtime=metrics(0.4833984375,0.053735751658678055,0.0003006960323546082)
    fp32=metrics(0.0004,0.00001,1e-9)
    decision=parity_decision(runtime,fp32)
    assert decision['passed'] and decision['fp32_algebra_pass']
    assert decision['reduced_precision_pass']


def test_large_numeric_gap_rejected_even_with_good_top1():
    assert not parity_decision(metrics(0.5,0.18,0.01),metrics(0.0001,1e-6,1e-10))['passed']
    assert not parity_decision(metrics(0.48,0.054,0.0003),metrics(0.12,0.003,0.0002))['passed']
    assert not parity_decision(metrics(0.48,0.054,0.0003,0.9),metrics(0.0001,1e-6,1e-10))['passed']
    assert not parity_decision(metrics(0.48,0.054,0.0003),metrics(float('nan'),1e-6,1e-10))['passed']
    assert not parity_decision(dict(metrics(0.48,0.054,0.0003),context_fits=False),metrics(1e-4,1e-6,1e-10))['passed']


class DummyTokenizer:
    def __call__(self,text,return_tensors,add_special_tokens):
        return {'input_ids':torch.tensor([list(range(1,9))],dtype=torch.long)}


class MockAttention(NativeLike):
    def forward(self, hidden_states, *args, **kwargs):
        return super().forward(hidden_states,kwargs['position_embeddings']),None


class MockLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed=nn.Embedding(64,cfg.hidden_size)
        self.out=nn.Linear(cfg.hidden_size,64,bias=False)
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn=MockAttention(cfg)
        self.cfg=cfg
    def forward(self,input_ids,use_cache=False):
        x=self.embed(input_ids)
        pe=tuple(p.to(device=x.device,dtype=x.dtype) for p in cos_sin(x.shape[0],x.shape[1],self.cfg.head_dim))
        y,_=self.model.layers[0].self_attn(x,position_embeddings=pe)
        return SimpleNamespace(logits=self.out(y+x))


def test_fp32_check_restores_runtime_precision_and_matches_math():
    torch.manual_seed(100)
    cfg=RotaryConsensusConfig(hidden_size=24,query_heads=6,key_value_heads=2,
                              head_dim=4,slots=16,query_block=3,correction_rank=4)
    model=MockLM(cfg).half()
    native=model.model.layers[0].self_attn
    candidate=RotaryCellularConsensus.from_pretrained_attention(native,cfg).half()
    candidate.correct_up.float();candidate.correct_down.float()
    r=fp32_parity_check(model,native,candidate,DummyTokenizer(),torch.device('cpu'),
                        torch.float16,max_tokens=8)
    assert r['context_fits']
    assert r['max_abs_logits']<1e-3,r
    assert r['native_vs_candidate_relative_mse']<1e-8,r
    assert next(model.parameters()).dtype==torch.float16
    assert candidate.q_proj.weight.dtype==torch.float16
    assert candidate.correct_down.weight.dtype==torch.float32
    assert candidate.correct_up.weight.dtype==torch.float32
