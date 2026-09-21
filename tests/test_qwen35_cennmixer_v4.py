"""CPU regressions without pretrained weights or dataset downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tinycenn_lm.qwen35_cennmixer_v4 import (
    CeNNMixerV4, CeNNMixerV4Config, ProgressiveCeNNMixerV4,
    install_cenn_mixer_v4, set_alpha_v4, finalize_cenn_only_v4,
    reset_stream_state_v4, freeze_all_except_cenn_v4,
)
from tinycenn_lm.qwen35_cennmixer_v4_train import stage_ok, stage_violation, topk_rank_loss


def config(kernel=4):
    return CeNNMixerV4Config(hidden_size=32, groups=2, cell_dim=8,
                            assoc_heads=2, key_dim=8, value_dim=8, conv_kernel=kernel)


@pytest.mark.parametrize('kernel', [1, 4])
@pytest.mark.parametrize('prefix', [1, 2, 3, 5])
def test_streaming_matches_full_sequence(kernel, prefix):
    torch.manual_seed(7)
    m=CeNNMixerV4(config(kernel)).eval()
    x=torch.randn(2,9,32)
    with torch.no_grad():
        full=m(x)
        parts=[m(x[:,:prefix],streaming=True)]
        parts.extend(m(x[:,i:i+1],streaming=True) for i in range(prefix,9))
    torch.testing.assert_close(torch.cat(parts,1),full,atol=2e-6,rtol=2e-5)


def test_frozen_original_preserves_input_gradients():
    original=nn.Linear(32,32,bias=False)
    m=ProgressiveCeNNMixerV4(original,CeNNMixerV4(config()),'linear_attention')
    x=torch.randn(1,3,32,requires_grad=True)
    m.set_alpha(0)
    m(x).sum().backward()
    torch.testing.assert_close(x.grad,original.weight.sum(0).expand_as(x))
    assert original.weight.grad is None


def test_nonfinite_gate_and_topk_one():
    a=SimpleNamespace(quick_smoke=True)
    m=dict(top1=1.,kl=0.,hidden_mse=0.,mixer_mse=0.,mixer_cosine=0.,mixer_delta=0.,student_ce=1.)
    assert stage_ok(m,0,a)
    m['student_ce']=float('nan')
    assert not stage_ok(m,0,a)
    assert stage_violation(m,0,a)==float('inf')
    assert torch.isfinite(topk_rank_loss(torch.randn(1,2,5),torch.randn(1,2,5),1))


def test_real_qwen_tiny_backward_and_finalized_cache():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(1)
    qc=Qwen3_5TextConfig(vocab_size=48,hidden_size=32,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_key_head_dim=8,linear_value_head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=2,
        layer_types=['linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.},
        eos_token_id=2,pad_token_id=0)
    model=Qwen3_5ForCausalLM(qc).eval()
    install_cenn_mixer_v4(model,0,config())
    trainable=freeze_all_except_cenn_v4(model)
    set_alpha_v4(model,.5)
    ids=torch.tensor([[3,4,5,6]])
    model(input_ids=ids,use_cache=False).logits.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in trainable)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
    set_alpha_v4(model,1.)
    with torch.no_grad():
        before=model(input_ids=ids,use_cache=False).logits
        finalize_cenn_only_v4(model)
        after=model(input_ids=ids,use_cache=False).logits
        torch.testing.assert_close(before,after)
        reset_stream_state_v4(model)
        prefill=model(input_ids=ids[:,:1],use_cache=True)
        pieces=[prefill.logits]
        cache=prefill.past_key_values
        for i in range(1,ids.shape[1]):
            out=model(input_ids=ids[:,i:i+1],past_key_values=cache,use_cache=True)
            pieces.append(out.logits)
            cache=out.past_key_values
        torch.testing.assert_close(torch.cat(pieces,1),after,rtol=2e-4,atol=2e-5)


def test_notebook_compiles_and_upload_is_opt_in():
    p=Path(__file__).resolve().parents[1]/'notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb'
    nb=json.loads(p.read_text())
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            compile(''.join(c['source']),f'cell_{i}','exec')
            assert not c['outputs']
    source=''.join(nb['cells'][-1]['source'])
    assert 'UPLOAD_TO_HF=False' in source
    exec(source,{})
