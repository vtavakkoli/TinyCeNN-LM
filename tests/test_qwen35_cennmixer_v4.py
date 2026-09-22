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


def test_chunked_losses_match_dense_values_and_gradients():
    from tinycenn_lm.qwen35_cennmixer_v4_train import causal_ce,forward_kl,reverse_kl
    import torch.nn.functional as F
    torch.manual_seed(3)
    s=torch.randn(2,35,17,requires_grad=True)
    t=torch.randn_like(s)
    y=torch.randint(0,17,(2,35))
    ls=F.log_softmax(s,-1);lt=F.log_softmax(t,-1)
    dense=F.cross_entropy(s.reshape(-1,17),y.flatten())+(lt.exp()*(lt-ls)).sum(-1).mean()+(ls.exp()*(ls-lt)).sum(-1).mean()
    chunk=causal_ce(s,y)+forward_kl(s,t)+reverse_kl(s,t)
    torch.testing.assert_close(chunk,dense)
    torch.testing.assert_close(torch.autograd.grad(chunk,s)[0],torch.autograd.grad(dense,s)[0])


def test_warmup_ranks_mixer_not_identity_roundoff():
    from tinycenn_lm.qwen35_cennmixer_v4_train import quality_key
    a=SimpleNamespace(quick_smoke=False)
    m=dict(top1=1.,kl=0.,hidden_mse=0.,mixer_mse=.2,mixer_cosine=.1,mixer_delta=.3,student_ce=1.)
    improved={**m,'mixer_mse':.1,'mixer_cosine':.05,'mixer_delta':.15,'kl':1e-9}
    assert quality_key(improved,0,a)<quality_key(m,0,a)


def test_generation_failure_blocks_numerical_pass():
    from tinycenn_lm.qwen35_cennmixer_v4_train import generation_health
    a=SimpleNamespace(quick_smoke=True)
    m=dict(top1=1.,kl=0.,hidden_mse=0.,mixer_mse=0.,mixer_cosine=0.,mixer_delta=0.,student_ce=1.,generation_ok=0.)
    assert not stage_ok(m,.7,a)
    assert not generation_health([])
    row=dict(expected='42',teacher_correct=True,student_correct=False,cenn='wrong',repetition=0.)
    assert not generation_health([row])
    assert generation_health([{**row,'student_correct':True,'cenn':'42'}])


def test_context_curriculum_and_bounds():
    import random
    from tinycenn_lm.qwen35_cennmixer_v4_train import context_lengths,curriculum_window,blocks
    a=SimpleNamespace(seq_len=512,context_lengths='128,256')
    assert context_lengths(a)==[128,256,512]
    a.seq_len=64
    with pytest.raises(ValueError):context_lengths(a)
    rng=random.Random(1); cpu=torch.arange(513)[None]
    sizes={curriculum_window(cpu,100,100,[128,256,512],rng).shape[1]-1 for _ in range(40)}
    assert sizes=={128,256,512}
    with pytest.raises(ValueError):blocks(None,[],1,128,1)


def test_both_notebooks_match_and_preserve_quality_defaults():
    r=Path(__file__).resolve().parents[1]
    name='Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb'
    a=(r/'notebooks'/name).read_text()
    assert a==(r/'cennmixer-v4-runtime-training/notebooks'/name).read_text()
    n=json.loads(a);s=''.join(''.join(c['source']) for c in n['cells'])
    assert 'QUICK_SMOKE=False' in s and 'EXPLORE_HIGH_ALPHA=False' in s
    assert 'SEQ_LEN=512' in s and '--context-lengths' in s


def test_runner_offline_onpolicy_and_final_evaluation(tmp_path,monkeypatch):
    """Exercise actual tiny-Qwen optimization; gate thresholds tested separately."""
    import importlib.util
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from tinycenn_lm.qwen35_cennmixer_v4_train import on_policy_distill_loss
    path=Path(__file__).resolve().parents[1]/'scripts/run_qwen35_cennmixer_v4.py'
    spec=importlib.util.spec_from_file_location('runner_v4',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    monkeypatch.setattr('sys.argv',['test'])
    a=runner.parse_args()
    for k,v in dict(seq_len=32,context_lengths='32',train_blocks=2,val_blocks=1,
                    alphas='0,1',groups=2,cell_dim=8,assoc_heads=2,key_dim=8,value_dim=8,
                    stage_updates=2,max_stage_updates=2,min_stage_updates=2,probe_every=1,
                    on_policy_every=1,on_policy_tokens=2,output_dir=str(tmp_path)).items():
        setattr(a,k,v)
    monkeypatch.setattr(runner,'parse_args',lambda:a)
    qc=Qwen3_5TextConfig(vocab_size=48,hidden_size=32,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,linear_num_value_heads=2,
        layer_types=['linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.},
        eos_token_id=2,pad_token_id=0)
    def factory(*args,**kwargs):
        torch.manual_seed(8)
        return Qwen3_5ForCausalLM(qc)
    monkeypatch.setattr(runner.AutoModelForCausalLM,'from_pretrained',factory)
    monkeypatch.setattr(runner.AutoTokenizer,'from_pretrained',lambda *args,**kwargs:SimpleNamespace(pad_token_id=0,eos_token_id=2))
    batch=torch.randint(3,48,(1,33))
    monkeypatch.setattr(runner,'load_data',lambda *args:([batch,batch],{32:[batch]}))
    monkeypatch.setattr(runner,'stage_ok',lambda *args:True)
    monkeypatch.setattr(runner,'retrieval_prompt',lambda *args:('prompt','42'))
    monkeypatch.setattr(runner,'chat_ids',lambda *args:batch[:,:8])
    row=dict(prompt='test',qwen='42',cenn='42',expected='42',teacher_correct=True,
             student_correct=True,repetition=0.,exact=True,jaccard=1.)
    monkeypatch.setattr(runner,'generation_suite',lambda *args,**kwargs:[row])
    import datasets
    import tinycenn_lm.qwen35_cennmixer_v4_train as train
    monkeypatch.setattr(datasets,'load_dataset',lambda *args,**kwargs:[{'text':'test'}])
    monkeypatch.setattr(train,'blocks',lambda *args:[batch])
    runner.main()
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['test_contexts'] and report['final_cenn_only_probe']
    assert all(r['trained_steps']==2 for r in report['stage_summary'])
    assert (tmp_path/'cennmixer_v4_alpha_0p0_best.pt').exists()
    assert (tmp_path/'cennmixer_v4_final_cenn_only.pt').exists()
