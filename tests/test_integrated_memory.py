import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinycenn_lm.integrated_memory import (
    build_student, wrappers, inference_mode, new_cache, adapter_payload,
    restore_student, greedy_generate, native_dtype,
)
from scripts import benchmark_smollm2_integrated_memory as bench


def tiny_model():
    torch.manual_seed(101)
    c=LlamaConfig(vocab_size=41,hidden_size=32,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    c._attn_implementation='sdpa'
    return LlamaForCausalLM(c).eval()


@pytest.mark.parametrize('layers',[[0],[1],[0,1]])
@pytest.mark.parametrize('variant',['cenn_partition','sink_window','transformer_readout'])
def test_complete_model_cache_matches_full_and_fused_logits(layers,variant):
    teacher=tiny_model()
    student=build_student(teacher,layers,variant,features=8,block_size=4,sinks=1)
    ids=torch.randint(0,41,(1,23))
    with torch.no_grad():
        for a in wrappers(student):
            if variant!='sink_window':
                a.core.readout.add_(.05*torch.randn_like(a.core.readout))
        ordinary=student(input_ids=ids,use_cache=False).logits
    with inference_mode(student):
        full=student(input_ids=ids,use_cache=False).logits
        cache=new_cache(student)
        first=student(input_ids=ids[:,:5],past_key_values=cache,use_cache=True).logits
        pieces=[first]
        for start,end in [(5,8),(8,13),(13,23)]:
            pieces.append(student(input_ids=ids[:,start:end],past_key_values=cache,use_cache=True).logits)
        torch.testing.assert_close(torch.cat(pieces,1),full,atol=4e-6,rtol=4e-5)
        assert cache.get_seq_length()==23
        assert cache.nbytes>0
        for state in cache.memory_states.values():
            assert state.keys.shape[2]<=8
        generated,state=greedy_generate(student,ids[:,:7],tokens=5)
        direct=ids[:,:7]
        for _ in range(5):
            token=student(input_ids=direct,use_cache=False).logits[:,-1].argmax(-1,keepdim=True)
            direct=torch.cat((direct,token),1)
        assert torch.equal(generated,direct[:,7:])
    torch.testing.assert_close(ordinary,full,atol=4e-6,rtol=4e-5)


def test_joint_updates_reach_both_cores_without_changing_teacher_and_reload():
    teacher=tiny_model()
    before=copy.deepcopy(teacher.state_dict())
    student=build_student(teacher,[0,1],features=8,block_size=4,sinks=1)
    ids=torch.randint(0,41,(1,17))
    with torch.no_grad(): target=teacher(input_ids=ids,use_cache=False).logits
    loss,_,_=bench.joint_loss(student(input_ids=ids,use_cache=False).logits,target,ids)
    loss.backward()
    for a in wrappers(student):
        assert a.core.wq.grad is not None and a.core.wq.grad.abs().sum()>0
    for n,p in student.named_parameters():
        if '.core.' not in n: assert p.grad is None
    for n,p in teacher.state_dict().items(): torch.testing.assert_close(p,before[n],atol=0,rtol=0)
    recovered=restore_student(teacher,adapter_payload(student))
    with inference_mode(student),inference_mode(recovered):
        torch.testing.assert_close(student(input_ids=ids,use_cache=False).logits,recovered(input_ids=ids,use_cache=False).logits)


def test_t4_uses_fp16_even_if_emulated_bf16_is_reported(monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:(7,5))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda *a,**k:True)
    assert native_dtype('cuda')==torch.float16
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda *a:(8,0))
    assert native_dtype('cuda')==torch.bfloat16
    assert native_dtype('cpu')==torch.float32


def test_end_to_end_joint_training_selection_reload_generation_and_timing(tmp_path,monkeypatch):
    import datasets,huggingface_hub,transformers
    teacher=tiny_model()
    original=copy.deepcopy(teacher.state_dict())
    class Tokenizer:
        def __call__(self,text,**kwargs):
            x=sum(text.encode())%41
            return {'input_ids':[(x+i)%41 for i in range(kwargs['max_length'])]}
        def decode(self,ids): return ' '.join(map(str,ids))
    class Stream:
        def shuffle(self,**kw): return self
        def __iter__(self): return iter({'text':f'new v3 document {i}'} for i in range(5000))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:teacher)
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**kw:Stream())
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:SimpleNamespace(
        model_info=lambda *a,**kw:SimpleNamespace(sha='a'*40),
        dataset_info=lambda *a,**kw:SimpleNamespace(sha='b'*40)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    args=SimpleNamespace(base_model='offline-llama',model_revision='main',dataset='offline',dataset_config='default',
        dataset_revision='main',conservative_layers=[0],expanded_layers=[0,1],train_contexts=[8,12],test_contexts=[8,16],
        train_documents=3,validation_documents=2,test_documents=2,warm_documents=2,warm_steps=1,joint_steps=2,
        eval_every=1,features=8,block_size=2,sinks=1,seed=2028,dataset_seed=9317,decode_tokens=2,timing_documents=1,
        timing_repeats=1,lr=.002,kl_weight=1.,temperature=1.,nll_margin=.02,exclude_manifest=[],output_dir=str(tmp_path/'run'))
    monkeypatch.setattr(bench,'parse_args',lambda:args)
    bench.main()
    root=tmp_path/'run'
    report=json.loads((root/'integrated_report.json').read_text())
    assert report['status']=='completed'
    assert len(report['rows'])==12
    assert len(report['candidates'])==5
    assert json.loads((root/'selection.json').read_text())['test_not_used_for_selection']
    assert all(not row.get('beats_both_quality',False) for row in report['rows'])
    for r in report['candidates']:
        assert r['validation_nll']<=r['before_joint_validation_nll']+1e-7
        assert (root/r['checkpoint']).is_file()
        assert r['cached_logits_nmse']<.001
    h=report['document_hashes']
    assert len(set(sum(h.values(),[])))==7
    assert all(40<=int(x[:8],16)%100<50 for x in h['test'])
    assert not set(sum(h.values(),[])) & bench.read_exclusions([bench.ROOT/'configs/smollm2_v3_prior_exclusions.json'])
    assert len(json.loads((root/'generation_examples.json').read_text()))==12
    for key,value in teacher.state_dict().items(): torch.testing.assert_close(value,original[key],atol=0,rtol=0)

    import importlib.util
    from pathlib import Path
    if all(importlib.util.find_spec(x) for x in ('pandas','matplotlib')):
        import matplotlib
        matplotlib.use('Agg')
        path=bench.ROOT/'notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb'
        notebook=json.loads(path.read_text())
        cells=[c for c in notebook['cells'] if 'results' in c.get('metadata',{}).get('tags',[])]
        assert len(cells)==1
        exec(compile(''.join(cells[0]['source']),str(path),'exec'),
             {'OUT':root,'json':json,'display':lambda *args:None})
        assert (root/'decision_table.csv').is_file()
        assert (root/'integrated-T16.png').stat().st_size>1000


def test_cli_starts_from_an_unrelated_directory(tmp_path):
    import os, subprocess, sys
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(bench.ROOT/'scripts/benchmark_smollm2_integrated_memory.py'),'--help'],
                          cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert '--train-contexts' in result.stdout
