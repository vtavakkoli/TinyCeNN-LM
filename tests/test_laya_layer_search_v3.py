"""Correctness of structural edits, case partitioning and quality gates."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'standalone/laya_1p2_speedup'))
from test_laya_fast_inference import tiny_model, Tokenizer
import modeling_laya_standalone as runtime
from layer_search import NarrowMLP, apply_plan, channel_specs, MODES
from search_laya_layers_v3 import (prepare_case, split_cases, metrics, quality_ok,
    calibrate, predict_cases, Search, paired_case_interval, timing_cases, predict_request)


def row(i=0):
    return dict(id=str(i), workflow='w' + str(i % 2), state={'x': i},
        questions={'a': {'criteria':['left','right']},
                   'b': {'type':'score','criteria':['low','medium','high']},
                   'c': {'type':'noul','instructions':'Is this true?'}},
        gold={'a':dict(label='left',probabilities={'left':.7,'right':.3}),
              'b':dict(label=2.0,probabilities={'0':.1,'1':.2,'2':.7}),
              'c':dict(label=True,probabilities={'false':.1,'true':.9})})


def cases():
    return [prepare_case(row(i),runtime,Tokenizer(),128) for i in range(3)]


@torch.inference_mode()
def test_full_width_glu_and_pruning_match_manual():
    m = tiny_model().encoder.layers[1].mlp
    x = torch.randn(2,7,64)
    torch.testing.assert_close(NarrowMLP(m,list(range(96)))(x),m(x))
    indices = [1,7,12,45,91]
    value,gate=m.Wi(x).chunk(2,-1)
    expected=torch.nn.functional.linear((m.act(value)*gate)[...,indices],m.Wo.weight[:,indices],m.Wo.bias)
    torch.testing.assert_close(NarrowMLP(m,indices)(x),expected)
    for bad in ([],[1,1],[-1],[96]):
        with pytest.raises(ValueError): NarrowMLP(m,bad)


@pytest.mark.parametrize('mode',MODES)
@torch.inference_mode()
def test_edits_match_independent_manual_residual_and_do_not_mutate(mode):
    model=tiny_model()
    before={k:v.clone() for k,v in model.state_dict().items()}
    original_layer=model.encoder.layers[1]
    spec=dict(mode=mode)
    if mode.startswith('mlp'): spec['indices']=list(range(48))
    fast=apply_plan(model,{'1':spec})
    changed=fast.model.encoder.layers[1]
    x=torch.randn(2,7,64)
    kw=dict(attention_mask=torch.zeros(2,1,1,7),sliding_window_mask=torch.zeros(2,1,7,7),
            position_ids=torch.arange(7)[None],output_attentions=False)
    expected=x
    if mode not in ('drop_attn','drop_block'):
        expected=expected+original_layer.attn(original_layer.attn_norm(expected),**kw)[0]
    if mode not in ('drop_mlp','drop_block'):
        mlp=NarrowMLP(original_layer.mlp,spec['indices']) if mode.startswith('mlp') else original_layer.mlp
        expected=expected+mlp(original_layer.mlp_norm(expected))
    torch.testing.assert_close(changed(x,**kw)[0],expected)
    assert model.encoder.layers[1] is original_layer
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
    assert sum(p.numel() for p in fast.parameters()) < sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def test_identity_plan_and_roundtrip_preserve_outputs():
    model=tiny_model()
    c=cases()
    ref=predict_cases(model,c)
    actual=predict_cases(apply_plan(model,{}),c)
    for a,b in zip(actual,ref): torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    plan={'0':{'mode':'drop_attn'},'1':{'mode':'mlp50','indices':list(range(48))}}
    a=predict_cases(apply_plan(model,plan),c)
    b=predict_cases(apply_plan(model,json.loads(json.dumps(plan))),c)
    for x,y in zip(a,b): assert torch.equal(x,y)


def test_gold_option_mapping_and_error():
    c=prepare_case(row(),runtime,Tokenizer(),128)
    assert c['target'].tolist()==[0,2,1]
    assert c['batch']['qtype'].tolist()==[0,1,2]
    bad=row();bad['gold']['b']['label']=8
    with pytest.raises(ValueError,match='gold'): prepare_case(bad,runtime,Tokenizer(),128)
    with pytest.raises(ValueError,match='Truncated'): prepare_case(row(),runtime,Tokenizer(),4)


def test_case_split_is_disjoint_balanced_and_reproducible():
    rows=[row(i) for i in range(40)]
    groups=split_cases(rows,[4,8,12],17)
    assert groups==split_cases(rows,[4,8,12],17)
    assert len({r['id'] for group in groups for r in group})==24
    assert all(len({r['workflow'] for r in group})==2 for group in groups)
    with pytest.raises(ValueError): split_cases(rows+[rows[0]],[4,8,12],17)


def test_quality_gate_rejects_gold_group_and_agreement_regressions():
    c=cases();out=[]
    for case in c:
        x=torch.full(case['batch']['marker_mask'].shape,-9.)
        x[torch.arange(3),case['target']]=9
        out.append(x)
    m=metrics(c,out,out)
    args=SimpleNamespace(min_agreement=.98,max_accuracy_drop=.005,max_js=.005,max_group_drop=.02)
    assert quality_ok(m,args) and m['accuracy']==1
    broken=[x.clone() for x in out];broken[0][0]=torch.tensor([-9.,9.,-9.])
    b=metrics(c,broken,out)
    assert not quality_ok(b,args)
    assert b['groups']['type:choice']['accuracy_delta']<0
    lo,hi=paired_case_interval(b,17,200)
    assert lo<=b['accuracy_delta']<=hi


@torch.inference_mode()
def test_calibration_search_resume_and_every_layer_mode(tmp_path):
    model=tiny_model();c=cases()
    scores=calibrate(model,c)
    assert len(scores)==3 and all(torch.isfinite(x).all() for x in scores)
    assert all(not layer.mlp.Wo._forward_pre_hooks for layer in model.encoder.layers)
    specs=channel_specs(model,scores)
    args=SimpleNamespace(output=tmp_path,min_agreement=0.,max_accuracy_drop=1.,max_js=1.,
        max_group_drop=1.,pair_top=4,combine_top=4,beam=2,depth=2)
    search=Search(model,c,predict_cases(model,c),args,specs)
    results=search.run()
    assert sum(r['phase']=='single' for r in results)==3*len(MODES)
    assert any(len(r['plan'])==3 for r in results)
    assert all(r['status']=='ok' for r in results)
    restored=Search(model,c,search.refs,args,specs)
    assert len(restored.run())==len(results)


@torch.inference_mode()
def test_changed_model_fullgraph_capture():
    model=apply_plan(tiny_model(),{'0':{'mode':'drop_attn'},'1':{'mode':'mlp50','indices':list(range(48))}})
    compiled=torch.compile(model,backend='eager',fullgraph=True,dynamic=True)
    c=cases()
    for a,b in zip(predict_cases(model,c),predict_cases(compiled,c)):
        torch.testing.assert_close(a,b)


@torch.inference_mode()
def test_complete_api_matches_forward_for_all_question_types():
    model=apply_plan(tiny_model(),{'0':{'mode':'drop_attn'}})
    tok=Tokenizer()
    case=prepare_case(row(),runtime,tok,128)
    expected=predict_cases(model,[case])[0].masked_fill(~case['batch']['marker_mask'],-1e9).softmax(-1)
    actual=predict_request(model,row(),runtime,tok,128)
    assert list(actual)==['a','b','c']
    for i,answer in enumerate(actual.values()):
        p=torch.tensor(list(answer['probabilities'].values()))
        torch.testing.assert_close(p,expected[i,:len(p)])


def test_full_pipeline_with_tiny_checkpoint_and_mock_hub(tmp_path,monkeypatch):
    """Exercise search -> validation -> selection freeze -> test -> recipe -> resume.

    Hub and Inductor are replaced, never advertised as a real checkpoint/GPU test.
    """
    import types
    import huggingface_hub
    import search_laya_layers_v3 as v3
    from transformers import ModernBertConfig
    torch.set_num_threads(2)
    cfg=ModernBertConfig(vocab_size=128,hidden_size=64,intermediate_size=96,
        num_hidden_layers=28,num_attention_heads=4,local_attention=8,
        global_attn_every_n_layers=3,max_position_embeddings=128,pad_token_id=0,reference_compile=False)
    model=runtime.LayaStandalone(cfg).half().eval().requires_grad_(False)
    fake_runtime=SimpleNamespace(**{k:getattr(runtime,k) for k in
        ['normalize_question','build_sequence','collate_items','QTYPES']})
    fake_runtime.load_model=lambda *a,**k:(model,Tokenizer())
    data=types.ModuleType('datasets')
    events=[]
    def load_dataset(*a,split,**k):
        if split=='test':
            assert (tmp_path/'selected_before_test.json').exists()
        events.append(split)
        return [row(i) for i in (range(8) if split=='train' else range(100,102))]
    data.load_dataset=load_dataset
    monkeypatch.setitem(sys.modules,'datasets',data)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:str(tmp_path))
    monkeypatch.setattr(v3.v1,'load_runtime',lambda *a,**k:fake_runtime)
    monkeypatch.setattr(v3,'compile_model',lambda m,args:m)
    def timing(*a,**k):
        return dict(results={'baseline':dict(passed=True,median_ms=12,round_ms=[12]*7),
                             'candidate':dict(passed=True,median_ms=10,round_ms=[10]*7)})
    monkeypatch.setattr(v3,'timed_models',timing)
    args=v3.parser().parse_args(['--output',str(tmp_path),'--device','cpu','--dataset-revision','fixture',
        '--calibration','2','--screen','2','--validation','2','--max-len','128',
        '--pair-top','2','--combine-top','2','--depth','1','--beam','1',
        '--validation-top','1','--finalists','1','--timing-cases','2','--warmup','1','--repeats','1',
        '--min-agreement','0','--max-accuracy-drop','1','--max-group-drop','1','--max-js','1'])
    v3.run(args)
    report=json.loads((tmp_path/'final_report.json').read_text())
    assert report['search_trials']>=145
    assert report['changed_layers']>0
    assert report['export_roundtrip']['agreement']==1
    assert report['final_compile']['passed']
    assert report['end_to_end_api']['results']['selected_compiled']['passed']
    assert events==['train','test']
    v3.run(args)
    assert events==['train','test','train']
