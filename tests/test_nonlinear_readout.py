import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinycenn_lm.nonlinear_readout import (
    ReadoutConfig, SharedNonlinearReadout, NonlinearMemory, local_attention,
    build_student, new_cache, adapter_payload, restore_student, wrappers,
)
from tinycenn_lm.research_layers import delta_recurrence, delta_token_reference


def llama():
    torch.manual_seed(83)
    cfg = LlamaConfig(vocab_size=47, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, attention_dropout=0.)
    cfg._attn_implementation = 'sdpa'
    return LlamaForCausalLM(cfg).eval()


def qwen():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(83)
    cfg = Qwen3_5TextConfig(vocab_size=47, hidden_size=64, intermediate_size=96,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        layer_types=['linear_attention','full_attention','linear_attention','full_attention'],
        max_position_embeddings=256, attention_dropout=0., linear_conv_kernel_dim=2,
        linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=4,linear_num_value_heads=8,
        rope_parameters={'rope_type':'default','rope_theta':10000,'mrope_section':[1,1,0],
                         'mrope_interleaved':True,'partial_rotary_factor':0.25})
    cfg._attn_implementation = 'sdpa'
    return Qwen3_5ForCausalLM(cfg).eval()


@pytest.mark.parametrize('window',[1,4,8])
def test_local_window_matches_dense_outputs_and_gradients(window):
    torch.manual_seed(5)
    q = torch.randn(2,4,13,8,requires_grad=True)
    k = torch.randn(2,2,13,8,requires_grad=True)
    v = torch.randn(2,2,13,8,requires_grad=True)
    got = local_attention(q,k,v,window,2)
    distance = torch.arange(13)[:,None]-torch.arange(13)[None,:]
    mask = (distance>=0)&(distance<window)
    ref = torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask)
    torch.testing.assert_close(got,ref,atol=1e-6,rtol=1e-5)
    gg = torch.autograd.grad(got.square().sum(),(q,k,v),retain_graph=True)
    gr = torch.autograd.grad(ref.square().sum(),(q,k,v))
    for a,b in zip(gg,gr):torch.testing.assert_close(a,b,atol=4e-6,rtol=1e-4)
    prefix = local_attention(q[:,:,7:],k[:,:,max(0,8-window):],v[:,:,max(0,8-window):],window,2)
    torch.testing.assert_close(prefix,ref[:,:,7:],atol=1e-6,rtol=1e-5)


def test_one_token_fast_update_and_gradients_match_oracle():
    torch.manual_seed(9)
    shapes = [(2,4,1,8),(2,2,1,8),(2,2,1,6),(2,2,1,8),(2,2,1,8),(2,2,8,6)]
    args = [(.1*torch.randn(shape)).requires_grad_() for shape in shapes]
    got, gs = delta_recurrence(*args,2)
    ref, rs = delta_token_reference(*args,2)
    torch.testing.assert_close(got,ref)
    torch.testing.assert_close(gs,rs)
    gg = torch.autograd.grad(got.square().sum()+gs.square().sum(),args,retain_graph=True)
    gr = torch.autograd.grad(ref.square().sum()+rs.square().sum(),args)
    for a,b in zip(gg,gr):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)


def test_refinement_is_identity_initially_and_weights_are_shared():
    one = SharedNonlinearReadout(4,8,3,1)
    two = SharedNonlinearReadout(4,8,3,2)
    assert sum(p.numel() for p in one.parameters())==sum(p.numel() for p in two.parameters())
    x = torch.randn(2,4,7,8)
    torch.testing.assert_close(two(x,x),x,rtol=0,atol=0)
    two(x,x).square().mean().backward()
    assert two.a.grad.abs().sum()>0


@pytest.mark.parametrize('steps',[0,1,2])
def test_core_streaming_causality_and_bounded_storage(steps):
    torch.manual_seed(14)
    cfg = ReadoutConfig(feature_dim=8,window=4,rank=3,steps=steps,chunk_size=4)
    core = NonlinearMemory(4,2,8,cfg)
    q,k,v = torch.randn(1,4,21,8),torch.randn(1,2,21,8),torch.randn(1,2,21,8)
    with torch.no_grad():
        if steps:core.refine.a.normal_(std=.02)
        full = core(q,k,v)
        first,state = core(q[:,:,:7],k[:,:,:7],v[:,:,:7],return_state=True)
        size = state.nbytes
        outs=[first]
        for start,end in [(7,8),(8,12),(12,21)]:
            out,state=core(q[:,:,start:end],k[:,:,start:end],v[:,:,start:end],state=state,return_state=True)
            outs.append(out)
            assert state.nbytes==size
            assert state.keys.untyped_storage().nbytes()==state.keys.numel()*state.keys.element_size()
        torch.testing.assert_close(torch.cat(outs,2),full,atol=3e-6,rtol=4e-5)
        changed=v.clone();changed[:,:,12:]+=10
        torch.testing.assert_close(core(q,k,changed)[:,:,:12],full[:,:,:12],atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('family',['llama','qwen'])
@pytest.mark.parametrize('variant',['recurrent','attention_control'])
def test_real_model_cache_checkpoint_and_joint_gradients(family,variant):
    teacher=llama() if family=='llama' else qwen()
    before=copy.deepcopy(teacher.state_dict())
    layers=[0,1] if family=='llama' else [1,3]
    config=ReadoutConfig(feature_dim=8,window=4,rank=3,steps=2,chunk_size=4,variant=variant)
    student=build_student(teacher,layers,config)
    ids=torch.randint(0,47,(1,19))
    with torch.no_grad():
        full=student(input_ids=ids,use_cache=False).logits
        if variant=='attention_control':
            torch.testing.assert_close(full,teacher(input_ids=ids,use_cache=False).logits,atol=3e-6,rtol=3e-5)
        cache=new_cache(student)
        outs=[student(input_ids=ids[:,:5],past_key_values=cache,use_cache=True).logits]
        for i in range(5,19):
            outs.append(student(input_ids=ids[:,i:i+1],past_key_values=cache,use_cache=True).logits)
        torch.testing.assert_close(torch.cat(outs,1),full,atol=2e-5,rtol=3e-4)
        assert cache.get_seq_length()==19
        assert cache.nbytes>0
        assert all(a.last_value is None for a in wrappers(student))
    loss=student(input_ids=ids,labels=ids,use_cache=False).loss
    loss.backward()
    for a in wrappers(student):
        if a.core is not None:
            assert a.core.refine.a.grad.abs().sum()>0
            assert a.core.memory.wq.grad.abs().sum()>0
        assert all(p.grad is None for p in a.original.parameters())
    recovered=restore_student(teacher,adapter_payload(student))
    with torch.no_grad():
        torch.testing.assert_close(recovered(input_ids=ids,use_cache=False).logits,full,rtol=0,atol=0)
    for name,value in teacher.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)


@pytest.mark.parametrize('family',['smollm2','qwen35'])
def test_offline_experiment_train_resume_select_evaluate(tmp_path,monkeypatch,family):
    import datasets,huggingface_hub,transformers
    from scripts import benchmark_nonlinear_readout as run
    teacher=llama() if family=='smollm2' else qwen()
    original=copy.deepcopy(teacher.state_dict())
    class Tokenizer:
        eos_token_id=46
        def __call__(self,text,**kw):
            length=kw.get('max_length',5)
            ids=[(sum(text.encode())+i)%46 for i in range(length)]
            return {'input_ids':torch.tensor([ids]) if kw.get('return_tensors') else ids}
        def decode(self,ids,**kw):return ' '.join(map(str,ids))
        def apply_chat_template(self,*a,**kw):return 'chat prompt'
    class Stream:
        def shuffle(self,**kw):return self
        def __iter__(self):return iter({'text':f'nonlinear readout unique document {i}'} for i in range(10000))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    # Return fresh objects like from_pretrained; evaluation deliberately offloads teacher.
    factory=lambda *a,**kw:copy.deepcopy(teacher)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',factory)
    monkeypatch.setattr(transformers.Qwen3_5ForCausalLM,'from_pretrained',factory)
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**kw:Stream())
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:SimpleNamespace(
        model_info=lambda *a,**kw:SimpleNamespace(sha='a'*40),dataset_info=lambda *a,**kw:SimpleNamespace(sha='b'*40)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    args=SimpleNamespace(family=family,base_model='offline',model_revision='main',dataset='offline',dataset_config='default',
        dataset_revision='main',layers=[0,1] if family=='smollm2' else [1,3],features=8,window=4,rank=3,
        refinement_steps=[0,1,2],state_dtype='fp32',train_contexts=[8,12],test_contexts=[8,16],
        train_documents=3,validation_documents=2,test_documents=2,warm_steps=1,joint_steps=1,eval_every=1,log_every=1,
        decode_tokens=2,timing_documents=1,timing_repeats=1,generation_tokens=2,seed=2031,lr=.002,kl_weight=1.,
        cache_nmse_limit=.001,nll_margin=.02,exclude_manifest=[],resume=False,allow_cpu=True,output_dir=str(tmp_path/'run'))
    monkeypatch.setattr(run,'parse_args',lambda:copy.deepcopy(args))
    run.main()
    out=Path(args.output_dir)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='completed' and len(report['rows'])==10
    assert all(not r.get('meets_requested_target',False) for r in report['rows'])
    assert json.loads((out/'selection.json').read_text())['test_not_used_for_selection']
    hashes=report['document_hashes']
    assert len(set(sum(hashes.values(),[])))==7
    import nbformat
    import matplotlib
    matplotlib.use("Agg")
    notebook_name = "SmolLM2_Nonlinear_Recurrent_Readout_Colab.ipynb" if family=="smollm2" else "Qwen3_5_0_8B_Nonlinear_Recurrent_Readout_Colab.ipynb"
    notebook=nbformat.read(run.ROOT/"notebooks"/notebook_name,as_version=4)
    result_cell=next(c for c in notebook.cells if "results" in c.metadata.get("tags",[]))
    exec(compile(result_cell.source,notebook_name,"exec"), {"OUT":out,"json":json,"display":lambda *a:None})
    assert (out/"comparison-T8.png").stat().st_size>1000
    import matplotlib.pyplot as plt
    plt.close("all")
    args.resume=True
    run.main()
    again=json.loads((out/'report.json').read_text())
    assert [r['test_nll'] for r in again['rows']]==[r['test_nll'] for r in report['rows']]
    for name,value in teacher.state_dict().items():torch.testing.assert_close(value,original[name],rtol=0,atol=0)


def test_matched_memory_initialization_across_refinement_depth():
    teacher=llama()
    models=[build_student(teacher,[0,1,2],ReadoutConfig(feature_dim=8,steps=r,rank=3)) for r in (0,1,2)]
    for index in range(3):
        states=[wrappers(model)[index].core.memory.state_dict() for model in models]
        for key in states[0]:
            torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
            torch.testing.assert_close(states[0][key],states[2][key],rtol=0,atol=0)


def test_colab_schema_all_cells_and_optional_prompt_program():
    import nbformat
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'notebooks').glob('*Nonlinear_Recurrent_Readout_Colab.ipynb'))
    assert len(paths)==2
    for path in paths:
        n=nbformat.read(path,as_version=4)
        nbformat.validate(n)
        for i,c in enumerate(n.cells):
            if c.cell_type=='code':
                assert c.execution_count is None and not c.outputs
                compile(c.source,f'{path}:{i}','exec')
                if 'prompt' in c.metadata.get('tags',[]):
                    # Check the nested fresh-process program too.
                    import ast
                    tree=ast.parse(c.source)
                    program=next(node.value.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='code_text' for t in node.targets))
                    compile(program,str(path)+':custom-prompt','exec')
