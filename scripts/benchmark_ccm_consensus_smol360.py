"""CCM-v2 vs first SmolLM2-360M-Instruct attention mixer, with qualitative chat.

Candidate is an independent finite cellular tape with mass-conserving merges.
Mathematically the merge is equivalent to softmax attention over the bounded
cells; no unique attention-free novelty or performance improvement is claimed.
This is ONE LAYER, not a standalone all-CCM language model.

Default run: CUDA, teacher activation distillation then optional teacher-logit
fine-tuning through the frozen remainder. Choose on development only; assess
held-out test once; collect paired GPU latency and no-cache greedy chat demos.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.competitive_consensus_v2 import ConsensusConfig, CompetitiveConsensusCeNN

MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_REVISION = 'a10cc1512eabd3dde888204e902eca88bddb4951'
DEFAULT_CHATS = [
    {'id':'short_fact','user':'What is the capital of Austria? Answer in one sentence.'},
    {'id':'simple_math','user':'What is 17 plus 25? Reply with the number only.'},
    {'id':'explanation','user':'Explain the difference between RAM and SSD in two short sentences.'},
    {'id':'instruction','user':'Respond with exactly three words about the Moon.'},
    {'id':'recall','user':'Remember this code: AURORA-731. What was the code I just gave you?'},
]


class ConsensusAttentionAdapter(nn.Module):
    """Replaces the attention SUBMODULE only; decoder supplies its own residual."""
    def __init__(self, cell: CompetitiveConsensusCeNN):
        super().__init__()
        self.cell=cell
    def forward(self, hidden_states, *args, **kwargs):
        if kwargs.get('past_key_value') is not None or kwargs.get('past_key_values') is not None or kwargs.get('use_cache', False):
            raise NotImplementedError('CCM-v2 model integration has no KV/recurrent cache support')
        if kwargs.get('output_attentions',False):
            raise NotImplementedError('CCM-v2 does not expose native attention weights')
        result=self.cell(hidden_states)
        return result.to(hidden_states.dtype),None


@contextmanager
def attention_slot(model, module):
    layer=model.model.layers[0]
    old=layer.self_attn
    layer.self_attn=module
    try: yield
    finally: layer.self_attn=old


def load_blocks(tokenizer, split, count, length):
    from datasets import load_dataset
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split)
    ids=[]
    for item in ds:
        string=item['text'].strip()
        if not string: continue
        ids+=tokenizer.encode(string,add_special_tokens=False)
        if tokenizer.eos_token_id is not None:ids.append(tokenizer.eos_token_id)
        if len(ids)>=length*count: break
    if len(ids)<length*count:raise RuntimeError(f'{split}: insufficient tokens')
    return torch.tensor(ids[:count*length],dtype=torch.long).reshape(count,length)


def batched(ids, size):
    for i in range(0,len(ids),size):yield ids[i:i+size]


def capture_teacher_attention(model, ids, batch_size, device):
    native=model.model.layers[0].self_attn
    examples=[];cached={'x':None}
    def before(_,args,kwargs):
        x=kwargs.get('hidden_states',args[0] if args else None)
        if x is None:raise RuntimeError('No native first-layer hidden_states')
        cached['x']=x.detach().cpu().half()
    def after(_,args,kwargs,result):
        out=result[0] if isinstance(result,tuple) else result
        assert cached['x'] is not None
        examples.append((cached['x'],out.detach().cpu().half()))
        cached['x']=None
    h1=native.register_forward_pre_hook(before,with_kwargs=True)
    h2=native.register_forward_hook(after,with_kwargs=True)
    try:
        with torch.inference_mode():
            for block in batched(ids,batch_size):
                model(input_ids=block.to(device),use_cache=False,return_dict=True)
    finally:
        h1.remove();h2.remove()
    if len(examples)!=(len(ids)+batch_size-1)//batch_size:raise RuntimeError('incomplete teacher capture')
    return examples


def local_metrics(cell,examples,device):
    cell.eval();errors=[];cosines=[]
    with torch.no_grad():
        for x,y in examples:
            p=cell(x.to(device)).float();ref=y.to(device).float()
            errors.append(float(((p-ref).square().mean()/ref.square().mean().clamp_min(1e-8)).item()))
            cosines.append(float(F.cosine_similarity(p,ref,dim=-1,eps=1e-6).mean().item()))
    return {'relative_mse':statistics.mean(errors),'cosine':statistics.mean(cosines)}


def compare_full_model(model,native,adapter,ids,batch_size,device):
    model.eval();adapter.eval()
    sums={'teacher_nll':0.,'student_nll':0.,'teacher_agreement':0,
          'student_gold_correct':0,'teacher_gold_correct':0,'kl_sum':0.,'tokens':0}
    with torch.inference_mode():
        for block in batched(ids,batch_size):
            tokens=block.to(device)
            with attention_slot(model,native):
                t=model(input_ids=tokens,use_cache=False,return_dict=True).logits[:,:-1].float()
            with attention_slot(model,adapter):
                s=model(input_ids=tokens,use_cache=False,return_dict=True).logits[:,:-1].float()
            gold=tokens[:,1:];tlog=F.log_softmax(t,-1);slog=F.log_softmax(s,-1)
            sums['teacher_nll']+=float(-tlog.gather(-1,gold[...,None]).sum())
            sums['student_nll']+=float(-slog.gather(-1,gold[...,None]).sum())
            sums['teacher_agreement']+=int((t.argmax(-1)==s.argmax(-1)).sum())
            sums['student_gold_correct']+=int((s.argmax(-1)==gold).sum())
            sums['teacher_gold_correct']+=int((t.argmax(-1)==gold).sum())
            sums['kl_sum']+=float((tlog.exp()*(tlog-slog)).sum())
            sums['tokens']+=gold.numel()
    n=sums['tokens']
    return {'tokens':n,'teacher_nll':sums['teacher_nll']/n,
            'student_nll':sums['student_nll']/n,
            'delta_nll':(sums['student_nll']-sums['teacher_nll'])/n,
            'teacher_agreement':sums['teacher_agreement']/n,
            'student_gold_accuracy':sums['student_gold_correct']/n,
            'teacher_gold_accuracy':sums['teacher_gold_correct']/n,
            'teacher_student_kl':sums['kl_sum']/n}

def rng(seed):
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def local_loss(cell,x,y,device):
    got=cell(x.to(device)).float()
    target=y.to(device).float()
    mse=(got-target).square().mean()/target.square().mean().clamp_min(1e-8)
    cos=1-F.cosine_similarity(got,target,dim=-1,eps=1e-6).mean()
    return mse+0.15*cos


def logits_positions(length, count=12):
    return sorted({max(0, int(v)) for v in torch.linspace(0,length-2,count).tolist()})


def logit_training_step(model,native,adapter,tokens,optimizer, *, temperature=2.0):
    dev=tokens.device
    pos=logits_positions(tokens.shape[1]);gold=tokens[:,[i+1 for i in pos]]
    model.eval();adapter.train()
    with torch.no_grad(),attention_slot(model,native):
        teacher=model(input_ids=tokens,use_cache=False,return_dict=True).logits[:,pos].float().detach()
    with torch.enable_grad(),attention_slot(model,adapter):
        student=model(input_ids=tokens,use_cache=False,return_dict=True).logits[:,pos].float()
        # matched positions, teacher detached; backprop only through adapter
        kd=F.kl_div(F.log_softmax(student/temperature,-1),
                    F.softmax(teacher/temperature,-1),reduction='batchmean')
        kd=kd / len(pos) * temperature**2
        ce=F.cross_entropy(student.reshape(-1,student.shape[-1]),gold.reshape(-1))
        loss=kd+0.05*ce
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0);optimizer.step()
    return {'kd_loss':float(kd.detach()),'gold_ce':float(ce.detach()),'loss':float(loss.detach())}


def paired_latency(model,native,adapter,input_ids,device):
    if device.type!='cuda':return {'available':False,'reason':'CUDA required'}
    x=input_ids.to(device);model.eval();adapter.eval()
    timings={'native_ms':[],'ccm_consensus_ms':[]}
    with torch.inference_mode():
        for order in ((('native',native),('ccm_consensus',adapter)),
                      (('ccm_consensus',adapter),('native',native))):
            for name,module in order:
                with attention_slot(model,module):
                    for _ in range(2):model(input_ids=x,use_cache=False,return_dict=True)
                    for _ in range(5):
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();a.record()
                        model(input_ids=x,use_cache=False,return_dict=True)
                        b.record();b.synchronize()
                        timings[name+'_ms'].append(a.elapsed_time(b))
    t=statistics.median(timings['native_ms']);s=statistics.median(timings['ccm_consensus_ms'])
    return {'available':True,'gpu':torch.cuda.get_device_name(device),
            'native_ms':t,'ccm_consensus_ms':s,'speedup':t/s,
            'samples':timings,'scope':'full-model no-cache prefill, matched shapes, CUDA events'}


def chat_input(tokenizer, user):
    messages=[{'role':'user','content':user}]
    try:
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                          return_tensors='pt')
        if isinstance(ids,dict):ids=ids['input_ids']
        return ids
    except (ValueError,AttributeError,TypeError):
        return tokenizer('User: '+user+'\nAssistant:',return_tensors='pt')['input_ids']


@torch.inference_mode()
def greedy_chat(model, tokenizer, prompt, *, max_new_tokens=24, max_context=512):
    """Both models use identical full-prefix no-cache decode for fair outputs."""
    ids=chat_input(tokenizer,prompt).to(next(model.parameters()).device)
    if ids.shape[1]>=max_context:
        raise ValueError('chat prompt too long; adjust max_context')
    prefix=ids.shape[1]
    new_tokens=[]
    for _ in range(max_new_tokens):
        logits=model(input_ids=ids,use_cache=False,return_dict=True).logits[:,-1]
        nxt=logits.argmax(-1,keepdim=True)
        new_tokens.append(int(nxt.item()))
        ids=torch.cat((ids,nxt),dim=1)
        if tokenizer.eos_token_id is not None and int(nxt.item())==tokenizer.eos_token_id:break
        if ids.shape[1]>=max_context:break
    return {'response':tokenizer.decode(new_tokens,skip_special_tokens=True),
            'tokens_generated':len(new_tokens),'prompt_tokens':prefix}


def chat_comparison(model,native,adapter,tokenizer,prompts,*,max_new_tokens=24):
    model.eval();adapter.eval();out=[]
    for case in prompts:
        with attention_slot(model,native):
            teacher=greedy_chat(model,tokenizer,case['user'],max_new_tokens=max_new_tokens)
        with attention_slot(model,adapter):
            student=greedy_chat(model,tokenizer,case['user'],max_new_tokens=max_new_tokens)
        row={'case':case['id'],'prompt':case['user'],'teacher':teacher,'ccm':student,
             'exact_response_match':student['response']==teacher['response']}
        out.append(row)
        print('\nCHAT',case['id'],'\nUSER:',case['user'],'\nNATIVE:',teacher['response'],
              '\nCCM-V2:',student['response'])
    return out


def get_prompts(path):
    if not path:return DEFAULT_CHATS
    val=json.loads(Path(path).read_text())
    if not isinstance(val,list) or not val:
        raise ValueError('chat prompt JSON must be a nonempty list')
    result=[]
    for i,item in enumerate(val):
        if isinstance(item,str):result.append({'id':f'custom_{i}','user':item})
        elif isinstance(item,dict) and isinstance(item.get('user'),str):
            result.append({'id':str(item.get('id',f'custom_{i}')),'user':item['user']})
        else:raise ValueError('each chat prompt must be a string or {id,user}')
    return result


def build_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-id',default=MODEL)
    ap.add_argument('--revision',default=MODEL_REVISION)
    ap.add_argument('--profile',choices=('smoke','quick','research'),default='quick')
    ap.add_argument('--output-dir',default='/content/ccm_consensus_smol360')
    ap.add_argument('--slots',type=int,default=128)
    ap.add_argument('--tag-width',type=int,default=64)
    ap.add_argument('--cell-width',type=int,default=256)
    ap.add_argument('--score-scale',type=float,default=2.0)
    ap.add_argument('--lr',type=float,default=0.0007)
    ap.add_argument('--seed',type=int,default=2031)
    ap.add_argument('--length',type=int,default=None)
    ap.add_argument('--train-blocks',type=int,default=None)
    ap.add_argument('--dev-blocks',type=int,default=None)
    ap.add_argument('--test-blocks',type=int,default=None)
    ap.add_argument('--local-steps',type=int,default=None)
    ap.add_argument('--logit-steps',type=int,default=None)
    ap.add_argument('--chat-new-tokens',type=int,default=24)
    ap.add_argument('--chat-prompts-json',default=None)
    ap.add_argument('--skip-chat',action='store_true')
    ap.add_argument('--allow-cpu',action='store_true')
    return ap


def main(args=None, *, return_runtime=False):
    args=build_parser().parse_args(args)
    profiles={'smoke':(64,8,4,4,8,0,4),
              'quick':(128,48,12,16,120,12,40),
              'research':(256,192,32,48,600,64,100)}
    length,ntrain,ndev,ntest,nlocal,nlogit,eval_every=profiles[args.profile]
    length=args.length or length
    ntrain=args.train_blocks or ntrain
    ndev=args.dev_blocks or ndev
    ntest=args.test_blocks or ntest
    if args.local_steps is not None:nlocal=args.local_steps
    if args.logit_steps is not None:nlogit=args.logit_steps
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not args.allow_cpu:raise RuntimeError('A CUDA GPU is required for this Colab experiment')
    if min(length,ntrain,ndev,ntest)<1 or nlocal<0 or nlogit<0 or nlocal+nlogit<1:
        raise ValueError('invalid training/evaluation counts')
    if length<3:raise ValueError('sequence length must be >=3')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    rng(args.seed)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from huggingface_hub import model_info
    pinned=args.revision or model_info(args.model_id).sha
    if args.revision and model_info(args.model_id).sha!=pinned:
        print('Pinned HF revision',pinned,'(not auto-updating to latest)')
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if device.type=='cuda' else torch.float32)
    tok=AutoTokenizer.from_pretrained(args.model_id,revision=pinned,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.model_id,revision=pinned,
        torch_dtype=dtype,attn_implementation='sdpa',trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    if model.config.model_type!='llama' or not hasattr(model.model.layers[0],'self_attn'):
        raise ValueError('Expected Llama-derived SmolLM2 with layer[0].self_attn')
    native=model.model.layers[0].self_attn
    cfg=ConsensusConfig(hidden_size=int(model.config.hidden_size), slots=args.slots,
                        tag_width=args.tag_width, cell_width=args.cell_width,
                        score_scale=args.score_scale)
    core=CompetitiveConsensusCeNN(cfg).to(device)
    adapter=ConsensusAttentionAdapter(core)
    optimizer=torch.optim.AdamW(core.parameters(),lr=args.lr,weight_decay=0.001)
    print('MODEL',args.model_id,pinned,'GPU',str(device),'precision',str(dtype))
    print('NATIVE_PARAMETERS',sum(p.numel() for p in native.parameters()),
          'CCM_PARAMETERS',sum(p.numel() for p in core.parameters()),
          'CELL_SLOTS',args.slots,'LENGTH',length)
    train_ids=load_blocks(tok,'train',ntrain,length)
    dev_ids=load_blocks(tok,'validation',ndev,length)
    train_cache=capture_teacher_attention(model,train_ids,1,device)
    dev_cache=capture_teacher_attention(model,dev_ids,1,device)
    best=float('inf');best_path=out/'best_ccm_consensus_layer0.pt'
    history=[];started=time.perf_counter()

    def evaluate(step,phase,loss):
        nonlocal best
        core.eval()
        local=local_metrics(core,dev_cache,device)
        full=compare_full_model(model,native,adapter,dev_ids,1,device)
        row={'step':step,'phase':phase,'train_loss':loss,**local,**full}
        history.append(row)
        print('DEV',json.dumps({k:row[k] for k in ('step','phase','cosine','delta_nll','teacher_agreement')}))
        (out/'dev_history.json').write_text(json.dumps(history,indent=2))
        if math.isfinite(full['student_nll']) and full['student_nll'] < best:
            best=full['student_nll']
            torch.save({'core':core.state_dict(),'config':cfg.__dict__,'step':step,
                        'phase':phase,'revision':pinned},best_path)
    for step in range(1,nlocal+1):
        core.train()
        x,y=train_cache[random.randrange(len(train_cache))]
        loss=local_loss(core,x,y,device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(),1.0)
        optimizer.step()
        if step==1 or step%eval_every==0 or step==nlocal:
            evaluate(step,'local',float(loss.detach()))
    for step in range(1,nlogit+1):
        i=random.randrange(len(train_ids))
        token=train_ids[i:i+1].to(device)
        ld=logit_training_step(model,native,adapter,token,optimizer)
        if step==nlogit or step%max(1,nlogit//3)==0:
            evaluate(nlocal+step,'end_to_end',ld['loss'])
    assert best_path.exists(),'No acceptable finite DEV NLL was found; do not evaluate test'
    chosen=torch.load(best_path,map_location='cpu',weights_only=True)
    core.load_state_dict(chosen['core']);core.eval()
    dev=compare_full_model(model,native,adapter,dev_ids,1,device)
    dev_local=local_metrics(core,dev_cache,device)
    test_ids=load_blocks(tok,'test',ntest,length)  # sealed until after DEV selection
    test=compare_full_model(model,native,adapter,test_ids,1,device)
    latency=paired_latency(model,native,adapter,dev_ids[:1],device)
    accepted=(dev['delta_nll']<=0.02 and test['delta_nll']<=0.02 and
              dev['teacher_agreement']>=0.98 and test['teacher_agreement']>=0.98 and
              dev_local['cosine']>=0.95 and latency.get('speedup',0)>1.0)
    prompts=get_prompts(args.chat_prompts_json)
    # Qualitative prompts are inspected after selection and cannot influence the checkpoint.
    chats=[] if args.skip_chat else chat_comparison(model,native,adapter,tok,prompts,
                                                   max_new_tokens=args.chat_new_tokens)
    report={'model':args.model_id,'revision':pinned,'architecture':'cellular consensus, bounded memory',
      'novelty_claim':False,'pure_cellular_layer':True,'other_layers_native':True,
      'soft_cellular_merges_attention_equivalent_over_tape':True,
      'selected_step':chosen['step'],'selected_phase':chosen['phase'],
      'native_attention_params':sum(p.numel() for p in native.parameters()),
      'ccm_params':sum(p.numel() for p in core.parameters()),'slots':args.slots,
      'profile':args.profile,'sequence_length':length,'training':{'local_steps':nlocal,'logit_steps':nlogit},
      'dev':dev,'dev_local':dev_local,'test':test,'paired_latency':latency,
      'chat_cases':len(chats),'chat_exact_response_match_rate':
        sum(x['exact_response_match'] for x in chats)/len(chats) if chats else None,
      'cached_decode_supported':False,'chat_decode':'identical full-prefix greedy no-cache',
      'accepted':accepted,'experimental_only':True,'elapsed_seconds':time.perf_counter()-started,
      'acceptance':{'max_delta_nll':0.02,'min_agreement':0.98,'min_local_cosine':0.95,
                    'min_full_model_speedup':1.0}}
    (out/'final_report.json').write_text(json.dumps(report,indent=2))
    (out/'chat_comparison.json').write_text(json.dumps(chats,indent=2))
    with (out/'dev_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys()))
        w.writeheader();w.writerows(history)
    print('FINAL',json.dumps({k:report[k] for k in ('selected_step','accepted','chat_cases')}))
    print('The pretrained native layer remains the deployment fallback.')
    if return_runtime:
        return report, {'model':model,'tokenizer':tok,'native':native,
                        'adapter':adapter,'cell':core}
    return report


if __name__=='__main__':main()
