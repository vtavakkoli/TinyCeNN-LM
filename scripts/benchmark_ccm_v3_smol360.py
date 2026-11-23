"""CCM-v3 SmolLM2 layer-0 parity-first experiment: native projections + RoPE.

This is an ATTENTION-EQUIVALENT bounded cellular control, not an efficient or
novel attention-free module. Evaluate real labels/behavior instead of teacher
matching; log parity and speed explicitly. Refuse to train on parity failure.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
import zipfile

import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.ccm_v3_rotary_consensus import RotaryConsensusConfig,RotaryCellularConsensus
from scripts.benchmark_ccm_consensus_smol360 import attention_slot,load_blocks,compare_full_model,paired_latency
from scripts.real_fast_eval import CHAT_CASES, arc_easy_eval,real_chat_eval

MODEL_ID='HuggingFaceTB/SmolLM2-360M-Instruct'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','quick','research'),default='quick')
    p.add_argument('--output-dir',default='/content/CCM_v3_Smol360_Results')
    p.add_argument('--revision',default='a10cc1512eabd3dde888204e902eca88bddb4951')
    p.add_argument('--slots',type=int,default=256)
    p.add_argument('--query-block',type=int,default=16)
    p.add_argument('--correction-rank',type=int,default=16)
    p.add_argument('--steps',type=int,default=None)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=2031)
    p.add_argument('--eval-every',type=int,default=16)
    p.add_argument('--train-blocks',type=int,default=None)
    p.add_argument('--dev-blocks',type=int,default=None)
    p.add_argument('--test-blocks',type=int,default=None)
    p.add_argument('--length',type=int,default=None)
    p.add_argument('--arc-questions',type=int,default=16)
    p.add_argument('--chat-new-tokens',type=int,default=48)
    # FP32 checks prove algebraic wiring; reduced-precision checks detect numerical drift.
    # Do not gate exclusively on a single maximum over millions of vocabulary logits.
    p.add_argument('--max-abs-parity',type=float,default=0.02,
                   help='Maximum FP32 full-model logit gap; strict algebraic control')
    p.add_argument('--max-fp32-rel-mse',type=float,default=1e-6)
    p.add_argument('--max-runtime-rel-mse',type=float,default=1e-3)
    p.add_argument('--max-runtime-mean-abs',type=float,default=0.08)
    p.add_argument('--min-parity-agreement',type=float,default=.99)
    p.add_argument('--resume-checkpoint',default=None,
                   help='Only a native-initialized v3 checkpoint, NOT a CCM-v2 checkpoint')
    p.add_argument('--skip-chat',action='store_true')
    p.add_argument('--skip-arc',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    return p.parse_args(argv)


def parity_check(model,native,candidate,tokenizer,device,max_tokens=96):
    text=('The capital of Austria is Vienna. A machine learning model needs '
          'accurate positions and pre-trained query key and value matrices. ')*10
    probe=tokenizer(text,return_tensors='pt',add_special_tokens=True)['input_ids'][:,:max_tokens].to(device)
    with torch.inference_mode():
        with attention_slot(model,native):a=model(input_ids=probe,use_cache=False).logits.float()
        with attention_slot(model,candidate):b=model(input_ids=probe,use_cache=False).logits.float()
    err=(a-b).abs()
    return {'max_abs_logits':float(err.max().item()),'mean_abs_logits':float(err.mean().item()),
            'top1_agreement':float((a.argmax(-1)==b.argmax(-1)).float().mean().item()),
            'tokens':int(probe.numel()),'context_fits':probe.shape[1]<=candidate.config.slots,
            'native_vs_candidate_relative_mse':float(((a-b).square().mean()/a.square().mean().clamp_min(1e-9)).item())}


def fp32_parity_check(model, native, candidate, tokenizer, device, original_dtype, max_tokens=96):
    """Check full-model native-vs-cellular algebra with reduced-precision rounding removed.

    Converts the *same already loaded (possibly quantized-to-FP16)* weights to FP32
    temporarily on GPU. Always restores the original runtime dtype even on failure.
    Temporarily disables TF32 matmul, so this is an actual FP32 reference check.
    This diagnostic does not touch the optimizer; caller constructs it afterwards.
    """
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.to(dtype=torch.float32)
        candidate.to(dtype=torch.float32)
        return parity_check(model, native, candidate, tokenizer, device, max_tokens=max_tokens)
    finally:
        model.to(dtype=original_dtype)
        candidate.to(dtype=original_dtype)
        # Trainable adapter correction remains FP32 on FP16/BF16 Colab GPU.
        if candidate.config.correction_rank:
            candidate.correct_down.float()
            candidate.correct_up.float()
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


def parity_decision(runtime, fp32, *, max_fp32_abs=0.02, max_fp32_rel_mse=1e-6,
                    max_runtime_rel_mse=1e-3, max_runtime_mean_abs=0.08,
                    min_top1=0.99):
    """Require both strict FP32 algebraic fidelity AND practical dtype stability."""
    for label, measure in (('runtime', runtime), ('fp32', fp32)):
        fields=('max_abs_logits','mean_abs_logits','top1_agreement',
                'native_vs_candidate_relative_mse')
        if not measure.get('context_fits', False) or not all(
                math.isfinite(float(measure.get(field, float('nan')))) for field in fields):
            return {'passed':False, 'reason':label+' parity is nonfinite or context exceeds capacity'}
    fp32_ok=(fp32['max_abs_logits']<=max_fp32_abs
             and fp32['native_vs_candidate_relative_mse']<=max_fp32_rel_mse
             and fp32['top1_agreement']>=min_top1)
    runtime_ok=(runtime['native_vs_candidate_relative_mse']<=max_runtime_rel_mse
                and runtime['mean_abs_logits']<=max_runtime_mean_abs
                and runtime['top1_agreement']>=min_top1)
    return {'passed':bool(fp32_ok and runtime_ok),'fp32_algebra_pass':bool(fp32_ok),
            'reduced_precision_pass':bool(runtime_ok),
            'reason':'passed' if (fp32_ok and runtime_ok) else
                     ('FP32 algebraic mismatch' if not fp32_ok else 'reduced-precision drift'),
            'thresholds':{'fp32_max_abs':max_fp32_abs,'fp32_max_relative_mse':max_fp32_rel_mse,
                          'runtime_max_relative_mse':max_runtime_rel_mse,
                          'runtime_max_mean_abs':max_runtime_mean_abs,'min_top1':min_top1}}


def train_gold_step(model,native,adapter,optimizer,batch):
    model.eval();adapter.train();optimizer.zero_grad(set_to_none=True)
    with attention_slot(model,adapter):
        logits=model(input_ids=batch,use_cache=False).logits[:,:-1].float()
    gold=batch[:,1:]
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),gold.reshape(-1))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite gold cross-entropy')
    loss.backward()
    grad=float(nn.utils.clip_grad_norm_((p for p in adapter.parameters() if p.requires_grad),1.0))
    optimizer.step()
    return {'train_nll':float(loss.detach()),'grad_norm':grad}


def main(argv=None,*,return_runtime=False):
    a=parse_args(argv)
    if a.slots & (a.slots-1) or a.slots<2:raise ValueError('Slots must be power of two')
    if min(a.query_block,a.arc_questions,a.eval_every)<1:raise ValueError('invalid experiment settings')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.allow_cpu:raise RuntimeError('GPU runtime required for full experiment')
    random.seed(a.seed);torch.manual_seed(a.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(a.seed)
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    settings={'smoke':(64,12,4,4,0),'quick':(128,64,12,16,64),'research':(256,256,32,48,320)}
    length,ntrain,ndev,ntest,steps=settings[a.profile]
    length=a.length or length;ntrain=a.train_blocks or ntrain;ndev=a.dev_blocks or ndev
    ntest=a.test_blocks or ntest;steps=a.steps if a.steps is not None else steps
    if length>a.slots:
        print('WARNING: training length exceeds cellular memory. Expect native divergence.')
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if device.type=='cuda' else torch.float32)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=a.revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=a.revision,
                         dtype=dtype,attn_implementation='sdpa',trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    cfg=model.config
    if cfg.model_type!='llama' or not hasattr(model.model.layers[0],'self_attn'):
        raise RuntimeError('Expected SmolLM2 Llama layer zero')
    native=model.model.layers[0].self_attn
    dims=RotaryConsensusConfig(hidden_size=int(cfg.hidden_size),
        query_heads=int(cfg.num_attention_heads),key_value_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)),
        slots=a.slots,query_block=a.query_block,correction_rank=a.correction_rank)
    student=RotaryCellularConsensus.from_pretrained_attention(native,dims).to(device=device,dtype=dtype).eval()
    # Keep trainable LoRA-style correction in FP32 even on T4 FP16.
    if dims.correction_rank:
        student.correct_down.float(); student.correct_up.float()
    # All full native Q/K/V/output projections are frozen. Optional low-rank
    # correction is trainable and zero on first step (exact pretrained warm start).
    resumed_step=0
    cp=None
    if a.resume_checkpoint:
        cp=torch.load(a.resume_checkpoint,map_location='cpu',weights_only=True)
        if cp.get('method')!='CCM-v3-rope-cellular-parity' or cp.get('model_revision')!=a.revision or cp.get('cfg')!=dims.__dict__:
            raise ValueError('Only same-config, same-revision CCM-v3 checkpoints can resume. CCM-v2 weights are INCOMPATIBLE.')
        student.load_state_dict(cp['core'])
        resumed_step=int(cp['step'])
    native_params=sum(p.numel() for p in native.parameters())
    student_params_count=sum(p.numel() for p in student.parameters())
    # The supplied Colab parity failed with max_abs 0.4834, but only 0.0003007
    # relative MSE and 100% argmax agreement. That is insufficient evidence of
    # miswiring: compare an FP32 reference to isolate FP16/BF16 SDPA rounding.
    parity=parity_check(model,native,student,tokenizer,device,max_tokens=min(96,a.slots))
    fp32_parity=fp32_parity_check(model,native,student,tokenizer,device,dtype,
                                  max_tokens=min(96,a.slots))
    decision=parity_decision(parity,fp32_parity,
                            max_fp32_abs=a.max_abs_parity,
                            max_fp32_rel_mse=a.max_fp32_rel_mse,
                            max_runtime_rel_mse=a.max_runtime_rel_mse,
                            max_runtime_mean_abs=a.max_runtime_mean_abs,
                            min_top1=a.min_parity_agreement)
    (output/'initial_parity.json').write_text(json.dumps(
        {'runtime':parity,'fp32_reference':fp32_parity,'decision':decision,
         'runtime_dtype':str(dtype)},indent=2))
    print('INIT RUNTIME PARITY',json.dumps(parity))
    print('INIT FP32 PARITY',json.dumps(fp32_parity))
    print('PARITY DECISION',json.dumps(decision))
    if a.resume_checkpoint:
        print('Resumed a trained v3 correction: strict initial native parity is informative, not a checkpoint rejection gate.')
    elif not decision['passed']:
        raise RuntimeError('Validated FP32 + reduced-precision parity failed: '+decision['reason']+
                           '. See initial_parity.json. Do not train a miswired attention replacement.')
    # Create/reload the optimizer AFTER temporary dtype changes to preserve correct
    # parameter identity and FP32 optimizer moment estimates.
    student_params=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(student_params,lr=a.lr,weight_decay=0.001) if student_params else None
    if cp is not None and optimizer is not None and 'optimizer' in cp:
        optimizer.load_state_dict(cp['optimizer'])
    train_ids=load_blocks(tokenizer,'train',ntrain,length) if steps else None
    dev_ids=load_blocks(tokenizer,'validation',ndev,length)
    history=[];best=float('inf');best_path=output/'best_ccm_v3_layer0.pt'

    def evaluate(step,train_stats=None):
        nonlocal best
        student.eval()
        m=compare_full_model(model,native,student,dev_ids,1,device)
        row={'step':step,**(train_stats or {}),
             'dev_gold_nll':m['student_nll'],'native_gold_nll':m['teacher_nll'],
             'dev_delta_nll':m['delta_nll'],'dev_gold_accuracy':m['student_gold_accuracy'],
             'native_gold_accuracy':m['teacher_gold_accuracy'],
             'teacher_agreement_DIAGNOSTIC':m['teacher_agreement']}
        history.append(row)
        (output/'dev_history.json').write_text(json.dumps(history,indent=2))
        print('DEV',json.dumps(row))
        if math.isfinite(m['student_nll']) and m['student_nll']<best:
            best=m['student_nll']
            cp={'method':'CCM-v3-rope-cellular-parity','core':{k:v.detach().cpu().clone() for k,v in student.state_dict().items()},
                'model_revision':a.revision,'cfg':dims.__dict__,'step':step,
                'dev_gold_nll':best}
            if optimizer is not None:cp['optimizer']=optimizer.state_dict()
            torch.save(cp,best_path)
        return row
    evaluate(resumed_step)
    if steps and optimizer is None:print('No trainable low-rank correction: evaluation-only')
    for i in range(1,steps+1) if optimizer is not None else []:
        batch=train_ids[random.randrange(len(train_ids)):][:1].to(device)
        stats=train_gold_step(model,native,student,optimizer,batch)
        if i==1 or i%a.eval_every==0 or i==steps:evaluate(resumed_step+i,stats)
    if not best_path.is_file():raise RuntimeError('No finite validation checkpoint')
    selected=torch.load(best_path,map_location='cpu',weights_only=True)
    student.load_state_dict(selected['core']);student.eval()
    dev=compare_full_model(model,native,student,dev_ids,1,device)
    # TEST never opened until checkpoint selected.
    test_ids=load_blocks(tokenizer,'test',ntest,length)
    test=compare_full_model(model,native,student,test_ids,1,device)
    latency=paired_latency(model,native,student,dev_ids[:1],device)
    from datasets import load_dataset
    arc=None
    if not a.skip_arc:
        arc_ds=load_dataset('allenai/ai2_arc','ARC-Easy',split='validation')
        with attention_slot(model,native):arc_native=arc_easy_eval(model,tokenizer,limit=a.arc_questions,dataset=arc_ds,seed=a.seed)
        with attention_slot(model,student):arc_student=arc_easy_eval(model,tokenizer,limit=a.arc_questions,dataset=arc_ds,seed=a.seed)
        arc={'native':arc_native,'ccm_v3':arc_student}
        (output/'arc_comparison.json').write_text(json.dumps(arc,indent=2))
    chat=None
    if not a.skip_chat:
        with attention_slot(model,native):chat_native=real_chat_eval(model,tokenizer,cases=CHAT_CASES,max_new_tokens=a.chat_new_tokens)
        with attention_slot(model,student):chat_student=real_chat_eval(model,tokenizer,cases=CHAT_CASES,max_new_tokens=a.chat_new_tokens)
        chat={'native':chat_native,'ccm_v3':chat_student}
        (output/'chat_comparison.json').write_text(json.dumps(chat,indent=2))
    report={'model':MODEL_ID,'model_revision':a.revision,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
            'cfg':dims.__dict__,'selected_step':selected['step'],'parity':parity,
            'fp32_parity':fp32_parity,'parity_decision':decision,
            'architecture':'RoPE grouped-head conserved cellular log-mass; attention-equivalent bounded tape',
            'pretrained_projection_warmstart':True,'old_v2_weights_compatible':False,
            'native_params':native_params,'student_params':student_params_count,
            'trainable_params':sum(p.numel() for p in student_params),
            'native_attention_called_inside_candidate':False,'other_transformer_layers_unchanged':True,
            'dev':dev,'test':test,'paired_latency':latency,
            'arc_score':{k:v['accuracy'] for k,v in arc.items()} if arc else None,
            'chat_score':{k:v['accuracy'] for k,v in chat.items()} if chat else None,
            'teacher_agreement_used_as_gate':False,
            'attention_free_novelty_claim':False,'gpu_speedup_claim':False,
            'quality_pass':bool(dev['delta_nll']<=0.02 and test['delta_nll']<=0.02 and
              dev['student_gold_accuracy']>=dev['teacher_gold_accuracy']-0.02 and
              test['student_gold_accuracy']>=test['teacher_gold_accuracy']-0.02),
            'limitations':'No KV cache, no parameter reduction; bounded normalized consensus is attention-equivalent. Small ARC/chat probes.'}
    (output/'final_report.json').write_text(json.dumps(report,indent=2))
    if history:
        with (output/'dev_history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0].keys())+sorted(set().union(*(set(r) for r in history))-set(history[0])))
            w.writeheader();w.writerows(history)
    archive=output.parent/(output.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as f:
        for p in sorted(output.rglob('*')):
            if p.is_file():f.write(p,p.relative_to(output))
    print('RESULT ARCHIVE',archive,'REPORT',json.dumps({k:report[k] for k in ('selected_step','quality_pass','arc_score','chat_score')}))
    if return_runtime:return report,archive,{'model':model,'native':native,'student':student,'tokenizer':tokenizer}
    return report,archive

if __name__=='__main__':main()
