"""Prefill-only FIRST-attention-layer viability probe for SmolLM2-360M-Instruct.

Reports attention concentration and the fidelity of *oracle* top-k truncation.
No model modifications, no training, no claims of a faster attention replacement.
Install transformers==4.57.6; use a CUDA GPU for the 360M experiment.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def analyze_probability_value(probs: torch.Tensor, values: torch.Tensor, *, window=32):
    """probs:[B,H,T,T]; values:[B,H,T,D], with GQA already expanded.
    Returns attention concentration and oracle sparse output error.
    """
    if probs.ndim != 4 or values.ndim != 4:
        raise ValueError('Expected [B,H,T,T] and [B,H,T,D]')
    B,H,T,T2=probs.shape
    if T!=T2 or values.shape[:3]!=(B,H,T):
        raise ValueError('QK and value dimensions do not agree')
    start=min(T-1,max(16,T//4))
    p=probs.float()
    v=values.float()
    norm=p.sum(-1)
    if not torch.allclose(norm,torch.ones_like(norm),atol=3e-3,rtol=3e-3):
        raise ValueError('Probability rows are not normalized (check masks/implementation)')
    output=torch.matmul(p,v)
    result={'length':T,'heads':H,'query_start':start,'topk':{}}
    for k in (1,4,8,16):
        n=min(k,T)
        weight,index=p.topk(n,dim=-1)
        mass=weight.sum(-1)
        # Evaluate grouped gathers without materializing [B,H,T,T,D] intermediary.
        flat_v=v.reshape(B*H,T,v.shape[-1])
        flat_ids=index.reshape(B*H,T,n)
        selected=torch.gather(flat_v[:,None].expand(-1,T,-1,-1),2,
                              flat_ids[...,None].expand(-1,-1,-1,v.shape[-1]))
        pred=((weight.reshape(B*H,T,n)/mass.reshape(B*H,T,1).clamp_min(1e-9))
              [...,None]*selected).sum(2).reshape(B,H,T,v.shape[-1])
        aa=output[:,:,start:]
        bb=pred[:,:,start:]
        err=(bb-aa).square().mean().item()/(aa.square().mean().item()+1e-9)
        mask=mass[:,:,start:]
        per_head=mask.mean(dim=(0,2)).tolist()
        result['topk'][str(k)]={'mean_probability_mass':mask.mean().item(),
                  'min_head_mass':min(per_head), 'mass_per_head':per_head,
                  'output_relative_mse':err}
    entropy=-(p.clamp_min(1e-20)*p.clamp_min(1e-20).log()).sum(-1)
    result['effective_context_size_mean']=entropy[:,:,start:].exp().mean().item()
    qi=torch.arange(T,device=p.device)[:,None]
    kj=torch.arange(T,device=p.device)[None,:]
    recent=(kj<=qi)&(kj>qi-window)
    result['local_window_mass_mean']=(p*recent[None,None,:,:]).sum(-1)[:,:,start:].mean().item()
    return result


@torch.inference_mode()
def run(args):
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU requested but unavailable')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device=torch.device(args.device)
    dtype=torch.float16 if device.type=='cuda' else torch.float32
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype=dtype,
                                               attn_implementation='eager',trust_remote_code=False).to(device).eval()
    first=model.model.layers[0]
    H=model.config.num_attention_heads
    HK=model.config.num_key_value_heads
    if H%HK:
        raise ValueError('Invalid GQA head grouping')
    texts=[
        'A small robot navigates a narrow hallway, records the route in memory, and later answers which door was near the staircase. ',
        'Consider an experiment in which the sequence contains dates, names, repeated identifiers, distances, and numerical values. ',
        'The programmer inspects a recursive algorithm, writes a unit test, fixes the incorrect index and compares the outputs. ',
        'The scientist described a cellular dynamical system with distributed local memory and feedback between neighboring cells. ',
    ]
    report={'model':args.model,'revision':args.revision,'device':str(device), 'precision':str(dtype),
            'probe':'native layer-0 attention concentration and oracle top-k truncation; no WaveFlux or CCM training',
            'results':[]}
    for length in args.lengths:
        for case,text in enumerate(texts[:args.texts]):
            ids=tokenizer(text*80,add_special_tokens=True,return_tensors='pt').input_ids[:,:length]
            if ids.shape[1]!=length:
                raise RuntimeError('Insufficient tokens for requested context')
            ids=ids.to(device)
            result=model(input_ids=ids,use_cache=False,output_attentions=True,output_hidden_states=True,return_dict=True)
            p=result.attentions[0].float()
            normed=first.input_layernorm(result.hidden_states[0])
            v=first.self_attn.v_proj(normed)
            B,T=ids.shape
            head_dim=v.shape[-1]//HK
            v=v.reshape(B,T,HK,head_dim).transpose(1,2).repeat_interleave(H//HK,dim=1)
            entry=analyze_probability_value(p,v,window=args.window)
            entry['text_case']=case
            report['results'].append(entry)
            print(f"context={length} case={case} top1mass={entry['topk']['1']['mean_probability_mass']:.3f} "
                  f"top4mass={entry['topk']['4']['mean_probability_mass']:.3f} "
                  f"top4mse={entry['topk']['4']['output_relative_mse']:.3f}")
            del result,p,v
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print('Saved',output)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='HuggingFaceTB/SmolLM2-360M-Instruct')
    parser.add_argument('--revision',default='a10cc1512eabd3dde888204e902eca88bddb4951')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--lengths',type=int,nargs='+',default=[128,256])
    parser.add_argument('--texts',type=int,default=3)
    parser.add_argument('--window',type=int,default=32)
    parser.add_argument('--output',default='/content/cenn_ccm_feasibility/smol360_attention_diagnostic.json')
    run(parser.parse_args())
