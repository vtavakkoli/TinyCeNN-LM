#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_cennmixer_v1 import (
    CeNNMixerV1,
    CeNNMixerV1Config,
    install_cenn_mixer_v1,
    freeze_all_except_cenn,
)


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--base-model",default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--layers",default="0",help="comma-separated decoder layer indices")
    p.add_argument("--seq-len",type=int,default=128)
    p.add_argument("--train-blocks",type=int,default=512)
    p.add_argument("--val-blocks",type=int,default=32)
    p.add_argument("--max-updates",type=int,default=1000)
    p.add_argument("--probe-every",type=int,default=50)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--groups",type=int,default=16)
    p.add_argument("--cell-dim",type=int,default=32)
    p.add_argument("--graph-steps",type=int,default=1)
    p.add_argument("--min-top1",type=float,default=0.97)
    p.add_argument("--max-kl",type=float,default=0.03)
    p.add_argument("--max-hidden-mse",type=float,default=0.03)
    p.add_argument("--output-dir",default="results/cennmixer_v1_qwen35_08b")
    p.add_argument("--seed",type=int,default=8621)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dtype_for(device):
    if device.type!="cuda": return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def blocks(tokenizer,texts,n,seq_len,seed):
    rng=random.Random(seed)
    clean=[x.strip() for x in texts if isinstance(x,str) and len(x.strip())>40]
    out=[]
    while len(out)<n:
        s=" ".join(rng.choice(clean) for _ in range(8))
        ids=tokenizer(s,return_tensors="pt",truncation=False).input_ids[0]
        if ids.numel()<seq_len+1: continue
        mx=ids.numel()-(seq_len+1)
        st=rng.randint(0,mx) if mx>0 else 0
        out.append(ids[st:st+seq_len+1].unsqueeze(0))
    return out


def load_data(tokenizer,a):
    from datasets import load_dataset
    tr=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    train=blocks(tokenizer,[x["text"] for x in tr],a.train_blocks,a.seq_len,a.seed+11)
    val=blocks(tokenizer,[x["text"] for x in va],a.val_blocks,a.seq_len,a.seed+777)
    return train,val


def ce(logits,y):
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1))


def kl(student,teacher):
    s=F.log_softmax(student.float(),dim=-1)
    t=F.softmax(teacher.float(),dim=-1)
    return F.kl_div(s,t,reduction="batchmean")/max(student.shape[1],1)


def rel_mse(a,b):
    den=b.float().square().mean().clamp_min(1e-12)
    return (a.float()-b.float()).square().mean()/den


def hidden_losses(sh,th,layers):
    hm=0.0; dm=0.0
    for li in layers:
        s=sh[li+1]; t=th[li+1]
        hm=hm+rel_mse(s,t)
        if s.shape[1]>1:
            dm=dm+rel_mse(s[:,1:]-s[:,:-1],t[:,1:]-t[:,:-1])
    n=max(len(layers),1)
    return hm/n,dm/n


@torch.no_grad()
def probe(student,teacher,batches,layers,device):
    student.eval(); teacher.eval()
    vals={"teacher_ce":0.0,"student_ce":0.0,"kl":0.0,"hidden_mse":0.0,"delta_mse":0.0}
    top=tot=0
    for cpu in batches:
        ids=cpu.to(device); x=ids[:,:-1]; y=ids[:,1:]
        to=teacher(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
        so=student(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
        h,d=hidden_losses(so.hidden_states,to.hidden_states,layers)
        vals["teacher_ce"]+=float(ce(to.logits,y))
        vals["student_ce"]+=float(ce(so.logits,y))
        vals["kl"]+=float(kl(so.logits,to.logits))
        vals["hidden_mse"]+=float(h); vals["delta_mse"]+=float(d)
        top+=int((so.logits.argmax(-1)==to.logits.argmax(-1)).sum())
        tot+=so.logits.shape[0]*so.logits.shape[1]
    n=max(len(batches),1)
    for k in vals: vals[k]/=n
    vals["ce_gap"]=vals["student_ce"]-vals["teacher_ce"]
    vals["top1"]=top/max(tot,1)
    return vals


def quality_key(m,a):
    violation=(
        max(a.min_top1-m["top1"],0.0)/max(1-a.min_top1,1e-6)
        +max(m["kl"]-a.max_kl,0.0)/max(a.max_kl,1e-8)
        +max(m["hidden_mse"]-a.max_hidden_mse,0.0)/max(a.max_hidden_mse,1e-8)
    )
    return (violation,m["kl"],1-m["top1"],m["hidden_mse"],m["student_ce"])


def clone_cenn_state(student):
    return {
        n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        for n,m in student.named_modules() if isinstance(m,CeNNMixerV1)
    }


def load_cenn_state(student,state):
    mods=dict(student.named_modules())
    for n,sd in state.items(): mods[n].load_state_dict(sd,strict=True)


def reset_stream(model):
    for m in model.modules():
        if isinstance(m,CeNNMixerV1): m.reset_stream_state()


@torch.no_grad()
def gen_suite(student,teacher,tok,device):
    prompts=[
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "Explain in one sentence what an API is.",
        "The secret number is 81427. Remember it. What is the secret number? Give only the number.",
    ]
    rows=[]
    for p in prompts:
        text=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
        enc=tok(text,return_tensors="pt").to(device)
        def run(m):
            reset_stream(m)
            y=m.generate(**enc,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
            z=y[0,enc.input_ids.shape[1]:]
            return z.tolist(),tok.decode(z,skip_special_tokens=True).strip()
        ti,tt=run(teacher); si,st=run(student)
        a,b=set(ti),set(si)
        rows.append({"prompt":p,"qwen":tt,"cenn":st,"exact":ti==si,"jaccard":len(a&b)/max(len(a|b),1)})
    return rows


def main():
    a=args(); set_seed(a.seed)
    layers=[int(x) for x in a.layers.split(",") if x.strip()]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype=dtype_for(device)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    print("DEVICE",device,"dtype",dtype,"layers",layers,flush=True)

    tok=AutoTokenizer.from_pretrained(a.base_model,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token

    teacher=AutoModelForCausalLM.from_pretrained(a.base_model,dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    student=AutoModelForCausalLM.from_pretrained(a.base_model,dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    cfg=CeNNMixerV1Config(
        hidden_size=int(student.config.hidden_size),
        groups=a.groups,cell_dim=a.cell_dim,graph_steps=a.graph_steps,
    )
    layer_kinds={}
    for li in layers:
        layer=student.model.layers[li]
        layer_kinds[str(li)]=getattr(layer,"block_type","unknown")
        install_cenn_mixer_v1(student,li,cfg)

    trainable=freeze_all_except_cenn(student)
    opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0.0)
    train,val=load_data(tok,a)

    initial=probe(student,teacher,val,layers,device)
    best_m=dict(initial); best_state=clone_cenn_state(student); best_key=quality_key(initial,a); best_step=0
    hist=[]
    print("INITIAL",json.dumps(initial),flush=True)

    for step in range(1,a.max_updates+1):
        student.train()
        ids=train[(step-1)%len(train)].to(device); x=ids[:,:-1]; y=ids[:,1:]
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            to=teacher(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
        so=student(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
        hm,dm=hidden_losses(so.hidden_states,to.hidden_states,layers)
        lkl=kl(so.logits,to.logits); lce=ce(so.logits,y)
        loss=0.45*hm+0.20*dm+0.25*lkl+0.10*lce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,0.5)
        opt.step()

        if step==1 or step%a.probe_every==0 or step==a.max_updates:
            vm=probe(student,teacher,val,layers,device)
            row={"step":step,"train_loss":float(loss.detach()),**vm}
            hist.append(row)
            print("PROBE",json.dumps(row),flush=True)
            key=quality_key(vm,a)
            if key<best_key:
                best_key=key; best_m=dict(vm); best_state=clone_cenn_state(student); best_step=step
                torch.save({"state":best_state,"config":cfg.to_dict(),"layers":layers,"metrics":best_m},
                           out/"cennmixer_v1_best.pt")
                print("✓ NEW BEST step",step,flush=True)

    load_cenn_state(student,best_state)
    final=probe(student,teacher,val,layers,device)
    gens=gen_suite(student,teacher,tok,device)

    cenn_params=sum(p.numel() for p in trainable)
    replaced_params=0
    for li in layers:
        tl=teacher.model.layers[li]
        mod=tl.linear_attn if getattr(tl,"block_type",None)=="linear_attention" else tl.self_attn
        replaced_params+=sum(p.numel() for p in mod.parameters())

    report={
        "architecture":"CeNNMixer-v1 progressive Qwen3.5 sequence-mixer replacement",
        "layers":layers,
        "layer_kinds":layer_kinds,
        "config":cfg.to_dict(),
        "initial":initial,
        "best_step":best_step,
        "final":final,
        "generation":gens,
        "cenn_trainable_params":cenn_params,
        "replaced_qwen_mixer_params":replaced_params,
        "mixer_param_reduction_pct":100*(1-cenn_params/max(replaced_params,1)),
        "quality_gate":bool(final["top1"]>=a.min_top1 and final["kl"]<=a.max_kl and final["hidden_mse"]<=a.max_hidden_mse),
    }
    pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False)
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FINAL",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
