from __future__ import annotations
import random
import torch
import torch.nn.functional as F

from .qwen35_cennmixer_v3 import (
    direct_mixer_losses_v3,
    reset_stream_state_v3,
)


def blocks(tokenizer, texts, n, seq_len, seed):
    rng=random.Random(seed)
    clean=[x.strip() for x in texts if isinstance(x,str) and len(x.strip())>40]
    out=[]
    while len(out)<n:
        s=" ".join(rng.choice(clean) for _ in range(10))
        ids=tokenizer(s,return_tensors="pt",truncation=False).input_ids[0]
        if ids.numel()<seq_len+1:
            continue
        mx=ids.numel()-(seq_len+1)
        st=rng.randint(0,mx) if mx>0 else 0
        out.append(ids[st:st+seq_len+1].unsqueeze(0))
    return out


def load_data(tokenizer,a):
    from datasets import load_dataset
    tr=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    return (
        blocks(tokenizer,[x["text"] for x in tr],a.train_blocks,a.seq_len,a.seed+11),
        blocks(tokenizer,[x["text"] for x in va],a.val_blocks,a.seq_len,a.seed+777),
    )


def causal_ce(logits,y):
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1))


def distill_kl(student,teacher):
    s=F.log_softmax(student.float(),dim=-1)
    t=F.softmax(teacher.float(),dim=-1)
    return F.kl_div(s,t,reduction="batchmean")/max(student.shape[1],1)


def rel_mse(student,teacher):
    t=teacher.float(); s=student.float()
    return (s-t).square().mean()/t.square().mean().clamp_min(1e-12)


def hidden_losses(student_hidden,teacher_hidden,layers):
    hm=student_hidden[0].new_zeros((),dtype=torch.float32)
    dm=student_hidden[0].new_zeros((),dtype=torch.float32)
    for li in layers:
        s,t=student_hidden[li+1],teacher_hidden[li+1]
        hm=hm+rel_mse(s,t)
        if s.shape[1]>1:
            dm=dm+rel_mse(s[:,1:]-s[:,:-1],t[:,1:]-t[:,:-1])
    n=max(len(layers),1)
    return hm/n,dm/n


def topk_rank_loss(student_logits,teacher_logits,k):
    k=min(int(k),int(teacher_logits.shape[-1]))
    with torch.no_grad():
        idx=teacher_logits.topk(k,dim=-1).indices
        t=teacher_logits.gather(-1,idx).float()
        t=t-t.mean(-1,keepdim=True)
        scale=t.std(-1,keepdim=True).clamp_min(0.25)
        t=t/scale
    s=student_logits.gather(-1,idx).float()
    s=(s-s.mean(-1,keepdim=True))/scale
    return F.smooth_l1_loss(s,t)


def stage_targets(alpha,a):
    alpha=float(alpha)
    if a.quick_smoke:
        local={
            "max_mixer_mse":0.30,
            "max_mixer_cosine":0.15,
            "max_mixer_delta":0.45,
        }
        if alpha<=0:
            return {"min_top1":0.999,"max_kl":1e-4,"max_hidden_mse":1e-4,**local}
        if alpha<1:
            return {
                "min_top1":max(0.82,0.98-0.22*alpha),
                "max_kl":0.04+0.30*alpha,
                "max_hidden_mse":0.02+0.30*alpha,
                **local,
            }
        return {
            "min_top1":0.82,"max_kl":0.45,"max_hidden_mse":0.35,
            "max_mixer_mse":0.32,"max_mixer_cosine":0.16,"max_mixer_delta":0.48,
        }
    if alpha<=0:
        return {
            "min_top1":0.999,"max_kl":1e-4,"max_hidden_mse":1e-4,
            "max_mixer_mse":a.max_mixer_mse,"max_mixer_cosine":0.08,"max_mixer_delta":0.25,
        }
    if alpha<1:
        return {
            "min_top1":max(a.min_top1,0.99-0.02*alpha),
            "max_kl":max(a.max_kl,0.05),
            "max_hidden_mse":max(a.max_hidden_mse,0.08),
            "max_mixer_mse":a.max_mixer_mse,"max_mixer_cosine":0.08,"max_mixer_delta":0.25,
        }
    return {
        "min_top1":a.min_top1,"max_kl":a.max_kl,"max_hidden_mse":a.max_hidden_mse,
        "max_mixer_mse":a.max_mixer_mse,"max_mixer_cosine":0.08,"max_mixer_delta":0.25,
    }


def stage_ok(m,alpha,a):
    t=stage_targets(alpha,a)
    return (
        m["top1"]>=t["min_top1"] and m["kl"]<=t["max_kl"]
        and m["hidden_mse"]<=t["max_hidden_mse"]
        and m["mixer_mse"]<=t["max_mixer_mse"]
        and m["mixer_cosine"]<=t["max_mixer_cosine"]
        and m["mixer_delta"]<=t["max_mixer_delta"]
    )


def stage_violation(m,alpha,a):
    t=stage_targets(alpha,a)
    return (
        max(t["min_top1"]-m["top1"],0)/max(1-t["min_top1"],1e-5)
        +max(m["kl"]-t["max_kl"],0)/max(t["max_kl"],1e-6)
        +max(m["hidden_mse"]-t["max_hidden_mse"],0)/max(t["max_hidden_mse"],1e-6)
        +max(m["mixer_mse"]-t["max_mixer_mse"],0)/max(t["max_mixer_mse"],1e-6)
        +max(m["mixer_cosine"]-t["max_mixer_cosine"],0)/max(t["max_mixer_cosine"],1e-6)
        +max(m["mixer_delta"]-t["max_mixer_delta"],0)/max(t["max_mixer_delta"],1e-6)
    )


def quality_key(m,alpha,a):
    return (
        0 if stage_ok(m,alpha,a) else 1,
        stage_violation(m,alpha,a),m["kl"],1-m["top1"],
        m["hidden_mse"],m["mixer_mse"],m["mixer_cosine"],m["mixer_delta"],m["student_ce"],
    )


@torch.no_grad()
def probe(student,teacher,batches,layers,device,local_teacher=True):
    student.eval(); teacher.eval()
    totals={"teacher_ce":0.0,"student_ce":0.0,"kl":0.0,"hidden_mse":0.0,"delta_mse":0.0}
    if local_teacher:
        totals.update({"mixer_mse":0.0,"mixer_cosine":0.0,"mixer_delta":0.0})
    top=tot=0
    for cpu in batches:
        ids=cpu.to(device); x,y=ids[:,:-1],ids[:,1:]
        to=teacher(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
        so=student(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
        hm,dm=hidden_losses(so.hidden_states,to.hidden_states,layers)
        totals["teacher_ce"]+=float(causal_ce(to.logits,y))
        totals["student_ce"]+=float(causal_ce(so.logits,y))
        totals["kl"]+=float(distill_kl(so.logits,to.logits))
        totals["hidden_mse"]+=float(hm); totals["delta_mse"]+=float(dm)
        if local_teacher:
            loc=direct_mixer_losses_v3(student)
            totals["mixer_mse"]+=float(loc["mse"])
            totals["mixer_cosine"]+=float(loc["cosine"])
            totals["mixer_delta"]+=float(loc["delta"])
        top+=int((so.logits.argmax(-1)==to.logits.argmax(-1)).sum())
        tot+=so.logits.shape[0]*so.logits.shape[1]
    n=max(len(batches),1)
    for k in totals: totals[k]/=n
    totals["ce_gap"]=totals["student_ce"]-totals["teacher_ce"]
    totals["top1"]=top/max(tot,1)
    return totals


@torch.no_grad()
def generation_suite(student,teacher,tok,device):
    prompts=[
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "Explain in one sentence what an API is.",
        "Translate 'Good morning' into German. Give only the translation.",
        "The secret number is 81427. Remember it. What is the secret number? Give only the number.",
    ]
    rows=[]
    for p in prompts:
        chat=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
        enc=tok(chat,return_tensors="pt").to(device)
        def run(m):
            reset_stream_state_v3(m)
            y=m.generate(**enc,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
            z=y[0,enc.input_ids.shape[1]:]
            return z.tolist(),tok.decode(z,skip_special_tokens=True).strip()
        ti,tt=run(teacher); si,st=run(student)
        a,b=set(ti),set(si)
        rows.append({"prompt":p,"qwen":tt,"cenn":st,"exact":ti==si,"jaccard":len(a&b)/max(len(a|b),1)})
    return rows
