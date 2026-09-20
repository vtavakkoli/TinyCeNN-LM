from __future__ import annotations
import random

import torch
import torch.nn.functional as F

from .qwen35_cennmixer_v4 import direct_mixer_losses_v4, reset_stream_state_v4


def blocks(tokenizer,texts,n,seq_len,seed):
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


def forward_kl(student,teacher):
    log_s=F.log_softmax(student.float(),dim=-1)
    p_t=F.softmax(teacher.float(),dim=-1)
    return F.kl_div(log_s,p_t,reduction="batchmean")/max(student.shape[1],1)


def reverse_kl(student,teacher):
    log_s=F.log_softmax(student.float(),dim=-1)
    log_t=F.log_softmax(teacher.float(),dim=-1)
    p_s=log_s.exp()
    return (p_s*(log_s-log_t)).sum(-1).mean()


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


def top1_margin_loss(student_logits,teacher_logits):
    s=student_logits.float(); t=teacher_logits.float()
    with torch.no_grad():
        tv,ti=t.topk(2,dim=-1)
        target=ti[...,0]
        teacher_gap=(tv[...,0]-tv[...,1]).clamp(0.05,1.5)
    target_logit=s.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    sv,si=s.topk(2,dim=-1)
    other=torch.where(si[...,0].eq(target),sv[...,1],sv[...,0])
    return F.relu(teacher_gap+other-target_logit).mean()


def stage_targets(alpha,a):
    alpha=float(alpha)
    if a.quick_smoke:
        local={"max_mixer_mse":0.25,"max_mixer_cosine":0.13,"max_mixer_delta":0.38}
        if alpha<=0:
            return {"min_top1":0.999,"max_kl":1e-4,"max_hidden_mse":1e-4,**local}
        if alpha<1:
            return {
                "min_top1":max(0.84,0.985-0.22*alpha),
                "max_kl":0.035+0.24*alpha,
                "max_hidden_mse":0.015+0.22*alpha,
                **local,
            }
        return {
            "min_top1":0.84,"max_kl":0.30,"max_hidden_mse":0.25,
            "max_mixer_mse":0.28,"max_mixer_cosine":0.145,"max_mixer_delta":0.42,
        }

    if alpha<=0:
        return {
            "min_top1":0.999,"max_kl":1e-4,"max_hidden_mse":1e-4,
            "max_mixer_mse":a.max_mixer_mse,"max_mixer_cosine":0.07,"max_mixer_delta":0.22,
        }
    if alpha<1:
        return {
            "min_top1":max(a.min_top1,0.995-0.02*alpha),
            "max_kl":max(a.max_kl,0.045),
            "max_hidden_mse":max(a.max_hidden_mse,0.07),
            "max_mixer_mse":a.max_mixer_mse,"max_mixer_cosine":0.07,"max_mixer_delta":0.22,
        }
    return {
        "min_top1":a.min_top1,"max_kl":a.max_kl,"max_hidden_mse":a.max_hidden_mse,
        "max_mixer_mse":a.max_mixer_mse,"max_mixer_cosine":0.07,"max_mixer_delta":0.22,
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
    totals={"teacher_ce":0.0,"student_ce":0.0,"kl":0.0,"reverse_kl":0.0,
            "hidden_mse":0.0,"delta_mse":0.0,"top1_margin_loss":0.0}
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
        totals["kl"]+=float(forward_kl(so.logits,to.logits))
        totals["reverse_kl"]+=float(reverse_kl(so.logits,to.logits))
        totals["hidden_mse"]+=float(hm); totals["delta_mse"]+=float(dm)
        totals["top1_margin_loss"]+=float(top1_margin_loss(so.logits,to.logits))
        if local_teacher:
            loc=direct_mixer_losses_v4(student)
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


def on_policy_distill_loss(student,teacher,prefix,device,max_new_tokens=12):
    """Teacher feedback on prefixes actually visited by the student."""
    student.eval()
    reset_stream_state_v4(student)
    with torch.no_grad():
        gen=student.generate(
            input_ids=prefix.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(student.config,"eos_token_id",None),
        )
    reset_stream_state_v4(student)

    x=gen[:,:-1]
    with torch.no_grad():
        to=teacher(input_ids=x,use_cache=False,return_dict=True)
    student.train()
    so=student(input_ids=x,use_cache=False,return_dict=True)

    # Emphasize the self-generated suffix; prefix is only context.
    start=max(prefix.shape[1]-1,0)
    sl=so.logits[:,start:]
    tl=to.logits[:,start:]
    return (
        0.45*reverse_kl(sl,tl)
        +0.30*forward_kl(sl,tl)
        +0.25*top1_margin_loss(sl,tl)
    )


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
            reset_stream_state_v4(m)
            y=m.generate(**enc,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
            z=y[0,enc.input_ids.shape[1]:]
            return z.tolist(),tok.decode(z,skip_special_tokens=True).strip()
        ti,tt=run(teacher); si,st=run(student)
        a,b=set(ti),set(si)
        rows.append({"prompt":p,"qwen":tt,"cenn":st,"exact":ti==si,"jaccard":len(a&b)/max(len(a|b),1)})
    return rows
