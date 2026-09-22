from __future__ import annotations
import random
import math

import torch
import torch.nn.functional as F

from .qwen35_cennmixer_v4 import direct_mixer_losses_v4, reset_stream_state_v4


def blocks(tokenizer,texts,n,seq_len,seed):
    """Contiguous corpus windows, bounded work, explicit document boundaries."""
    if n < 1 or seq_len < 2:
        raise ValueError("n >= 1 and seq_len >= 2 required")
    clean=[x.strip() for x in texts if isinstance(x,str) and x.strip()]
    if not clean:
        raise ValueError("No nonempty training documents")
    # Keep source order: randomly concatenating ten paragraphs destroys context.
    ids=tokenizer("\n\n".join(clean),return_tensors="pt",truncation=False).input_ids[0]
    if ids.numel()<seq_len+1:
        raise ValueError(f"Corpus has {ids.numel()} tokens, needs {seq_len+1}")
    rng=random.Random(seed)
    return [ids[st:st+seq_len+1].unsqueeze(0) for st in
            (rng.randint(0,ids.numel()-seq_len-1) for _ in range(n))]


def context_lengths(a):
    lengths=sorted(set(int(x) for x in a.context_lengths.split(",") if x.strip()))
    if not lengths or lengths[0]<32 or lengths[-1]>a.seq_len:
        raise ValueError("context lengths must be >=32 and <= seq-len")
    if lengths[-1] != a.seq_len:
        lengths.append(a.seq_len)
    return lengths


def retrieval_prompt(tokenizer,length,seed):
    rng=random.Random(seed)
    key=f"{rng.randrange(10000,100000)}"
    fillers=[f"Archive entry {i}: the parcel is stored on shelf {rng.randrange(100)}."
             for i in range(max(8,length//8))]
    # Vary the location, with a fresh target on each training example.
    pos=rng.randrange(max(1,len(fillers)//2))
    fillers.insert(pos,f"The access code is {key}.")
    prefix="\n".join(fillers)
    ids=tokenizer(prefix,add_special_tokens=False).input_ids
    # Keep the early target; leave room for the question and chat wrapper.
    prefix=tokenizer.decode(ids[:max(8,length-70)],skip_special_tokens=True)
    if key not in prefix:
        prefix=f"The access code is {key}.\n"+prefix
    return prefix+"\nWhat is the access code? Reply with the code only.",key


def chat_ids(tokenizer,prompt):
    return tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_tensors="pt")


def load_data(tokenizer,a):
    from datasets import load_dataset
    tr=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    train=blocks(tokenizer,[x["text"] for x in tr],a.train_blocks,a.seq_len,a.seed+11)
    validation={length:blocks(tokenizer,[x["text"] for x in va],a.val_blocks,length,a.seed+777+length)
                for length in context_lengths(a)}
    return train,validation


def curriculum_window(cpu,step,cap,lengths,rng):
    # All lengths become available in the first half of EACH stage.
    available=min(len(lengths),1+int(2*step*len(lengths)/max(cap,1)))
    length=rng.choice(lengths[:available])
    start=rng.randrange(cpu.shape[1]-length)
    return cpu[:,start:start+length+1]


def _token_chunks(fn,*tensors,chunk_size=32):
    from torch.utils.checkpoint import checkpoint
    flattened=[t.reshape(-1,t.shape[-1]) for t in tensors]
    count=flattened[0].shape[0]
    total=flattened[0].new_zeros((),dtype=torch.float32)
    for start in range(0,count,chunk_size):
        chunk=[t[start:start+chunk_size] for t in flattened]
        value=checkpoint(fn,*chunk,use_reentrant=False) if torch.is_grad_enabled() else fn(*chunk)
        total=total+value*chunk[0].shape[0]/count
    return total


def causal_ce(logits,y):
    return _token_chunks(lambda s,t:F.cross_entropy(s.float(),t[:,0]),logits,y.unsqueeze(-1))


def forward_kl(student,teacher):
    def loss(s,t):
        ls=F.log_softmax(s.float(),-1); lt=F.log_softmax(t.float(),-1)
        return (lt.exp()*(lt-ls)).sum(-1).mean().clamp_min(0)
    return _token_chunks(loss,student,teacher)


def reverse_kl(student,teacher):
    def loss(s,t):
        ls=F.log_softmax(s.float(),-1); lt=F.log_softmax(t.float(),-1)
        return (ls.exp()*(ls-lt)).sum(-1).mean().clamp_min(0)
    return _token_chunks(loss,student,teacher)


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
        scale=t.std(-1,keepdim=True,correction=0).clamp_min(0.25)
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
    if alpha<=0:
        return {
            "min_top1":0.999,"max_kl":1e-4,"max_hidden_mse":1e-4,
            "max_mixer_mse":getattr(a,"max_mixer_mse",0.12),"max_mixer_cosine":0.07,"max_mixer_delta":0.22,
        }
    if alpha<1:
        return {
            "min_top1":max(getattr(a,"min_top1",0.97),0.995-0.02*alpha),
            "max_kl":getattr(a,"max_kl",0.03),
            "max_hidden_mse":getattr(a,"max_hidden_mse",0.05),
            "max_mixer_mse":getattr(a,"max_mixer_mse",0.12),"max_mixer_cosine":0.07,"max_mixer_delta":0.22,
        }
    return {
        "min_top1":getattr(a,"min_top1",0.97),"max_kl":getattr(a,"max_kl",0.03),"max_hidden_mse":getattr(a,"max_hidden_mse",0.05),
        "max_mixer_mse":getattr(a,"max_mixer_mse",0.12),"max_mixer_cosine":0.07,"max_mixer_delta":0.22,
    }


def stage_ok(m,alpha,a):
    t=stage_targets(alpha,a)
    if not all(math.isfinite(float(v)) for v in m.values()):
        return False
    return (
        m.get("generation_ok",1)==1 and m.get("ce_gap",0)<=getattr(a,"max_ce_gap",0.05) and m["top1"]>=t["min_top1"] and m["kl"]<=t["max_kl"]
        and m["hidden_mse"]<=t["max_hidden_mse"]
        and m["mixer_mse"]<=t["max_mixer_mse"]
        and m["mixer_cosine"]<=t["max_mixer_cosine"]
        and m["mixer_delta"]<=t["max_mixer_delta"]
    )


def stage_violation(m,alpha,a):
    t=stage_targets(alpha,a)
    if not all(math.isfinite(float(v)) for v in m.values()):
        return float("inf")
    return (
        (1-m.get("generation_ok",1))
        +max(m.get("ce_gap",0)-getattr(a,"max_ce_gap",0.05),0)/max(getattr(a,"max_ce_gap",0.05),1e-6)
        +max(t["min_top1"]-m["top1"],0)/max(1-t["min_top1"],1e-5)
        +max(m["kl"]-t["max_kl"],0)/max(t["max_kl"],1e-6)
        +max(m["hidden_mse"]-t["max_hidden_mse"],0)/max(t["max_hidden_mse"],1e-6)
        +max(m["mixer_mse"]-t["max_mixer_mse"],0)/max(t["max_mixer_mse"],1e-6)
        +max(m["mixer_cosine"]-t["max_mixer_cosine"],0)/max(t["max_mixer_cosine"],1e-6)
        +max(m["mixer_delta"]-t["max_mixer_delta"],0)/max(t["max_mixer_delta"],1e-6)
    )


def quality_key(m,alpha,a):
    if float(alpha)==0:
        return (stage_violation(m,alpha,a),m["mixer_mse"],m["mixer_cosine"],m["mixer_delta"])
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
    was_training=student.training
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
    student.train(was_training)
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
def generation_suite(student,teacher,tok,device,context_sizes=(128,),seed=9001):
    prompts=[
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "Explain in one sentence what an API is.",
        "Translate 'Good morning' into German. Give only the translation.",
        "The secret number is 81427. Remember it. What is the secret number? Give only the number.",
    ]
    cases=[(p,None) for p in prompts]
    cases[0]=(prompts[0],"42")
    cases[-1]=(prompts[-1],"81427")
    cases.extend(retrieval_prompt(tok,n,seed+i) for i,n in enumerate(context_sizes))
    rows=[]
    for p,expected in cases:
        chat=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(chat,return_tensors="pt").to(device)
        def run(m):
            reset_stream_state_v4(m)
            y=m.generate(**enc,max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
            z=y[0,enc.input_ids.shape[1]:]
            return z.tolist(),tok.decode(z,skip_special_tokens=True).strip()
        ti,tt=run(teacher); si,st=run(student)
        a,b=set(ti),set(si)
        rows.append({"prompt":p,"qwen":tt,"cenn":st,"exact":ti==si,"jaccard":len(a&b)/max(len(a|b),1),
                     "expected":expected,"teacher_correct":tt.strip()==expected,
                     "student_correct":st.strip()==expected,"repetition":repetition_rate(si),
                     "prompt_tokens":int(enc.input_ids.shape[1])})
    return rows


def probe_contexts(student,teacher,validation,layers,device,local_teacher=True):
    per={str(n):probe(student,teacher,b,layers,device,local_teacher) for n,b in validation.items()}
    # Worst context drives acceptance, so short windows cannot hide a long-context failure.
    worst={k:((min if k=="top1" else max)(m[k] for m in per.values())
              if all(math.isfinite(m[k]) for m in per.values()) else float("nan"))
           for k in next(iter(per.values()))}
    return worst,per


def generation_health(rows):
    scored=[r for r in rows if r.get("expected") is not None and r.get("teacher_correct")]
    return bool(scored) and all(r["student_correct"] for r in scored) and all(
        bool(r["cenn"].strip()) and r["repetition"]<0.5 for r in rows)


def repetition_rate(ids,n=4):
    grams=[tuple(ids[i:i+n]) for i in range(max(0,len(ids)-n+1))]
    return 1-len(set(grams))/len(grams) if grams else 0.0
