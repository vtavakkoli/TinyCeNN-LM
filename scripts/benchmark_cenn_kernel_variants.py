#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from tinycenn_lm.smollm2_amcenn import DEFAULT_SMOLLM2


RANDOM_VARIANTS = ("current_prf", "stable_prf", "stable_orf")
FIXED_VARIANTS = ("taylor2", "elu1")
ALL_VARIANTS = RANDOM_VARIANTS + FIXED_VARIANTS


def int_list(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def str_list(value: str) -> list[str]:
    values = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in values if x not in ALL_VARIANTS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown variants: {bad}; choose from {ALL_VARIANTS}")
    return values


def parse_args():
    p = argparse.ArgumentParser(description="Compare CeNN/linear-attention kernel variants against frozen SmolLM2 attention")
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--num-sequences", type=int, default=2)
    p.add_argument("--layers", default="0,6,14,18,20,23,29", help="comma list or all")
    p.add_argument("--variants", type=str_list, default=list(ALL_VARIANTS))
    p.add_argument("--feature-dims", type=int_list, default=[256,512,1024,2048])
    p.add_argument("--feature-seed", type=int, default=2026)
    p.add_argument("--dataset-seed", type=int, default=9107)
    p.add_argument("--query-samples", type=int, default=10)
    p.add_argument("--head-samples", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--gradient-check", action="store_true")
    p.add_argument("--gradient-layers", default="0,18,29")
    p.add_argument("--gradient-feature-dim", type=int, default=1024)
    p.add_argument("--skip-weight-metrics", action="store_true")
    p.add_argument("--output-dir", default="result/cenn-kernel-variants")
    return p.parse_args()


def choose_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def token_blocks(dataset, tokenizer, context_length):
    buf, eos = [], tokenizer.eos_token_id
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= context_length:
            yield torch.tensor(buf[:context_length], dtype=torch.long)
            del buf[:context_length]


def exact_attention(q, k, v, groups):
    kh = k.float().repeat_interleave(groups, dim=1)
    vh = v.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhtd,bhsd->bhts", q.float(), kh) / math.sqrt(q.shape[-1])
    t = q.shape[-2]
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal.view(1,1,t,t), float("-inf"))
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.einsum("bhts,bhsd->bhtd", weights, vh)
    return out, weights, torch.logsumexp(scores, dim=-1), scores


def antithetic_projection(feature_dim, head_dim, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    half = feature_dim // 2
    p = torch.randn(half, head_dim, generator=g)
    if feature_dim % 2 == 0:
        p = torch.cat([p, -p], dim=0)
    else:
        p = torch.cat([p, -p, torch.randn(1, head_dim, generator=g)], dim=0)[:feature_dim]
    return p.to(device=device, dtype=torch.float32)


def orthogonal_gaussian_projection(feature_dim, head_dim, seed, device):
    """Block-orthogonal Gaussian-like rows (ORF-style variance reduction)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks = []
    remaining = feature_dim
    while remaining > 0:
        a = torch.randn(head_dim, head_dim, generator=g)
        q, r = torch.linalg.qr(a)
        signs = torch.sign(torch.diag(r)).clamp(min=-1, max=1)
        signs[signs == 0] = 1
        q = q * signs.unsqueeze(0)
        radii = torch.randn(head_dim, head_dim, generator=g).norm(dim=1)
        block = q.T * radii.unsqueeze(1)
        take = min(remaining, head_dim)
        blocks.append(block[:take])
        remaining -= take
    return torch.cat(blocks, dim=0).to(device=device, dtype=torch.float32)


def exp_logits(x, projection):
    m = projection.shape[0]
    work = x.float() * (x.shape[-1] ** -0.25)
    projected = torch.einsum("...d,fd->...f", work, projection)
    norm = 0.5 * work.square().sum(dim=-1, keepdim=True)
    return projected - norm - 0.5 * math.log(float(m))


@dataclass
class FeatureRepresentation:
    q_phi: torch.Tensor
    k_phi: torch.Tensor
    q_log_scale: torch.Tensor
    k_log_scale: torch.Tensor


def current_prf_attention(q, k, v, groups, projection):
    aq, ak = exp_logits(q, projection), exp_logits(k, projection)
    pq = torch.exp(aq.clamp(-20.0, 20.0))
    pk = torch.exp(ak.clamp(-20.0, 20.0))
    values = v.transpose(1,2).float()
    pk_t = pk.transpose(1,2)
    pq_t = pq.transpose(1,2)
    writes = torch.einsum("btkf,btkd->btkfd", pk_t, values)
    s = writes.cumsum(1).repeat_interleave(groups, dim=2)
    z = pk_t.cumsum(1).repeat_interleave(groups, dim=2)
    num = torch.einsum("bthf,bthfd->bthd", pq_t, s)
    den = torch.einsum("bthf,bthf->bth", pq_t, z)
    out = (num / (den.unsqueeze(-1) + 1e-12)).transpose(1,2)
    rep = FeatureRepresentation(pq, pk, torch.zeros_like(aq[...,0]), torch.zeros_like(ak[...,0]))
    diag = {
        "upper_clip_fraction": float((torch.cat([aq.reshape(-1),ak.reshape(-1)]) > 20).float().mean().detach().cpu()),
        "lower_clip_fraction": float((torch.cat([aq.reshape(-1),ak.reshape(-1)]) < -20).float().mean().detach().cpu()),
    }
    return out, den.clamp_min(1e-30).log().transpose(1,2), rep, projection.shape[0], diag


def stable_prf_attention(q, k, v, groups, projection):
    """Exact rescaling of the recurrent PRF state; no hard exponent clipping."""
    aq, ak = exp_logits(q, projection), exp_logits(k, projection)
    q_shift = aq.amax(-1).detach()
    k_shift = ak.amax(-1).detach()
    pq = torch.exp(aq - q_shift.unsqueeze(-1))
    pk = torch.exp(ak - k_shift.unsqueeze(-1))
    b, h, t, f = pq.shape
    kvh, d = pk.shape[1], v.shape[-1]
    s = torch.zeros(b, kvh, f, d, device=q.device, dtype=torch.float32)
    z = torch.zeros(b, kvh, f, device=q.device, dtype=torch.float32)
    running = torch.full((b,kvh), -float("inf"), device=q.device)
    kv_index = torch.arange(h, device=q.device) // groups
    outs, logz = [], []
    for i in range(t):
        ks = k_shift[:,:,i]
        new_running = torch.maximum(running, ks).detach()
        old_scale = torch.where(torch.isfinite(running), torch.exp(running-new_running), torch.zeros_like(running))
        write_scale = torch.exp(ks-new_running)
        kfi = pk[:,:,i,:] * write_scale.unsqueeze(-1)
        s = s * old_scale.unsqueeze(-1).unsqueeze(-1) + kfi.unsqueeze(-1) * v[:,:,i,:].float().unsqueeze(-2)
        z = z * old_scale.unsqueeze(-1) + kfi
        sh, zh = s.index_select(1,kv_index), z.index_select(1,kv_index)
        qfi = pq[:,:,i,:]
        num = torch.einsum("bhf,bhfd->bhd",qfi,sh)
        den = torch.einsum("bhf,bhf->bh",qfi,zh)
        outs.append(num/(den.unsqueeze(-1)+1e-12))
        mh = new_running.index_select(1,kv_index)
        logz.append(den.clamp_min(1e-30).log()+q_shift[:,:,i]+mh)
        running = new_running
    rep = FeatureRepresentation(pq,pk,q_shift,k_shift)
    diag = {
        "upper_clip_fraction": 0.0,
        "lower_clip_fraction": 0.0,
        "raw_logit_max": float(torch.maximum(aq.max(),ak.max()).detach().cpu()),
        "raw_logit_min": float(torch.minimum(aq.min(),ak.min()).detach().cpu()),
    }
    return torch.stack(outs,2), torch.stack(logz,2), rep, f, diag


def taylor2_features(x):
    """Feature map whose dot product is 1+s+s^2/2, s=q^T k/sqrt(d)."""
    u = x.float() * (x.shape[-1] ** -0.25)
    d = u.shape[-1]
    i,j = torch.triu_indices(d,d,offset=1,device=x.device)
    linear = u
    diag = u.square() / math.sqrt(2.0)
    off = u[...,i] * u[...,j]
    one = torch.ones(*u.shape[:-1],1,device=u.device,dtype=u.dtype)
    return torch.cat([one,linear,diag,off],dim=-1)


def linear_attention(q, k, v, groups, variant):
    if variant == "taylor2":
        pq, pk = taylor2_features(q), taylor2_features(k)
    elif variant == "elu1":
        scale = q.shape[-1] ** -0.25
        pq = F.elu(q.float()*scale)+1.0
        pk = F.elu(k.float()*scale)+1.0
    else:
        raise ValueError(variant)
    pk_t, pq_t = pk.transpose(1,2), pq.transpose(1,2)
    values = v.transpose(1,2).float()
    writes = torch.einsum("btkf,btkd->btkfd",pk_t,values)
    s = writes.cumsum(1).repeat_interleave(groups,dim=2)
    z = pk_t.cumsum(1).repeat_interleave(groups,dim=2)
    num = torch.einsum("bthf,bthfd->bthd",pq_t,s)
    den = torch.einsum("bthf,bthf->bth",pq_t,z)
    out = (num/(den.unsqueeze(-1)+1e-12)).transpose(1,2)
    rep = FeatureRepresentation(pq,pk,torch.zeros_like(pq[...,0]),torch.zeros_like(pk[...,0]))
    return out, den.clamp_min(1e-30).log().transpose(1,2), rep, pq.shape[-1], {"upper_clip_fraction":0.0,"lower_clip_fraction":0.0}


def approximate_attention(variant,q,k,v,groups,feature_dim,seed,projection_cache=None):
    if variant in RANDOM_VARIANTS:
        key=(variant,feature_dim,q.shape[-1],seed,str(q.device))
        projection = None if projection_cache is None else projection_cache.get(key)
        if projection is None:
            if variant == "stable_orf":
                projection=orthogonal_gaussian_projection(feature_dim,q.shape[-1],seed,q.device)
            else:
                projection=antithetic_projection(feature_dim,q.shape[-1],seed,q.device)
            if projection_cache is not None: projection_cache[key]=projection
        if variant == "current_prf":
            return current_prf_attention(q,k,v,groups,projection)
        return stable_prf_attention(q,k,v,groups,projection)
    return linear_attention(q,k,v,groups,variant)


def pearson(a,b):
    x,y=a.float().reshape(-1),b.float().reshape(-1)
    x,y=x-x.mean(),y-y.mean()
    den=x.square().sum().sqrt()*y.square().sum().sqrt()
    return float(((x*y).sum()/den.clamp_min(1e-12)).detach().cpu())


def tensor_metrics(ref,app):
    ref,app=ref.float(),app.float(); err=app-ref; ae=err.abs().reshape(-1)
    return {
        "cosine":float(F.cosine_similarity(ref.reshape(-1,ref.shape[-1]),app.reshape(-1,app.shape[-1]),dim=-1).mean().detach().cpu()),
        "nmse":float((err.square().mean()/ref.square().mean().clamp_min(1e-12)).detach().cpu()),
        "mae":float(ae.mean().detach().cpu()),
        "pearson":pearson(ref,app),
    }


def weight_metrics(exact_w, rep, groups, query_samples, head_samples, top_k):
    b,h,t,_=exact_w.shape
    qs=torch.linspace(max(1,t//4),t-1,steps=min(query_samples,max(1,t-t//4)),device=exact_w.device).round().long().unique()
    hs=torch.linspace(0,h-1,steps=min(head_samples,h),device=exact_w.device).round().long().unique()
    kls,jss,ovs=[],[],[]; eps=1e-9
    for head in hs.tolist():
        kv=head//groups
        qphi=rep.q_phi[:,head,qs,:]
        kphi=rep.k_phi[:,kv,:,:]
        ker=torch.einsum("bqf,bsf->bqs",qphi,kphi).clamp_min(1e-30)
        logker=ker.log()+rep.q_log_scale[:,head,qs].unsqueeze(-1)+rep.k_log_scale[:,kv,:].unsqueeze(1)
        causal=torch.arange(t,device=logker.device).view(1,1,-1)<=qs.view(1,-1,1)
        logker=logker.masked_fill(~causal,float("-inf"))
        app=torch.softmax(logker,dim=-1)
        ref=exact_w[:,head,qs,:].float()
        p,q=ref.clamp_min(eps),app.clamp_min(eps)
        kl=(p*(p.log()-q.log())).sum(-1); m=.5*(p+q)
        js=.5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1))
        kls += kl.detach().cpu().reshape(-1).tolist(); jss += js.detach().cpu().reshape(-1).tolist()
        for bi in range(b):
            for qi,pos in enumerate(qs.tolist()):
                kk=min(top_k,pos+1); n=pos+1
                a=set(torch.topk(ref[bi,qi,:n],kk).indices.tolist())
                c=set(torch.topk(app[bi,qi,:n],kk).indices.tolist())
                ovs.append(len(a&c)/kk)
    return {"attention_kl":statistics.fmean(kls),"attention_js":statistics.fmean(jss),"topk_overlap":statistics.fmean(ovs)}


def memory_stats(feature_dim,context,kvh,d):
    kv=2*context*kvh*d; cenn=kvh*feature_dim*(d+1)
    return {
        "state_vs_kv_ratio":cenn/max(kv,1),
        "break_even_tokens":math.ceil(feature_dim*(d+1)/(2*d)),
        "cenn_state_mib_fp32":cenn*4/(1024**2),
    }


def score_distribution(scores,q,k):
    finite=scores[torch.isfinite(scores)].detach().float()
    qn=q.detach().float().norm(dim=-1).reshape(-1); kn=k.detach().float().norm(dim=-1).reshape(-1)
    return {
        "score_p01":float(torch.quantile(finite,.01).cpu()),"score_p50":float(torch.quantile(finite,.5).cpu()),
        "score_p99":float(torch.quantile(finite,.99).cpu()),"score_min":float(finite.min().cpu()),"score_max":float(finite.max().cpu()),
        "q_norm_mean":float(qn.mean().cpu()),"k_norm_mean":float(kn.mean().cpu()),
    }


def grad_pair_metrics(ref,app):
    r,a=ref.float().reshape(-1),app.float().reshape(-1)
    return {
        "cosine":float(F.cosine_similarity(r.unsqueeze(0),a.unsqueeze(0)).detach().cpu()),
        "nmse":float(((a-r).square().mean()/r.square().mean().clamp_min(1e-12)).detach().cpu()),
    }


def gradient_fidelity(variant,q,k,v,groups,feature_dim,seed,projection_cache):
    gen=torch.Generator(device="cpu").manual_seed(12345)
    probe=torch.randn(q.shape,generator=gen).to(q.device)
    qr=q.detach().float().requires_grad_(True); kr=k.detach().float().requires_grad_(True); vr=v.detach().float().requires_grad_(True)
    exact,_,_,_=exact_attention(qr,kr,vr,groups)
    loss=(exact*probe).mean(); eg=torch.autograd.grad(loss,(qr,kr,vr))
    qa=q.detach().float().requires_grad_(True); ka=k.detach().float().requires_grad_(True); va=v.detach().float().requires_grad_(True)
    app,_,_,_,_=approximate_attention(variant,qa,ka,va,groups,feature_dim,seed,projection_cache)
    loss2=(app*probe).mean(); ag=torch.autograd.grad(loss2,(qa,ka,va))
    names=("q","k","v"); row={}
    cos=[]; nm=[]
    for name,r,a in zip(names,eg,ag):
        m=grad_pair_metrics(r,a); row[f"grad_{name}_cosine"]=m["cosine"]; row[f"grad_{name}_nmse"]=m["nmse"]
        cos.append(m["cosine"]); nm.append(m["nmse"])
    row["grad_mean_cosine"]=statistics.fmean(cos); row["grad_mean_nmse"]=statistics.fmean(nm)
    return row


def avg(rows,key):
    vals=[float(r[key]) for r in rows if key in r and r[key] is not None and math.isfinite(float(r[key]))]
    return statistics.fmean(vals) if vals else None


def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def settings(variants,feature_dims):
    out=[]
    for variant in variants:
        if variant in RANDOM_VARIANTS:
            out.extend((variant,f) for f in feature_dims)
        else:
            out.append((variant,0))
    return out


def main():
    args=parse_args(); outdir=Path(args.output_dir).resolve(); outdir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype=choose_dtype(device)
    print(f"device={device} dtype={dtype}")
    tok=AutoTokenizer.from_pretrained(args.base_model,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.base_model,dtype=dtype).to(device).eval(); model.config.use_cache=False
    n_layers=int(model.config.num_hidden_layers); layers=list(range(n_layers)) if args.layers.lower()=="all" else int_list(args.layers)
    grad_layers=set(int_list(args.gradient_layers)); grad_layers &= set(layers)
    data=load_dataset(args.dataset,name=args.dataset_config,split=args.split,streaming=True).shuffle(seed=args.dataset_seed,buffer_size=2048)
    blocks=token_blocks(data,tok,args.context_length)
    captured={}; hooks=[]; projection_cache={}
    for idx in layers:
        def make_hook(i):
            def hook(_m,_a,o): captured[i]=(o[0] if isinstance(o,tuple) else o).detach().float()
            return hook
        hooks.append(model.model.layers[idx].self_attn.register_forward_hook(make_hook(idx)))
    rows=[]; grad_rows=[]; recon=[]; score_rows=[]; started=time.time()
    try:
        for seq in range(args.num_sequences):
            ids=next(blocks).unsqueeze(0).to(device); captured.clear()
            with torch.no_grad(): out=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
            t=ids.shape[1]; pos=torch.arange(t,device=device).unsqueeze(0)
            for idx in layers:
                layer=model.model.layers[idx]; h0=out.hidden_states[idx]; x=layer.input_layernorm(h0); pe=model.model.rotary_emb(x,pos); attn=layer.self_attn; b=x.shape[0]
                q=attn.q_proj(x).view(b,t,model.config.num_attention_heads,-1).transpose(1,2)
                k=attn.k_proj(x).view(b,t,model.config.num_key_value_heads,-1).transpose(1,2)
                v=attn.v_proj(x).view(b,t,model.config.num_key_value_heads,-1).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,*pe); groups=model.config.num_attention_heads//model.config.num_key_value_heads
                with torch.no_grad():
                    exact,ew,elogz,scores=exact_attention(q,k,v,groups)
                    ef=exact.transpose(1,2).contiguous().view(b,t,model.config.hidden_size); eo=attn.o_proj(ef.to(x.dtype)).float()
                    rm=tensor_metrics(captured[idx],eo); recon.append({"sequence":seq,"layer":idx,"cosine":rm["cosine"],"nmse":rm["nmse"]})
                    sr=score_distribution(scores,q,k); sr.update({"sequence":seq,"layer":idx}); score_rows.append(sr)
                for variant,fd in settings(args.variants,args.feature_dims):
                    if device.type=="cuda": torch.cuda.synchronize()
                    tic=time.perf_counter()
                    with torch.no_grad(): app,alogz,rep,eff_f,diag=approximate_attention(variant,q,k,v,groups,fd,args.feature_seed+1009*idx,projection_cache)
                    if device.type=="cuda": torch.cuda.synchronize()
                    runtime_ms=(time.perf_counter()-tic)*1000.0
                    af=app.transpose(1,2).contiguous().view(b,t,model.config.hidden_size); ao=attn.o_proj(af.to(x.dtype)).float()
                    m=tensor_metrics(exact,app); om=tensor_metrics(eo,ao); res=tensor_metrics(h0.float()+eo,h0.float()+ao)
                    row={"sequence":seq,"layer":idx,"variant":variant,"requested_feature_dim":fd,"effective_feature_dim":eff_f,
                         "output_cosine":m["cosine"],"output_nmse":m["nmse"],"output_mae":m["mae"],"output_pearson":m["pearson"],
                         "o_projection_cosine":om["cosine"],"o_projection_nmse":om["nmse"],"residual_cosine":res["cosine"],"residual_nmse":res["nmse"],
                         "partition_log_mae":float((alogz-elogz.float()).abs().mean().cpu()),"partition_ratio_median":float(torch.exp((alogz-elogz.float()).clamp(-30,30)).median().cpu()),
                         "runtime_ms":runtime_ms,**diag,**memory_stats(eff_f,args.context_length,model.config.num_key_value_heads,q.shape[-1])}
                    if not args.skip_weight_metrics: row.update(weight_metrics(ew,rep,groups,args.query_samples,args.head_samples,args.top_k))
                    rows.append(row)
                    print(f"seq={seq+1}/{args.num_sequences} layer={idx:02d} {variant:12s} F={eff_f:4d} cos={row['output_cosine']:.4f} nmse={row['output_nmse']:.4f} logZ={row['partition_log_mae']:.3f}")
                    if args.gradient_check and seq==0 and idx in grad_layers and (variant not in RANDOM_VARIANTS or fd==args.gradient_feature_dim):
                        gm=gradient_fidelity(variant,q,k,v,groups,fd,args.feature_seed+1009*idx,projection_cache)
                        gm.update({"layer":idx,"variant":variant,"requested_feature_dim":fd,"effective_feature_dim":eff_f}); grad_rows.append(gm)
                    del app,rep,af,ao
                    if device.type=="cuda": torch.cuda.empty_cache()
    finally:
        for h in hooks: h.remove()

    metric_keys=["output_cosine","output_nmse","output_pearson","o_projection_cosine","o_projection_nmse","residual_cosine","residual_nmse","partition_log_mae","partition_ratio_median","attention_kl","attention_js","topk_overlap","runtime_ms","state_vs_kv_ratio","break_even_tokens","cenn_state_mib_fp32","upper_clip_fraction","lower_clip_fraction"]
    summary=[]
    for variant,fd in settings(args.variants,args.feature_dims):
        sub=[r for r in rows if r["variant"]==variant and r["requested_feature_dim"]==fd]
        if not sub: continue
        s={"variant":variant,"requested_feature_dim":fd,"effective_feature_dim":sub[0]["effective_feature_dim"]}
        s.update({k:avg(sub,k) for k in metric_keys if avg(sub,k) is not None})
        gs=[g for g in grad_rows if g["variant"]==variant and g["requested_feature_dim"]==fd]
        for k in ("grad_mean_cosine","grad_mean_nmse","grad_q_cosine","grad_k_cosine","grad_v_cosine"):
            if avg(gs,k) is not None: s[k]=avg(gs,k)
        gc=s.get("grad_mean_cosine",0.5)
        s["heuristic_selection_score"]=(0.30*s.get("output_cosine",0)+0.20/(1+s.get("output_nmse",99))+0.15/(1+s.get("partition_log_mae",99))+0.10/(1+s.get("attention_kl",99))+0.25*max(min(gc,1.0),-1.0))
        summary.append(s)
    summary.sort(key=lambda r:r["heuristic_selection_score"],reverse=True)
    best=summary[0] if summary else None
    strict=[r for r in summary if r.get("output_cosine",0)>=.90 and r.get("output_nmse",99)<=.10 and r.get("partition_log_mae",99)<=.5 and r.get("grad_mean_cosine",1)>=.80]
    report={"status":"PASS","experiment":"cenn-kernel-variants","base_model":args.base_model,"context_length":args.context_length,"num_sequences":args.num_sequences,"layers":layers,"variants":args.variants,"feature_dims":args.feature_dims,
            "teacher_reconstruction_sanity":{"cosine":avg(recon,"cosine"),"nmse":avg(recon,"nmse")},"score_distribution_summary":{k:avg(score_rows,k) for k in score_rows[0] if k not in ("sequence","layer")} if score_rows else {},
            "summary":summary,"strict_candidates":strict,"best_by_heuristic":best,"gradient_checked":bool(args.gradient_check),
            "heuristic_note":"Selection score is only a ranking aid; forward fidelity, partition fidelity, gradient fidelity, state size and runtime should be inspected separately.",
            "elapsed_minutes":(time.time()-started)/60.0}
    write_csv(outdir/"kernel_variant_rows.csv",rows); write_csv(outdir/"kernel_variant_summary.csv",summary)
    if grad_rows: write_csv(outdir/"kernel_variant_gradients.csv",grad_rows)
    write_csv(outdir/"score_distribution.csv",score_rows)
    (outdir/"kernel_variant_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
