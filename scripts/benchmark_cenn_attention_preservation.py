#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from tinycenn_lm.smollm2_amcenn import DEFAULT_SMOLLM2
from tinycenn_lm.smollm2_amcenn_v2 import AdaptivePositiveSoftmaxFeatures


def int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return out


def args_parser():
    p = argparse.ArgumentParser(description="CeNN vs Transformer attention preservation benchmark")
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--num-sequences", type=int, default=2)
    p.add_argument("--layers", default="0,6,14,18,20,23,29", help="comma list or all")
    p.add_argument("--feature-dims", type=int_list, default=[128,256,512,1024,2048,4096])
    p.add_argument("--query-samples", type=int, default=12)
    p.add_argument("--head-samples", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--feature-seed", type=int, default=2026)
    p.add_argument("--dataset-seed", type=int, default=9107)
    p.add_argument("--output-dir", default="result/cenn-attention-preservation")
    p.add_argument("--skip-weight-metrics", action="store_true")
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
    return out, weights, torch.logsumexp(scores, dim=-1)


def cenn_attention(q, k, v, groups, feature_dim, seed):
    fmap = AdaptivePositiveSoftmaxFeatures(
        q.shape[-1], feature_dim, seed=seed,
        antithetic=(feature_dim % 2 == 0), learnable_correction=False,
    ).to(q.device)
    pq, pk = fmap(q).float(), fmap(k).float()
    b, h, t, _ = q.shape
    kvh, d = k.shape[1], q.shape[-1]
    s = torch.zeros(b, kvh, feature_dim, d, device=q.device)
    z = torch.zeros(b, kvh, feature_dim, device=q.device)
    kv_index = torch.arange(h, device=q.device) // groups
    outs, logz = [], []
    for i in range(t):
        kf, vv = pk[:,:,i,:], v[:,:,i,:].float()
        s = s + kf.unsqueeze(-1) * vv.unsqueeze(-2)
        z = z + kf
        sh, zh = s.index_select(1, kv_index), z.index_select(1, kv_index)
        qf = pq[:,:,i,:]
        num = torch.einsum("bhf,bhfd->bhd", qf, sh)
        den = torch.einsum("bhf,bhf->bh", qf, zh).clamp_min(1e-12)
        outs.append(num / den.unsqueeze(-1))
        logz.append(den.log())
    return torch.stack(outs, 2), torch.stack(logz, 2), pq, pk


def pearson(a, b):
    x, y = a.float().reshape(-1), b.float().reshape(-1)
    x, y = x-x.mean(), y-y.mean()
    den = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float(((x*y).sum()/den.clamp_min(1e-12)).cpu())


def tensor_metrics(ref, app):
    ref, app = ref.float(), app.float()
    err = app-ref
    ae = err.abs().reshape(-1)
    return {
        "cosine": float(F.cosine_similarity(ref.reshape(-1, ref.shape[-1]), app.reshape(-1, app.shape[-1]), dim=-1).mean().cpu()),
        "nmse": float((err.square().mean()/ref.square().mean().clamp_min(1e-12)).cpu()),
        "mae": float(ae.mean().cpu()),
        "p95_abs": float(torch.quantile(ae, 0.95).cpu()),
        "max_abs": float(ae.max().cpu()),
        "pearson": pearson(ref, app),
    }


def weight_metrics(exact_w, pq, pk, groups, query_samples, head_samples, top_k):
    b,h,t,_ = exact_w.shape
    qs = torch.linspace(max(1,t//4), t-1, steps=min(query_samples, max(1,t-t//4)), device=exact_w.device).round().long().unique()
    hs = torch.linspace(0,h-1,steps=min(head_samples,h),device=exact_w.device).round().long().unique()
    kls, jss, ovs = [], [], []
    eps = 1e-9
    for head in hs.tolist():
        kv = head//groups
        ker = torch.einsum("bqf,bsf->bqs", pq[:,head,qs,:], pk[:,kv,:,:]).clamp_min(0)
        causal = torch.arange(t,device=ker.device).view(1,1,-1) <= qs.view(1,-1,1)
        ker = ker.masked_fill(~causal,0)
        app = ker/ker.sum(-1,keepdim=True).clamp_min(eps)
        ref = exact_w[:,head,qs,:].float()
        p,q = ref.clamp_min(eps), app.clamp_min(eps)
        kl = (p*(p.log()-q.log())).sum(-1)
        m = 0.5*(p+q)
        js = 0.5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1))
        kls += kl.cpu().reshape(-1).tolist(); jss += js.cpu().reshape(-1).tolist()
        for bi in range(b):
            for qi,pos in enumerate(qs.tolist()):
                n, kk = pos+1, min(top_k,pos+1)
                a = set(torch.topk(ref[bi,qi,:n],kk).indices.tolist())
                c = set(torch.topk(app[bi,qi,:n],kk).indices.tolist())
                ovs.append(len(a & c)/kk)
    return {"attention_kl":statistics.fmean(kls),"attention_js":statistics.fmean(jss),"topk_overlap":statistics.fmean(ovs)}


def memory_stats(f, context, kvh, d):
    kv = 2*context*kvh*d
    cenn = kvh*f*(d+1)
    return {
        "state_vs_kv_ratio": cenn/max(kv,1),
        "break_even_tokens": math.ceil(f*(d+1)/(2*d)),
        "cenn_state_mib_fp32": cenn*4/(1024**2),
    }


def avg(rows, key):
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return statistics.fmean(vals) if vals else None


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    args = args_parser()
    outdir = Path(args.output_dir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device).eval()
    model.config.use_cache = False
    n_layers = int(model.config.num_hidden_layers)
    layers = list(range(n_layers)) if args.layers.lower()=="all" else int_list(args.layers)
    if any(i<0 or i>=n_layers for i in layers): raise ValueError(f"invalid layer list for {n_layers} layers")

    data = load_dataset(args.dataset,name=args.dataset_config,split=args.split,streaming=True).shuffle(seed=args.dataset_seed,buffer_size=2048)
    blocks = token_blocks(data,tok,args.context_length)
    captured = {}
    hooks=[]
    for idx in layers:
        def make_hook(i):
            def hook(_m,_a,o):
                captured[i]=(o[0] if isinstance(o,tuple) else o).detach().float()
            return hook
        hooks.append(model.model.layers[idx].self_attn.register_forward_hook(make_hook(idx)))

    rows=[]; recon=[]; started=time.time()
    try:
        for seq in range(args.num_sequences):
            ids=next(blocks).unsqueeze(0).to(device); captured.clear()
            with torch.no_grad():
                out=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
            t=ids.shape[1]; pos=torch.arange(t,device=device).unsqueeze(0)
            for idx in layers:
                layer=model.model.layers[idx]; h0=out.hidden_states[idx]; x=layer.input_layernorm(h0)
                pe=model.model.rotary_emb(x,pos); attn=layer.self_attn; b=x.shape[0]
                q=attn.q_proj(x).view(b,t,model.config.num_attention_heads,-1).transpose(1,2)
                k=attn.k_proj(x).view(b,t,model.config.num_key_value_heads,-1).transpose(1,2)
                v=attn.v_proj(x).view(b,t,model.config.num_key_value_heads,-1).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,*pe)
                groups=model.config.num_attention_heads//model.config.num_key_value_heads
                with torch.no_grad():
                    exact,ew,elogz=exact_attention(q,k,v,groups)
                    ef=exact.transpose(1,2).contiguous().view(b,t,model.config.hidden_size)
                    eo=attn.o_proj(ef.to(x.dtype)).float()
                    rm=tensor_metrics(captured[idx],eo)
                    recon.append({"sequence":seq,"layer":idx,"cosine":rm["cosine"],"nmse":rm["nmse"]})
                    for fd in args.feature_dims:
                        ca,clogz,pq,pk=cenn_attention(q,k,v,groups,fd,args.feature_seed+1009*idx)
                        cf=ca.transpose(1,2).contiguous().view(b,t,model.config.hidden_size)
                        co=attn.o_proj(cf.to(x.dtype)).float()
                        m=tensor_metrics(exact,ca); om=tensor_metrics(eo,co); res=tensor_metrics(h0.float()+eo,h0.float()+co)
                        row={"sequence":seq,"layer":idx,"feature_dim":fd,
                             "output_cosine":m["cosine"],"output_nmse":m["nmse"],"output_mae":m["mae"],
                             "output_p95_abs":m["p95_abs"],"output_max_abs":m["max_abs"],"output_pearson":m["pearson"],
                             "o_projection_cosine":om["cosine"],"o_projection_nmse":om["nmse"],
                             "residual_cosine":res["cosine"],"residual_nmse":res["nmse"],
                             "partition_log_mae":float((clogz-elogz.float()).abs().mean().cpu()),
                             "partition_ratio_median":float(torch.exp((clogz-elogz.float()).clamp(-20,20)).median().cpu())}
                        if not args.skip_weight_metrics:
                            row.update(weight_metrics(ew,pq,pk,groups,args.query_samples,args.head_samples,args.top_k))
                        row.update(memory_stats(fd,args.context_length,model.config.num_key_value_heads,q.shape[-1]))
                        rows.append(row)
                        print(f"seq={seq+1}/{args.num_sequences} layer={idx:02d} F={fd:4d} cos={row['output_cosine']:.5f} nmse={row['output_nmse']:.5f} logZ={row['partition_log_mae']:.4f}")
                        del ca,clogz,pq,pk,cf,co
                if device.type=="cuda": torch.cuda.empty_cache()
    finally:
        for h in hooks: h.remove()

    metric_keys=["output_cosine","output_nmse","output_mae","output_p95_abs","output_max_abs","output_pearson","o_projection_cosine","o_projection_nmse","residual_cosine","residual_nmse","partition_log_mae","partition_ratio_median","attention_kl","attention_js","topk_overlap"]
    by_feature=[]; by_layer=[]
    for fd in args.feature_dims:
        sub=[r for r in rows if r["feature_dim"]==fd]
        s={"feature_dim":fd, **{k:avg(sub,k) for k in metric_keys if avg(sub,k) is not None}}
        if sub:
            s.update({k:sub[0][k] for k in ["state_vs_kv_ratio","break_even_tokens","cenn_state_mib_fp32"]})
        by_feature.append(s)
    for idx in layers:
        for fd in args.feature_dims:
            sub=[r for r in rows if r["layer"]==idx and r["feature_dim"]==fd]
            by_layer.append({"layer":idx,"feature_dim":fd,**{k:avg(sub,k) for k in metric_keys if avg(sub,k) is not None}})

    candidates=[r for r in by_feature if r.get("output_cosine",0)>=0.99 and r.get("output_nmse",9e9)<=0.02]
    report={
        "status":"PASS","experiment":"cenn-attention-preservation","base_model":args.base_model,
        "context_length":args.context_length,"num_sequences":args.num_sequences,"layers":layers,"feature_dims":args.feature_dims,
        "teacher_reconstruction_sanity":{"cosine":avg(recon,"cosine"),"nmse":avg(recon,"nmse")},
        "summary_by_feature_dim":by_feature,"summary_by_layer_feature":by_layer,
        "recommended_min_feature_dim_for_0_99_cos_and_0_02_nmse":min((r["feature_dim"] for r in candidates),default=None),
        "transformer_element_equivalence":{
            "Q_projection":"identical pretrained weights","K_projection":"identical pretrained weights","V_projection":"identical pretrained weights",
            "RoPE":"identical teacher rotary embedding","attention_kernel":"CeNN approximation under test",
            "attention_normalization":"exact log partition vs phi(q)^T z measured","O_projection":"identical pretrained weights",
            "residual":"same path, measured after attention addition","RMSNorm":"identical pretrained RMSNorm","MLP":"unchanged / not tested here"},
        "elapsed_minutes":(time.time()-started)/60.0}
    (outdir/"attention_preservation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    write_csv(outdir/"attention_preservation_raw.csv",rows); write_csv(outdir/"attention_preservation_by_feature.csv",by_feature); write_csv(outdir/"attention_preservation_by_layer.csv",by_layer)
    print("\n=== summary ===")
    for r in by_feature:
        print(f"F={r['feature_dim']:4d} cos={r.get('output_cosine',float('nan')):.5f} nmse={r.get('output_nmse',float('nan')):.5f} logZ={r.get('partition_log_mae',float('nan')):.4f} break-even≈{r.get('break_even_tokens',0)} tokens")
    print("saved:",outdir)


if __name__ == "__main__":
    main()
