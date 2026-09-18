#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.smollm2_flyffn_v2 import (
    FlyFFNV2Config,
    anchor_layer_indices,
    assert_flyffn_v2_replacement,
    flyffn_v2_modules,
    flyffn_v2_parameter_groups,
    flyffn_v2_router_regularizer,
    flyffn_v2_stats,
    replace_ffns_with_fly_v2,
    routing_schedule,
    set_route_state,
)

_V1_PATH = Path(__file__).with_name("run_smollm2_flyffn.py")
_spec = importlib.util.spec_from_file_location("flyffn_v1_runner", _V1_PATH)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

BASE_MODEL = base.BASE_MODEL


def parse_args():
    p = argparse.ArgumentParser(description="FlyFFN v2 progressive SmolLM2 FFN experiment")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--router-rank", type=int, default=64)
    p.add_argument("--max-edges", type=int, default=2048)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix-init", type=float, default=0.50)
    p.add_argument("--anchor-every", type=int, default=4)
    p.add_argument("--max-ce-gap", type=float, default=None)
    p.add_argument("--rewired", action="store_true")
    p.add_argument("--output-dir", default="results/flyffn_v2_smollm2_135m")
    p.add_argument("--seed", type=int, default=4321)
    return p.parse_args()


def stage_schedule(num_shards: int):
    if num_shards == 8:
        return [(6, 0.10), (4, 0.25), (4, 0.50), (3, 0.70), (2, 1.00)]
    ks = sorted(set(max(1, round(num_shards * f)) for f in (0.75, 0.50, 0.375, 0.25)), reverse=True)
    return list(zip(ks, np.linspace(0.15, 1.0, len(ks)).tolist()))


def build_student(dtype, device, cfg, adjacency, seed):
    base.set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype if device.type == "cuda" else torch.float32
    ).to(device)
    replace_ffns_with_fly_v2(model, cfg, adjacency)
    assert_flyffn_v2_replacement(model, cfg.anchor_every)
    return model


def snapshot_group(model, layer_group):
    prefixes = tuple(f"model.layers.{i}.mlp." for i in layer_group)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(prefixes)}


def restore_group(model, state):
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def capture_student_inputs(student, x, layer_indices, device, dtype):
    captures, handles = {}, []
    for idx in layer_indices:
        module = student.model.layers[idx].mlp
        def hook(_module, args, idx=idx):
            captures[idx] = args[0].detach()
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with base.amp_ctx(device, dtype):
            student(input_ids=x, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return captures


@torch.no_grad()
def probe_ce(student, teacher, batches, device, dtype):
    student.eval(); teacher.eval()
    tce, sce = [], []
    for cpu_ids in batches:
        ids = cpu_ids.to(device); x = ids[:, :-1]
        with base.amp_ctx(device, dtype):
            t = teacher(input_ids=x, use_cache=False, return_dict=True)
            s = student(input_ids=x, use_cache=False, return_dict=True)
        tce.append(float(base.causal_ce(t.logits, ids)))
        sce.append(float(base.causal_ce(s.logits, ids)))
    return float(np.mean(tce)), float(np.mean(sce))


def calibrate_progressively(name, student, teacher, calib_batches, probe_batches, device, dtype,
                            steps_per_stage, group_size, max_ce_gap, num_shards):
    layers = [m.layer_idx for m in flyffn_v2_modules(student)]
    groups = [layers[i:i+group_size] for i in range(0, len(layers), group_size)]
    stages = stage_schedule(num_shards)
    history, cursor = [], 0
    set_route_state(student, active_k=num_shards, route_mix=0.0)

    for gi, layer_group in enumerate(groups, 1):
        for si, (k, mix) in enumerate(stages, 1):
            before = snapshot_group(student, layer_group)
            set_route_state(student, layer_group, active_k=k, route_mix=mix)
            param_groups, trainable = flyffn_v2_parameter_groups(
                student, layer_group, router_lr=7e-4, shard_lr=1e-5, weight_decay=0.0
            )
            opt = base.make_optimizer(param_groups, device)
            scaler = base.make_scaler(device, dtype)
            student.train()

            for step in range(1, steps_per_stage + 1):
                ids = calib_batches[cursor % len(calib_batches)].to(device); cursor += 1
                x = ids[:, :-1]
                teacher_caps = base.capture_teacher_mlp_io(
                    teacher, x, layer_group, lambda: base.amp_ctx(device, dtype)
                )
                student_inputs = capture_student_inputs(student, x, layer_group, device, dtype)
                opt.zero_grad(set_to_none=True)
                losses = []
                with base.amp_ctx(device, dtype):
                    for idx in layer_group:
                        mlp = student.model.layers[idx].mlp
                        p_teacher = mlp(teacher_caps[idx]["hidden"])
                        p_student = mlp(student_inputs[idx])
                        losses.append(base.mlp_alignment_loss(p_teacher, teacher_caps[idx]["target"]))
                        losses.append(base.mlp_alignment_loss(p_student, teacher_caps[idx]["target"]))
                    align = torch.stack(losses).mean()
                    reg = flyffn_v2_router_regularizer(student, layer_group)
                    loss = align + 0.01 * reg
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt); scaler.update()
                if step == 1 or step == steps_per_stage:
                    print(f"CALIB {name} group={gi}/{len(groups)} stage={si}/{len(stages)} "
                          f"k={k} mix={mix:.2f} step={step}/{steps_per_stage} loss={float(loss):.5f}", flush=True)

            teacher_ce, student_ce = probe_ce(student, teacher, probe_batches, device, dtype)
            gap = student_ce - teacher_ce
            accepted = gap <= max_ce_gap
            print(f"GATE {name} group={gi}/{len(groups)} stage={si}/{len(stages)} k={k} mix={mix:.2f} "
                  f"teacher_ce={teacher_ce:.4f} student_ce={student_ce:.4f} gap={gap:.4f} "
                  f"{'ACCEPT' if accepted else 'ROLLBACK'}", flush=True)
            history.append({"group":gi,"stage":si,"layers":str(layer_group),"active_k":k,
                            "route_mix":mix,"teacher_ce":teacher_ce,"student_ce":student_ce,
                            "ce_gap":gap,"accepted":accepted})
            if not accepted:
                restore_group(student, before)
                break
    return history


@torch.no_grad()
def evaluate(student, teacher, eval_batches, device, dtype, seq_len, batch_size):
    student.eval(); teacher.eval(); rows=[]
    for cpu_ids in eval_batches:
        ids=cpu_ids.to(device); x=ids[:,:-1]
        with base.amp_ctx(device,dtype):
            t=teacher(input_ids=x,use_cache=False,return_dict=True)
            s=student(input_ids=x,use_cache=False,return_dict=True)
        rows.append((float(base.causal_ce(t.logits,ids)), float(base.causal_ce(s.logits,ids)),
                     float(base.distill_kl(s.logits,t.logits))))
    a=np.asarray(rows); tce=float(a[:,0].mean()); sce=float(a[:,1].mean())
    return {"teacher_ce":tce,"teacher_perplexity":math.exp(min(tce,20)),"ce":sce,
            "perplexity":math.exp(min(sce,20)),"teacher_kl":float(a[:,2].mean()),
            "eval_tokens":len(eval_batches)*batch_size*seq_len, **flyffn_v2_stats(student)}


def global_train(name, student, teacher, train_batches, eval_batches, args, device, dtype,
                 train_updates, grad_accum):
    layers=[m.layer_idx for m in flyffn_v2_modules(student) if m.route_mix>0.0]
    if not layers:
        print(f"GLOBAL {name}: no groups passed quality gate; exact dense fallback retained",flush=True)
        return evaluate(student,teacher,eval_batches,device,dtype,args.seq_len,args.batch_size), []
    groups, trainable=flyffn_v2_parameter_groups(student,layers,router_lr=5e-4,shard_lr=8e-6,weight_decay=.01)
    opt=base.make_optimizer(groups,device); scaler=base.make_scaler(device,dtype)
    warmup=max(20,train_updates//20); history=[]; micro=0; t0=time.perf_counter(); opt.zero_grad(set_to_none=True)
    student.train()
    for update in range(1,train_updates+1):
        ce_a=kl_a=hid_a=reg_a=loss_a=0.0
        for _ in range(grad_accum):
            ids=train_batches[micro%len(train_batches)].to(device); micro+=1; x=ids[:,:-1]
            with torch.no_grad(), base.amp_ctx(device,dtype):
                t=teacher(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
            with base.amp_ctx(device,dtype):
                s=student(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
                ce=base.causal_ce(s.logits,ids); kl=base.distill_kl(s.logits,t.logits)
                hid=base.hidden_alignment(s.hidden_states,t.hidden_states)
                reg=flyffn_v2_router_regularizer(student,layers)
                raw=.35*ce+.45*kl+.20*hid+.005*reg; loss=raw/grad_accum
            scaler.scale(loss).backward()
            ce_a+=float(ce.detach())/grad_accum; kl_a+=float(kl.detach())/grad_accum
            hid_a+=float(hid.detach())/grad_accum; reg_a+=float(reg.detach())/grad_accum
            loss_a+=float(raw.detach())/grad_accum
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable,1.0)
        if update<=warmup:
            mult=max(update/warmup,1e-3)
        else:
            p=(update-warmup)/max(train_updates-warmup,1)
            mult=.10+.90*.5*(1+math.cos(math.pi*min(p,1.0)))
        opt.param_groups[0]["lr"]=5e-4*mult; opt.param_groups[1]["lr"]=8e-6*mult
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        history.append({"update":update,"loss":loss_a,"ce":ce_a,"kl":kl_a,"hidden":hid_a,"router_reg":reg_a})
        if update==1 or update%25==0 or update==train_updates:
            elapsed=time.perf_counter()-t0; ups=update/max(elapsed,1e-9); eta=(train_updates-update)/max(ups,1e-9)
            print(f"GLOBAL {name} {update}/{train_updates} loss={loss_a:.4f} ce={ce_a:.4f} kl={kl_a:.4f} "
                  f"hid={hid_a:.4f} reg={reg_a:.4f} ups={ups:.3f} eta_s={eta:.1f}",flush=True)
    metrics=evaluate(student,teacher,eval_batches,device,dtype,args.seq_len,args.batch_size)
    metrics["train_tokens_s"]=train_updates*grad_accum*args.batch_size*args.seq_len/max(time.perf_counter()-t0,1e-9)
    metrics["params_m"]=sum(p.numel() for p in student.parameters())/1e6
    return metrics, history


def v2_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if ".mlp." in k or k.startswith("flyffn_shared_graph.")}


def load_v2_state(model,state):
    inc=model.load_state_dict(state,strict=False)
    missing=[k for k in inc.missing_keys if ".mlp." in k or k.startswith("flyffn_shared_graph.")]
    if missing:
        raise RuntimeError(f"missing FlyFFN-v2 keys: {missing[:8]}")


@torch.no_grad()
def dense_equivalence(student,teacher,ids,device,dtype):
    set_route_state(student,active_k=8,route_mix=0.0)
    with base.amp_ctx(device,dtype):
        t=teacher(input_ids=ids,use_cache=False,return_dict=True).logits
        s=student(input_ids=ids,use_cache=False,return_dict=True).logits
    return float((t.float()-s.float()).abs().max().cpu())


def train_variant(name, adjacency, teacher, tokenizer, cfg, calib_batches, probe_batches, train_batches,
                  eval_batches,args,device,dtype,steps_per_stage,group_size,max_ce_gap,train_updates,grad_accum):
    print(f"STAGE building {name} FlyFFN-v2 student",flush=True)
    student=build_student(dtype,device,cfg,adjacency,args.seed)
    eq=dense_equivalence(student,teacher,eval_batches[0][:,:-1].to(device),device,dtype)
    print(f"DENSE_EQ {name} max_abs_logit_diff={eq:.8g}",flush=True)
    hist=calibrate_progressively(name,student,teacher,calib_batches,probe_batches,device,dtype,
                                 steps_per_stage,group_size,max_ce_gap,args.num_shards)
    print(f"STAGE global distillation: {name}",flush=True)
    metrics,train_hist=global_train(name,student,teacher,train_batches,eval_batches,args,device,dtype,train_updates,grad_accum)
    return student,metrics,hist,train_hist,v2_state(student),eq


def main():
    args=parse_args(); base.set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype=base.choose_dtype(device)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True
    print(f"DEVICE {device} | dtype={dtype} | gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}",flush=True)
    if args.run_mode=="quick":
        steps_per_stage,group_size,train_updates,grad_accum,eval_count,probe_count=4,3,600,2,12,3
        max_ce_gap=.45 if args.max_ce_gap is None else args.max_ce_gap
    else:
        steps_per_stage,group_size,train_updates,grad_accum,eval_count,probe_count=15,3,5000,4,24,6
        max_ce_gap=.30 if args.max_ce_gap is None else args.max_ce_gap

    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    bio_adj,rew_adj,edge_count=base.extract_graph(args.fly_nodes,args.max_edges,args.seed,Path("/content/flyffn_v2_cache"))
    print("STAGE loading standard SmolLM2-135M teacher/tokenizer",flush=True)
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    teacher=AutoModelForCausalLM.from_pretrained(BASE_MODEL,dtype=dtype if device.type=="cuda" else torch.float32).to(device)
    teacher.eval(); teacher.config.use_cache=True
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg=FlyFFNV2Config(fly_nodes=args.fly_nodes,router_rank=args.router_rank,num_shards=args.num_shards,
                       graph_steps=args.graph_steps,graph_mix_init=args.graph_mix_init,anchor_every=args.anchor_every)
    temp=build_student(dtype,device,cfg,bio_adj,args.seed)
    fly_layers=[m.layer_idx for m in flyffn_v2_modules(temp)]
    anchors=anchor_layer_indices(temp,args.anchor_every)
    n_groups=math.ceil(len(fly_layers)/group_size); del temp; gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()
    print(f"Architecture: attention unchanged; {len(fly_layers)} FlyFFN-v2 + {len(anchors)} dense anchors={anchors}",flush=True)
    print(f"Progressive schedule: {stage_schedule(args.num_shards)} | CE gate={max_ce_gap:.3f}",flush=True)

    print("STAGE preparing FineWeb-Edu calibration/probe/train/eval blocks",flush=True)
    calib_needed=n_groups*len(stage_schedule(args.num_shards))*steps_per_stage*args.batch_size
    train_needed=train_updates*grad_accum*args.batch_size
    calib_batches=base.make_batches(list(base.token_blocks(tokenizer,args.seed+5,calib_needed,args.seq_len)),args.batch_size)
    probe_batches=base.make_batches(list(base.token_blocks(tokenizer,args.seed+777,probe_count*args.batch_size,args.seq_len)),args.batch_size)
    train_batches=base.make_batches(list(base.token_blocks(tokenizer,args.seed+10,train_needed,args.seq_len)),args.batch_size)
    eval_batches=base.make_batches(list(base.token_blocks(tokenizer,args.seed+999,eval_count*args.batch_size,args.seq_len)),args.batch_size)

    bio,bio_metrics,bio_calib,bio_hist,bio_state,bio_eq=train_variant(
        "biological",bio_adj,teacher,tokenizer,cfg,calib_batches,probe_batches,train_batches,eval_batches,
        args,device,dtype,steps_per_stage,group_size,max_ce_gap,train_updates,grad_accum)
    pd.DataFrame(bio_calib).to_csv(out_dir/"bio_progressive_calibration.csv",index=False)
    pd.DataFrame(bio_hist).to_csv(out_dir/"bio_training_history.csv",index=False)
    bio_schedule=routing_schedule(bio)

    rewired_metrics=rew_schedule=rew_eq=None
    if args.rewired:
        del bio; gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
        rew,rewired_metrics,rew_calib,rew_hist,_,rew_eq=train_variant(
            "rewired",rew_adj,teacher,tokenizer,cfg,calib_batches,probe_batches,train_batches,eval_batches,
            args,device,dtype,steps_per_stage,group_size,max_ce_gap,train_updates,grad_accum)
        pd.DataFrame(rew_calib).to_csv(out_dir/"rewired_progressive_calibration.csv",index=False)
        pd.DataFrame(rew_hist).to_csv(out_dir/"rewired_training_history.csv",index=False)
        rew_schedule=routing_schedule(rew); del rew; gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
        bio=build_student(dtype,device,cfg,bio_adj,args.seed); load_v2_state(bio,bio_state)

    assert_flyffn_v2_replacement(bio,args.anchor_every)
    test_ids=eval_batches[0][:,:-1].to(device)
    print("STAGE benchmarking standard SmolLM2 vs FlyFFN-v2",flush=True)
    tbench=base.benchmark(teacher,test_ids,device); fbench=base.benchmark(bio,test_ids,device)
    print("STAGE generating qualitative samples",flush=True)
    samples=base.generate_samples(bio,tokenizer,device)
    (out_dir/"samples.json").write_text(json.dumps(samples,indent=2),encoding="utf-8")

    report={
        "architecture":"FlyFFN v2: standard SmolLM2 attention + progressive FlyWire sparse FFN + dense anchors",
        "base_model":BASE_MODEL,"attention_unchanged":True,"flyffn_layers":len(fly_layers),"dense_anchor_layers":anchors,
        "dense_equivalence_biological_max_abs_logit_diff":bio_eq,"dense_equivalence_rewired_max_abs_logit_diff":rew_eq,
        "implementation_note":"quality prototype computes all shards during dense/sparse blend; fused selected-shard dispatch is a later speed optimization",
        "device":str(device),"dtype":str(dtype),
        "config":{"run_mode":args.run_mode,"seq_len":args.seq_len,"fly_nodes":args.fly_nodes,"fly_edges":edge_count,
                  "router_rank":args.router_rank,"num_shards":args.num_shards,"graph_steps":args.graph_steps,
                  "graph_mix_init":args.graph_mix_init,"anchor_every":args.anchor_every,"quality_gate_max_ce_gap":max_ce_gap,
                  "stage_schedule":stage_schedule(args.num_shards),"steps_per_stage":steps_per_stage,
                  "global_train_updates":train_updates,"grad_accum":grad_accum},
        "biological":bio_metrics,"rewired":rewired_metrics,"biological_routing_schedule":bio_schedule,
        "rewired_routing_schedule":rew_schedule,"benchmark_teacher":tbench,"benchmark_biological":fbench,
        "fly_ce_gap_vs_smollm2":bio_metrics["ce"]-bio_metrics["teacher_ce"],
        "fly_ppl_ratio_vs_smollm2":bio_metrics["perplexity"]/bio_metrics["teacher_perplexity"],
        "decode_speed_ratio_fly_over_smollm2":fbench["decode_tokens_s"]/tbench["decode_tokens_s"],
        "parameter_ratio_fly_over_smollm2":sum(p.numel() for p in bio.parameters())/sum(p.numel() for p in teacher.parameters()),
    }
    if rewired_metrics:
        report["biological_topology_ce_gain"]=rewired_metrics["ce"]-bio_metrics["ce"]
        report["biological_topology_ppl_gain_pct"]=100*(rewired_metrics["perplexity"]-bio_metrics["perplexity"])/rewired_metrics["perplexity"]
    rows={
        "SmolLM2-135M":{"ce":bio_metrics["teacher_ce"],"perplexity":bio_metrics["teacher_perplexity"],**tbench},
        "FlyFFN-v2 biological":{"ce":bio_metrics["ce"],"perplexity":bio_metrics["perplexity"],"teacher_kl":bio_metrics["teacher_kl"],**fbench},
    }
    if rewired_metrics:
        rows["FlyFFN-v2 rewired"]={"ce":rewired_metrics["ce"],"perplexity":rewired_metrics["perplexity"],"teacher_kl":rewired_metrics["teacher_kl"]}
    pd.DataFrame(rows).T.to_csv(out_dir/"summary.csv")
    (out_dir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    torch.save(bio_state,out_dir/"biological_flyffn_v2.pt")

    print("\nFLYFFN-V2 CHECK",flush=True)
    print("Attention unchanged: True",flush=True)
    print(f"FlyFFN-v2 layers: {len(fly_layers)} | dense anchors: {anchors}",flush=True)
    print(f"Dense-equivalence max logit diff: {bio_eq}",flush=True)
    print("\nSUMMARY",flush=True); print(pd.DataFrame(rows).T,flush=True)
    print("\nKEY REPORT",flush=True)
    keys=("fly_ce_gap_vs_smollm2","fly_ppl_ratio_vs_smollm2","parameter_ratio_fly_over_smollm2",
          "decode_speed_ratio_fly_over_smollm2","biological_topology_ce_gain","biological_topology_ppl_gain_pct")
    print(json.dumps({k:report[k] for k in keys if k in report},indent=2),flush=True)
    print("\nFINAL BIOLOGICAL ROUTING SCHEDULE",flush=True)
    print(json.dumps(bio_schedule,indent=2),flush=True)
    print("\nSAMPLES",flush=True)
    for item in samples:
        print("="*88,flush=True); print("PROMPT:",item["prompt"],flush=True); print(item["text"],flush=True)
    print("\nSaved:",out_dir,flush=True)


if __name__=="__main__":
    main()
