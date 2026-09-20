#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_cennmixer_v3 import (
    CeNNMixerV3Config, clone_cenn_state_v3, direct_mixer_losses_v3,
    finalize_cenn_only_v3, freeze_all_except_cenn_v3, install_cenn_mixer_v3,
    load_cenn_state_v3, reset_stream_state_v3, set_alpha_v3,
)
from tinycenn_lm.qwen35_cennmixer_v3_train import (
    causal_ce, distill_kl, generation_suite, hidden_losses, load_data, probe,
    quality_key, stage_ok, stage_targets, stage_violation, topk_rank_loss,
)


def parse_args():
    p=argparse.ArgumentParser(description="CeNNMixer-v3 global-memory progressive takeover")
    p.add_argument("--base-model",default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--layers",default="0")
    p.add_argument("--alphas",default="0,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0")
    p.add_argument("--seq-len",type=int,default=128)
    p.add_argument("--train-blocks",type=int,default=1024)
    p.add_argument("--val-blocks",type=int,default=48)

    p.add_argument("--groups",type=int,default=32)
    p.add_argument("--cell-dim",type=int,default=48)
    p.add_argument("--graph-steps",type=int,default=1)
    p.add_argument("--memory-slots",type=int,default=8)
    p.add_argument("--memory-dim",type=int,default=64)
    p.add_argument("--memory-topk",type=int,default=4)

    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--stage-updates",type=int,default=300)
    p.add_argument("--extend-updates",type=int,default=200)
    p.add_argument("--max-stage-updates",type=int,default=3000)
    p.add_argument("--probe-every",type=int,default=50)
    p.add_argument("--patience-probes",type=int,default=8)
    p.add_argument("--min-lr",type=float,default=1.25e-5)

    p.add_argument("--topk",type=int,default=64)
    p.add_argument("--min-top1",type=float,default=0.97)
    p.add_argument("--max-kl",type=float,default=0.03)
    p.add_argument("--max-hidden-mse",type=float,default=0.05)
    p.add_argument("--max-mixer-mse",type=float,default=0.12)
    p.add_argument("--quick-smoke",action="store_true")
    p.add_argument("--output-dir",default="results/cennmixer_v3_qwen35_08b")
    p.add_argument("--seed",type=int,default=8621)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dtype_for(device):
    if device.type!="cuda": return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def smoke_overrides(a):
    if not a.quick_smoke: return
    a.alphas="0,0.10,0.20,0.30,0.40,0.50,0.60,0.75,1.0"
    a.seq_len=64
    a.train_blocks=256
    a.val_blocks=16
    a.stage_updates=80
    a.extend_updates=40
    a.max_stage_updates=260
    a.probe_every=20
    a.patience_probes=4
    a.topk=16
    print("QUICK_SMOKE V3",{
        "alphas":a.alphas,"seq_len":a.seq_len,
        "train_blocks":a.train_blocks,"val_blocks":a.val_blocks,
        "memory_slots":a.memory_slots,"memory_dim":a.memory_dim,
        "memory_topk":a.memory_topk,
    },flush=True)


def stage_cap(alpha,a):
    if not a.quick_smoke: return a.max_stage_updates
    if alpha==0.0: return 520
    if alpha in (0.4,0.5,0.6): return 300
    if alpha==1.0: return 360
    return 220


def main():
    a=parse_args()
    smoke_overrides(a)
    seed_all(a.seed)

    layers=[int(x) for x in a.layers.split(",") if x.strip()]
    alphas=[float(x) for x in a.alphas.split(",") if x.strip()]
    if not alphas or alphas[0]!=0.0 or alphas[-1]!=1.0:
        raise ValueError("alpha schedule must start at 0 and end at 1")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype=dtype_for(device)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    print("DEVICE",device,"dtype",dtype,"layers",layers,"alphas",alphas,flush=True)

    tok=AutoTokenizer.from_pretrained(a.base_model,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token

    teacher=AutoModelForCausalLM.from_pretrained(
        a.base_model,dtype=dtype,low_cpu_mem_usage=True
    ).to(device).eval()
    student=AutoModelForCausalLM.from_pretrained(
        a.base_model,dtype=dtype,low_cpu_mem_usage=True
    ).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    cfg=CeNNMixerV3Config(
        hidden_size=int(student.config.hidden_size),
        groups=a.groups,cell_dim=a.cell_dim,graph_steps=a.graph_steps,
        memory_slots=a.memory_slots,memory_dim=a.memory_dim,memory_topk=a.memory_topk,
    )
    layer_kinds={}
    for li in layers:
        layer=student.model.layers[li]
        layer_kinds[str(li)]=getattr(layer,"block_type","unknown")
        install_cenn_mixer_v3(student,li,cfg)

    train,val=load_data(tok,a)
    trainable=freeze_all_except_cenn_v3(student)
    cenn_params=sum(p.numel() for p in trainable)
    replaced_params=0
    for li in layers:
        tl=teacher.model.layers[li]
        mod=tl.linear_attn if getattr(tl,"block_type",None)=="linear_attention" else tl.self_attn
        replaced_params+=sum(p.numel() for p in mod.parameters())
    print("CeNN-v3 params",cenn_params,"Qwen mixer params",replaced_params,flush=True)

    history=[]; stages=[]; global_step=0
    previous_best=clone_cenn_state_v3(student)

    for alpha in alphas:
        load_cenn_state_v3(student,previous_best)
        set_alpha_v3(student,alpha)
        reset_stream_state_v3(student)

        initial=probe(student,teacher,val,layers,device,local_teacher=True)
        best_m=dict(initial); best_state=clone_cenn_state_v3(student)
        best_key=quality_key(initial,alpha,a); best_step=0
        stage_steps=0; no_improve=0
        cap=stage_cap(alpha,a)
        budget=min(a.stage_updates,cap)

        print("\n=== ALPHA",alpha,"===",flush=True)
        print("TARGETS",json.dumps(stage_targets(alpha,a)),flush=True)
        print("INITIAL",json.dumps(initial),flush=True)

        # Do not perturb a checkpoint that already passes the next alpha.
        if not stage_ok(initial,alpha,a):
            trainable=freeze_all_except_cenn_v3(student)
            opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0.0)

            while stage_steps<cap:
                while stage_steps<min(budget,cap):
                    global_step+=1; stage_steps+=1
                    student.train()
                    ids=train[(global_step-1)%len(train)].to(device)
                    x,y=ids[:,:-1],ids[:,1:]

                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        to=teacher(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
                    so=student(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)

                    loc=direct_mixer_losses_v3(student)
                    hm,dm=hidden_losses(so.hidden_states,to.hidden_states,layers)
                    lkl=distill_kl(so.logits,to.logits)
                    rank=topk_rank_loss(so.logits,to.logits,a.topk)
                    lce=causal_ce(so.logits,y)

                    local_loss=0.65*loc["mse"]+0.20*loc["cosine"]+0.15*loc["delta"]
                    global_loss=0.40*lkl+0.25*rank+0.20*hm+0.10*dm+0.05*lce
                    local_weight=0.82-0.22*float(alpha)
                    loss=local_weight*local_loss+(1.0-local_weight)*global_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable,0.5)
                    opt.step()

                    if stage_steps==1 or stage_steps%a.probe_every==0:
                        vm=probe(student,teacher,val,layers,device,local_teacher=True)
                        row={
                            "global_step":global_step,"alpha":alpha,"stage_step":stage_steps,
                            "train_loss":float(loss.detach()),
                            "train_mixer_mse":float(loc["mse"].detach()),
                            "train_mixer_cosine":float(loc["cosine"].detach()),
                            "train_mixer_delta":float(loc["delta"].detach()),
                            "train_kl":float(lkl.detach()),"train_rank_loss":float(rank.detach()),
                            **vm,
                        }
                        history.append(row)
                        print("PROBE",json.dumps(row),flush=True)
                        key=quality_key(vm,alpha,a)
                        if key<best_key:
                            best_key=key; best_m=dict(vm)
                            best_state=clone_cenn_state_v3(student); best_step=stage_steps
                            no_improve=0
                            torch.save({
                                "state":best_state,"config":cfg.to_dict(),"layers":layers,
                                "layer_kinds":layer_kinds,"alpha":alpha,
                                "stage_step":best_step,"metrics":best_m,
                            },out/f"cennmixer_v3_alpha_{str(alpha).replace('.','p')}_best.pt")
                            print("✓ NEW BEST",json.dumps({
                                "alpha":alpha,"step":best_step,
                                "violation":stage_violation(best_m,alpha,a),
                                "pass":stage_ok(best_m,alpha,a),
                            }),flush=True)
                        else:
                            no_improve+=1
                        if stage_ok(best_m,alpha,a): break

                if stage_ok(best_m,alpha,a) or stage_steps>=cap: break
                if no_improve>=a.patience_probes:
                    for g in opt.param_groups:
                        g["lr"]=max(float(g["lr"])*0.5,a.min_lr)
                    no_improve=0
                    print("↘ LR",opt.param_groups[0]["lr"],flush=True)
                budget=min(budget+a.extend_updates,cap)
                print("↻ extend alpha",alpha,"to",budget,flush=True)

        load_cenn_state_v3(student,best_state)
        previous_best=clone_cenn_state_v3(student)
        set_alpha_v3(student,alpha)
        restored=probe(student,teacher,val,layers,device,local_teacher=True)
        gens=generation_suite(student,teacher,tok,device)
        stage_row={
            "alpha":alpha,"best_step":best_step,"trained_steps":stage_steps,
            "pass":stage_ok(restored,alpha,a),
            "violation":stage_violation(restored,alpha,a),**restored,
            "generation_exact_rate":sum(int(x["exact"]) for x in gens)/len(gens),
            "generation_mean_jaccard":sum(x["jaccard"] for x in gens)/len(gens),
        }
        stages.append(stage_row)
        print("RESTORED BEST",json.dumps(stage_row),flush=True)
        (out/f"generation_alpha_{str(alpha).replace('.','p')}.json").write_text(
            json.dumps(gens,indent=2),encoding="utf-8"
        )
        if not stage_row["pass"]:
            print("STOPPING: alpha",alpha,"did not pass readiness gate.",flush=True)
            break

    reached_alpha=stages[-1]["alpha"] if stages else None
    progression_complete=bool(stages and reached_alpha==1.0 and stages[-1]["pass"])
    alpha1_probe=None; alpha1_generation=[]; compact_probe=None; compact_generation=[]
    strict_quality=False

    if progression_complete:
        set_alpha_v3(student,1.0)
        alpha1_probe=probe(student,teacher,val,layers,device,local_teacher=True)
        alpha1_generation=generation_suite(student,teacher,tok,device)
        finalize_cenn_only_v3(student)
        reset_stream_state_v3(student)
        compact_probe=probe(student,teacher,val,layers,device,local_teacher=False)
        compact_generation=generation_suite(student,teacher,tok,device)
        strict_quality=(
            compact_probe["top1"]>=a.min_top1
            and compact_probe["kl"]<=a.max_kl
            and compact_probe["hidden_mse"]<=a.max_hidden_mse
        )
        torch.save({
            "state":clone_cenn_state_v3(student),"config":cfg.to_dict(),
            "layers":layers,"layer_kinds":layer_kinds,"base_model":a.base_model,
            "alpha":1.0,"cenn_only":True,
        },out/"cennmixer_v3_final_cenn_only.pt")

    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    pd.DataFrame(stages).to_csv(out/"stage_summary.csv",index=False)
    report={
        "architecture":"CeNNMixer-v3 cellular + global recurrent memory progressive takeover",
        "base_model":a.base_model,"layers":layers,"layer_kinds":layer_kinds,
        "alpha_schedule":alphas,"config":cfg.to_dict(),
        "cenn_params":cenn_params,"replaced_qwen_mixer_params":replaced_params,
        "mixer_param_reduction_pct":100*(1-cenn_params/max(replaced_params,1)),
        "stage_summary":stages,"reached_alpha":reached_alpha,
        "progression_complete":progression_complete,
        "alpha1_before_removing_qwen_mixer":alpha1_probe,
        "alpha1_generation_before_removal":alpha1_generation,
        "final_cenn_only_probe":compact_probe,
        "final_cenn_only_generation":compact_generation,
        "strict_quality_gate":strict_quality,
        "thresholds":{"min_top1":a.min_top1,"max_kl":a.max_kl,
                      "max_hidden_mse":a.max_hidden_mse,"max_mixer_mse":a.max_mixer_mse},
        "args":vars(a),
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FINAL",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
