#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_cennmixer_v4 import (
    CeNNMixerV4Config,clone_cenn_state_v4,direct_mixer_losses_v4,
    finalize_cenn_only_v4,freeze_all_except_cenn_v4,install_cenn_mixer_v4,
    load_cenn_state_v4,reset_stream_state_v4,set_alpha_v4,
)
from tinycenn_lm.qwen35_cennmixer_v4_train import (
    context_lengths,curriculum_window,chat_ids,retrieval_prompt,probe_contexts,generation_health,
    causal_ce,forward_kl,generation_suite,hidden_losses,load_data,
    on_policy_distill_loss,probe,quality_key,reverse_kl,stage_ok,
    stage_targets,stage_violation,top1_margin_loss,topk_rank_loss,
)


def parse_args():
    p=argparse.ArgumentParser(description="CeNNMixer-v4 DeltaCell progressive distillation")
    p.add_argument("--base-model",default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--layers",default="0")
    p.add_argument("--alphas",default="0,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0")
    p.add_argument("--seq-len",type=int,default=512)
    p.add_argument("--context-lengths",default="128,256,512")
    p.add_argument("--min-stage-updates",type=int,default=100)
    p.add_argument("--max-ce-gap",type=float,default=0.05)
    p.add_argument("--train-blocks",type=int,default=1536)
    p.add_argument("--val-blocks",type=int,default=8)

    p.add_argument("--groups",type=int,default=24)
    p.add_argument("--cell-dim",type=int,default=32)
    p.add_argument("--graph-steps",type=int,default=1)
    p.add_argument("--assoc-heads",type=int,default=8)
    p.add_argument("--key-dim",type=int,default=32)
    p.add_argument("--value-dim",type=int,default=64)
    p.add_argument("--conv-kernel",type=int,default=4)

    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--stage-updates",type=int,default=300)
    p.add_argument("--extend-updates",type=int,default=200)
    p.add_argument("--max-stage-updates",type=int,default=3000)
    p.add_argument("--probe-every",type=int,default=50)
    p.add_argument("--patience-probes",type=int,default=8)
    p.add_argument("--min-lr",type=float,default=1.25e-5)

    p.add_argument("--topk",type=int,default=64)
    p.add_argument("--on-policy-every",type=int,default=4)
    p.add_argument("--on-policy-tokens",type=int,default=32)

    p.add_argument("--min-top1",type=float,default=0.97)
    p.add_argument("--max-kl",type=float,default=0.03)
    p.add_argument("--max-hidden-mse",type=float,default=0.05)
    p.add_argument("--max-mixer-mse",type=float,default=0.12)

    p.add_argument("--quick-smoke",action="store_true")
    p.add_argument("--explore-high-alpha",action="store_true", help="Continue through soft smoke-gate misses while generation remains meaningful")
    p.add_argument("--resume-alpha",type=float,default=None, help="Load and revalidate/retrain this alpha before progressing")
    p.add_argument("--output-dir",default="results/cennmixer_v4_qwen35_08b")
    p.add_argument("--seed",type=int,default=8621)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dtype_for(device):
    if device.type!="cuda": return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def smoke_overrides(a):
    if not a.quick_smoke: return
    a.alphas="0,0.10,0.20,0.30,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0"
    # Smoke changes the budget, never the explicitly requested context.
    a.train_blocks=384
    a.val_blocks=16
    a.stage_updates=80
    a.extend_updates=40
    a.max_stage_updates=280
    a.probe_every=20
    a.patience_probes=4
    a.min_stage_updates=min(a.min_stage_updates,40)
    a.topk=16
    print("QUICK_SMOKE V4",{
        "alphas":a.alphas,"seq_len":a.seq_len,
        "train_blocks":a.train_blocks,"val_blocks":a.val_blocks,
        "assoc":f"{a.assoc_heads}x{a.key_dim}x{a.value_dim}",
        "conv_kernel":a.conv_kernel,
        "on_policy_every":a.on_policy_every,
    },flush=True)


def stage_cap(alpha,a):
    if not a.quick_smoke: return a.max_stage_updates
    if alpha==0.0: return 600
    if alpha>=0.5: return 320 if alpha<1.0 else 420
    return 240


def main():
    a=parse_args(); smoke_overrides(a); seed_all(a.seed)

    lengths=context_lengths(a)
    rng=random.Random(a.seed)
    layers=[int(x) for x in a.layers.split(",") if x.strip()]
    alphas=[float(x) for x in a.alphas.split(",") if x.strip()]
    if not alphas or alphas[0]!=0.0 or alphas[-1]!=1.0:
        raise ValueError("alpha schedule must start at 0 and end at 1")

    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    if any(not 0 <= x <= 1 for x in alphas) or any(x >= y for x,y in zip(alphas,alphas[1:])):
        raise ValueError("alphas must be strictly increasing in [0,1]")
    for name in ("min_stage_updates","seq_len","train_blocks","val_blocks","stage_updates","extend_updates",
                 "max_stage_updates","probe_every","patience_probes","topk","on_policy_tokens"):
        if getattr(a,name) <= 0:
            raise ValueError(f"{name} must be positive")

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

    cfg=CeNNMixerV4Config(
        hidden_size=int(student.config.hidden_size),
        groups=a.groups,cell_dim=a.cell_dim,graph_steps=a.graph_steps,
        assoc_heads=a.assoc_heads,key_dim=a.key_dim,value_dim=a.value_dim,
        conv_kernel=a.conv_kernel,
    )

    layer_kinds={}
    for li in layers:
        if not 0 <= li < len(student.model.layers):
            raise ValueError(f"Layer index out of range: {li}")
        layer=student.model.layers[li]
        layer_kinds[str(li)]=getattr(layer,"block_type","unknown")
        install_cenn_mixer_v4(student,li,cfg)

    train,val=load_data(tok,a)
    print("CONTEXTS",lengths,"validation blocks per length",a.val_blocks,flush=True)
    def evaluate(local_teacher=True):
        metrics,per_context=probe_contexts(student,teacher,val,layers,device,local_teacher)
        rows=generation_suite(student,teacher,tok,device,lengths)
        metrics["generation_ok"]=float(generation_health(rows))
        return metrics,per_context,rows

    trainable=freeze_all_except_cenn_v4(student)
    cenn_params=sum(p.numel() for p in trainable)
    replaced_params=0
    for li in layers:
        tl=teacher.model.layers[li]
        mod=tl.linear_attn if getattr(tl,"block_type",None)=="linear_attention" else tl.self_attn
        replaced_params+=sum(p.numel() for p in mod.parameters())
    print("CeNN-v4 params",cenn_params,"Qwen mixer params",replaced_params,flush=True)

    history=[]; stages=[]; global_step=0
    previous_best=clone_cenn_state_v4(student)
    run_alphas=list(alphas)

    if a.resume_alpha is not None:
        tag=str(float(a.resume_alpha)).replace(".","p")
        ck_path=out/f"cennmixer_v4_alpha_{tag}_best.pt"
        if not ck_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {ck_path}. "
                "Use the same Colab runtime/output directory or rerun from alpha=0."
            )
        ck=torch.load(ck_path,map_location="cpu",weights_only=True)
        if ck.get("config") != cfg.to_dict() or ck.get("layers") != layers:
            raise ValueError("Resume checkpoint architecture/layers differ from this run")
        if ck.get("base_model",a.base_model) != a.base_model:
            raise ValueError("Resume checkpoint uses a different base model")
        load_cenn_state_v4(student,ck["state"])
        previous_best=clone_cenn_state_v4(student)
        run_alphas=[x for x in alphas if x>=float(a.resume_alpha)-1e-12]
        print(
            "RESUME:",
            "loaded",ck_path,
            "best step",ck.get("stage_step"),
            "metrics",json.dumps(ck.get("metrics",{})),
            "next alphas",run_alphas,
            flush=True,
        )
        if not run_alphas:
            raise ValueError("No alpha values remain after --resume-alpha")

    for alpha in run_alphas:
        load_cenn_state_v4(student,previous_best)
        set_alpha_v4(student,alpha)
        reset_stream_state_v4(student)

        initial,initial_contexts,initial_gens=evaluate()
        best_m=dict(initial); best_state=clone_cenn_state_v4(student)
        best_key=quality_key(initial,alpha,a); best_step=0
        stage_steps=0; no_improve=0
        cap=stage_cap(alpha,a); budget=min(a.stage_updates,cap)

        print("\n=== ALPHA",alpha,"===",flush=True)
        print("TARGETS",json.dumps(stage_targets(alpha,a)),flush=True)
        print("INITIAL",json.dumps(initial),flush=True)

        checkpoint_path=out/f"cennmixer_v4_alpha_{str(alpha).replace('.', 'p')}_best.pt"
        # Even an initially passing stage must be resumable.
        torch.save({"state":best_state,"config":cfg.to_dict(),"layers":layers,
                    "layer_kinds":layer_kinds,"base_model":a.base_model,
                    "alpha":alpha,"stage_step":0,"metrics":best_m},checkpoint_path)

        if a.min_stage_updates>0 or not stage_ok(initial,alpha,a):
            trainable=freeze_all_except_cenn_v4(student)
            opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0.0)

            while stage_steps<cap:
                while stage_steps<min(budget,cap):
                    global_step+=1; stage_steps+=1
                    student.eval()  # Gradients enabled; frozen backbone stays deterministic.

                    cpu=curriculum_window(rng.choice(train),stage_steps,a.min_stage_updates,lengths,rng)
                    ids=cpu.to(device)
                    x,y=ids[:,:-1],ids[:,1:]
                    opt.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        to=teacher(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
                    so=student(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)

                    loc=direct_mixer_losses_v4(student)
                    hm,dm=hidden_losses(so.hidden_states,to.hidden_states,layers)
                    fkl=forward_kl(so.logits,to.logits)
                    rkl=reverse_kl(so.logits,to.logits)
                    rank=topk_rank_loss(so.logits,to.logits,a.topk)
                    margin=top1_margin_loss(so.logits,to.logits)
                    ce=causal_ce(so.logits,y)

                    local_loss=0.62*loc["mse"]+0.20*loc["cosine"]+0.18*loc["delta"]
                    global_loss=(
                        0.22*fkl+0.20*rkl+0.18*rank+0.20*margin
                        +0.12*hm+0.05*dm+0.03*ce
                    )
                    local_weight=1.0 if alpha==0 else 0.30
                    loss=local_weight*local_loss+(1.0-local_weight)*global_loss

                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite offline loss")
                    loss.backward()
                    # Release the large offline graph before the on-policy forward.
                    del to,so
                    onp=torch.zeros((),device=device)
                    onp_scale=0.0
                    if alpha>0 and a.on_policy_every>0 and stage_steps%a.on_policy_every==0:
                        if (stage_steps//a.on_policy_every)%2:
                            prompt,_=retrieval_prompt(tok,int(x.shape[1]),a.seed+global_step)
                            prefix=chat_ids(tok,prompt)
                        else:
                            prefix=x[:,:max(16,x.shape[1]//2)]
                        onp=on_policy_distill_loss(
                            student,teacher,prefix,device,max_new_tokens=a.on_policy_tokens
                        )
                        # Keep on-policy feedback useful without allowing one bad
                        # generated prefix to dominate the optimizer. The detached
                        # scale caps its effective scalar magnitude at ~0.25.
                        raw_onp=float(onp.detach())
                        onp_scale=min(1.0,0.25/max(raw_onp,1e-8))
                        on_policy_weight=0.30
                        (on_policy_weight*onp*onp_scale).backward()

                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at alpha={alpha}, step={stage_steps}; best checkpoint retained")
                    torch.nn.utils.clip_grad_norm_(trainable,0.5,error_if_nonfinite=True)
                    opt.step()

                    if stage_steps==1 or stage_steps%a.probe_every==0:
                        vm,context_metrics,probe_gens=evaluate()
                        row={
                            "global_step":global_step,"alpha":alpha,"stage_step":stage_steps,
                            "train_loss":float(loss.detach()),
                            "train_mixer_mse":float(loc["mse"].detach()),
                            "train_mixer_cosine":float(loc["cosine"].detach()),
                            "train_mixer_delta":float(loc["delta"].detach()),
                            "train_forward_kl":float(fkl.detach()),
                            "train_reverse_kl":float(rkl.detach()),
                            "train_rank_loss":float(rank.detach()),
                            "train_margin_loss":float(margin.detach()),
                            "train_on_policy_loss":float(onp.detach()),
                            "train_on_policy_scale":float(onp_scale),
                            **vm,
                        }
                        history.append(row)
                        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
                        print("PROBE",json.dumps(row),flush=True)

                        key=quality_key(vm,alpha,a)
                        if key<best_key:
                            best_key=key; best_m=dict(vm)
                            best_state=clone_cenn_state_v4(student); best_step=stage_steps
                            no_improve=0
                            torch.save({
                                "state":best_state,"config":cfg.to_dict(),"layers":layers,
                                "layer_kinds":layer_kinds,"base_model":a.base_model,"alpha":alpha,
                                "stage_step":best_step,"metrics":best_m,
                            },out/f"cennmixer_v4_alpha_{str(alpha).replace('.','p')}_best.pt")
                            print("✓ NEW BEST",json.dumps({
                                "alpha":alpha,"step":best_step,
                                "violation":stage_violation(best_m,alpha,a),
                                "pass":stage_ok(best_m,alpha,a),
                            }),flush=True)
                        else:
                            no_improve+=1
                        if stage_steps>=a.min_stage_updates and stage_ok(best_m,alpha,a): break

                if (stage_steps>=a.min_stage_updates and stage_ok(best_m,alpha,a)) or stage_steps>=cap: break
                if no_improve>=a.patience_probes:
                    for g in opt.param_groups:
                        g["lr"]=max(float(g["lr"])*0.5,a.min_lr)
                    no_improve=0
                    print("↘ LR",opt.param_groups[0]["lr"],flush=True)
                budget=min(budget+a.extend_updates,cap)
                print("↻ extend alpha",alpha,"to",budget,flush=True)

        load_cenn_state_v4(student,best_state)
        previous_best=clone_cenn_state_v4(student)
        set_alpha_v4(student,alpha)

        restored,context_metrics,gens=evaluate()
        hard_pass=stage_ok(restored,alpha,a)
        violation=stage_violation(restored,alpha,a)
        gen_exact=sum(int(x["exact"]) for x in gens)/len(gens)
        gen_jaccard=sum(x["jaccard"] for x in gens)/len(gens)

        # Exploration mode is deliberately separate from the quality gate.
        # It allows higher-alpha diagnosis only while the model remains locally
        # faithful and still generates recognizably related text.
        exploration_continue=False
        if a.quick_smoke and a.explore_high_alpha and alpha<1.0 and not hard_pass:
            exploration_continue=(
                generation_health(gens)
                and violation<=0.20
                and restored["top1"]>=0.82
                and restored["kl"]<=0.20
                and restored["hidden_mse"]<=0.18
                and restored["mixer_mse"]<=0.30
                and restored["mixer_cosine"]<=0.15
                and restored["mixer_delta"]<=0.45
            )

        stage_row={
            "alpha":alpha,"best_step":best_step,"trained_steps":stage_steps,
            "pass":hard_pass,
            "exploration_continue":exploration_continue,
            "violation":violation,**restored,
            "generation_exact_rate":gen_exact,
            "generation_mean_jaccard":gen_jaccard,
            "generation_warning":not generation_health(gens),
            "context_metrics":context_metrics,
        }
        stages.append(stage_row)
        pd.DataFrame(stages).to_csv(out/"stage_summary.csv",index=False)
        print("RESTORED BEST",json.dumps(stage_row),flush=True)
        (out/f"generation_alpha_{str(alpha).replace('.','p')}.json").write_text(
            json.dumps(gens,indent=2),encoding="utf-8"
        )

        if not hard_pass:
            if exploration_continue:
                print(
                    "⚠ SOFT EXPLORATION CONTINUE:",
                    "alpha",alpha,
                    "missed the strict smoke gate but numerical stability is sufficient to test higher alpha.",
                    flush=True,
                )
            else:
                print("STOPPING: alpha",alpha,"did not pass readiness/exploration gate.",flush=True)
                break

    reached_alpha=stages[-1]["alpha"] if stages else None
    progression_complete=bool(stages and reached_alpha==1.0 and stages[-1]["pass"])
    alpha1_probe=None; alpha1_generation=[]; compact_probe=None; compact_generation=[]
    strict_quality=False
    test_probe=None; test_contexts={}; test_generation=[]

    if progression_complete:
        set_alpha_v4(student,1.0)
        alpha1_probe,_,alpha1_generation=evaluate()

        finalize_cenn_only_v4(student)
        reset_stream_state_v4(student)
        compact_probe,_,compact_generation=evaluate(local_teacher=False)
        # A separate Wikitext test split is touched only after conversion.
        from datasets import load_dataset
        from tinycenn_lm.qwen35_cennmixer_v4_train import blocks
        test_texts=[x["text"] for x in load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")]
        test_batches={n:blocks(tok,test_texts,a.val_blocks,n,a.seed+19000+n) for n in lengths}
        test_probe,test_contexts=probe_contexts(student,teacher,test_batches,layers,device,False)
        test_generation=generation_suite(student,teacher,tok,device,lengths,seed=27001)
        strict_quality=(
            all(math.isfinite(v) for v in compact_probe.values())
            and all(math.isfinite(v) for v in test_probe.values())
            and compact_probe["top1"]>=a.min_top1
            and compact_probe["kl"]<=a.max_kl
            and compact_probe["hidden_mse"]<=a.max_hidden_mse
            and compact_probe["ce_gap"]<=a.max_ce_gap
            and generation_health(compact_generation)
            and test_probe["top1"]>=a.min_top1 and test_probe["kl"]<=a.max_kl
            and test_probe["hidden_mse"]<=a.max_hidden_mse and test_probe["ce_gap"]<=a.max_ce_gap
            and generation_health(test_generation) and not a.quick_smoke
        )
        torch.save({
            "state":clone_cenn_state_v4(student),"config":cfg.to_dict(),
            "layers":layers,"layer_kinds":layer_kinds,"base_model":a.base_model,
            "alpha":1.0,"cenn_only":True,
        },out/"cennmixer_v4_final_cenn_only.pt")

    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    pd.DataFrame(stages).to_csv(out/"stage_summary.csv",index=False)

    report={
        "architecture":"CeNNMixer-v4 DeltaCell: CeNN + compact gated-delta matrix memory",
        "base_model":a.base_model,"layers":layers,"layer_kinds":layer_kinds,
        "alpha_schedule":alphas,"config":cfg.to_dict(),
        "cenn_params":cenn_params,"replaced_qwen_mixer_params":replaced_params,
        "mixer_param_reduction_pct":100*(1-cenn_params/max(replaced_params,1)),
        "stage_summary":stages,"reached_alpha":reached_alpha,
        "progression_complete":progression_complete,
        "exploration_mode":bool(a.explore_high_alpha),
        "exploration_reached_alpha":reached_alpha,
        "alpha1_before_removing_qwen_mixer":alpha1_probe,
        "alpha1_generation_before_removal":alpha1_generation,
        "final_cenn_only_probe":compact_probe,
        "final_cenn_only_generation":compact_generation,
        "strict_quality_gate":strict_quality,
        "test_probe":test_probe,"test_contexts":test_contexts,"test_generation":test_generation,
        "evaluated_context_lengths":lengths,
        "last_passing_alpha":next((s["alpha"] for s in reversed(stages) if s["pass"]),None),
        "args":vars(a),
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FINAL",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
