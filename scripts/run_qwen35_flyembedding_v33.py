#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinycenn_lm.qwen35_flyembedding_v33 import (
    FlyEmbeddingV33Config,
    factorize_vocab_weight_v33,
    install_fly_embedding_v33,
    set_progressive_beta_v33,
    freeze_qwen_train_compact_v33,
    export_compact_only_v33,
)

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args():
    p = argparse.ArgumentParser(description="FlyEmbedding-v3.3 progressive full vocabulary replacement")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--run-mode", choices=["quick","strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--fly-nodes", type=int, default=256)
    p.add_argument("--graph-steps", type=int, default=1)
    p.add_argument("--graph-mix", type=float, default=0.05)
    p.add_argument("--max-fly-scale", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--stage-updates", type=int, default=100)
    p.add_argument("--probe-every", type=int, default=20)
    p.add_argument("--extend-updates", type=int, default=100)
    p.add_argument("--max-stage-updates", type=int, default=1200)
    p.add_argument("--patience-probes", type=int, default=12)
    p.add_argument("--min-top1", type=float, default=0.97)
    p.add_argument("--max-kl", type=float, default=0.03)
    p.add_argument("--max-ce-gap", type=float, default=0.08)
    p.add_argument("--output-dir", default="results/flyembedding_v33_qwen35_08b")
    p.add_argument("--seed", type=int, default=8621)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_for(device):
    if device.type != "cuda": return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def synthetic_texts():
    return [
        "Artificial intelligence systems learn patterns from data and use those patterns to make predictions.",
        "Vienna is the capital of Austria and is known for music, architecture, science, and public transport.",
        "A neural network transforms an input through a sequence of learned linear and nonlinear operations.",
        "Efficient language models try to reduce memory and computation while preserving quality.",
        "The sky appears blue because shorter wavelengths of sunlight are scattered more strongly by the atmosphere.",
        "Seventeen plus twenty five equals forty two.",
        "Software architecture describes components, interfaces, constraints, data flows, and operational qualities.",
        "Machine learning evaluation should separate training data from held out validation and test data.",
        "Scientific experiments should report methods, baselines, uncertainty, and reproducible measurements.",
        "Graph neural computation propagates local state through sparse connections and nonlinear transformations.",
        "Residual adapters preserve a pretrained model while learning a compact correction.",
        "Language models predict the next token from a sequence of preceding tokens.",
    ]


def blocks_from_texts(tokenizer, texts, count, seq_len, seed):
    rng = random.Random(seed)
    clean = [x.strip() for x in texts if isinstance(x,str) and len(x.strip()) > 30]
    if not clean: clean = synthetic_texts()
    out = []
    while len(out) < count:
        s = " ".join(rng.choice(clean) for _ in range(10))
        ids = tokenizer(s, return_tensors="pt", truncation=False).input_ids[0]
        if ids.numel() < seq_len+1: continue
        hi = ids.numel()-(seq_len+1)
        st = rng.randint(0,hi) if hi > 0 else 0
        out.append(ids[st:st+seq_len+1].unsqueeze(0))
    return out


def load_blocks(tokenizer, run_mode, seq_len, seed):
    ntrain = 640 if run_mode=="quick" else 2400
    nval = 24 if run_mode=="quick" else 64
    try:
        from datasets import load_dataset
        tr = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
        va = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
        train = blocks_from_texts(tokenizer,[x["text"] for x in tr],ntrain,seq_len,seed+11)
        val = blocks_from_texts(tokenizer,[x["text"] for x in va],nval,seq_len,seed+777)
        return train,val,"Salesforce/WikiText-2"
    except Exception as e:
        print("WARNING: WikiText load failed:",repr(e),flush=True)
        base=synthetic_texts()
        return (
            blocks_from_texts(tokenizer,base,ntrain,seq_len,seed+11),
            blocks_from_texts(tokenizer,base,nval,seq_len,seed+777),
            "synthetic-fallback",
        )


def causal_ce(logits, target):
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1,logits.shape[-1]),
        target.reshape(-1),
    )


def distill_kl(student, teacher):
    s = torch.log_softmax(student.float(),dim=-1)
    t = torch.softmax(teacher.float(),dim=-1)
    return torch.nn.functional.kl_div(s,t,reduction="batchmean")/max(student.shape[1],1)


@torch.no_grad()
def probe(student,teacher,batches,device):
    student.eval(); teacher.eval()
    tce=sce=kl=emb=0.0; top=tot=0
    for cpu in batches:
        ids=cpu.to(device); x=ids[:,:-1]; y=ids[:,1:]
        t=teacher(input_ids=x,use_cache=False,return_dict=True)
        s=student(input_ids=x,use_cache=False,return_dict=True)
        te=teacher.model.embed_tokens(x)
        se=student.model.embed_tokens(x)
        tce += float(causal_ce(t.logits,y)); sce += float(causal_ce(s.logits,y))
        kl += float(distill_kl(s.logits,t.logits))
        den=te.float().square().mean().clamp_min(1e-12)
        emb += float(((se.float()-te.float()).square().mean()/den).item())
        top += int((s.logits.argmax(-1)==t.logits.argmax(-1)).sum())
        tot += int(s.logits.shape[0]*s.logits.shape[1])
    n=max(len(batches),1)
    return {
        "teacher_ce":tce/n, "student_ce":sce/n, "ce_gap":(sce-tce)/n,
        "teacher_kl":kl/n, "embedding_relative_mse":emb/n,
        "top1_logit_agreement":top/max(tot,1),
    }


@torch.no_grad()
def generation_suite(student,teacher,tokenizer,device):
    prompts=[
        "Explain in two sentences why the sky is blue.",
        "What is 17 + 25? Give only the answer.",
        "Write one short sentence about Vienna.",
        "What is the capital of France? Give only the city.",
        "Explain in one sentence what an API is.",
        "Translate 'Good morning' into German. Give only the translation.",
        "Complete the pattern: 2, 4, 8, 16, ?",
        "In one sentence, explain why validation data should be separate from training data.",
    ]
    rows=[]
    for p in prompts:
        txt=tokenizer.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
        enc=tokenizer(txt,return_tensors="pt").to(device)
        def gen(m):
            o=m.generate(**enc,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            z=o[0,enc.input_ids.shape[1]:]
            return z.tolist(),tokenizer.decode(z,skip_special_tokens=True).strip()
        ti,tt=gen(teacher); si,st=gen(student)
        a,b=set(ti),set(si); jac=len(a&b)/max(len(a|b),1)
        rows.append({"prompt":p,"qwen":tt,"fly":st,"exact":ti==si,"jaccard":jac,"valid":bool(st) and "�" not in st})
    return rows


def unique_params(model):
    seen=set(); n=0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); n += p.numel()
    return n


def stage_ok(m,args):
    return (
        m["top1_logit_agreement"] >= args.min_top1
        and m["teacher_kl"] <= args.max_kl
        and m["ce_gap"] <= args.max_ce_gap
    )



def stage_violation_score(m,args):
    """Lower is better; zero means all preservation constraints are satisfied."""
    top1_v=max(args.min_top1-float(m["top1_logit_agreement"]),0.0)/max(1.0-args.min_top1,1e-6)
    kl_v=max(float(m["teacher_kl"])-args.max_kl,0.0)/max(args.max_kl,1e-8)
    ce_v=max(float(m["ce_gap"])-args.max_ce_gap,0.0)/max(args.max_ce_gap,1e-8)
    return top1_v + kl_v + ce_v


def stage_quality_key(m,args):
    """Lexicographic key: satisfy constraints first, then preserve Qwen, then improve CE."""
    safe=stage_ok(m,args)
    return (
        0 if safe else 1,
        stage_violation_score(m,args),
        float(m["teacher_kl"]),
        1.0-float(m["top1_logit_agreement"]),
        float(m["student_ce"]),
    )


def clone_core_state(core):
    return {k:v.detach().cpu().clone() for k,v in core.state_dict().items()}


def main():
    args=parse_args(); set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype=dtype_for(device)
    print("DEVICE",device,"dtype",dtype,flush=True)

    tokenizer=AutoTokenizer.from_pretrained(args.base_model,use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token

    teacher=AutoModelForCausalLM.from_pretrained(args.base_model,dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    student=AutoModelForCausalLM.from_pretrained(args.base_model,dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    original_unique=unique_params(student)
    vocab_params=int(student.model.embed_tokens.weight.numel())
    print("Original unique params:",original_unique,"vocab params:",vocab_params,flush=True)

    print("Factorizing vocabulary at rank",args.latent_dim,flush=True)
    fac=factorize_vocab_weight_v33(student.model.embed_tokens.weight,args.latent_dim)
    fac_stats={k:v for k,v in fac.items() if not isinstance(v,torch.Tensor)}
    print("FACTORIZATION",json.dumps(fac_stats,indent=2),flush=True)

    cfg=FlyEmbeddingV33Config(
        latent_dim=args.latent_dim,fly_nodes=args.fly_nodes,graph_steps=args.graph_steps,
        graph_mix=args.graph_mix,max_fly_scale=args.max_fly_scale
    )
    install_fly_embedding_v33(student,cfg,fac)
    trainable=freeze_qwen_train_compact_v33(student)

    train,val,data_source=load_blocks(tokenizer,args.run_mode,args.seq_len,args.seed)
    betas=[0.0,0.05,0.10,0.20,0.35,0.50,0.70,0.85,1.0]
    base_updates=args.stage_updates if args.run_mode=="quick" else args.stage_updates*2
    max_stage_updates=args.max_stage_updates if args.run_mode=="quick" else args.max_stage_updates*2
    hist=[]; stages=[]; global_step=0
    best_safe_beta=0.0; best_safe_state=None
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    # This state is what the next beta inherits. It is always the BEST validation
    # checkpoint from the preceding beta, never simply the last training step.
    previous_stage_best=clone_core_state(student.fly_embedding_v33_core)

    for beta in betas:
        student.fly_embedding_v33_core.load_state_dict(previous_stage_best,strict=True)
        set_progressive_beta_v33(student,beta)
        print("\n=== BETA",beta,"===",flush=True)

        # Fresh optimizer per beta: do not carry Adam momentum from a worse stage.
        trainable=freeze_qwen_train_compact_v33(student)
        opt=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=0.0)

        initial_m=probe(student,teacher,val,device)
        stage_best_m=dict(initial_m)
        stage_best_state=clone_core_state(student.fly_embedding_v33_core)
        stage_best_key=stage_quality_key(initial_m,args)
        stage_best_step=0
        probes_without_improvement=0
        stage_steps=0
        target_budget=0 if beta==0.0 else base_updates

        print("STAGE INITIAL",json.dumps({"beta":beta,**initial_m,"safe":stage_ok(initial_m,args)}),flush=True)

        while beta>0.0 and stage_steps < max_stage_updates:
            # Train at least the current target budget. If quality is still out
            # of bounds afterwards, automatically extend by extend_updates.
            while stage_steps < min(target_budget,max_stage_updates):
                global_step += 1
                stage_steps += 1
                ids=train[(global_step-1)%len(train)].to(device)
                x=ids[:,:-1]; y=ids[:,1:]
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    t=teacher(input_ids=x,use_cache=False,return_dict=True)
                    te=teacher.model.embed_tokens(x)
                st=student(input_ids=x,use_cache=False,return_dict=True)
                se=student.model.embed_tokens(x)
                ce=causal_ce(st.logits,y)
                kl=distill_kl(st.logits,t.logits)
                den=te.float().square().mean().clamp_min(1e-12)
                em=(se.float()-te.float()).square().mean()/den
                loss=0.15*ce + 0.55*kl + 0.30*em
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable,0.5)
                opt.step()

                if stage_steps==1 or stage_steps%args.probe_every==0 or stage_steps==target_budget:
                    train_row={
                        "global_step":global_step,"beta":beta,"stage_step":stage_steps,
                        "loss":float(loss.detach()),"ce":float(ce.detach()),
                        "kl":float(kl.detach()),"embedding_mse":float(em.detach()),
                        "fly_scale":float(student.fly_embedding_v33_core.fly_scale.detach()),
                    }
                    hist.append(train_row)
                    print(
                        f"beta={beta:.2f} step={stage_steps}/{target_budget} "
                        f"loss={train_row['loss']:.4f} ce={train_row['ce']:.4f} "
                        f"kl={train_row['kl']:.4f} emb={train_row['embedding_mse']:.4f}",
                        flush=True
                    )

                    vm=probe(student,teacher,val,device)
                    key=stage_quality_key(vm,args)
                    improved=key < stage_best_key
                    if improved:
                        stage_best_key=key
                        stage_best_m=dict(vm)
                        stage_best_state=clone_core_state(student.fly_embedding_v33_core)
                        stage_best_step=stage_steps
                        probes_without_improvement=0
                        torch.save(
                            {"core":stage_best_state,"config":cfg.to_dict(),"base_model":args.base_model,
                             "beta":beta,"stage_step":stage_best_step,"metrics":stage_best_m},
                            out/f"fly_v33_beta_{str(beta).replace('.','p')}_best.pt"
                        )
                        print("  ✓ NEW STAGE BEST",json.dumps({
                            "step":stage_best_step,
                            "safe":stage_ok(stage_best_m,args),
                            "violation_score":stage_violation_score(stage_best_m,args),
                            **stage_best_m,
                        }),flush=True)
                    else:
                        probes_without_improvement += 1

                    # Once the target constraints are met, retain the best safe
                    # checkpoint and stop this beta stage.
                    if stage_ok(stage_best_m,args):
                        print(f"  ✓ BETA {beta:.2f} fulfilled target at best step {stage_best_step}",flush=True)
                        break

            if stage_ok(stage_best_m,args):
                break

            if stage_steps >= max_stage_updates:
                print(
                    f"  ! BETA {beta:.2f} reached max-stage-updates={max_stage_updates}; "
                    f"restoring best step {stage_best_step}",
                    flush=True
                )
                break

            # Extend training when teacher/student gap is still too high.
            old_budget=target_budget
            target_budget=min(target_budget+args.extend_updates,max_stage_updates)
            print(
                f"  ↻ gap not fulfilled at beta={beta:.2f}; extending training "
                f"{old_budget} -> {target_budget} updates. "
                f"best violation={stage_violation_score(stage_best_m,args):.4f}",
                flush=True
            )

            # If many validation probes do not improve, lower LR before continuing
            # rather than blindly stepping with the same learning rate.
            if probes_without_improvement >= args.patience_probes:
                for group in opt.param_groups:
                    group["lr"] *= 0.5
                print(
                    f"  ↘ plateau detected; reducing LR to {opt.param_groups[0]['lr']:.3e}",
                    flush=True
                )
                probes_without_improvement=0

        # CRITICAL: use best weights found at this beta for the next beta.
        student.fly_embedding_v33_core.load_state_dict(stage_best_state,strict=True)
        previous_stage_best=clone_core_state(student.fly_embedding_v33_core)
        set_progressive_beta_v33(student,beta)
        restored_m=probe(student,teacher,val,device)
        ok=stage_ok(restored_m,args)
        stage_row={
            "beta":beta,
            "initial_student_ce":initial_m["student_ce"],
            "best_step":stage_best_step,
            "trained_steps":stage_steps,
            **restored_m,
            "safe":ok,
            "violation_score":stage_violation_score(restored_m,args),
        }
        stages.append(stage_row)
        print("STAGE RESTORED BEST",json.dumps(stage_row),flush=True)

        if ok:
            best_safe_beta=beta
            best_safe_state=clone_core_state(student.fly_embedding_v33_core)

    # Always inspect the fully compact beta=1 candidate.
    set_progressive_beta_v33(student,1.0)
    final_beta1_probe=probe(student,teacher,val,device)
    final_gen=generation_suite(student,teacher,tokenizer,device)
    beta1_quality=stage_ok(final_beta1_probe,args) and sum(x["valid"] for x in final_gen)==len(final_gen)

    # Save beta=1 compact candidate before optional safe rollback.
    beta1_state={k:v.detach().cpu() for k,v in student.fly_embedding_v33_core.state_dict().items()}
    torch.save({"core":beta1_state,"config":cfg.to_dict(),"base_model":args.base_model,"beta":1.0},out/"fly_v33_beta1_compact.pt")

    export_compact_only_v33(student)
    compact_unique=unique_params(student)
    reduction_pct=100.0*(1.0-compact_unique/max(original_unique,1))
    torch.save({
        "model_state_dict":{k:v.detach().cpu() for k,v in student.state_dict().items()},
        "config":cfg.to_dict(),"base_model":args.base_model,
        "compact_only":True,"beta":1.0,
    },out/"fly_v33_compact_only_model.pt")

    if best_safe_state is not None:
        torch.save({"core":best_safe_state,"config":cfg.to_dict(),"base_model":args.base_model,"beta":best_safe_beta},out/"fly_v33_best_safe_core.pt")

    pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False)
    pd.DataFrame(stages).to_csv(out/"stage_validation.csv",index=False)
    (out/"generation_beta1.json").write_text(json.dumps(final_gen,indent=2),encoding="utf-8")

    report={
        "architecture":"Qwen3.5-0.8B + FlyEmbedding-v3.3 progressive full vocabulary replacement",
        "data_source":data_source,
        "factorization":fac_stats,
        "schedule":betas,
        "adaptive_stage_training":{"base_updates":base_updates,"extend_updates":args.extend_updates,"max_stage_updates":max_stage_updates,"probe_every":args.probe_every,"patience_probes":args.patience_probes},
        "best_safe_beta":best_safe_beta,
        "beta1_probe":final_beta1_probe,
        "beta1_quality_gate_passed":beta1_quality,
        "beta1_generation":final_gen,
        "original_unique_params":original_unique,
        "compact_only_unique_params":compact_unique,
        "actual_total_param_reduction_pct":reduction_pct,
        "original_vocab_params":vocab_params,
        "compact_vocab_stats":student.fly_embedding_v33_core.stats(vocab_params),
        "target_half_size_reached":bool(compact_unique <= 0.5*original_unique),
        "important_note":"Embedding replacement alone may not halve total model parameters; report uses actual unique parameter counts.",
        "config":vars(args),
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FINAL",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
