"""Controlled Qwen3.5-0.8B Delta-only and Full-only attention experiments.

Designed for Colab GPU. Always writes a statistics ZIP, including on error.
Uses Qwen3.5 no-cache text-only causal attention, never CeNN.
"""
import copy
import gc
import json
import math
import random
import re
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tinycenn_lm import qwen35_isolated as iso
from tinycenn_lm import qwen35_window_anchor_v3 as wa


def run(mode="delta", seed=20261019, delta_window=128, full_window=512,
        anchors=16, chunk_size=64, max_context=192,
        train_n=96, dev_n=48, holdout_n=96,
        train_delta=True, train_steps=120, train_lr=0.0005,
        save_model=True, download_stats=True, download_model=False,
        upload_if_good=True, hf_private=True, hf_name=None,
        repeats=10):
    if mode not in ("delta", "full"):
        raise ValueError("Choose mode='delta' or mode='full'")
    root=Path("/content/Qwen35_V4_"+mode+"_Only")
    out=root/"statistics"
    checkpoint=root/"model"
    out.mkdir(parents=True,exist_ok=True)
    checkpoint.mkdir(parents=True,exist_ok=True)
    stats_zip=root/("Qwen35_0p8B_V4_"+mode.upper()+"_Only_Statistics.zip")
    model_zip=root/("Qwen35_0p8B_V4_"+mode.upper()+"_Only_Model.zip")
    chosen_hf_name=hf_name or ("qwen35-0p8b-"+mode+"-only-window-anchor-v4")
    settings={"delta_window":int(delta_window),"full_window":int(full_window),
              "anchors":int(anchors),"chunk_size":int(chunk_size)}
    state={"mode":mode,"source":"Qwen/Qwen3.5-0.8B","settings":settings,
           "seed":seed,"created_utc":datetime.now(timezone.utc).isoformat(),
           "model_saved":False,"reload_verified":False,"qualified":False}
    def write(filename,obj):
        (out/filename).write_text(
            json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False),
            encoding="utf-8")
    def archives():
        reports=sorted(p for p in out.rglob("*") if p.is_file())
        with zipfile.ZipFile(stats_zip,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for p in reports:z.write(p,"statistics/"+p.relative_to(out).as_posix())
        if state["model_saved"] and (checkpoint/"model.safetensors").exists():
            with zipfile.ZipFile(model_zip,"w",allowZip64=True) as z:
                for p in sorted(p for p in checkpoint.rglob("*") if p.is_file()):
                    ct=zipfile.ZIP_STORED if p.suffix==".safetensors" else zipfile.ZIP_DEFLATED
                    z.write(p,"model/"+p.relative_to(checkpoint).as_posix(),compress_type=ct)
                for p in reports:z.write(p,"statistics/"+p.relative_to(out).as_posix())
        for path in (stats_zip,model_zip):
            if not path.exists():continue
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
            print("VERIFIED ZIP:",path,"MiB",round(path.stat().st_size/2**20,2),flush=True)
        from IPython.display import display,FileLink
        display(FileLink(str(stats_zip)))
        if model_zip.exists():display(FileLink(str(model_zip)))
        try:
            from google.colab import files
            if download_stats:files.download(str(stats_zip))
            if download_model and model_zip.exists():files.download(str(model_zip))
        except (ImportError,RuntimeError,AttributeError,Exception) as exc:
            print("Browser auto-download unavailable; use the FileLinks:",repr(exc))
    try:
        assert torch.cuda.is_available(),"Enable Colab GPU in Runtime > Change runtime type"
        assert 0<=anchors<=delta_window and delta_window>=16
        assert full_window>=max_context and chunk_size>=8
        assert dev_n>=12 and holdout_n>=24 and train_n>=16
        from datasets import load_dataset
        from huggingface_hub import HfApi,get_token,login
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device=torch.device("cuda",torch.cuda.current_device())
        print("GPU",torch.cuda.get_device_name(device),
              "MODE",mode,flush=True)
        data_status="official_wikitext"
        def select_text(split,needed):
            dataset=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
            seen=set();txt=[]
            for row in dataset:
                val=" ".join(str(row["text"]).split())
                if 85<=len(val)<=760 and not val.startswith("=") and val not in seen:
                    txt.append(val);seen.add(val)
            assert len(txt)>=needed,(split,len(txt),needed)
            random.Random(seed+{"train":1,"validation":2,"test":3}[split]).shuffle(txt)
            return txt[:needed]
        try:
            texts_train=select_text("train",train_n)
            texts_dev=select_text("validation",dev_n)
            texts_hold=select_text("test",holdout_n)
        except Exception as exc:
            print("Dataset fallback - INVALID for publication:",repr(exc))
            data_status="synthetic_debug"
            rows=[("A research report number "+str(i)+" compared the speed and quality "
                   "of different machine learning models with controlled experiments.")
                  for i in range(train_n+dev_n+holdout_n)]
            random.Random(seed).shuffle(rows)
            texts_train=rows[:train_n]
            texts_dev=rows[train_n:train_n+dev_n]
            texts_hold=rows[train_n+dev_n:]
        assert set(texts_train).isdisjoint(texts_dev)
        assert set(texts_train).isdisjoint(texts_hold)
        assert set(texts_dev).isdisjoint(texts_hold)
        state["dataset_status"]=data_status
        print("Data:",data_status,"train/dev/holdout",
              len(texts_train),len(texts_dev),len(texts_hold),flush=True)
        teacher,tokenizer,source_root=wa.load_official_text(
            "Qwen/Qwen3.5-0.8B",dtype=torch.float16)
        sha=iso.original_ffn_digest(teacher)
        def make_row(text):
            tokens=tokenizer(text,add_special_tokens=False)["input_ids"][:max_context+1]
            assert len(tokens)>=8
            return {"tokens":torch.tensor([tokens[:-1]],dtype=torch.long),
                    "gold":int(tokens[-1])}
        train_rows=[make_row(t) for t in texts_train]
        dev_rows=[make_row(t) for t in texts_dev]
        teacher.to(device).eval().requires_grad_(False)
        @torch.inference_mode()
        def reference(rows):
            for j,row in enumerate(rows):
                with torch.autocast("cuda",dtype=torch.float16):
                    a=teacher(input_ids=row["tokens"].to(device),
                              logits_to_keep=1,use_cache=False).logits[:,-1,:]
                row["teacher_logits"]=a.float().cpu()
                if (j+1)%24==0:print("Teacher cache",j+1,"/",len(rows),flush=True)
        reference(train_rows);reference(dev_rows)
        teacher.cpu()
        gc.collect();torch.cuda.empty_cache()
        student=copy.deepcopy(teacher).to(device).eval().requires_grad_(False)
        indices=iso.layer_ids(student,mode)
        assert len(indices)==(18 if mode=="delta" else 6)
        field="linear_attn" if mode=="delta" else "self_attn"
        originals={i:getattr(student.model.layers[i],field) for i in indices}
        @torch.inference_mode()
        def evaluate(rows):
            student.eval()
            agreement=0;js=kl=nll=teacher_nll=0.0
            for rec in rows:
                with torch.autocast("cuda",dtype=torch.float16):
                    b=student(input_ids=rec["tokens"].to(device),
                              use_cache=False,logits_to_keep=1).logits[:,-1].float()
                a=rec["teacher_logits"].to(device).float()
                pa=a.softmax(-1);pb=b.softmax(-1);mid=0.5*(pa+pb)
                js+=float(0.5*F.kl_div(mid.clamp_min(1e-12).log(),pa,reduction="sum")
                          +0.5*F.kl_div(mid.clamp_min(1e-12).log(),pb,reduction="sum"))
                kl+=float(F.kl_div(b.log_softmax(-1),pa,reduction="sum"))
                agreement+=int(a.argmax(-1).item()==b.argmax(-1).item())
                gold=torch.tensor([rec["gold"]],device=device)
                nll+=float(F.cross_entropy(b,gold))
                teacher_nll+=float(F.cross_entropy(a,gold))
            n=len(rows)
            return {"n":n,"teacher_agreement":agreement/n,
                    "mean_js":js/n,"mean_kl":kl/n,
                    "student_sampled_next_token_nll":nll/n,
                    "teacher_sampled_next_token_nll":teacher_nll/n,
                    "sampled_next_token_ppl_ratio":math.exp(max(-30,min(30,(nll-teacher_nll)/n)))}
        def configure(selected):
            selected=set(selected)
            for i in indices:
                setattr(student.model.layers[i],field,originals[i])
            for i in sorted(selected):
                iso.substitute(student,i,mode,**settings)
            student.to(device).eval().requires_grad_(False)
            topo=iso.describe(student)
            assert topo["replaced_delta"]==(len(selected) if mode=="delta" else 0)
            assert topo["replaced_full"]==(len(selected) if mode=="full" else 0)
            assert iso.original_ffn_digest(student)==sha
            return topo
        baseline=evaluate(dev_rows)
        assert baseline["teacher_agreement"]>=0.999,baseline
        write("baseline.json",{"dev":baseline,"ffn_norm_sha256":sha})
        sensitivity=[]
        for count,i in enumerate(indices,1):
            configure([i])
            metrics=evaluate(dev_rows)
            sensitivity.append({"layer":i,"dev":metrics})
            print(f"Sensitivity {count}/{len(indices)} layer {i} "
                  f"agreement {metrics['teacher_agreement']:.1%} JS {metrics['mean_js']:.5f}",flush=True)
        sensitivity.sort(key=lambda r:(-r["dev"]["teacher_agreement"],r["dev"]["mean_js"]))
        ranking=[r["layer"] for r in sensitivity]
        write("layer_sensitivity.json",{"mode":mode,"ranked_layers":sensitivity})
        milestone_sizes=[1,3,6,12,18] if mode=="delta" else [1,3,6]
        progressive=[]
        for count in milestone_sizes:
            selected=ranking[:count]
            topology=configure(selected)
            quality=evaluate(dev_rows)
            progressive.append({"n_replaced":count,"indices":selected,
                                "topology":topology,"dev":quality})
            print("Progressive",count,quality["teacher_agreement"],
                  quality["mean_js"],flush=True)
        selected_stage=(max(progressive,key=lambda r:(
            r["dev"]["teacher_agreement"],-r["dev"]["mean_js"]))
                        if mode=="delta" else progressive[-1])
        selected=selected_stage["indices"]
        topology=configure(selected)
        initial_dev=evaluate(dev_rows)
        write("progressive_stages.json",{"native_baseline":baseline,
            "stages":progressive,"selected":selected_stage})
        best_step=0
        finetune_history=[]
        if mode=="delta" and train_delta and train_steps>0:
            trainable=[]
            for i in selected:
                m=student.model.layers[i].linear_attn
                for name in ("log_temperature","log_decay","direct_mix_logit",
                             "output_gain","output_bias"):
                    p=nn.Parameter(getattr(m,name).detach().float().to(device))
                    setattr(m,name,p)
                    trainable.append(p)
            assert all(p.dtype==torch.float32 for p in trainable)
            student.eval()
            optimizer=torch.optim.AdamW(trainable,lr=train_lr,weight_decay=0.0)
            scaler=torch.amp.GradScaler("cuda")
            best=[p.detach().cpu().clone() for p in trainable]
            best_quality=initial_dev
            rng=random.Random(seed+810)
            for step in range(1,train_steps+1):
                rec=train_rows[rng.randrange(len(train_rows))]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=student(input_ids=rec["tokens"].to(device),
                                   logits_to_keep=1,use_cache=False).logits[:,-1,:].float()
                    ref=rec["teacher_logits"].to(device).float()
                    tau=2.0
                    loss=tau*tau*F.kl_div(
                        F.log_softmax(logits/tau,-1),F.softmax(ref/tau,-1),
                        reduction="batchmean")
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Nonfinite KD loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable,0.5)
                scaler.step(optimizer);scaler.update()
                if step%20==0 or step==train_steps:
                    quality=evaluate(dev_rows)
                    if (quality["teacher_agreement"],-quality["mean_js"]) > (
                        best_quality["teacher_agreement"],-best_quality["mean_js"]):
                        best_quality=quality
                        best_step=step
                        best=[p.detach().cpu().clone() for p in trainable]
                    finetune_history.append({"step":step,"loss":float(loss.detach()),
                                             "dev":quality,"best_step":best_step})
                    print("KD step",step,"agreement",quality["teacher_agreement"],flush=True)
            with torch.no_grad():
                for p,snap in zip(trainable,best):p.copy_(snap.to(device))
            student.half().eval().requires_grad_(False)
            del optimizer,scaler,best
            gc.collect();torch.cuda.empty_cache()
        final_dev=evaluate(dev_rows)
        assert iso.original_ffn_digest(student)==sha
        write("finetune.json",{"enabled":mode=="delta" and train_delta,
            "selected_best_step":best_step,"initial_dev":initial_dev,
            "final_dev":final_dev,"history":finetune_history})
        # SEALED holdout is touched only after all DEV decisions.
        holdout_rows=[make_row(t) for t in texts_hold]
        teacher.to(device).eval()
        reference(holdout_rows)
        holdout=evaluate(holdout_rows)
        @torch.inference_mode()
        def latency(model,tokens,reps):
            model.eval();x=tokens.to(device)
            for _ in range(3):
                with torch.autocast("cuda",dtype=torch.float16):
                    model(input_ids=x,use_cache=False,logits_to_keep=1)
            torch.cuda.synchronize()
            measurements=[]
            for _ in range(reps):
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast("cuda",dtype=torch.float16):
                    model(input_ids=x,use_cache=False,logits_to_keep=1)
                end.record();end.synchronize()
                measurements.append(float(start.elapsed_time(end)))
            return {"median_ms":float(np.median(measurements)),
                    "p95_ms":float(np.percentile(measurements,95)),
                    "repeats":reps}
        bench=[]
        for count in (64,128,192):
            tokens=holdout_rows[0]["tokens"]
            if tokens.size(1)<count:
                tokens=tokens.repeat(1,math.ceil(count/tokens.size(1)))[:,:count]
            else:tokens=tokens[:,:count]
            original=latency(teacher,tokens,repeats)
            candidate=latency(student,tokens,repeats)
            bench.append({"tokens":count,"batch":1,"dtype":"float16","use_cache":False,
                          "teacher":original,"student":candidate,
                          "speedup":original["median_ms"]/max(1e-9,candidate["median_ms"])})
        speed=float(np.median([b["speedup"] for b in bench]))
        def chat_prompt(prompt):
            msgs=[{"role":"user","content":prompt}]
            try:
                try:
                    a=tokenizer.apply_chat_template(
                        msgs,tokenize=True,add_generation_prompt=True,
                        return_tensors="pt",enable_thinking=False)
                except TypeError:
                    a=tokenizer.apply_chat_template(
                        msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt")
                if isinstance(a,dict):a=a["input_ids"]
                if hasattr(a,"input_ids"):a=a.input_ids
                if not isinstance(a,torch.Tensor):a=torch.as_tensor(a,dtype=torch.long)
            except Exception:
                a=tokenizer("User: "+prompt+"\nAssistant:",return_tensors="pt")["input_ids"]
            return a[None] if a.ndim==1 else a
        @torch.inference_mode()
        def generate(model,prompt,n=24):
            x=chat_prompt(prompt).to(device)
            origin=x.shape[-1]
            for _ in range(n):
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=model(input_ids=x,use_cache=False,logits_to_keep=1).logits[:,-1,:]
                nxt=logits.argmax(-1,keepdim=True)
                x=torch.cat((x,nxt),dim=-1)
                if int(nxt.item())==tokenizer.eos_token_id:break
            return tokenizer.decode(x[0,origin:],skip_special_tokens=True)
        prompts=["Explain why the sky is blue in two sentences.",
                 "What is 17 multiplied by 23?",
                 "Write a friendly greeting in German.",
                 "Explain two differences between memory and attention.",
                 "Give two practical safety tips near busy roads."]
        chats=[]
        for prompt in prompts:
            row={"prompt":prompt}
            try:
                row["teacher"]=generate(teacher,prompt)
                row["student"]=generate(student,prompt)
            except Exception as exc:
                row["error"]=repr(exc)
            words=re.findall(r"\w+",row.get("student","").lower())
            row["usable"]=len(words)>=4 and len(set(words))/max(1,len(words))>=0.45
            chats.append(row)
            print("CHAT",prompt,"STUDENT:",row.get("student",row.get("error")),flush=True)
        chat_pass=sum(c["usable"] for c in chats)
        teacher.cpu()
        gc.collect();torch.cuda.empty_cache()
        write("holdout.json",holdout)
        write("benchmark.json",bench)
        write("chat_examples.json",chats)
        qualified=(data_status=="official_wikitext"
                   and holdout["teacher_agreement"]>=(0.97 if mode=="full" else 0.95)
                   and holdout["mean_js"]<=(0.01 if mode=="full" else 0.04)
                   and holdout["sampled_next_token_ppl_ratio"]<=1.25
                   and speed>=0.95 and chat_pass>=4)
        state.update({"source_sha256_ffn":sha,"selected_indices":selected,
                      "topology":topology,"baseline_dev":baseline,
                      "post_selection_dev":final_dev,"holdout":holdout,
                      "latency":bench,"median_speedup":speed,
                      "chat_usable":chat_pass,"best_training_step":best_step,
                      "qualified":qualified})
        if save_model:
            try:
                student.half().cpu().eval()
                manifest={
                    "mode":mode,"model_dtype":"float16",
                    "attention_settings":settings,"selected_indices":selected,
                    "source_model":"Qwen/Qwen3.5-0.8B",
                    "original_ffn_sha256":sha,"dataset_status":data_status,
                    "dev":final_dev,"holdout":holdout,
                    "median_speedup":speed,
                    "no_cache":True,"text_only":True,"qualified":qualified
                }
                iso.save_isolated(student,tokenizer,checkpoint,manifest)
                state["model_saved"]=True
                loaded,loaded_tokenizer,loaded_topology=iso.load_isolated(
                    checkpoint,device=device)
                assert loaded_topology==topology
                assert iso.original_ffn_digest(loaded)==sha
                student.to(device)
                parity=[]
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                    for row in dev_rows[:3]:
                        x=row["tokens"].to(device)
                        a=student(input_ids=x,use_cache=False,logits_to_keep=1).logits.float()
                        b=loaded(input_ids=x,use_cache=False,logits_to_keep=1).logits.float()
                        parity.append({"maximum_abs_logit_delta":float((a-b).abs().max()),
                                       "argmax_same":bool(torch.equal(a.argmax(-1),b.argmax(-1)))})
                assert all(p["maximum_abs_logit_delta"]<=0.05
                           and p["argmax_same"] for p in parity),parity
                state["reload_verified"]=True
                state["reload_parity"]=parity
                del loaded;student.cpu();gc.collect();torch.cuda.empty_cache()
            except Exception as exc:
                state["reload_error"]=repr(exc)
                qualified=False
                print("Reload validation failed:",repr(exc),flush=True)
        qualified=bool(qualified and state["reload_verified"])
        state["qualified"]=qualified
        write("final_report.json",state)
        if state["model_saved"]:
            (checkpoint/"isolated_manifest.json").write_text(
                json.dumps({
                    "mode":mode,"model_dtype":"float16",
                    "attention_settings":settings,"selected_indices":selected,
                    "source_model":"Qwen/Qwen3.5-0.8B","qualified":qualified,
                    "original_ffn_sha256":sha,"text_only":True,"no_cache":True
                },indent=2))
            (checkpoint/"README.md").write_text(
                "# Qwen3.5-0.8B "+mode+"-only window-anchor V4\n\n"
                "Experimental text-only, no-KV-cache model. Load with "
                "tinycenn_lm.qwen35_isolated.load_isolated, not AutoModel.\n"
                "Replaced indices: "+str(selected)+
                ". This is not an all-24 replacement.\n"
                "Sealed holdout teacher agreement: "+str(holdout["teacher_agreement"])+".\n"
                "Qualified for publication: "+str(qualified)+".\n")
        upload={"qualified":qualified,"attempted":False,"uploaded":False,
                "reason":"Fidelity / latency / chat / reload gate not met"}
        if qualified and upload_if_good:
            try:
                token=get_token()
                if not token:login();token=get_token()
                assert token,"Hugging Face write token required"
                api=HfApi(token=token)
                user=api.whoami()["name"]
                repo=user+"/"+chosen_hf_name
                upload["attempted"]=True
                api.create_repo(repo,private=hf_private,repo_type="model",exist_ok=True)
                api.upload_folder(repo_id=repo,repo_type="model",
                                  folder_path=str(checkpoint),
                                  commit_message="Verified Qwen3.5 "+mode+"-only controlled substitution")
                required={"model.safetensors","isolated_manifest.json",
                          "qwen35_isolated.py","qwen35_window_anchor_v3.py"}
                assert required<=set(api.list_repo_files(repo,repo_type="model"))
                upload.update({"uploaded":True,"url":"https://huggingface.co/"+repo,
                               "reason":"All quality gates passed"})
            except Exception as exc:
                upload["reason"]=repr(exc)
                print("HF upload failure - local model retained:",repr(exc))
        write("hf_status.json",upload)
        print("FINAL",mode,"teacher agreement",holdout["teacher_agreement"],
              "speedup",speed,"HF uploaded",upload["uploaded"],flush=True)
    except Exception:
        err=traceback.format_exc()
        state["exception"]=err
        write("error.json",{"traceback":err})
        print("EXPERIMENT FAILED. Diagnostic ZIP will still download.\n",err,flush=True)
    finally:
        write("run_status.json",state)
        archives()
    return state
