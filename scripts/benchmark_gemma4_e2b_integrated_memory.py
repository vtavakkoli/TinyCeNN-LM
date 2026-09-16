#!/usr/bin/env python3
"""Gemma 4 E2B: replace independent global full-attention layers with TinyCeNN memory."""
import argparse
import contextlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, repeat_kv

from scripts import benchmark_smollm2_integrated_memory as base
from scripts.benchmark_cenn_research_layers import fit_transfer, write_csv, write_json
from tinycenn_lm.gemma4_integrated_memory import (
    adapter_payload, build_student, full_attention_layers, greedy_generate,
    inference_mode, native_dtype, new_cache, replaceable_full_attention_layers,
    restore_student, text_backbone, text_config, wrappers,
)
from tinycenn_lm.optimized_memory import ridge_calibrate

base.native_dtype = native_dtype
base.build_student = build_student
base.wrappers = wrappers
base.inference_mode = inference_mode
base.adapter_payload = adapter_payload
base.restore_student = restore_student
base.new_cache = new_cache
base.greedy_generate = greedy_generate


def project_qkv(attention, hidden, position_embeddings):
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    cos, sin = position_embeddings
    q = attention.q_proj(hidden).view(hidden_shape)
    q = attention.q_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    k = attention.k_proj(hidden).view(hidden_shape)
    v = attention.v_proj(hidden).view(hidden_shape) if attention.v_proj is not None else k
    k = attention.k_norm(k)
    k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    v = attention.v_norm(v).transpose(1, 2)
    q_core = q.float() * (float(attention.scaling) * math.sqrt(attention.head_dim))
    return q.float(), q_core, k.float(), v.float()


@torch.no_grad()
def capture_samples(model, blocks, layer_indices, context, device):
    samples = {index: [] for index in layer_indices}
    tm = text_backbone(model)
    replaceable = set(replaceable_full_attention_layers(model))
    for number, block in enumerate(blocks, 1):
        ids = block[:context].unsqueeze(0).to(device)
        result = tm(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
        positions = torch.arange(context, device=device)[None]
        for index in layer_indices:
            layer = tm.layers[index]
            attn = layer.self_attn
            if index not in replaceable:
                raise ValueError(f"Layer {index} is not independently replaceable in Gemma 4 V1")
            hidden = layer.input_layernorm(result.hidden_states[index])
            rope = tm.rotary_emb(hidden, positions, attn.layer_type)
            q_native, q_core, k, v = project_qkv(attn, hidden, rope)
            target = F.scaled_dot_product_attention(
                q_native,
                repeat_kv(k, attn.num_key_value_groups),
                repeat_kv(v, attn.num_key_value_groups),
                is_causal=True,
                scale=float(attn.scaling),
            ).float()
            samples[index].append(tuple(x.detach().cpu() for x in (q_core, k, v, target)))
        if number % 4 == 0 or number == len(blocks):
            print(f"captured {number}/{len(blocks)} at context {context}", flush=True)
    return samples


def joint_loss(student_logits, teacher_logits, targets, kl_weight=1., temperature=1., chunk=8):
    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    labels = targets.reshape(-1)
    ce = s.new_zeros((), dtype=torch.float32)
    kl = s.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), chunk):
        end = min(start + chunk, len(labels))
        logits = s[start:end].float()
        ce += F.cross_entropy(logits, labels[start:end], reduction="sum") / len(labels)
        logp = F.log_softmax(logits / temperature, -1)
        with torch.no_grad():
            teacher_logp = F.log_softmax(t[start:end].float() / temperature, -1)
        kl += F.kl_div(logp, teacher_logp, log_target=True, reduction="sum") * (
            temperature ** 2 / len(labels)
        )
    return ce + kl_weight * kl, ce.detach(), kl.detach()


base.joint_loss = joint_loss


def _nmse(reference, candidate):
    reference = reference.float(); candidate = candidate.float()
    return float((candidate-reference).square().mean()/reference.square().mean().clamp_min(1e-8))


def _top1(reference, candidate):
    return float((reference.argmax(-1) == candidate.argmax(-1)).float().mean())


@torch.no_grad()
def _full_cached(model, ids, cache, split, precision, cenn):
    manager = inference_mode(model, precision) if cenn else contextlib.nullcontext(model)
    with manager:
        full = model(input_ids=ids, use_cache=False).logits
        first = model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True).logits
        parts = [first]
        for i in range(split, ids.shape[1]):
            parts.append(model(input_ids=ids[:, i:i+1], past_key_values=cache, use_cache=True).logits)
        cached = torch.cat(parts, 1)
    return full, cached


@torch.no_grad()
def gemma4_cache_equivalence(teacher, model, block, device, precision, block_size):
    length = min(len(block)-1, 2*block_size+5)
    ids = block[:length][None].to(device)
    split = min(block_size+1, length-1)
    tf, tc = _full_cached(teacher, ids, DynamicCache(config=text_config(teacher)), split, precision, False)
    cf, cc = _full_cached(model, ids, new_cache(model), split, precision, True)
    teacher_nmse, candidate_nmse = _nmse(tf, tc), _nmse(cf, cc)
    teacher_top1, candidate_top1 = _top1(tf, tc), _top1(cf, cc)
    teacher_mismatch = int((tf.argmax(-1) != tc.argmax(-1)).sum().item())
    candidate_mismatch = int((cf.argmax(-1) != cc.argmax(-1)).sum().item())
    nmse_limit = max(2e-3, 4.0*teacher_nmse + 5e-4)
    allowed_mismatch = min(length, teacher_mismatch + max(2, math.ceil(.05*length)))
    metrics = {
        "cached_logits_nmse": candidate_nmse,
        "teacher_cached_logits_nmse": teacher_nmse,
        "cache_nmse_excess": candidate_nmse-teacher_nmse,
        "cache_equivalence_nmse_limit": nmse_limit,
        "cached_top1_agreement": candidate_top1,
        "teacher_cached_top1_agreement": teacher_top1,
        "candidate_top1_mismatches": candidate_mismatch,
        "teacher_top1_mismatches": teacher_mismatch,
        "allowed_top1_mismatches": allowed_mismatch,
        "cache_equivalence_top1_floor": .95,
        "cache_test_tokens": length,
    }
    metrics["cache_equivalence_passed"] = bool(
        candidate_nmse <= nmse_limit and candidate_mismatch <= allowed_mismatch and candidate_top1 >= .95
    )
    print("cache_equivalence_gemma4:", json.dumps(metrics), flush=True)
    if not all(math.isfinite(metrics[k]) for k in (
        "cached_logits_nmse", "teacher_cached_logits_nmse", "cached_top1_agreement", "teacher_cached_top1_agreement"
    )):
        raise RuntimeError(f"Non-finite Gemma 4 cache metrics: {metrics}")
    if candidate_nmse > nmse_limit:
        raise RuntimeError(f"Gemma 4 cache NMSE too high: {metrics}")
    if candidate_mismatch > allowed_mismatch or candidate_top1 < .95:
        raise RuntimeError(f"Gemma 4 cache top-1 drift too high: {metrics}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="google/gemma-4-E2B")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--conservative-layers", default="4")
    p.add_argument("--expanded-layers", default="4,9")
    p.add_argument("--train-contexts", default="128,256")
    p.add_argument("--test-contexts", default="128,256,512,1024")
    for name, value in [
        ("train-documents",24),("validation-documents",6),("test-documents",8),("warm-documents",4),
        ("warm-steps",30),("joint-steps",60),("eval-every",15),("features",64),("block-size",32),
        ("sinks",4),("seed",2041),("dataset-seed",9341),("decode-tokens",24),
        ("timing-documents",1),("timing-repeats",1),("loss-chunk",4),
    ]:
        p.add_argument("--"+name, type=int, default=value)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--nll-margin", type=float, default=.03)
    p.add_argument("--exclude-manifest", action="append", default=[])
    p.add_argument("--output-dir", default="result/gemma4-e2b-integrated-v1")
    a = p.parse_args()
    for key in ("conservative_layers","expanded_layers","train_contexts","test_contexts"):
        setattr(a,key,list(dict.fromkeys(int(x) for x in getattr(a,key).split(","))))
    if min(a.train_contexts+a.test_contexts) <= 2*a.block_size+a.sinks:
        p.error("Every context must exercise compressed history")
    if a.features < 2 or a.features % 2 or a.loss_chunk < 1:
        p.error("features must be positive/even and loss-chunk positive")
    return a


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, Gemma4ForCausalLM
    import transformers

    args = parse_args(); out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory")
    (out/"checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = native_dtype(args.device); args.precision = str(dtype).removeprefix("torch.")
    if args.device.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = False

    api = HfApi(); args.model_sha = api.model_info(args.base_model, revision=args.model_revision).sha
    data_sha = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    teacher = Gemma4ForCausalLM.from_pretrained(
        args.base_model, revision=args.model_sha, dtype=dtype, attn_implementation="sdpa"
    ).to(args.device).eval().requires_grad_(False)
    cfg = text_config(teacher)
    available_full = full_attention_layers(teacher)
    safe_full = replaceable_full_attention_layers(teacher)
    requested = sorted(set(args.conservative_layers + args.expanded_layers))
    if cfg.model_type != "gemma4_text" or not set(requested).issubset(safe_full):
        raise ValueError(f"Gemma4 full={available_full}; safely replaceable={safe_full}; requested={requested}")
    if max(args.train_contexts+args.test_contexts)+args.decode_tokens > cfg.max_position_embeddings:
        raise ValueError("Requested context plus decode exceeds position limit")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_sha)
    exclusions = base.read_exclusions(args.exclude_manifest) if args.exclude_manifest else set()
    stream = load_dataset(args.dataset,args.dataset_config,revision=data_sha,split="train",streaming=True).shuffle(
        seed=args.dataset_seed,buffer_size=2048)
    blocks, hashes = base.collect_documents(
        stream,tokenizer,{"train":args.train_documents,"validation":args.validation_documents,"test":args.test_documents},
        max(args.train_contexts+args.test_contexts)+args.decode_tokens,exclusions)
    manifest = {
        "status":"running","experiment":"gemma4-e2b-cenn-integrated-v1",
        "architecture":{"model_type":cfg.model_type,"num_hidden_layers":cfg.num_hidden_layers,
            "full_attention_layers":available_full,"replaceable_full_attention_layers":safe_full,
            "layer_types":list(cfg.layer_types),"sliding_window":cfg.sliding_window,
            "num_kv_shared_layers":cfg.num_kv_shared_layers,"num_attention_heads":cfg.num_attention_heads},
        "args":{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items()},
        "model_revision":args.model_sha,"dataset_revision":data_sha,"document_hashes":hashes,
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__,
        "native_dtype":args.precision,"gpu":torch.cuda.get_device_name(0) if args.device.type=="cuda" else None,
        "split_protocol":"test40..49; val50..59; train60..99; supplied prior hashes excluded",
        "kv_sharing_note":"Only independent non-shared global layers are replaceable in V1; shared-KV producer/consumers stay native.",
    }
    write_json(out/"manifest.json",manifest); torch.save(blocks,out/"token_blocks.pt")

    indices = sorted(set(requested)); warm_context = min(args.train_contexts)
    train = capture_samples(teacher,blocks["train"][:args.warm_documents],indices,warm_context,args.device)
    valid = capture_samples(teacher,blocks["validation"],indices,warm_context,args.device)
    records, history = [], []
    for name,variant,layers,control in base.configurations(args):
        torch.manual_seed(args.seed); model = build_student(teacher,layers,variant,args.features,args.block_size,args.sinks)
        started = time.perf_counter()
        if variant != "sink_window":
            for adapter in wrappers(model):
                ridge_calibrate(adapter.core,train[adapter.layer_idx],args.device)
                warmargs = SimpleNamespace(device=args.device,steps=args.warm_steps,
                    eval_every=max(1,min(args.eval_every,args.warm_steps)),lr=.002,seed=args.seed)
                fit_transfer(adapter.core,train[adapter.layer_idx],valid[adapter.layer_idx],warmargs,
                    out/"checkpoints"/f"{name}_warm_{adapter.layer_idx}.pt",f"{name}_layer{adapter.layer_idx}")
        before,_ = base.validation_score(model,blocks["validation"],args.train_contexts,args.device,args.precision)
        base.save_adapter(out/"checkpoints"/f"{name}_before_joint.pt",model,
            {"candidate":name,"stage":"before_joint","model_revision":args.model_sha})
        fit_history,best = base.train_joint(teacher,model,blocks,args,out,name); history.extend(fit_history)
        checkpoint = f"checkpoints/{name}.pt"; del model
        if args.device.type == "cuda": torch.cuda.empty_cache()
        model = restore_student(teacher,torch.load(out/checkpoint,map_location="cpu",weights_only=True))
        equivalence = gemma4_cache_equivalence(teacher,model,blocks["validation"][0],args.device,args.precision,args.block_size)
        remaining_full = len(available_full) if variant=="transformer_readout" else len(available_full)-len(layers)
        record = {"candidate":name,"variant":variant,"layers":layers,"matched_control":control,"checkpoint":checkpoint,
            "before_joint_validation_nll":before,"validation_nll":best,"training_seconds":time.perf_counter()-started,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters":sum(p.numel() for p in model.parameters()),"replacement_count":len(layers),
            "original_full_attention_layers":len(available_full),"remaining_full_attention_layers":remaining_full,
            "joint_updates":args.joint_steps if variant!="sink_window" else 0,**equivalence}
        records.append(record); write_csv(out/"validation_summary.csv",records); write_csv(out/"training_history.csv",history)
        write_json(out/"progress.json",{"status":"training","completed_candidates":records})
        del model
        if args.device.type == "cuda": torch.cuda.empty_cache()

    selected = min((r for r in records if r["variant"] == "cenn_partition"), key=lambda r:r["validation_nll"])["candidate"]
    write_json(out/"selection.json",{"selected":selected,"criterion":"mean validation NLL across training contexts","test_not_used_for_selection":True})
    rows,_,examples = base.evaluate_models(teacher,records,blocks,hashes,args,out,selected)
    for example in examples:
        example["prompt"] = tokenizer.decode(example["prompt_ids"]); example["continuation"] = tokenizer.decode(example["generated_ids"])
    write_json(out/"generation_examples.json",examples)
    manifest["status"]="completed"; write_json(out/"manifest.json",manifest)
    write_json(out/"progress.json",{"status":"completed","selected":selected})
    write_json(out/"integrated_report.json",{**manifest,"candidates":records,"rows":rows,"selected":selected,
        "interpretation":"Gemma 4 E2B text-backbone experiment. Sliding attention and all shared-KV producer/consumer layers remain native. TinyCeNN V1 replaces only independent original full-attention layers 4 and/or 9. Selection is validation-only."})
    print(f"Completed: {out.resolve()}",flush=True)


if __name__ == "__main__": main()
