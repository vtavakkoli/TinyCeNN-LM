#!/usr/bin/env python3
"""FunctionGemma 270M: replace global full-attention layers with TinyCeNN memory."""
import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import torch.nn.functional as F
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, repeat_kv

from scripts import benchmark_smollm2_integrated_memory as base
from scripts.benchmark_cenn_research_layers import fit_transfer, write_csv, write_json
from tinycenn_lm.gemma3_integrated_memory import (
    adapter_payload, build_student, full_attention_layers, greedy_generate,
    inference_mode, native_dtype, new_cache, restore_student, wrappers,
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
    b, t, _ = hidden.shape
    h = attention.config.num_attention_heads
    hk = attention.config.num_key_value_heads
    d = attention.head_dim
    q = attention.q_proj(hidden).view(b, t, h, d).transpose(1, 2)
    k = attention.k_proj(hidden).view(b, t, hk, d).transpose(1, 2)
    v = attention.v_proj(hidden).view(b, t, hk, d).transpose(1, 2)
    q = attention.q_norm(q)
    k = attention.k_norm(k)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q.float(), k.float(), v.float()


@torch.no_grad()
def capture_samples(model, blocks, layer_indices, context, device):
    samples = {index: [] for index in layer_indices}
    for number, block in enumerate(blocks, 1):
        ids = block[:context].unsqueeze(0).to(device)
        result = model.model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
        positions = torch.arange(context, device=device)[None]
        for index in layer_indices:
            layer = model.model.layers[index]
            if getattr(layer.self_attn, "is_sliding", False):
                raise ValueError(f"Layer {index} is sliding; only full attention is replaceable in V1")
            hidden = layer.input_layernorm(result.hidden_states[index])
            rope = model.model.rotary_emb(hidden, positions)
            q, k, v = project_qkv(layer.self_attn, hidden, rope)
            target = F.scaled_dot_product_attention(
                q, repeat_kv(k, q.shape[1] // k.shape[1]), repeat_kv(v, q.shape[1] // v.shape[1]),
                is_causal=True, scale=float(layer.self_attn.scaling)
            ).float()
            samples[index].append(tuple(x.detach().cpu() for x in (q, k, v, target)))
        if number % 8 == 0 or number == len(blocks):
            print(f"captured {number}/{len(blocks)} at context {context}", flush=True)
    return samples


def joint_loss(student_logits, teacher_logits, targets, kl_weight=1., temperature=1., chunk=16):
    """Small chunks keep FunctionGemma's 262k-vocabulary KL temporary memory bounded."""
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="vtava/functiongemma-270m-it-simple-tool-calling")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--conservative-layers", default="5,17")
    p.add_argument("--expanded-layers", default="5,11,17")
    p.add_argument("--train-contexts", default="128,256,512")
    p.add_argument("--test-contexts", default="128,256,512,1024,2048")
    for name, value in [
        ("train-documents",64),("validation-documents",8),("test-documents",16),("warm-documents",8),
        ("warm-steps",50),("joint-steps",150),("eval-every",25),("features",64),("block-size",32),
        ("sinks",4),("seed",2029),("dataset-seed",9318),("decode-tokens",32),
        ("timing-documents",2),("timing-repeats",2),("loss-chunk",16),
    ]:
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--kl-weight", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--nll-margin", type=float, default=.02)
    p.add_argument("--exclude-manifest", action="append", default=[])
    p.add_argument("--output-dir", default="result/functiongemma-integrated-v1")
    a = p.parse_args()
    for key in ("conservative_layers","expanded_layers","train_contexts","test_contexts"):
        setattr(a, key, list(dict.fromkeys(int(x) for x in getattr(a, key).split(","))))
    if min(a.train_contexts + a.test_contexts) <= 2 * a.block_size + a.sinks:
        p.error("Every context must exercise compressed history")
    if a.features < 2 or a.features % 2 or a.loss_chunk < 1:
        p.error("features must be positive/even and loss-chunk positive")
    return a


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = native_dtype(args.device)
    args.precision = str(dtype).removeprefix("torch.")
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    api = HfApi()
    args.model_sha = api.model_info(args.base_model, revision=args.model_revision).sha
    data_sha = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.model_sha, dtype=dtype, attn_implementation="sdpa"
    ).to(args.device).eval().requires_grad_(False)
    config = teacher.config.get_text_config(decoder=True)
    available_full = full_attention_layers(teacher)
    requested = sorted(set(args.conservative_layers + args.expanded_layers))
    if config.model_type != "gemma3_text" or not set(requested).issubset(available_full):
        raise ValueError(f"Expected Gemma3 full layers {available_full}; requested {requested}")
    if max(args.train_contexts + args.test_contexts) + args.decode_tokens > config.max_position_embeddings:
        raise ValueError("Requested context plus decode exceeds position limit")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_sha)
    exclusions = base.read_exclusions(args.exclude_manifest) if args.exclude_manifest else set()
    stream = load_dataset(args.dataset, args.dataset_config, revision=data_sha, split="train", streaming=True).shuffle(
        seed=args.dataset_seed, buffer_size=2048
    )
    blocks, hashes = base.collect_documents(
        stream, tokenizer,
        {"train":args.train_documents,"validation":args.validation_documents,"test":args.test_documents},
        max(args.train_contexts + args.test_contexts) + args.decode_tokens, exclusions
    )
    manifest = {
        "status":"running","experiment":"functiongemma-cenn-integrated-v1",
        "architecture":{"model_type":config.model_type,"num_hidden_layers":config.num_hidden_layers,
            "full_attention_layers":available_full,"sliding_window":config.sliding_window,
            "layer_types":list(config.layer_types),"head_dim":config.head_dim,
            "num_attention_heads":config.num_attention_heads,"num_key_value_heads":config.num_key_value_heads},
        "args":{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items()},
        "model_revision":args.model_sha,"dataset_revision":data_sha,"document_hashes":hashes,
        "prior_document_hashes_excluded":len(exclusions),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__,
        "native_dtype":args.precision,"gpu":torch.cuda.get_device_name(0) if args.device.type=="cuda" else None,
        "split_protocol":"exclude supplied prior hashes; test40..49; val50..59; train60..99",
        "unique_train_tokens_at_max_context":len(blocks["train"])*max(args.train_contexts)
    }
    write_json(out / "manifest.json", manifest)
    torch.save(blocks, out / "token_blocks.pt")

    indices = sorted(set(requested))
    warm_context = min(args.train_contexts)
    train = capture_samples(teacher, blocks["train"][:args.warm_documents], indices, warm_context, args.device)
    valid = capture_samples(teacher, blocks["validation"], indices, warm_context, args.device)
    records, history = [], []
    for name, variant, layers, control in base.configurations(args):
        torch.manual_seed(args.seed)
        model = build_student(teacher, layers, variant, args.features, args.block_size, args.sinks)
        started = time.perf_counter()
        if variant != "sink_window":
            for adapter in wrappers(model):
                ridge_calibrate(adapter.core, train[adapter.layer_idx], args.device)
                warmargs = SimpleNamespace(device=args.device, steps=args.warm_steps,
                    eval_every=max(1,min(args.eval_every,args.warm_steps)), lr=.002, seed=args.seed)
                fit_transfer(adapter.core, train[adapter.layer_idx], valid[adapter.layer_idx], warmargs,
                    out/"checkpoints"/f"{name}_warm_{adapter.layer_idx}.pt", f"{name}_layer{adapter.layer_idx}")
        before, _ = base.validation_score(model, blocks["validation"], args.train_contexts, args.device, args.precision)
        base.save_adapter(out/"checkpoints"/f"{name}_before_joint.pt", model,
            {"candidate":name,"stage":"before_joint","model_revision":args.model_sha})
        fit_history, best = base.train_joint(teacher, model, blocks, args, out, name)
        history.extend(fit_history)
        checkpoint = f"checkpoints/{name}.pt"
        del model
        if args.device.type == "cuda": torch.cuda.empty_cache()
        model = restore_student(teacher, torch.load(out/checkpoint, map_location="cpu", weights_only=True))
        equivalence = base.cache_equivalence(model, blocks["validation"][0], args.device, args.precision, args.block_size)
        remaining_full = len(available_full) if variant == "transformer_readout" else len(available_full)-len(layers)
        record = {"candidate":name,"variant":variant,"layers":layers,"matched_control":control,
            "checkpoint":checkpoint,"before_joint_validation_nll":before,"validation_nll":best,
            "training_seconds":time.perf_counter()-started,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters":sum(p.numel() for p in model.parameters()),"replacement_count":len(layers),
            "original_full_attention_layers":len(available_full),"remaining_full_attention_layers":remaining_full,
            "joint_updates":args.joint_steps if variant!="sink_window" else 0,
            "joint_token_presentations":sum(args.train_contexts[i%len(args.train_contexts)] for i in range(args.joint_steps)) if variant!="sink_window" else 0,
            **equivalence}
        records.append(record)
        write_csv(out/"validation_summary.csv", records); write_csv(out/"training_history.csv", history)
        write_json(out/"progress.json", {"status":"training","completed_candidates":records})
        del model
        if args.device.type == "cuda": torch.cuda.empty_cache()

    selected = min((r for r in records if r["matched_control"]), key=lambda r:r["validation_nll"])["candidate"]
    write_json(out/"selection.json", {"selected":selected,"criterion":"mean validation NLL across training contexts","test_not_used_for_selection":True})
    rows, _, examples = base.evaluate_models(teacher, records, blocks, hashes, args, out, selected)
    for example in examples:
        example["prompt"] = tokenizer.decode(example["prompt_ids"])
        example["continuation"] = tokenizer.decode(example["generated_ids"])
    write_json(out/"generation_examples.json", examples)
    manifest["status"] = "completed"; write_json(out/"manifest.json", manifest)
    write_json(out/"progress.json", {"status":"completed","selected":selected})
    write_json(out/"integrated_report.json", {**manifest,"candidates":records,"rows":rows,"selected":selected,
        "interpretation":"FunctionGemma/Gemma3 experiment replacing only original full-attention layers; 15 sliding-window layers remain unchanged. Selection is validation-only."})
    print(f"Completed: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
