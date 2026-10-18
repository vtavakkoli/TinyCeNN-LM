"""WaveFlux-CeNN vs the FIRST attention mixer in SmolLM2-360M-Instruct.

This is a *single-layer replacement experiment*, not a standalone full-LM release.
The replacement module itself performs no attention/QKV/DeltaNet/LSTM operation.
Every other pretrained decoder component remains unchanged during this experiment.

Train only on teacher mixer activations from WikiText-2 train, choose checkpoints
on validation, and evaluate an untouched test split once. Never promote a
candidate based solely on local loss or CPU execution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F
from torch import nn

from tinycenn_lm.waveflux_cenn import WaveFluxConfig, WaveFluxCeNN

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


class WaveFluxAttentionAdapter(nn.Module):
    """Native Llama attention-slot API; no native attention is computed.

    The output is an *attention-submodule update*; LlamaDecoderLayer supplies
    its ordinary outer residual, so this adapter must NOT add another one.
    Deliberately reject cached decoding until proper cache parity is validated.
    """

    def __init__(self, cell: WaveFluxCeNN):
        super().__init__()
        self.cell = cell

    def forward(self, hidden_states, *args, **kwargs):
        if (kwargs.get("past_key_value") is not None
                or kwargs.get("past_key_values") is not None
                or kwargs.get("cache_params") is not None
                or kwargs.get("use_cache", False)):
            raise NotImplementedError("WaveFlux attention-slot cache integration is not yet supported")
        if kwargs.get("output_attentions", False):
            raise NotImplementedError("WaveFlux has no attention weights to output")
        # An FP32 standalone cellular mixer with dtype-matched downstream output.
        return self.cell(hidden_states, mode="scan"), None


@contextmanager
def attention_slot(model, replacement):
    decoder = model.model.layers[0]
    prior = decoder.self_attn
    decoder.self_attn = replacement
    try:
        yield
    finally:
        decoder.self_attn = prior


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--profile", choices=("smoke", "quick", "research"), default="quick")
    p.add_argument("--output-dir", default="results/waveflux_smol360_l0")
    p.add_argument("--seed", type=int, default=2031)
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--train-blocks", type=int, default=None)
    p.add_argument("--dev-blocks", type=int, default=None)
    p.add_argument("--test-blocks", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--groups", type=int, default=24)
    p.add_argument("--cell-width", type=int, default=32)
    p.add_argument("--min-teacher-agreement", type=float, default=0.98)
    p.add_argument("--max-nll-increase", type=float, default=0.02)
    p.add_argument("--min-local-cosine", type=float, default=0.95)
    p.add_argument("--min-speedup", type=float, default=1.0)
    p.add_argument("--skip-gpu-required", action="store_true", help="CPU debug only; never accepted")
    return p.parse_args()


def load_blocks(tokenizer, split: str, count: int, seq_len: int):
    from datasets import load_dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    tokens = []
    needed = count * seq_len
    for item in dataset:
        s = item["text"].strip()
        if not s:
            continue
        tokens.extend(tokenizer.encode(s, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= needed:
            break
    if len(tokens) < needed:
        raise RuntimeError(f"Insufficient {split} tokens: {len(tokens)} < {needed}")
    return torch.tensor(tokens[:needed], dtype=torch.long).reshape(count, seq_len)


def batched(ids, batch_size):
    for start in range(0, len(ids), batch_size):
        yield ids[start:start + batch_size]


def capture_teacher_attention(model, ids, batch_size, device):
    """Collect real layer-0 pre-attention states and native mixer outputs."""
    attn = model.model.layers[0].self_attn
    captured = []
    slot = {"x": None}

    def before(_, args, kwargs):
        x = kwargs.get("hidden_states", None)
        if x is None and args:
            x = args[0]
        if x is None:
            raise RuntimeError("Llama attention inputs could not be captured")
        slot["x"] = x.detach().cpu().to(torch.float16)

    def after(_, args, kwargs, output):
        y = output[0] if isinstance(output, tuple) else output
        if slot["x"] is None or y is None:
            raise RuntimeError("Missing teacher attention IO")
        captured.append((slot["x"], y.detach().cpu().to(torch.float16)))
        slot["x"] = None

    h1 = attn.register_forward_pre_hook(before, with_kwargs=True)
    h2 = attn.register_forward_hook(after, with_kwargs=True)
    try:
        with torch.inference_mode():
            for batch in batched(ids, batch_size):
                model(input_ids=batch.to(device), use_cache=False, return_dict=True)
    finally:
        h1.remove()
        h2.remove()
    if len(captured) != math.ceil(len(ids) / batch_size):
        raise RuntimeError("Teacher IO capture count mismatch")
    return captured


def local_metrics(cell, data, device):
    cell.eval()
    n, mse_sum, cos_sum = 0, 0.0, 0.0
    with torch.no_grad():
        for x, y in data:
            got = cell(x.to(device)).float()
            target = y.to(device).float()
            mse_sum += float(((got-target).square().mean() /
                              target.square().mean().clamp_min(1e-8)).item())
            cos_sum += float(F.cosine_similarity(got, target, dim=-1, eps=1e-6).mean().item())
            n += 1
    return {"relative_mse": mse_sum/n, "cosine": cos_sum/n}


def compare_full_model(model, native, adapter, ids, batch_size, device):
    """Paired next-token scores; every valid position, no test-leakage tuning."""
    model.eval()
    adapter.eval()
    sums = {"teacher_nll": 0.0, "student_nll": 0.0,
            "teacher_agreement": 0, "student_gold_correct": 0,
            "teacher_gold_correct": 0, "kl_sum": 0.0, "tokens": 0}
    with torch.inference_mode():
        for batch in batched(ids, batch_size):
            tokens = batch.to(device)
            with attention_slot(model, native):
                teacher = model(input_ids=tokens, use_cache=False, return_dict=True).logits[:,:-1].float()
            with attention_slot(model, adapter):
                student = model(input_ids=tokens, use_cache=False, return_dict=True).logits[:,:-1].float()
            gold = tokens[:,1:]
            n = gold.numel()
            tlog = F.log_softmax(teacher, dim=-1)
            slog = F.log_softmax(student, dim=-1)
            tpred = teacher.argmax(-1)
            spred = student.argmax(-1)
            sums["teacher_nll"] += float(-tlog.gather(-1, gold[...,None]).sum().item())
            sums["student_nll"] += float(-slog.gather(-1, gold[...,None]).sum().item())
            sums["teacher_agreement"] += int((spred == tpred).sum().item())
            sums["student_gold_correct"] += int((spred == gold).sum().item())
            sums["teacher_gold_correct"] += int((tpred == gold).sum().item())
            sums["kl_sum"] += float((tlog.exp()*(tlog-slog)).sum().item())
            sums["tokens"] += n
    count = sums["tokens"]
    return {"tokens": count,
            "teacher_nll": sums["teacher_nll"]/count,
            "student_nll": sums["student_nll"]/count,
            "delta_nll": (sums["student_nll"]-sums["teacher_nll"])/count,
            "teacher_agreement": sums["teacher_agreement"]/count,
            "student_gold_accuracy": sums["student_gold_correct"]/count,
            "teacher_gold_accuracy": sums["teacher_gold_correct"]/count,
            "teacher_student_kl": sums["kl_sum"]/count}


def paired_latency(model, native, adapter, ids, device, repeats=8):
    if device.type != "cuda":
        return {"available": False, "reason": "CUDA required for speed acceptance"}
    ids = ids.to(device)
    model.eval()
    adapter.eval()
    readings = {"native_ms": [], "waveflux_ms": []}

    with torch.inference_mode():
        for order in (["native", "waveflux"], ["waveflux", "native"]):
            for name in order:
                module = native if name == "native" else adapter
                with attention_slot(model, module):
                    for _ in range(2):
                        model(input_ids=ids, use_cache=False, return_dict=True)
                    for _ in range(repeats):
                        torch.cuda.synchronize()
                        start, stop = (torch.cuda.Event(enable_timing=True),
                                       torch.cuda.Event(enable_timing=True))
                        start.record()
                        model(input_ids=ids, use_cache=False, return_dict=True)
                        stop.record()
                        stop.synchronize()
                        readings[name+"_ms"].append(start.elapsed_time(stop))
    native_ms = statistics.median(readings["native_ms"])
    wave_ms = statistics.median(readings["waveflux_ms"])
    return {"available": True, "native_ms": native_ms, "waveflux_ms": wave_ms,
            "speedup": native_ms/wave_ms, "gpu": torch.cuda.get_device_name(device),
            "samples": readings, "scope": "full-model prefill, no-cache, GPU CUDA events"}


def main():
    a = parse_args()
    profiles = {"smoke": (64, 12, 4, 4, 6, 2),
                "quick": (128, 64, 12, 16, 160, 40),
                "research": (256, 256, 32, 48, 800, 100)}
    length, ntrain, ndev, ntest, steps, eval_every = profiles[a.profile]
    length = a.length or length
    ntrain = a.train_blocks or ntrain
    ndev = a.dev_blocks or ndev
    ntest = a.test_blocks or ntest
    steps = a.steps if a.steps is not None else steps
    eval_every = a.eval_every or eval_every
    batch_size = a.batch_size or (1 if a.profile == "research" else 2)
    assert min(length, ntrain, ndev, ntest, steps, batch_size, eval_every) > 0
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.skip_gpu_required:
        raise RuntimeError("Select a GPU runtime or pass --skip-gpu-required for CPU diagnostic only")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type == "cuda" else torch.float32)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info
    revision = model_info(a.model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(a.model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, revision=revision,
                                                torch_dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    model.requires_grad_(False)
    if model.config.model_type != "llama" or not hasattr(model.model.layers[0], "self_attn"):
        raise ValueError("Expected SmolLM2 Llama architecture with layer-0 self_attn")
    native = model.model.layers[0].self_attn
    hidden = int(model.config.hidden_size)
    cfg = WaveFluxConfig(hidden_size=hidden, groups=a.groups, cell_width=a.cell_width,
                         velocities=(0,1,-1,2,-2), initial_decay=0.995,
                         initial_write=0.10, flux_strength=0.22, residual_gain=0.1)
    cell = WaveFluxCeNN(cfg).to(device)
    adapter = WaveFluxAttentionAdapter(cell)
    original_params = sum(p.numel() for p in native.parameters())
    wave_params = sum(p.numel() for p in cell.parameters())
    print("MODEL", a.model_id, "revision", revision)
    print("FIRST ATTENTION", type(native).__name__, "layer", 0)
    print("Hidden", hidden, "native_params", original_params, "cellular_params", wave_params)

    # Separate Wikitext-2 splits. The test split is not opened until AFTER
    # the best checkpoint has been selected on development data.
    train_ids = load_blocks(tokenizer,"train",ntrain,length)
    dev_ids = load_blocks(tokenizer,"validation",ndev,length)
    train_data = capture_teacher_attention(model, train_ids, batch_size, device)
    dev_data = capture_teacher_attention(model, dev_ids, batch_size, device)
    optim = torch.optim.AdamW(cell.parameters(),lr=a.lr,weight_decay=0.001)
    logs = []
    best = float("inf")
    best_path = out / "best_waveflux_layer0.pt"
    start_time = time.perf_counter()
    for step in range(1, steps+1):
        cell.train()
        inp, target = train_data[random.randrange(len(train_data))]
        pred = cell(inp.to(device)).float()
        reference = target.to(device).float()
        n_mse = (pred-reference).square().mean()/reference.square().mean().clamp_min(1e-8)
        cosine = 1-F.cosine_similarity(pred,reference,dim=-1,eps=1e-6).mean()
        loss = n_mse + 0.15*cosine
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(),1.0)
        optim.step()
        if step % eval_every == 0 or step == steps or step == 1:
            local = local_metrics(cell,dev_data,device)
            full = compare_full_model(model,native,adapter,dev_ids,batch_size,device)
            entry = {"step": step,"train_local_loss": float(loss.detach().item()),
                     "dev_local_cosine": local["cosine"],
                     "dev_local_rel_mse": local["relative_mse"],**full}
            logs.append(entry)
            print(json.dumps(entry))
            # Validation NLL controls checkpoint selection, never local train MSE.
            if math.isfinite(full["student_nll"]) and full["student_nll"] < best:
                best = full["student_nll"]
                torch.save({"core":cell.state_dict(),"config":cfg.__dict__,
                            "step":step,"model_revision":revision,
                            "layer":0,"native_params":original_params},best_path)
            with (out / "dev_history.json").open("w") as f:
                json.dump(logs,f,indent=2)
    best_payload = torch.load(best_path,map_location="cpu",weights_only=True)
    cell.load_state_dict(best_payload["core"])
    dev = compare_full_model(model,native,adapter,dev_ids,batch_size,device)
    dev_local = local_metrics(cell,dev_data,device)
    # Fresh untouched test set, loaded only after checkpoint selection.
    test_ids = load_blocks(tokenizer,"test",ntest,length)
    test = compare_full_model(model,native,adapter,test_ids,batch_size,device)
    timing = paired_latency(model,native,adapter,dev_ids[:batch_size],device)
    passed_dev = (dev["delta_nll"] <= a.max_nll_increase
                  and dev["teacher_agreement"] >= a.min_teacher_agreement
                  and dev_local["cosine"] >= a.min_local_cosine)
    passed_test = (test["delta_nll"] <= a.max_nll_increase
                   and test["teacher_agreement"] >= a.min_teacher_agreement)
    accepted = (passed_dev and passed_test and timing.get("available",False)
                and timing["speedup"] > a.min_speedup)
    report = {"model":a.model_id,"revision":revision,"replaced_layer":0,
              "replaced_component":"model.layers[0].self_attn", "method":"pure WaveFlux-CeNN",
              "other_layers_unchanged":True,"cached_decode_supported":False,
              "weights_published":False,"test_only_after_validation":True,
              "profile":a.profile,"length":length,"train_blocks":ntrain,
              "dev_blocks":ndev,"test_blocks":ntest,"steps":steps,"seed":a.seed,
              "original_attention_params":original_params,
              "replacement_params":wave_params,"original_attention_module":type(native).__name__,
              "core_state_bytes_per_sequence_fp32":len(cfg.velocities)*cfg.groups*cfg.cell_width*4,
              "best_step":best_payload["step"], "dev":dev,"dev_local":dev_local,
              "test":test,"latency":timing,"dev_quality_pass":passed_dev,
              "test_quality_pass":passed_test,"accepted":accepted,
              "experimental_only":not accepted,
              "elapsed_seconds":time.perf_counter()-start_time,
              "acceptance_thresholds":{"max_delta_nll":a.max_nll_increase,
                         "min_teacher_agreement":a.min_teacher_agreement,
                         "min_local_cosine":a.min_local_cosine,
                         "min_full_model_speedup":a.min_speedup}}
    (out / "final_report.json").write_text(json.dumps(report,indent=2))
    with (out / "dev_history.csv").open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=sorted(logs[0]))
        w.writeheader();w.writerows(logs)
    print("FINAL",json.dumps({k:report[k] for k in
           ("replaced_layer","accepted","dev_quality_pass","test_quality_pass","best_step")}))
    print("RESULTS",out.absolute())
    if not accepted:
        print("NOT ACCEPTED: retain native first attention for deployment.")
    return report


if __name__ == "__main__":
    main()
