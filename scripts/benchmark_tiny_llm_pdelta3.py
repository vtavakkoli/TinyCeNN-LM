#!/usr/bin/env python3
"""Test TinyCeNN PDelta3/GDN2 attention replacements in arnir0/Tiny-LLM.

Tiny-LLM has one Llama block, so true cross-layer value routing (CLVR) is
structurally unavailable.  The benchmark therefore compares the exact one-layer
Transformer attention against:

* conv4_pdelta_f96 -- current bounded-state control;
* conv4_channel_decay_f96 -- content-dependent channel decay;
* conv4_gdn2_f96 -- independent erase/write GDN2 recurrence;
* conv4_gdn2_inputroute_f96 -- a clearly labelled one-layer proxy that routes
  the current pre-convolution V through the CLVR projection/gate.  It tests the
  value-routing mechanism but is NOT a claim of cross-layer CLVR.

All architecture selection is validation-only.  Held-out test documents are used
once at the end with paired bootstrap confidence intervals.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import nn

from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer
from tinycenn_lm.research_layers import softmax_reference
from scripts.benchmark_cenn_research_layers import (
    collect_documents,
    cosine,
    evaluate_nll,
    nmse,
    paired_interval,
    project_qkv,
    quality_label,
    save_checkpoint,
    time_kernel,
    write_csv,
    write_json,
)

MODEL_REVISION = "b784a70a5e6908c9148820a245d60a3347279868"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def profile(name: str):
    return {
        "quick": dict(train=24, validation=8, test=12,
                      transfer=(35, 45, 55), lm=(8, 10, 12)),
        "balanced": dict(train=56, validation=20, test=36,
                         transfer=(70, 100, 140), lm=(16, 22, 32)),
        "strong": dict(train=112, validation=40, test=72,
                       transfer=(140, 200, 280), lm=(28, 40, 60)),
    }[name]


def candidate_specs():
    return [
        dict(name="conv4_pdelta_f96", variant="conv4_pdelta_f96",
             route_mode="none", ingredient="current_control"),
        dict(name="conv4_channel_decay_f96", variant="conv4_channel_decay_f96",
             route_mode="none", ingredient="channel_decay"),
        dict(name="conv4_gdn2_f96", variant="conv4_gdn2_f96",
             route_mode="none", ingredient="gdn2_erase_write"),
        dict(name="conv4_gdn2_inputroute_f96", variant="conv4_gdn2_clvr_f96",
             route_mode="current_v", ingredient="one_layer_value_route_proxy"),
    ]


def make_core(spec, model, device):
    heads = model.config.num_attention_heads
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // heads
    return FrontierPDelta3Layer(
        heads, kv, dim,
        feature_dim=96,
        variant=spec["variant"],
        chunk_size=32,
        conv_kernel=4,
        state_dtype="fp16",
    ).to(device)


def transformer_kv_bytes(model, context: int, bytes_per_element: int = 2):
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    return 2 * context * kv * dim * bytes_per_element


def route_for(spec, v):
    if spec["route_mode"] == "current_v":
        return v
    return None


class TinyLLMAttentionReplacement(nn.Module):
    def __init__(self, original, core, spec, num_heads, num_kv_heads):
        super().__init__()
        self.original = original
        self.core = core
        self.spec = spec
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.last_output = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if (kwargs.get("past_key_values") is not None
                or kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False)):
            raise ValueError("Tiny-LLM compatibility benchmark requires use_cache=False")
        if position_embeddings is None:
            raise ValueError("Llama position_embeddings are required")
        t = hidden_states.shape[1]
        if attention_mask is not None:
            if (attention_mask.ndim != 4 or attention_mask.shape[-1] != t
                    or bool((attention_mask[..., -1, :] < 0).any())):
                raise ValueError("Only unpadded full causal blocks are supported")
        q, k, v = project_qkv(
            self.original, hidden_states, position_embeddings,
            self.num_heads, self.num_kv_heads,
        )
        if self.core is None:
            output = softmax_reference(q, k, v, self.num_heads // self.num_kv_heads)
        else:
            output = self.core(q, k, v, routed_v=route_for(self.spec, v))
        self.last_output = output
        flat = output.transpose(1, 2).reshape(hidden_states.shape)
        return self.original.o_proj(flat.to(hidden_states.dtype)), None


@contextlib.contextmanager
def replace_attention(model, core, spec):
    original = model.model.layers[0].self_attn
    wrapper = TinyLLMAttentionReplacement(
        original, core, spec,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
    )
    model.model.layers[0].self_attn = wrapper
    try:
        yield wrapper
    finally:
        wrapper.last_output = None
        model.model.layers[0].self_attn = original


@torch.no_grad()
def capture_samples(model, blocks, max_context, device):
    samples = []
    heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    for number, block in enumerate(blocks, 1):
        ids = block[:max_context].unsqueeze(0).to(device)
        result = model.model(
            input_ids=ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        positions = torch.arange(max_context, device=device)[None]
        layer = model.model.layers[0]
        hidden = layer.input_layernorm(result.hidden_states[0])
        rope = model.model.rotary_emb(hidden, positions)
        q, k, v = project_qkv(layer.self_attn, hidden, rope, heads, kv_heads)
        target = softmax_reference(q, k, v, heads // kv_heads)
        samples.append(tuple(x.detach().cpu() for x in (q, k, v, target)))
        if number % 8 == 0 or number == len(blocks):
            print(f"captured {number}/{len(blocks)} at context {max_context}", flush=True)
    return samples


def slice_sample(sample, context, device):
    return tuple(x[:, :, :context].to(device) for x in sample)


@torch.no_grad()
def evaluate_transfer(core, spec, samples, context, device):
    errors, similarities = [], []
    for sample in samples:
        q, k, v, target = slice_sample(sample, context, device)
        output = core(q, k, v, routed_v=route_for(spec, v))
        errors.append(float(nmse(target, output)))
        similarities.append(float(cosine(target, output)))
    return {
        "output_nmse": statistics.fmean(errors),
        "output_cosine": statistics.fmean(similarities),
    }


def fit_transfer(core, spec, train, validation, contexts, steps, device, seed, checkpoint):
    rng = random.Random(seed)
    history = []
    for stage, (context, count) in enumerate(zip(contexts, steps), 1):
        optimizer = torch.optim.AdamW(core.parameters(), lr=2e-3 * (0.72 ** (stage - 1)), weight_decay=1e-4)
        best = float("inf")
        best_state = None
        for step in range(1, count + 1):
            q, k, v, target = slice_sample(train[rng.randrange(len(train))], context, device)
            optimizer.zero_grad(set_to_none=True)
            output = core(q, k, v, routed_v=route_for(spec, v))
            loss = nmse(target, output) + 0.1 * (1 - cosine(target, output))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{spec['name']}: non-finite transfer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % max(10, count // 3) == 0 or step == count:
                metrics = evaluate_transfer(core, spec, validation, context, device)
                history.append({
                    "candidate": spec["name"], "phase": "transfer", "stage": stage,
                    "context": context, "step": step, "train_loss": float(loss.detach()),
                    **metrics,
                })
                if metrics["output_nmse"] < best:
                    best = metrics["output_nmse"]
                    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
                print(f"{spec['name']} transfer ctx={context} {step}/{count}: {metrics}", flush=True)
        if best_state is not None:
            core.load_state_dict(best_state)
        save_checkpoint(checkpoint, core, {
            "candidate": spec["name"], "phase": "transfer", "stage": stage,
            "context": context, "validation_output_nmse": best,
        })
    return history


def fit_language(model, core, spec, blocks, samples, validation, contexts, steps,
                 device, seed, checkpoint):
    rng = random.Random(seed)
    history = []
    with replace_attention(model, core, spec) as wrapper:
        for stage, (context, count) in enumerate(zip(contexts, steps), 1):
            optimizer = torch.optim.AdamW(core.parameters(), lr=4e-4 * (0.72 ** (stage - 1)), weight_decay=1e-4)
            best = statistics.fmean(evaluate_nll(model, validation, context, device))
            best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
            history.append({
                "candidate": spec["name"], "phase": "lm", "stage": stage,
                "context": context, "step": 0, "validation_nll": best,
            })
            for step in range(1, count + 1):
                i = rng.randrange(len(blocks))
                optimizer.zero_grad(set_to_none=True)
                inputs = blocks[i][:context].unsqueeze(0).to(device)
                labels = blocks[i][1:context + 1].to(device)
                logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
                ce = F.cross_entropy(logits[0].float(), labels)
                _, _, _, teacher = slice_sample(samples[i], context, device)
                imitation = nmse(teacher, wrapper.last_output)
                alignment = 1 - cosine(teacher, wrapper.last_output)
                loss = 2.0 * ce + 0.08 * imitation + 0.02 * alignment
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{spec['name']}: non-finite LM loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                wrapper.last_output = None
                if step == 1 or step % max(6, count // 3) == 0 or step == count:
                    score = statistics.fmean(evaluate_nll(model, validation, context, device))
                    history.append({
                        "candidate": spec["name"], "phase": "lm", "stage": stage,
                        "context": context, "step": step,
                        "train_loss": float(loss.detach()), "train_ce": float(ce.detach()),
                        "validation_nll": score,
                    })
                    if score < best:
                        best = score
                        best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
                    print(f"{spec['name']} lm ctx={context} {step}/{count}: val NLL={score:.6f}", flush=True)
            core.load_state_dict(best_state)
            save_checkpoint(checkpoint, core, {
                "candidate": spec["name"], "phase": "lm", "stage": stage,
                "context": context, "validation_nll": best,
            })
    return history


@torch.no_grad()
def score(model, core, spec, blocks, reference, context, device, margin, seed):
    with replace_attention(model, core, spec):
        values = evaluate_nll(model, blocks, context, device)
    delta, low, high = paired_interval(values, reference, seed=seed, repeats=3000)
    mean = statistics.fmean(values)
    ref_mean = statistics.fmean(reference)
    state = core.recurrent_state_bytes(context=context)
    kv = transformer_kv_bytes(model, context)
    ratio = state / kv
    if high < 0:
        verdict = "strict_quality_win"
    elif delta < 0:
        verdict = "point_quality_win"
    elif low >= -margin and high <= margin and ratio < 0.5:
        verdict = "memory_efficient_parity"
    else:
        verdict = quality_label(delta, low, high, len(values), margin)
    return {
        "candidate_nll": mean,
        "transformer_nll": ref_mean,
        "candidate_ppl": math.exp(mean),
        "transformer_ppl": math.exp(ref_mean),
        "delta_nll": delta,
        "ci95_low": low,
        "ci95_high": high,
        "ppl_ratio": math.exp(delta),
        "state_bytes": state,
        "transformer_kv_bytes_fp16": kv,
        "state_vs_transformer_fp16": ratio,
        "verdict": verdict,
        "documents": len(values),
    }


@torch.no_grad()
def diagnostics(core, spec, sample, device):
    q, k, v, _ = (x.to(device) for x in sample)
    route = route_for(spec, v)
    eager_ms, eager_peak = time_kernel(
        lambda: core(q, k, v, routed_v=route), device, repeats=5
    )
    exact_ms, exact_peak = time_kernel(
        lambda: softmax_reference(q, k, v, core.groups), device, repeats=5
    )
    result = {
        "prefill_ms": eager_ms,
        "transformer_reference_ms": exact_ms,
        "prefill_speed_ratio": exact_ms / eager_ms,
        "peak_extra_bytes": eager_peak,
        "transformer_peak_extra_bytes": exact_peak,
        "state_bytes": core.recurrent_state_bytes(context=q.shape[2]),
    }
    result.update(core.diagnostics(q, k, v, routed_v=route))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "balanced", "strong"], default="balanced")
    parser.add_argument("--base-model", default="arnir0/Tiny-LLM")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--curriculum", default="256,512,1024")
    parser.add_argument("--validation-contexts", default="256,512,1024")
    parser.add_argument("--test-contexts", default="256,512,1024")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the Colab benchmark")
    device = torch.device("cuda")
    args.device = device
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.model_revision,
        torch_dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if model.config.num_hidden_layers != 1:
        raise RuntimeError(f"Expected the pinned Tiny-LLM to have 1 layer, got {model.config.num_hidden_layers}")
    if model.config.hidden_size // model.config.num_attention_heads != 96:
        raise RuntimeError("Pinned Tiny-LLM no longer has head_dim=96; review F96 experiment")

    curriculum = [int(x) for x in args.curriculum.split(",")]
    val_contexts = [int(x) for x in args.validation_contexts.split(",")]
    test_contexts = [int(x) for x in args.test_contexts.split(",")]
    max_context = max(curriculum + val_contexts + test_contexts)
    if max_context > model.config.max_position_embeddings:
        raise ValueError(f"Tiny-LLM max_position_embeddings={model.config.max_position_embeddings}, requested {max_context}")

    cfg = profile(args.profile)
    rows = load_dataset(
        args.dataset, split="train", streaming=True,
        revision=args.dataset_revision,
    ).shuffle(seed=9107, buffer_size=2048)
    counts = {"train": cfg["train"], "validation": cfg["validation"], "test": cfg["test"]}
    blocks, hashes = collect_documents(rows, tokenizer, counts, max_context, max_documents=250000)

    # Verify the benchmark wrapper reproduces exact softmax before training.
    ref_errors = []
    for block in blocks["validation"][:4]:
        baseline = evaluate_nll(model, [block], min(512, max_context), device)[0]
        with replace_attention(model, None, candidate_specs()[0]):
            wrapped = evaluate_nll(model, [block], min(512, max_context), device)[0]
        ref_errors.append(abs(wrapped - baseline))
    wrapper_error = max(ref_errors)
    if wrapper_error > 0.01:
        raise RuntimeError(f"exact-softmax wrapper error too high: {wrapper_error}")

    train_samples = capture_samples(model, blocks["train"], max_context, device)
    validation_samples = capture_samples(model, blocks["validation"], max_context, device)

    val_reference = {
        c: evaluate_nll(model, blocks["validation"], c, device) for c in val_contexts
    }
    test_reference = {
        c: evaluate_nll(model, blocks["test"], c, device) for c in test_contexts
    }

    trained = {}
    validation_rows = []
    history = []
    diagnostics_rows = []
    for spec in candidate_specs():
        print(f"\n=== {spec['name']} ===", flush=True)
        core = make_core(spec, model, device)
        checkpoint = output / "checkpoints" / f"{spec['name']}.pt"
        history += fit_transfer(
            core, spec, train_samples, validation_samples,
            curriculum, cfg["transfer"], device, args.seed, checkpoint,
        )
        history += fit_language(
            model, core, spec, blocks["train"], train_samples,
            blocks["validation"], curriculum, cfg["lm"],
            device, args.seed, checkpoint,
        )
        core.eval()
        trained[spec["name"]] = (core, spec)
        diag = diagnostics(core, spec, validation_samples[0], device)
        diagnostics_rows.append({"name": spec["name"], **diag})
        for context in val_contexts:
            with replace_attention(model, core, spec):
                vals = evaluate_nll(model, blocks["validation"], context, device)
            candidate = statistics.fmean(vals)
            reference = statistics.fmean(val_reference[context])
            validation_rows.append({
                "name": spec["name"], "context": context,
                "candidate_nll": candidate, "transformer_nll": reference,
                "delta_nll": candidate - reference,
                "candidate_ppl": math.exp(candidate),
                "transformer_ppl": math.exp(reference),
                "state_bytes": core.recurrent_state_bytes(context=context),
                "state_vs_transformer_fp16": core.recurrent_state_bytes(context=context) / transformer_kv_bytes(model, context),
            })

    # Select using all supported validation contexts; weight longer contexts more.
    weights = {256: 0.20, 512: 0.30, 1024: 0.50}
    composite = {}
    for name in trained:
        rows_for = [r for r in validation_rows if r["name"] == name]
        numerator = sum(weights.get(r["context"], 1.0) * r["delta_nll"] for r in rows_for)
        denominator = sum(weights.get(r["context"], 1.0) for r in rows_for)
        composite[name] = numerator / denominator
    quality_winner = min(composite, key=composite.get)
    efficiency_winner = min(
        trained,
        key=lambda n: next(r["prefill_ms"] for r in diagnostics_rows if r["name"] == n),
    )
    selection = {
        "quality_winner": quality_winner,
        "efficient_winner": efficiency_winner,
        "quality_criterion": "weighted validation delta NLL: 0.20@256 + 0.30@512 + 0.50@1024",
        "validation_composite_delta_nll": composite,
        "test_not_used_for_selection": True,
        "tiny_llm_has_one_layer": True,
        "true_clvr_available": False,
        "route_proxy": "conv4_gdn2_inputroute_f96 routes current pre-convolution V; it is not cross-layer CLVR",
    }

    test_rows = []
    for name, (core, spec) in trained.items():
        for context in test_contexts:
            result = score(
                model, core, spec, blocks["test"], test_reference[context],
                context, device, args.margin, args.seed + context,
            )
            test_rows.append({
                "candidate": name, "context": context,
                "selected_quality": name == quality_winner,
                "selected_efficiency": name == efficiency_winner,
                **result,
            })

    report = {
        "experiment": "tiny-llm-pdelta3-compatibility",
        "profile": args.profile,
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "host_config": {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.hidden_size // model.config.num_attention_heads,
            "max_position_embeddings": model.config.max_position_embeddings,
        },
        "document_hashes": hashes,
        "curriculum_contexts": curriculum,
        "validation_contexts": val_contexts,
        "test_contexts": test_contexts,
        "exact_softmax_wrapper_max_document_nll_error": wrapper_error,
        "selection": selection,
        "strict_quality_win_contexts_for_selected": [
            r["context"] for r in test_rows
            if r["candidate"] == quality_winner and r["verdict"] == "strict_quality_win"
        ],
        "limitations": (
            "arnir0/Tiny-LLM has one Transformer block, so true cross-layer CLVR cannot be tested. "
            "The input-route candidate is a one-layer proxy only. This is one seed and reference PyTorch kernels."
        ),
    }

    write_csv(output / "validation_context_summary.csv", validation_rows)
    write_csv(output / "test_summary.csv", test_rows)
    write_csv(output / "training_history.csv", history)
    write_csv(output / "candidate_diagnostics.csv", diagnostics_rows)
    write_json(output / "selection.json", selection)
    write_json(output / "tiny_llm_pdelta3_report.json", report)
    print("\nSELECTION")
    print(json.dumps(selection, indent=2))
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()
