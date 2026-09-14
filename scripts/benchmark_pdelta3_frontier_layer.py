#!/usr/bin/env python3
"""Benchmark four frontier-inspired Conv4 recurrent attention replacements."""
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

from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer, VARIANTS
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
)


def profile(name):
    return {
        "quick": dict(train=24, validation=8, test=12,
                      transfer=(40, 50, 60), lm=(8, 10, 12), every=20),
        "balanced": dict(train=48, validation=16, test=32,
                         transfer=(80, 110, 150), lm=(18, 24, 36), every=30),
        "strong": dict(train=96, validation=32, test=64,
                       transfer=(160, 220, 320), lm=(30, 45, 70), every=40),
    }[name]


def candidate_specs():
    return [
        dict(name="conv4_pdelta_f96", ingredient="current_control"),
        dict(name="conv4_channel_decay_f96", ingredient="kda_style_channel_decay"),
        dict(name="conv4_gdn2_f96", ingredient="independent_channel_erase_write"),
        dict(name="conv4_gdn2_clvr_f96", ingredient="gdn2_plus_previous_layer_value_routing"),
    ]


def transformer_kv_bytes(model, context, bytes_per_element=2):
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // model.config.num_attention_heads
    return 2 * context * kv * dim * bytes_per_element


def make_core(spec, model, device):
    heads = model.config.num_attention_heads
    kv = model.config.num_key_value_heads
    dim = model.config.hidden_size // heads
    return FrontierPDelta3Layer(
        heads, kv, dim,
        feature_dim=96,
        variant=spec["name"],
        chunk_size=32,
        conv_kernel=4,
        state_dtype="fp16",
    ).to(device)


class FrontierAttentionReplacement(nn.Module):
    """Frozen Q/K/V/O wrapper with optional previous-layer V routing."""
    def __init__(self, original, core, num_heads, num_kv_heads, route_holder):
        super().__init__()
        self.original = original
        self.core = core
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.route_holder = route_holder
        self.last_output = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if (kwargs.get("past_key_values") is not None
                or kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False)):
            raise ValueError("This benchmark wrapper requires use_cache=False")
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
            route = self.route_holder.get("v") if self.core.use_clvr else None
            output = self.core(q, k, v, routed_v=route)
        self.last_output = output
        flat = output.transpose(1, 2).reshape(hidden_states.shape)
        return self.original.o_proj(flat.to(hidden_states.dtype)), None


@contextlib.contextmanager
def replace_frontier_attention(model, index, core):
    if index < 1 and core is not None and core.use_clvr:
        raise ValueError("CLVR needs a preceding layer")
    original = model.model.layers[index].self_attn
    holder = {}
    hook = None
    if core is not None and core.use_clvr:
        previous = model.model.layers[index - 1].self_attn
        kv_heads = model.config.num_key_value_heads

        def capture_previous_v(module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                raise RuntimeError("could not capture previous-layer hidden states for CLVR")
            b, t, _ = hidden.shape
            value = module.v_proj(hidden).view(b, t, kv_heads, -1).transpose(1, 2)
            holder["v"] = value.float()

        hook = previous.register_forward_pre_hook(capture_previous_v, with_kwargs=True)

    wrapper = FrontierAttentionReplacement(
        original, core, model.config.num_attention_heads,
        model.config.num_key_value_heads, holder,
    )
    model.model.layers[index].self_attn = wrapper
    try:
        yield wrapper
    finally:
        wrapper.last_output = None
        model.model.layers[index].self_attn = original
        if hook is not None:
            hook.remove()


@torch.no_grad()
def capture_frontier_samples(model, blocks, layer_index, context, device):
    if layer_index < 1:
        raise ValueError("frontier lab requires layer >=1 for the CLVR control")
    samples = []
    heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    for number, block in enumerate(blocks, 1):
        ids = block[:context].unsqueeze(0).to(device)
        result = model.model(
            input_ids=ids, output_hidden_states=True,
            use_cache=False, return_dict=True,
        )
        positions = torch.arange(context, device=device)[None]

        layer = model.model.layers[layer_index]
        hidden = layer.input_layernorm(result.hidden_states[layer_index])
        rope = model.model.rotary_emb(hidden, positions)
        q, k, v = project_qkv(layer.self_attn, hidden, rope, heads, kv_heads)
        target = softmax_reference(q, k, v, heads // kv_heads)

        previous_layer = model.model.layers[layer_index - 1]
        previous_hidden = previous_layer.input_layernorm(
            result.hidden_states[layer_index - 1]
        )
        b, t, _ = previous_hidden.shape
        routed_v = previous_layer.self_attn.v_proj(previous_hidden).view(
            b, t, kv_heads, -1
        ).transpose(1, 2).float()

        samples.append(tuple(
            x.detach().cpu() for x in (q, k, v, target, routed_v)
        ))
        if number % 8 == 0 or number == len(blocks):
            print(f"captured {number}/{len(blocks)} at context {context}", flush=True)
    return samples


def slice_sample(sample, context, device):
    q, k, v, target, route = sample
    return tuple(x[:, :, :context].to(device) for x in (q, k, v, target, route))


@torch.no_grad()
def evaluate_transfer(core, samples, context, device):
    errors, similarities = [], []
    for sample in samples:
        q, k, v, target, route = slice_sample(sample, context, device)
        output = core(q, k, v, routed_v=route if core.use_clvr else None)
        errors.append(float(nmse(target, output)))
        similarities.append(float(cosine(target, output)))
    return {
        "output_nmse": statistics.fmean(errors),
        "output_cosine": statistics.fmean(similarities),
    }


def fit_progressive_transfer(core, train, validation, contexts, steps_by_stage,
                             device, seed, checkpoint, name):
    history = []
    rng = random.Random(seed)
    for stage, (context, stage_steps) in enumerate(zip(contexts, steps_by_stage), 1):
        core.train()
        lr = 2e-3 * (0.72 ** (stage - 1))
        optimizer = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=1e-4)
        best = float("inf")
        best_state = None
        for step in range(1, stage_steps + 1):
            q, k, v, target, route = slice_sample(
                train[rng.randrange(len(train))], context, device
            )
            optimizer.zero_grad(set_to_none=True)
            output = core(q, k, v, routed_v=route if core.use_clvr else None)
            loss = nmse(target, output) + 0.1 * (1.0 - cosine(target, output))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite transfer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
            ratio = step / max(stage_steps, 1)
            for group in optimizer.param_groups:
                group["lr"] = lr * (0.1 + 0.9 * (1 + math.cos(math.pi * ratio)) / 2)
            optimizer.step()

            if step == 1 or step % max(10, stage_steps // 3) == 0 or step == stage_steps:
                metrics = evaluate_transfer(core, validation, context, device)
                history.append({
                    "candidate": name, "phase": "transfer", "stage": stage,
                    "context": context, "step": step, "train_loss": float(loss.detach()),
                    **metrics,
                })
                if metrics["output_nmse"] < best:
                    best = metrics["output_nmse"]
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in core.state_dict().items()
                    }
                print(
                    f"{name} transfer ctx={context} {step}/{stage_steps}: {metrics}",
                    flush=True,
                )
        if best_state is not None:
            core.load_state_dict(best_state)
        save_checkpoint(checkpoint, core, {
            "phase": "transfer", "stage": stage, "context": context,
            "validation_output_nmse": best,
        })
    return history


def fit_progressive_lm(model, layer, core, blocks, samples, validation,
                       contexts, steps_by_stage, device, seed, checkpoint, name):
    history = []
    rng = random.Random(seed)
    with replace_frontier_attention(model, layer, core) as wrapper:
        for stage, (context, stage_steps) in enumerate(zip(contexts, steps_by_stage), 1):
            core.train()
            lr = 4e-4 * (0.72 ** (stage - 1))
            optimizer = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=1e-4)
            initial = statistics.fmean(evaluate_nll(model, validation, context, device))
            best = initial
            best_state = {
                key: value.detach().cpu().clone() for key, value in core.state_dict().items()
            }
            history.append({
                "candidate": name, "phase": "lm", "stage": stage,
                "context": context, "step": 0, "validation_nll": initial,
            })

            for step in range(1, stage_steps + 1):
                i = rng.randrange(len(blocks))
                optimizer.zero_grad(set_to_none=True)
                inputs = blocks[i][:context].unsqueeze(0).to(device)
                labels = blocks[i][1:context + 1].to(device)
                logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
                ce = F.cross_entropy(logits[0].float(), labels)

                _, _, _, teacher, _ = slice_sample(samples[i], context, device)
                imitation = nmse(teacher, wrapper.last_output)
                alignment = 1.0 - cosine(teacher, wrapper.last_output)
                loss = 2.0 * ce + 0.08 * imitation + 0.02 * alignment
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{name}: non-finite LM loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                wrapper.last_output = None

                if step == 1 or step % max(6, stage_steps // 3) == 0 or step == stage_steps:
                    score = statistics.fmean(evaluate_nll(model, validation, context, device))
                    history.append({
                        "candidate": name, "phase": "lm", "stage": stage,
                        "context": context, "step": step,
                        "train_loss": float(loss.detach()), "train_ce": float(ce.detach()),
                        "validation_nll": score,
                    })
                    if score < best:
                        best = score
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in core.state_dict().items()
                        }
                    print(
                        f"{name} lm ctx={context} {step}/{stage_steps}: val NLL={score:.6f}",
                        flush=True,
                    )
            core.load_state_dict(best_state)
            save_checkpoint(checkpoint, core, {
                "phase": "lm", "stage": stage, "context": context,
                "validation_nll": best,
            })
    return history


@torch.no_grad()
def score(model, layer, core, blocks, reference, context, device, margin, seed,
          bootstrap=True):
    with replace_frontier_attention(model, layer, core):
        values = evaluate_nll(model, blocks, context, device)
    mean = statistics.fmean(values)
    ref_mean = statistics.fmean(reference)
    if bootstrap:
        delta, low, high = paired_interval(values, reference, seed=seed, repeats=3000)
    else:
        delta = mean - ref_mean
        low = high = None
    state = core.recurrent_state_bytes(context=context)
    ratio = state / transformer_kv_bytes(model, context)
    verdict = None
    if bootstrap:
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
        "transformer_kv_bytes_fp16": transformer_kv_bytes(model, context),
        "state_vs_transformer_fp16": ratio,
        "verdict": verdict,
        "documents": len(values),
    }


@torch.no_grad()
def benchmark_core(core, sample, device, repeats=5):
    q, k, v, _, route = (x.to(device) for x in sample)
    exact = lambda: softmax_reference(q, k, v, core.groups)
    candidate = lambda: core(q, k, v, routed_v=route if core.use_clvr else None)
    exact_ms, exact_peak = time_kernel(exact, device, repeats)
    candidate_ms, candidate_peak = time_kernel(candidate, device, repeats)

    _, state = core(
        q[:, :, :-1], k[:, :, :-1], v[:, :, :-1],
        routed_v=route[:, :, :-1] if core.use_clvr else None,
        return_state=True,
    )
    step = lambda: core(
        q[:, :, -1:], k[:, :, -1:], v[:, :, -1:], state=state,
        routed_v=route[:, :, -1:] if core.use_clvr else None,
    )
    exact_step = lambda: F.scaled_dot_product_attention(
        q[:, :, -1:],
        k.repeat_interleave(core.groups, 1),
        v.repeat_interleave(core.groups, 1),
        dropout_p=0.0, is_causal=False,
    )
    step_ms, _ = time_kernel(step, device, repeats)
    exact_step_ms, _ = time_kernel(exact_step, device, repeats)
    return {
        "prefill_ms": candidate_ms,
        "transformer_prefill_ms": exact_ms,
        "prefill_speedup": exact_ms / candidate_ms,
        "decode_step_ms": step_ms,
        "transformer_decode_step_ms": exact_step_ms,
        "decode_speedup": exact_step_ms / step_ms,
        "peak_extra_bytes": candidate_peak,
        "transformer_peak_extra_bytes": exact_peak,
    }


@torch.no_grad()
def compiled_timing(core, sample, device):
    q, k, v, _, route = (x.to(device) for x in sample)
    args = (q, k, v)
    kwargs = {"routed_v": route if core.use_clvr else None}
    eager_ms, _ = time_kernel(lambda: core(*args, **kwargs), device, repeats=5)
    result = {"eager_prefill_ms": eager_ms, "compiled_prefill_ms": None,
              "compile_speedup": None}
    if not hasattr(torch, "compile"):
        return result
    try:
        compiled = torch.compile(core, mode="reduce-overhead", fullgraph=False)
        compiled(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compiled_ms, _ = time_kernel(lambda: compiled(*args, **kwargs), device, repeats=5)
        result.update(
            compiled_prefill_ms=compiled_ms,
            compile_speedup=eager_ms / compiled_ms,
        )
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"[:600]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "balanced", "strong"], default="balanced")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--curriculum-contexts", default="256,512,1024")
    parser.add_argument("--validation-contexts", default="512,1024,2048")
    parser.add_argument("--test-contexts", default="512,1024,2048,4096")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-seed", type=int, default=12031)
    parser.add_argument("--nll-margin", type=float, default=0.02)
    parser.add_argument("--output-dir", default="result/pdelta3-frontier-layer")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    curriculum = [int(x) for x in args.curriculum_contexts.split(",")]
    validation_contexts = [int(x) for x in args.validation_contexts.split(",")]
    test_contexts = [int(x) for x in args.test_contexts.split(",")]
    if curriculum != sorted(curriculum) or len(curriculum) != 3:
        raise ValueError("curriculum-contexts must contain three increasing contexts")
    if args.layer < 1:
        raise ValueError("layer must be >=1 for CLVR")
    cfg = profile(args.profile)
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    api = HfApi()
    model_sha = api.model_info(args.base_model).sha
    data_sha = api.dataset_info(args.dataset).sha
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=model_sha)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device.type == "cuda" else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=model_sha, torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    if args.layer >= len(model.model.layers):
        raise ValueError("layer is outside the model")

    max_context = max(curriculum + validation_contexts + test_contexts)
    stream = load_dataset(
        args.dataset, args.dataset_config, split="train", streaming=True,
        revision=data_sha,
    ).shuffle(seed=args.dataset_seed, buffer_size=4096)
    counts = {
        "train": cfg["train"],
        "validation": cfg["validation"],
        "test": cfg["test"],
    }
    blocks, hashes = collect_documents(
        stream, tokenizer, counts, length=max_context,
    )

    capture_context = max(curriculum)
    train_samples = capture_frontier_samples(
        model, blocks["train"], args.layer, capture_context, device
    )
    validation_samples = capture_frontier_samples(
        model, blocks["validation"], args.layer, capture_context, device
    )

    # Exact replacement plumbing sanity check.
    sanity_context = validation_contexts[0]
    teacher_sanity = evaluate_nll(model, blocks["validation"], sanity_context, device)
    with replace_frontier_attention(model, args.layer, None):
        exact_sanity = evaluate_nll(model, blocks["validation"], sanity_context, device)
    exact_error = max(abs(a - b) for a, b in zip(teacher_sanity, exact_sanity))
    if exact_error > 0.01:
        raise RuntimeError(f"Exact replacement control mismatch: {exact_error:.6f}")

    specs = candidate_specs()
    cores = {}
    records = []
    history = []

    for spec in specs:
        name = spec["name"]
        print("\n" + "=" * 96 + f"\n{name}", flush=True)
        torch.manual_seed(args.seed)
        core = make_core(spec, model, device)
        checkpoint = outdir / "checkpoints" / f"{name}.pt"
        history += fit_progressive_transfer(
            core, train_samples, validation_samples,
            curriculum, cfg["transfer"], device, args.seed,
            checkpoint, name,
        )
        history += fit_progressive_lm(
            model, args.layer, core,
            blocks["train"], train_samples, blocks["validation"],
            curriculum, cfg["lm"], device, args.seed,
            checkpoint, name,
        )
        core.eval()

        transfer = evaluate_transfer(core, validation_samples, curriculum[-1], device)
        timing = benchmark_core(core, validation_samples[0], device, repeats=5)
        record = {
            **spec,
            **transfer,
            **timing,
            **core.decay_statistics(),
            **core.gate_statistics(),
            "trainable_parameters": sum(p.numel() for p in core.parameters() if p.requires_grad),
            "state_bytes": core.recurrent_state_bytes(),
            "state_vs_transformer_fp16_1024": (
                core.recurrent_state_bytes() / transformer_kv_bytes(model, 1024)
            ),
        }
        records.append(record)
        save_checkpoint(checkpoint, core, {
            **record,
            "model_revision": model_sha,
            "dataset_revision": data_sha,
            "curriculum": curriculum,
        })

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reloaded = FrontierPDelta3Layer(**payload["config"]).to(device).eval()
        reloaded.load_state_dict(payload["state_dict"])
        check = evaluate_transfer(reloaded, validation_samples[:1], curriculum[-1], device)
        if not math.isfinite(check["output_nmse"]):
            raise RuntimeError(f"{name}: checkpoint reload produced non-finite output")
        cores[name] = core
        pd.DataFrame(records).to_csv(outdir / "candidate_diagnostics.csv", index=False)
        pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    teacher_validation = {
        context: evaluate_nll(model, blocks["validation"], context, device)
        for context in validation_contexts
    }
    validation_rows = []
    weights = {validation_contexts[0]: 0.4, validation_contexts[1]: 0.35,
               validation_contexts[2]: 0.25}
    composite = {name: 0.0 for name in cores}
    for name, core in cores.items():
        for context in validation_contexts:
            result = score(
                model, args.layer, core, blocks["validation"],
                teacher_validation[context], context, device,
                args.nll_margin, args.seed, bootstrap=False,
            )
            validation_rows.append({"candidate": name, "context": context, **result})
            composite[name] += weights[context] * result["delta_nll"]

    quality_winner = min(composite, key=composite.get)
    diagnostics_by_name = {row["name"]: row for row in records}
    eligible = [
        name for name, score_value in composite.items()
        if score_value <= composite[quality_winner] + 0.005
    ]
    efficient_winner = min(
        eligible,
        key=lambda name: diagnostics_by_name[name]["decode_step_ms"],
    )
    selection = {
        "quality_winner": quality_winner,
        "efficient_winner": efficient_winner,
        "quality_criterion": "0.40*deltaNLL@512 + 0.35*@1024 + 0.25*@2048 on validation only",
        "validation_composite_delta_nll": composite,
        "test_not_used_for_selection": True,
        "curriculum_contexts": curriculum,
        "validation_contexts": validation_contexts,
    }
    (outdir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    teacher_test = {
        context: evaluate_nll(model, blocks["test"], context, device)
        for context in test_contexts
    }
    test_rows = []
    for context in test_contexts:
        for name, core in cores.items():
            result = score(
                model, args.layer, core, blocks["test"], teacher_test[context],
                context, device, args.nll_margin,
                args.seed + context + sum(map(ord, name)), bootstrap=True,
            )
            test_rows.append({
                "candidate": name,
                "context": context,
                "selected_quality": name == quality_winner,
                "selected_efficiency": name == efficient_winner,
                **result,
            })
            print(
                f"TEST {name} ctx={context}: dNLL={result['delta_nll']:+.6f} "
                f"CI=[{result['ci95_low']:+.6f},{result['ci95_high']:+.6f}] "
                f"state={100*result['state_vs_transformer_fp16']:.2f}%",
                flush=True,
            )

    winner = cores[quality_winner]
    winner_diag = {
        **benchmark_core(winner, validation_samples[0], device, repeats=10),
        **compiled_timing(winner, validation_samples[0], device),
        **winner.decay_statistics(),
        **winner.gate_statistics(),
        "state_bytes": winner.recurrent_state_bytes(),
    }

    pd.DataFrame(validation_rows).to_csv(
        outdir / "validation_context_summary.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(outdir / "test_summary.csv", index=False)
    pd.DataFrame(records).to_csv(outdir / "candidate_diagnostics.csv", index=False)
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    strict = [
        row for row in test_rows
        if row["candidate"] == quality_winner and row["verdict"] == "strict_quality_win"
    ]
    report = {
        "experiment": "pdelta3-frontier-layer-lab",
        "profile": args.profile,
        "base_model": args.base_model,
        "model_revision": model_sha,
        "dataset": args.dataset,
        "dataset_revision": data_sha,
        "document_hashes": hashes,
        "layer": args.layer,
        "curriculum_contexts": curriculum,
        "validation_contexts": validation_contexts,
        "test_contexts": test_contexts,
        "exact_softmax_wrapper_max_document_nll_error": exact_error,
        "selection": selection,
        "winner_diagnostics": winner_diag,
        "candidate_specs": specs,
        "strict_quality_win_contexts_for_selected": [row["context"] for row in strict],
        "research_design": {
            "channel_decay": "content-dependent per-feature KDA/GDN2-style decay in fp32",
            "gdn2": "independent key-channel erase and value-channel write gates",
            "clvr": "previous layer V is projected/aligned and routed into the current write value; no extra temporal memory",
            "progressive_training": "256 -> 512 -> 1024",
            "selection": "long-context validation composite; held-out test never selects architecture",
        },
        "limitations": (
            "One pretrained attention layer is replaced, one seed is used, and kernels are PyTorch references. "
            "The GDN2/CLVR variants are independent adaptations to frozen SmolLM2 QKV rather than reproductions "
            "of full frontier-model pretraining. Whole-model claims require multi-layer training and independent seeds."
        ),
    }
    (outdir / "pdelta3_frontier_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nSELECTION", json.dumps(selection, indent=2), flush=True)
    print("STRICT SELECTED TRANSFORMER WINS:", report["strict_quality_win_contexts_for_selected"], flush=True)
    print("RESULT_DIR", outdir, flush=True)


if __name__ == "__main__":
    main()
