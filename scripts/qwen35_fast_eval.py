#!/usr/bin/env python3
"""Fast sampled evaluation for TinyCeNN Qwen3.5 standalone releases.

This is deliberately *not* an official benchmark reproduction. It samples a
small deterministic subset from public benchmarks and uses zero-shot next-token
letter scoring so that a 0.8B model can be checked quickly on a single GPU.

Default suite (50 examples each = 200 MCQs):
- MMLU-Pro
- PIQA
- MMMLU German (DE_DE)
- GPQA Diamond

It also runs a short greedy generation probe and reports latency, tokens/sec and
peak CUDA memory. Results are written to JSON and, by default, a concise FastEval
section is inserted into the standalone model's README.md/model card.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
CARD_START = "<!-- TINYCENN_FASTEVAL_START -->"
CARD_END = "<!-- TINYCENN_FASTEVAL_END -->"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--samples", type=int, default=50, help="examples per benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--compare-base", action="store_true")
    p.add_argument("--generation-prompts", type=int, default=3)
    p.add_argument("--generation-tokens", type=int, default=24)
    p.add_argument(
        "--update-model-card",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="insert/replace the TinyCeNN FastEval section in README.md",
    )
    return p.parse_args()


def device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_standalone(path: Path, device: torch.device):
    loader_path = path / "load_model.py"
    if not loader_path.exists():
        raise FileNotFoundError(loader_path)
    spec = importlib.util.spec_from_file_location("tinycenn_standalone_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.path.insert(0, str(path))
    try:
        spec.loader.exec_module(module)
        model, tokenizer = module.load_model(path, device=str(device))
    finally:
        if sys.path and sys.path[0] == str(path):
            sys.path.pop(0)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def load_base(device: torch.device, dtype: torch.dtype):
    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN") or None,
    ).to(device).eval()
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return model, tok


def take_rows(dataset, n: int, seed: int):
    # IterableDataset and Dataset both expose shuffle/take in recent datasets.
    try:
        shuffled = dataset.shuffle(seed=seed, buffer_size=max(1000, n * 20))
        if hasattr(shuffled, "take"):
            return list(shuffled.take(n))
        return [shuffled[i] for i in range(min(n, len(shuffled)))]
    except TypeError:
        # Non-streaming Dataset.shuffle has no buffer_size.
        shuffled = dataset.shuffle(seed=seed)
        return [shuffled[i] for i in range(min(n, len(shuffled)))]


def load_mmmlu_de(split="test"):
    errors = []
    for config in ("DE_DE", "de_de"):
        try:
            return load_dataset("openai/MMMLU", config, split=split, streaming=True)
        except Exception as exc:
            errors.append(f"{config}: {exc}")
    raise RuntimeError("Could not load German MMMLU: " + " | ".join(errors))


def build_examples(samples: int, seed: int):
    suites: dict[str, list[dict]] = {}

    # MMLU-Pro: question, options, answer_index.
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
    rows = take_rows(ds, samples, seed)
    suites["MMLU-Pro"] = [
        {
            "question": r["question"],
            "options": list(r["options"]),
            "answer_index": int(
                r.get(
                    "answer_index",
                    string.ascii_uppercase.index(str(r["answer"]).strip().upper()),
                )
            ),
        }
        for r in rows
    ]

    # PIQA: goal/sol1/sol2/label.
    ds = load_dataset("lighteval/piqa", split="validation", streaming=True)
    rows = take_rows(ds, samples, seed + 1)
    piqa = []
    for r in rows:
        question = r.get("goal", r.get("question", ""))
        a = r.get("sol1", r.get("choice_a"))
        b = r.get("sol2", r.get("choice_b"))
        label = r.get("label", r.get("answer"))
        if isinstance(label, str) and label.upper() in ("A", "B"):
            idx = 0 if label.upper() == "A" else 1
        else:
            idx = int(label)
        piqa.append({"question": question, "options": [a, b], "answer_index": idx})
    suites["PIQA"] = piqa

    # MMMLU German: Question/A/B/C/D/Answer.
    ds = load_mmmlu_de()
    rows = take_rows(ds, samples, seed + 2)
    mmmlu = []
    for r in rows:
        answer = str(r["Answer"]).strip().upper()
        idx = string.ascii_uppercase.index(answer)
        mmmlu.append(
            {
                "question": r["Question"],
                "options": [r["A"], r["B"], r["C"], r["D"]],
                "answer_index": idx,
            }
        )
    suites["MMMLU-DE"] = mmmlu

    # Public GPQA Diamond mirror. Shuffle the four options per item so the
    # original column order does not leak the correct answer position.
    ds = load_dataset("Wanfq/gpqa", "gpqa_diamond", split="train", streaming=True)
    rows = take_rows(ds, samples, seed + 3)
    gpqa = []
    for i, r in enumerate(rows):
        question = r.get("Question") or r.get("Pre-Revision Question")
        correct = r.get("Correct Answer") or r.get("Pre-Revision Correct Answer")
        wrong = [
            r.get("Incorrect Answer 1") or r.get("Pre-Revision Incorrect Answer 1"),
            r.get("Incorrect Answer 2") or r.get("Pre-Revision Incorrect Answer 2"),
            r.get("Incorrect Answer 3") or r.get("Pre-Revision Incorrect Answer 3"),
        ]
        tagged = [(correct, True)] + [(x, False) for x in wrong]
        rr = random.Random(seed * 100000 + i)
        rr.shuffle(tagged)
        gpqa.append(
            {
                "question": question,
                "options": [x[0] for x in tagged],
                "answer_index": next(j for j, x in enumerate(tagged) if x[1]),
            }
        )
    suites["GPQA-Diamond"] = gpqa
    return suites


def prompt_for(example: dict):
    letters = string.ascii_uppercase
    lines = [
        "Choose the best answer. Reply with only the option letter.",
        "",
        f"Question: {example['question']}",
    ]
    for i, option in enumerate(example["options"]):
        lines.append(f"{letters[i]}. {option}")
    lines.append("Answer:")
    return "\n".join(lines)


@torch.inference_mode()
def score_example(model, tokenizer, example: dict):
    device = next(model.parameters()).device
    prompt = prompt_for(example)
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded.input_ids.to(device)
    mask = encoded.get("attention_mask")
    if mask is not None:
        mask = mask.to(device)
    out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    logits = out.logits[0, -1].float()
    scores = []
    for i in range(len(example["options"])):
        letter = string.ascii_uppercase[i]
        token_ids = tokenizer(" " + letter, add_special_tokens=False).input_ids
        if not token_ids:
            token_ids = tokenizer(letter, add_special_tokens=False).input_ids
        scores.append(float(logits[token_ids[0]]))
    pred = max(range(len(scores)), key=scores.__getitem__)
    return pred, int(example["answer_index"]), int(ids.shape[1])


def evaluate_mcq(model, tokenizer, suites: dict[str, list[dict]]):
    benchmark_rows = []
    total_correct = 0
    total_items = 0
    total_input_tokens = 0
    start_all = time.perf_counter()

    for name, examples in suites.items():
        t0 = time.perf_counter()
        correct = 0
        tokens = 0
        for idx, example in enumerate(examples, 1):
            pred, gold, n_tokens = score_example(model, tokenizer, example)
            correct += int(pred == gold)
            tokens += n_tokens
            if idx % 10 == 0 or idx == len(examples):
                print(f"  {name}: {idx}/{len(examples)}", flush=True)
        elapsed = time.perf_counter() - t0
        acc = 100.0 * correct / max(1, len(examples))
        benchmark_rows.append(
            {
                "benchmark": name,
                "samples": len(examples),
                "correct": correct,
                "accuracy_pct": round(acc, 2),
                "seconds": round(elapsed, 2),
                "items_per_second": round(len(examples) / max(elapsed, 1e-9), 3),
                "input_tokens": tokens,
            }
        )
        total_correct += correct
        total_items += len(examples)
        total_input_tokens += tokens
        print(
            f"{name}: {correct}/{len(examples)} = {acc:.1f}% ({elapsed/60:.1f} min)",
            flush=True,
        )

    elapsed_all = time.perf_counter() - start_all
    return benchmark_rows, {
        "samples": total_items,
        "correct": total_correct,
        "accuracy_pct": round(100.0 * total_correct / max(1, total_items), 2),
        "seconds": round(elapsed_all, 2),
        "input_tokens": total_input_tokens,
    }


@torch.inference_mode()
def generation_probe(model, tokenizer, prompt_count=3, max_new_tokens=24):
    prompts = [
        "Explain in two sentences why Vienna is the capital of Austria.",
        "A robot has three batteries and uses one battery every two hours. Explain how long it can operate.",
        "In German, briefly explain what an API gateway does.",
        "Summarize the main advantage of local recurrent processing in a neural network.",
    ][: max(1, prompt_count)]
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    total_new = 0
    t0 = time.perf_counter()
    outputs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        start_len = ids.shape[1]
        for _ in range(max_new_tokens):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat((ids, nxt), dim=1)
            if (
                tokenizer.eos_token_id is not None
                and int(nxt.item()) == int(tokenizer.eos_token_id)
            ):
                break
        new_tokens = ids.shape[1] - start_len
        total_new += new_tokens
        outputs.append(tokenizer.decode(ids[0, start_len:], skip_special_tokens=True)[:300])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    return {
        "prompts": len(prompts),
        "generated_tokens": int(total_new),
        "seconds": round(elapsed, 3),
        "tokens_per_second": round(total_new / max(elapsed, 1e-9), 3),
        "peak_vram_gib": round(peak, 3),
        "sample_outputs": outputs,
    }


def run_one(name: str, model, tokenizer, suites, args):
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rows, overall = evaluate_mcq(model, tokenizer, suites)
    generation = generation_probe(
        model,
        tokenizer,
        prompt_count=args.generation_prompts,
        max_new_tokens=args.generation_tokens,
    )
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    return {
        "name": name,
        "benchmarks": rows,
        "overall": overall,
        "generation": generation,
        "peak_vram_gib_observed": round(peak, 3),
    }


def _fmt_pct(value):
    return "—" if value is None else f"{float(value):.2f}%"


def _fmt_delta(value):
    if value is None:
        return "—"
    return f"{float(value):+.2f} pp"


def build_model_card_section(report: dict):
    results = report.get("results", [])
    if not results:
        raise ValueError("FastEval report has no model results")

    custom = results[0]
    base = results[1] if len(results) > 1 else None
    base_rows = {
        row["benchmark"]: row for row in (base.get("benchmarks", []) if base else [])
    }

    lines = [
        CARD_START,
        "## TinyCeNN FastEval",
        "",
        "> **Sampled evaluation, not an official full benchmark reproduction.** "
        "Scores use deterministic zero-shot next-token option-letter scoring on small public subsets.",
        "",
        f"- Suite: `{report.get('suite', 'TinyCeNN FastEval')}`",
        f"- Samples per benchmark: **{report.get('samples_per_benchmark', '—')}**",
        f"- Seed: `{report.get('seed', '—')}`",
        f"- Method: {report.get('method', '—')}",
        f"- Device: `{report.get('device', '—')}`",
        f"- Generated: `{report.get('generated_at_utc', '—')}`",
        "",
    ]

    if base:
        lines += [
            "| Benchmark | n | TinyCeNN | Qwen3.5-0.8B base | Δ |",
            "|---|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            "| Benchmark | n | Accuracy |",
            "|---|---:|---:|",
        ]

    for row in custom.get("benchmarks", []):
        name = row["benchmark"]
        acc = row.get("accuracy_pct")
        n = row.get("samples")
        if base:
            b = base_rows.get(name, {})
            bacc = b.get("accuracy_pct")
            delta = None if bacc is None or acc is None else float(acc) - float(bacc)
            lines.append(
                f"| {name} | {n} | **{_fmt_pct(acc)}** | {_fmt_pct(bacc)} | {_fmt_delta(delta)} |"
            )
        else:
            lines.append(f"| {name} | {n} | **{_fmt_pct(acc)}** |")

    custom_overall = custom.get("overall", {})
    if base:
        base_overall = base.get("overall", {})
        cacc = custom_overall.get("accuracy_pct")
        bacc = base_overall.get("accuracy_pct")
        delta = None if cacc is None or bacc is None else float(cacc) - float(bacc)
        lines.append(
            f"| **Overall sampled** | **{custom_overall.get('samples', '—')}** | "
            f"**{_fmt_pct(cacc)}** | **{_fmt_pct(bacc)}** | **{_fmt_delta(delta)}** |"
        )
    else:
        lines.append(
            f"| **Overall sampled** | **{custom_overall.get('samples', '—')}** | "
            f"**{_fmt_pct(custom_overall.get('accuracy_pct'))}** |"
        )

    lines += ["", "### Fast inference probe", ""]
    if base:
        lines += [
            "| Model | Generated tokens | Tokens/s | Peak VRAM (GiB) |",
            "|---|---:|---:|---:|",
        ]
        for result in (custom, base):
            g = result.get("generation", {})
            lines.append(
                f"| {result.get('name', 'model')} | {g.get('generated_tokens', '—')} | "
                f"{g.get('tokens_per_second', '—')} | {g.get('peak_vram_gib', '—')} |"
            )
    else:
        g = custom.get("generation", {})
        lines += [
            "| Generated tokens | Tokens/s | Peak VRAM (GiB) |",
            "|---:|---:|---:|",
            f"| {g.get('generated_tokens', '—')} | {g.get('tokens_per_second', '—')} | {g.get('peak_vram_gib', '—')} |",
        ]

    lines += [
        "",
        "The generation probe uses greedy decoding with `use_cache=False`, matching the current TinyCeNN runtime. "
        "These sampled scores are intended for rapid regression/comparison testing and should not be reported as "
        "the official full-dataset benchmark results.",
        "",
        "Detailed machine-readable results: [`fast_eval.json`](./fast_eval.json).",
        CARD_END,
    ]
    return "\n".join(lines)


def update_model_card(model_dir: Path, report: dict):
    readme = model_dir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
    else:
        text = f"# {model_dir.name}\n"

    section = build_model_card_section(report)
    if CARD_START in text and CARD_END in text:
        before = text.split(CARD_START, 1)[0].rstrip()
        after = text.split(CARD_END, 1)[1].lstrip()
        text = before + "\n\n" + section
        if after:
            text += "\n\n" + after
        text += "\n"
    else:
        text = text.rstrip() + "\n\n" + section + "\n"

    readme.write_text(text, encoding="utf-8")
    return readme


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    output = Path(args.output) if args.output else model_dir / "fast_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    device, dtype = device_dtype()
    print(f"device={device} dtype={dtype} samples_per_benchmark={args.samples}")
    print("Loading benchmark samples ...", flush=True)
    suites = build_examples(args.samples, args.seed)
    print("FastEval items:", {k: len(v) for k, v in suites.items()})

    model, tok = load_standalone(model_dir, device)
    results = [run_one(model_dir.name, model, tok, suites, args)]
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.compare_base:
        base, base_tok = load_base(device, dtype)
        results.append(run_one(BASE_MODEL, base, base_tok, suites, args))
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "suite": "TinyCeNN FastEval v1",
        "official_full_benchmark": False,
        "method": "deterministic sampled zero-shot next-token letter scoring",
        "samples_per_benchmark": args.samples,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {
            "MMLU-Pro": "TIGER-Lab/MMLU-Pro:test",
            "PIQA": "lighteval/piqa:validation",
            "MMMLU-DE": "openai/MMMLU:DE_DE:test",
            "GPQA-Diamond": "Wanfq/gpqa:gpqa_diamond:train",
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.update_model_card:
        card = update_model_card(model_dir, report)
        print("Updated model card:", card)

    print("\nFAST EVAL SUMMARY")
    for result in results:
        print(result["name"])
        for row in result["benchmarks"]:
            print(
                f"  {row['benchmark']:14s} {row['accuracy_pct']:6.2f}%  "
                f"n={row['samples']:3d}  {row['seconds']/60:5.1f} min"
            )
        print(
            f"  OVERALL        {result['overall']['accuracy_pct']:6.2f}%  "
            f"n={result['overall']['samples']}"
        )
        print(
            f"  GENERATION     {result['generation']['tokens_per_second']:.2f} tok/s  "
            f"peak={result['generation']['peak_vram_gib']:.2f} GiB"
        )
    print("Saved:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
