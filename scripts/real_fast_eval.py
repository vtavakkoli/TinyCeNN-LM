"""Teacher-independent fast evaluation for experimental causal language models.

Reports ground-truth ARC-Easy multiple-choice accuracy and a small explicitly
specified chat task suite. Scores answer continuation *likelihoods* rather than
string-agreement with a teacher. Not a replacement for official lm-eval runs.

Important: models must be in eval mode; no cache or speculative decoding is used.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
import random
import re
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F


CHAT_CASES = (
    {"id": "vienna", "prompt": "What is the capital of Austria? Answer in one short sentence.", "kind": "contains", "answer": "vienna"},
    {"id": "math_17_25", "prompt": "What is 17 plus 25? Reply with the number only.", "kind": "number_only", "answer": "42"},
    {"id": "math_23_19", "prompt": "What is 23 plus 19? Reply with the number only.", "kind": "number_only", "answer": "42"},
    {"id": "exact_three_words", "prompt": "Describe the Moon using exactly three words and no punctuation.", "kind": "three_words"},
    {"id": "recall_code", "prompt": "Remember this exact code: AURORA-731. Repeat only the code.", "kind": "contains", "answer": "aurora-731"},
)


def score_chat_response(case, response):
    """Return an objective *limited* rubric result, never an LLM-judge guess."""
    content = response.strip()
    kind = case["kind"]
    if kind == "number_only":
        ok = bool(re.fullmatch(r"\s*[+]?" + re.escape(case["answer"]) + r"(?:\.0+)?\s*", content))
    elif kind == "contains":
        ok = case["answer"].lower() in content.casefold()
    elif kind == "three_words":
        words = re.findall(r"[A-Za-z]+", content)
        ok = len(words) == 3 and bool(re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){2}", content))
    else:
        raise ValueError(f"Unknown chat rubric: {kind}")
    return bool(ok)


def _as_input_ids(rendered):
    """Normalize HF tensor/list/BatchEncoding chat templates to [batch, tokens].

    Transformers may return BatchEncoding (a Mapping but NOT a dict) even
    with return_tensors='pt'. Calling .shape on that object fails; select the
    input_ids field *before* moving it to the model device.
    """
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise ValueError("Chat tokenizer output is missing input_ids")
        rendered = rendered["input_ids"]
    elif not torch.is_tensor(rendered) and hasattr(rendered, "__getitem__"):
        # Defensive support for mapping-like wrappers from older tokenizers.
        try:
            rendered = rendered["input_ids"]
        except (KeyError, TypeError, IndexError):
            pass
    if not torch.is_tensor(rendered):
        rendered = torch.as_tensor(rendered, dtype=torch.long)
    if rendered.ndim == 1:
        rendered = rendered.unsqueeze(0)
    if rendered.ndim != 2 or rendered.size(0) != 1 or rendered.size(1) < 1:
        raise ValueError(f"Expected a single nonempty tokenized chat, got {tuple(rendered.shape)}")
    return rendered.to(dtype=torch.long)


def render_chat_input(tokenizer, user):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt")
        return _as_input_ids(rendered)
    return _as_input_ids(tokenizer("User: " + user + "\nAssistant:", return_tensors="pt"))


@torch.inference_mode()
def greedy_no_cache(model, tokenizer, user, *, max_new_tokens=48, max_context=512):
    ids = render_chat_input(tokenizer, user).to(next(model.parameters()).device)
    prefix = ids.shape[1]
    if prefix > max_context:
        raise ValueError("Chat prompt exceeds max_context; no silent truncation")
    output_ids = []
    elapsed = 0.0
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        out = model(input_ids=ids, use_cache=False, return_dict=True)
        elapsed += time.perf_counter() - start
        nxt = int(out.logits[0, -1].argmax().item())
        if nxt in {v for v in [getattr(tokenizer, "eos_token_id", None)] if v is not None}:
            break
        output_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], -1)
        if ids.shape[-1] >= max_context:
            break
    return {"response": tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            "generated_tokens": len(output_ids), "prompt_tokens": prefix,
            "seconds": round(elapsed, 4)}


@torch.inference_mode()
def real_chat_eval(model, tokenizer, *, cases=CHAT_CASES, max_new_tokens=48):
    output = []
    for case in cases:
        generated = greedy_no_cache(model, tokenizer, case["prompt"], max_new_tokens=max_new_tokens)
        output.append({"id": case["id"], "prompt": case["prompt"],
                       "expected_rule": case["kind"], "expected": case.get("answer"),
                       "pass": score_chat_response(case, generated["response"]), **generated})
    return {"accuracy": sum(x["pass"] for x in output) / len(output) if output else None,
            "n": len(output), "rows": output,
            "warning": "Five transparent prompt rules; not a comprehensive measure of intelligence."}


def prepare_arc_example(row, *, max_options=5):
    """Read genuine labeled ARC-Easy question/choice fields, preserving reference label."""
    keys = row["choices"]["label"]
    values = row["choices"]["text"]
    if len(values) < 2 or len(values) != len(keys) or len(values) > max_options:
        return None
    if row.get("answerKey") not in keys:
        return None
    letters = "ABCDE"[:len(values)]
    correct = letters[keys.index(row["answerKey"])]
    prompt = ("Question: " + row["question"].strip() + "\n" +
              "\n".join(f"{letter}. {value.strip()}" for letter, value in zip(letters, values))+
              "\nAnswer:")
    return {"id": str(row.get("id", "")), "prompt": prompt, "options": list(letters),
            "answer": correct}


def _forward(model, ids):
    return model(input_ids=ids, use_cache=False, return_dict=True).logits


@torch.inference_mode()
def score_continuations(model, tokenizer, prompt, options, *, max_length=512):
    """Compute unnormalized sum of conditional continuation log probs, no teacher.

    This avoids treating a possibly different wording as an error. The prompt
    is tokenized separately and each alternative is appended as encoded tokens,
    to avoid tokenizer boundary-dependent mask mistakes.
    """
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prefix) < 1:
        raise ValueError("empty tokenized prompt")
    result = {}
    device = next(model.parameters()).device
    for choice in options:
        suffix = tokenizer.encode(" " + str(choice), add_special_tokens=False)
        if not suffix:
            raise ValueError("empty option tokens")
        if len(prefix) + len(suffix) > max_length:
            raise ValueError("ARC example too long")
        ids = torch.tensor([prefix + suffix], dtype=torch.long, device=device)
        logits = _forward(model, ids)[0, len(prefix) - 1:len(prefix) + len(suffix)-1].float()
        targets = torch.tensor(suffix, device=device)
        result[str(choice)] = float(F.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).sum().item())
    return result


@torch.inference_mode()
def arc_easy_eval(model, tokenizer, *, limit=20, seed=2031, split="validation", dataset=None, max_length=512):
    """Ground-truth multiple-choice diagnostic, reproducible, no teacher logits.

    *External validation* comes from ARC-Easy labels, not from the teacher.
    Fails visibly if the dataset cannot load; caller must not treat missing eval as passing.
    """
    if dataset is None:
        from datasets import load_dataset
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    if limit < 1:
        raise ValueError("limit must be positive")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows = []
    for i in indices:
        row = prepare_arc_example(dataset[i])
        if row is None:
            continue
        try:
            probs = score_continuations(model, tokenizer, row["prompt"], row["options"], max_length=max_length)
        except ValueError as exc:
            if "too long" in str(exc):
                continue
            raise
        answer = max(probs, key=probs.get)
        rows.append({"id": row["id"], "gold": row["answer"], "pred": answer,
                     "correct": answer == row["answer"], "log_likelihoods": probs})
        if len(rows) >= limit:
            break
    return {"benchmark": "ARC-Easy", "split": split, "requested": limit,
            "evaluated": len(rows), "seed": seed,
            "accuracy": sum(x["correct"] for x in rows)/len(rows) if rows else None,
            "rows": rows,
            "scoring": "unnormalized conditional log-likelihood of answer-label continuations",
            "warning": "Small deterministic diagnostic subset; not official ARC score."}


def with_native(model, context):
    """Helper provided for uniform explicit native/student comparison context."""
    return context() if context is not None else nullcontext()


def quality_gate(native_metrics, student_metrics, latency, *, max_delta_nll=0.05,
                 max_gold_accuracy_drop=0.02, min_speedup=1.0):
    """No teacher-agreement or activation cosine criterion: use real gold data."""
    required = ("teacher_nll", "student_nll", "teacher_gold_accuracy", "student_gold_accuracy")
    if not all(k in student_metrics for k in required):
        raise KeyError(f"Missing ground-truth metrics: {required}")
    nll_gap = student_metrics["student_nll"]-student_metrics["teacher_nll"]
    gold_drop = student_metrics["teacher_gold_accuracy"]-student_metrics["student_gold_accuracy"]
    return {"max_delta_nll": max_delta_nll, "max_gold_accuracy_drop":max_gold_accuracy_drop,
            "measured_delta_nll": nll_gap, "measured_gold_accuracy_drop":gold_drop,
            "quality_pass": nll_gap <= max_delta_nll and gold_drop <= max_gold_accuracy_drop,
            "speed_pass": bool(latency.get("available",False) and latency.get("speedup",0)>min_speedup),
            "teacher_agreement_used": False}
