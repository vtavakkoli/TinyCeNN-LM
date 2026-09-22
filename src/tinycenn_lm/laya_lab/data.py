from __future__ import annotations
import json, random

def _load_typed_split(split: str):
    from datasets import load_dataset
    return load_dataset("LocalLLaMA/typed-decisions", "all", split=split)

def _parse_jsonish(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

def build_training_items(agent, ds, limit_cases: int, max_len: int, seed: int):
    from laya.common import QTYPES, build_sequence, render_options
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:min(limit_cases, len(rows))]
    items = []
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    for row in rows:
        state = _parse_jsonish(row["state"])
        questions = _parse_jsonish(row["questions"])
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            try:
                ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            except Exception:
                continue
            if len(markers) != len(render_options(q)):
                continue
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "qid": qid})
    rng.shuffle(items)
    return items

def _batch_from_items(items, offset: int, batch_size: int, pad_id: int):
    from laya.common import collate_items
    if not items:
        raise RuntimeError("no Laya training items were built")
    sel = [items[(offset + i) % len(items)] for i in range(batch_size)]
    return collate_items([sel], pad_id)

def dataset_cases(ds, limit_cases: int | None = None):
    cases = []
    for row in list(ds)[:limit_cases]:
        state = _parse_jsonish(row["state"])
        questions = _parse_jsonish(row["questions"])
        gold = _parse_jsonish(row["gold"])
        cases.append((state, questions, gold, row.get("workflow", "unknown")))
    return cases
