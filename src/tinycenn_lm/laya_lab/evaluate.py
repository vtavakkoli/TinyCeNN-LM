from __future__ import annotations
import math, time
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .core import LayaLabConfig, BaseLayaReplacementAttention

def _gold_for_question(qdef: dict, gold: dict):
    if qdef["type"] == "choice":
        keys = list(qdef["criteria"].keys())
        idx = keys.index(str(gold["label"]))
        soft = np.array([float(gold.get("probabilities", {}).get(k, 0.0)) for k in keys], dtype=float)
        return idx, soft, None
    if qdef["type"] == "noul":
        pt = float(gold.get("probabilities", {}).get("true", gold.get("noul", 0.5)))
        idx = 1 if str(gold["label"]).lower() == "true" else 0
        return idx, np.array([1.0 - pt, pt]), None
    n = len(qdef["criteria"])
    soft = np.array([float(gold.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)], dtype=float)
    return int(gold["label"]), soft, float(gold.get("score", gold["label"]))

def _answer_probs(answer: dict, qdef: dict):
    if qdef["type"] == "choice":
        keys = list(qdef["criteria"].keys())
        return np.array([float(answer["probabilities"][k]) for k in keys], dtype=float)
    if qdef["type"] == "score":
        n = len(qdef["criteria"])
        return np.array([float(answer["probabilities"][str(i)]) for i in range(n)], dtype=float)
    p = float(answer["noul"])
    return np.array([1.0 - p, p], dtype=float)

def _ece(conf, correct, bins: int = 15):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if not len(conf):
        return float("nan")
    total = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            total += float(s.mean()) * abs(float(conf[s].mean()) - float(correct[s].mean()))
    return total

def evaluate_agent(agent, cases, teacher_agent=None, label: str = "model"):
    rows, teacher_rows = [], []
    by_type: dict[str, list] = {}
    by_workflow: dict[str, list] = {}
    t0 = time.perf_counter()
    for state, questions, gold, workflow in cases:
        pred = agent.predict(state, questions)
        teach = teacher_agent.predict(state, questions) if teacher_agent is not None else None
        for qid, qdef in questions.items():
            p = _answer_probs(pred["answers"][qid], qdef)
            idx, soft, gold_score = _gold_for_question(qdef, gold[qid])
            row = (idx, p, soft, gold_score, qdef["type"], workflow)
            rows.append(row)
            by_type.setdefault(qdef["type"], []).append(row)
            by_workflow.setdefault(workflow, []).append(row)
            if teach is not None:
                teacher_rows.append((_answer_probs(teach["answers"][qid], qdef), p))
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    elapsed = time.perf_counter() - t0

    def metrics(sub):
        if not sub:
            return {"n": 0}
        correct, conf, brier, nll, soft_acc, brier_soft, score_mae = [], [], [], [], [], [], []
        for idx, p, soft, gold_score, qtype, _ in sub:
            p = np.asarray(p, float)
            p = p / max(p.sum(), 1e-12)
            correct.append(int(np.argmax(p)) == idx)
            conf.append(float(p.max()))
            onehot = np.eye(len(p))[idx]
            brier.append(float(((p - onehot) ** 2).sum()))
            nll.append(-math.log(max(float(p[idx]), 1e-12)))
            if soft.sum() > 0:
                s = soft / soft.sum()
                soft_acc.append(float((p * s).sum()))
                brier_soft.append(float(((p - s) ** 2).sum()))
            if gold_score is not None:
                score_mae.append(abs(float((np.arange(len(p)) * p).sum()) - gold_score))
        return {
            "n": len(sub),
            "accuracy": float(np.mean(correct)),
            "brier": float(np.mean(brier)),
            "nll": float(np.mean(nll)),
            "ece": float(_ece(conf, correct)),
            "soft_accuracy": float(np.mean(soft_acc)) if soft_acc else None,
            "brier_vs_soft": float(np.mean(brier_soft)) if brier_soft else None,
            "score_mae": float(np.mean(score_mae)) if score_mae else None,
        }

    out = metrics(rows)
    out.update(
        label=label,
        seconds=elapsed,
        ms_per_case=1000.0 * elapsed / max(len(cases), 1),
        by_type={k: metrics(v) for k, v in sorted(by_type.items())},
        by_workflow={k: metrics(v) for k, v in sorted(by_workflow.items())},
    )
    if teacher_rows:
        agree, kls, l1 = [], [], []
        for tp, sp in teacher_rows:
            tp = np.asarray(tp, float)
            sp = np.asarray(sp, float)
            tp = tp / max(tp.sum(), 1e-12)
            sp = sp / max(sp.sum(), 1e-12)
            agree.append(int(tp.argmax()) == int(sp.argmax()))
            kls.append(float(np.sum(
                tp * (np.log(np.clip(tp, 1e-9, 1)) - np.log(np.clip(sp, 1e-9, 1)))
            )))
            l1.append(float(np.abs(tp - sp).mean()))
        out["teacher_agreement"] = float(np.mean(agree))
        out["mean_teacher_kl"] = float(np.mean(kls))
        out["mean_probability_l1"] = float(np.mean(l1))
    return out

def _accept(local: dict, teacher_metrics: dict, student_metrics: dict, cfg: LayaLabConfig):
    agreement = student_metrics.get("teacher_agreement", 0.0)
    kl = student_metrics.get("mean_teacher_kl", float("inf"))
    accuracy_drop = teacher_metrics["accuracy"] - student_metrics["accuracy"]
    checks = {
        "local_nmse": local["nmse"] <= cfg.max_local_nmse,
        "local_cosine": local["cosine"] >= cfg.min_local_cosine,
        "teacher_agreement": agreement >= cfg.min_teacher_agreement,
        "teacher_kl": kl <= cfg.max_mean_kl,
        "accuracy_drop": accuracy_drop <= cfg.max_accuracy_drop,
    }
    return all(checks.values()), checks, accuracy_drop

def benchmark_latency(agent, state, questions, warmup: int = 3, repeats: int = 20):
    for _ in range(warmup):
        agent.predict(state, questions)
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        agent.predict(state, questions)
        if agent.device.type == "cuda":
            torch.cuda.synchronize(agent.device)
        values.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "repeats": repeats,
    }

def adapter_payload(model: nn.Module, cfg: LayaLabConfig, report: dict):
    adapters = {}
    for i, layer in enumerate(model.encoder.layers):
        if isinstance(layer.attn, BaseLayaReplacementAttention):
            adapters[str(i)] = {
                "config": layer.attn.config_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in layer.attn.state_dict().items()},
            }
    return {
        "format": "tinycenn-laya-attention-lab-v1",
        "lab_config": asdict(cfg),
        "report": report,
        "adapters": adapters,
    }
