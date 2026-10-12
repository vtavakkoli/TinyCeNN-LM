"""Objective and split guards for a selective CeNN 3-to-1 encoder experiment.

Designed to prevent acceptance of a harmful 3->1 layer replacement even
when its local cosine is relatively high. All thresholds apply to a DEV
selection split; the independent HOLDOUT is not part of selection.
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence


def verify_case_partition(train_rows, selection_rows, holdout_rows):
    """Enforce case-ID and exact-state isolation before creating examples."""
    seen = set()
    sizes=[]
    for label, rows in (("train",train_rows),("selection",selection_rows),
                        ("holdout",holdout_rows)):
        ids=[str(x["id"]) for x in rows]
        states=[str(x["state"]) for x in rows]
        if len(ids)!=len(set(ids)) or len(states)!=len(set(states)):
            raise ValueError(f"{label}: duplicate case ID or state")
        tokens={("id",x) for x in ids}|{("state",x) for x in states}
        if seen & tokens:
            raise ValueError(f"{label}: case ID/state overlaps another partition")
        seen.update(tokens)
        sizes.append(len(rows))
    if any(size<1 for size in sizes):
        raise ValueError("Every case partition must be nonempty")
    return {"train_cases":sizes[0],"selection_cases":sizes[1],
            "holdout_cases":sizes[2],"all_partitions_disjoint":True}


def finite_quality(quality: Mapping) -> None:
    for key in ("agreement","teacher_gold_accuracy",
                "student_gold_accuracy","mean_js"):
        if key not in quality or not math.isfinite(float(quality[key])):
            raise ValueError(f"Missing/nonfinite quality statistic: {key}")
    if int(quality.get("decisions",0)) < 1:
        raise ValueError("Evaluation requires >=1 decision")


def judge_replacement(
    baseline: Mapping, candidate: Mapping, *,
    local_cosine: float, local_nmse: float,
    min_agreement: float = 0.99,
    max_incremental_agreement_drop: float = 0.01,
    max_gold_drop: float = 0.015,
    min_local_cosine: float = 0.99,
    max_local_nmse: float = 0.025,
    speedup: float | None = None,
    require_speedup: bool = False,
) -> dict:
    """Only select a replacement if its *cumulative* accuracy survives.

    The baseline is the entire CURRENTLY compressed student, not an
    isolated teacher block. A candidate is compared on the same DEV cases.
    """
    finite_quality(baseline)
    finite_quality(candidate)
    if int(baseline["decisions"])!=int(candidate["decisions"]):
        raise ValueError("Compare exactly the same development decision count")
    if not all(math.isfinite(float(z)) for z in (local_cosine,local_nmse)):
        raise ValueError("Nonfinite local fidelity")
    if require_speedup and (speedup is None or not math.isfinite(speedup)):
        raise ValueError("Speed requirement needs a measured finite speedup")
    reasons=[]
    def flag(condition:bool,reason:str):
        if not condition:reasons.append(reason)
    flag(local_cosine>=min_local_cosine,"local_cosine_below_gate")
    flag(local_nmse<=max_local_nmse,"local_nmse_above_gate")
    flag(candidate["agreement"]>=min_agreement,"teacher_agreement_below_gate")
    flag(candidate["agreement"]>=baseline["agreement"]-max_incremental_agreement_drop,
         "incremental_teacher_agreement_drop")
    flag(candidate["student_gold_accuracy"]>=
         baseline["student_gold_accuracy"]-max_gold_drop,
         "incremental_gold_accuracy_drop")
    flag(candidate["student_gold_accuracy"]>=
         candidate["teacher_gold_accuracy"]-max_gold_drop,
         "teacher_gold_accuracy_drop")
    if require_speedup:
        flag(float(speedup)>1.0,"candidate_not_faster")
    return {
        "accepted":not reasons,"reasons":reasons,
        "baseline_agreement":float(baseline["agreement"]),
        "candidate_agreement":float(candidate["agreement"]),
        "delta_agreement":float(candidate["agreement"]-baseline["agreement"]),
        "delta_gold":float(candidate["student_gold_accuracy"]-
                           baseline["student_gold_accuracy"]),
        "local_cosine":float(local_cosine),
        "local_nmse":float(local_nmse),
        "measured_speedup":None if speedup is None else float(speedup),
    }


def plan_active_layers(original_count:int,accepted_groups:Sequence):
    groups=[tuple(x) for x in accepted_groups]
    covered=set()
    for group in groups:
        if len(group)!=3 or group!=(group[0],group[0]+1,group[0]+2):
            raise ValueError("All replacements must be consecutive triples")
        if any(x in covered or x<0 or x>=original_count for x in group):
            raise ValueError("Invalid/overlapping group")
        covered.update(group)
    return {
        "original_encoder_blocks":original_count,
        "accepted_groups":len(groups),
        "active_encoder_blocks":original_count-2*len(groups),
        "replaced_original_blocks":3*len(groups),
        "bypassed_blocks":2*len(groups),
        "native_layers":[i for i in range(original_count) if i not in covered],
    }
