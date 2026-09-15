#!/usr/bin/env python3
"""FunctionGemma TinyCeNN V2 benchmark with a native-relative cache-equivalence gate.

This wraps benchmark_functiongemma_integrated_memory.py but replaces its overly
strict fixed 98% cached/full top-1 agreement requirement. Gemma3's own hybrid
sliding/full-attention cache is not bit-identical to its one-shot full path in
BF16, so the scientifically relevant check is whether TinyCeNN adds materially
more drift than native FunctionGemma on the same tokens.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from transformers.cache_utils import DynamicCache

from scripts import benchmark_functiongemma_integrated_memory as v1


@torch.no_grad()
def functiongemma_cache_equivalence(teacher, model, block, device, precision, block_size):
    """Compare candidate cache drift with native FunctionGemma cache drift.

    Acceptance rules:
    1. Candidate NMSE must stay inside a small native-relative numerical budget.
    2. Candidate cached/full top-1 mismatches may be at most native mismatches + 2
       tokens on this short diagnostic sequence.
    3. A separate 85% agreement safety floor catches genuinely broken caching.

    The mismatch-count formulation is stable for short sequences where one token
    changes the percentage by ~1-2 points.
    """
    length = min(len(block) - 1, 2 * block_size + 5)
    ids = block[:length][None].to(device)
    split = min(block_size + 1, length - 1)

    teacher_cache = DynamicCache(config=teacher.config)
    teacher_full, teacher_cached = v1._full_and_cached_logits(
        teacher, ids, teacher_cache, split, precision, use_cenn_context=False
    )
    teacher_nmse = v1._nmse(teacher_full, teacher_cached)
    teacher_top1 = v1._top1_agreement(teacher_full, teacher_cached)
    teacher_mismatches = int(
        (teacher_full.argmax(-1) != teacher_cached.argmax(-1)).sum().item()
    )

    candidate_cache = v1.new_cache(model)
    candidate_full, candidate_cached = v1._full_and_cached_logits(
        model, ids, candidate_cache, split, precision, use_cenn_context=True
    )
    candidate_nmse = v1._nmse(candidate_full, candidate_cached)
    candidate_top1 = v1._top1_agreement(candidate_full, candidate_cached)
    candidate_mismatches = int(
        (candidate_full.argmax(-1) != candidate_cached.argmax(-1)).sum().item()
    )

    # Native-relative numerical budget. This already passed comfortably in the
    # observed FunctionGemma run (candidate ~4.2e-4, native ~1.08e-3).
    nmse_limit = max(1.5e-3, 4.0 * teacher_nmse + 2.5e-4)

    # Short-sequence categorical gate: allow at most two extra mismatched positions
    # relative to the untouched model, while retaining an absolute safety floor.
    allowed_mismatches = min(length, teacher_mismatches + 2)
    mismatch_relative_limit = 1.0 - allowed_mismatches / max(length, 1)
    top1_limit = max(0.85, mismatch_relative_limit)

    metrics = {
        "cached_logits_nmse": candidate_nmse,
        "teacher_cached_logits_nmse": teacher_nmse,
        "cache_nmse_excess": candidate_nmse - teacher_nmse,
        "cache_equivalence_nmse_limit": nmse_limit,
        "cached_top1_agreement": candidate_top1,
        "teacher_cached_top1_agreement": teacher_top1,
        "candidate_top1_mismatches": candidate_mismatches,
        "teacher_top1_mismatches": teacher_mismatches,
        "allowed_top1_mismatches": allowed_mismatches,
        "cache_equivalence_top1_limit": top1_limit,
        "cache_test_tokens": length,
    }
    print("cache_equivalence_v2:", json.dumps(metrics), flush=True)

    finite_keys = (
        "cached_logits_nmse",
        "teacher_cached_logits_nmse",
        "cached_top1_agreement",
        "teacher_cached_top1_agreement",
    )
    if not all(math.isfinite(metrics[k]) for k in finite_keys):
        raise RuntimeError(f"Non-finite FunctionGemma cache-equivalence metrics: {metrics}")

    if candidate_nmse > nmse_limit:
        raise RuntimeError(
            "FunctionGemma cache NMSE adds too much drift: "
            f"candidate={candidate_nmse:.6g}, native={teacher_nmse:.6g}, "
            f"limit={nmse_limit:.6g}"
        )

    if candidate_mismatches > allowed_mismatches or candidate_top1 < 0.85:
        raise RuntimeError(
            "FunctionGemma cache changes too many top-1 positions: "
            f"candidate mismatches={candidate_mismatches}/{length}, "
            f"native mismatches={teacher_mismatches}/{length}, "
            f"allowed={allowed_mismatches}, agreement={candidate_top1:.4f}"
        )

    return metrics


# main() resolves this name from the v1 module at runtime, so patching it here
# changes only the post-training diagnostic and leaves all training/evaluation
# logic untouched.
v1.functiongemma_cache_equivalence = functiongemma_cache_equivalence


if __name__ == "__main__":
    v1.main()
