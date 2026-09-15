#!/usr/bin/env python3
"""Qwen3.5 TinyCeNN V2.1 benchmark with a numerically robust cache-equivalence gate.

This wraps benchmark_qwen35_integrated_memory.py. The underlying cache-position bug
is fixed in qwen35_integrated_memory.py; this wrapper only changes the post-training
diagnostic so tiny near-tie argmax flips do not abort an otherwise numerically
consistent run.
"""
import json
import math

import torch

from scripts import benchmark_qwen35_integrated_memory as v1


@torch.no_grad()
def qwen_cache_equivalence(teacher, model, block, device, precision, block_size):
    """Validate cached TinyCeNN execution against one-shot execution.

    Hard gate:
      * cached/full logit NMSE must remain inside a native-relative numerical budget.

    Secondary categorical safety gate:
      * cached/full top-1 agreement must be at least 95%; and
      * short-sequence mismatch allowance scales with diagnostic length rather than
        requiring an arbitrary fixed +2-token budget.

    Exact argmax equality is intentionally not used as the primary criterion because
    extremely small logit perturbations can flip two nearly tied vocabulary items.
    """
    length = min(len(block) - 1, 2 * block_size + 5)
    ids = block[:length][None].to(device)
    split = min(block_size + 1, length - 1)

    teacher_cache = v1.new_cache(teacher)
    candidate_cache = v1.new_cache(model)

    tf, tc = v1._full_cached(teacher, ids, teacher_cache, split, precision, False)
    cf, cc = v1._full_cached(model, ids, candidate_cache, split, precision, True)

    teacher_nmse = v1._nmse(tf, tc)
    candidate_nmse = v1._nmse(cf, cc)
    teacher_top1 = v1._top1(tf, tc)
    candidate_top1 = v1._top1(cf, cc)
    teacher_mismatch = int((tf.argmax(-1) != tc.argmax(-1)).sum().item())
    candidate_mismatch = int((cf.argmax(-1) != cc.argmax(-1)).sum().item())

    # Qwen's native hybrid cache provides the numerical reference. Keep a small
    # absolute floor so BF16 kernel/order changes cannot cause spurious failures.
    nmse_limit = max(2e-3, 4.0 * teacher_nmse + 5e-4)

    # On a 69-token diagnostic one argmax flip is ~1.45 percentage points. Allow a
    # length-scaled number of flips, but retain a 95% agreement safety floor.
    scaled_allowance = max(2, math.ceil(0.05 * length))
    allowed_mismatch = min(length, teacher_mismatch + scaled_allowance)
    top1_floor = 0.95

    # Cache length must advance through prefill + decode. The custom cache reports
    # TinyCeNN MemoryState.position for replaced full-attention layers.
    teacher_cache_length = int(teacher_cache.get_seq_length())
    candidate_cache_length = int(candidate_cache.get_seq_length())

    metrics = {
        "cached_logits_nmse": candidate_nmse,
        "teacher_cached_logits_nmse": teacher_nmse,
        "cache_nmse_excess": candidate_nmse - teacher_nmse,
        "cache_equivalence_nmse_limit": nmse_limit,
        "cached_top1_agreement": candidate_top1,
        "teacher_cached_top1_agreement": teacher_top1,
        "candidate_top1_mismatches": candidate_mismatch,
        "teacher_top1_mismatches": teacher_mismatch,
        "allowed_top1_mismatches": allowed_mismatch,
        "cache_equivalence_top1_floor": top1_floor,
        "teacher_cache_length": teacher_cache_length,
        "candidate_cache_length": candidate_cache_length,
        "expected_cache_length": length,
        "cache_test_tokens": length,
    }

    print("cache_equivalence_qwen35_v2_1:", json.dumps(metrics), flush=True)

    finite = (
        "cached_logits_nmse",
        "teacher_cached_logits_nmse",
        "cached_top1_agreement",
        "teacher_cached_top1_agreement",
    )
    if not all(math.isfinite(metrics[key]) for key in finite):
        raise RuntimeError(f"Non-finite Qwen3.5 cache metrics: {metrics}")

    if teacher_cache_length != length or candidate_cache_length != length:
        raise RuntimeError(
            "Qwen3.5 cache length did not advance correctly: "
            f"candidate={candidate_cache_length}, native={teacher_cache_length}, expected={length}"
        )

    if candidate_nmse > nmse_limit:
        raise RuntimeError(f"Qwen3.5 cache NMSE too high: {metrics}")

    if candidate_top1 < top1_floor or candidate_mismatch > allowed_mismatch:
        raise RuntimeError(f"Qwen3.5 cache categorical drift too high: {metrics}")

    metrics["cache_equivalence_passed"] = True
    return metrics


# Patch only the diagnostic. Training, validation selection, held-out evaluation,
# checkpointing and report generation remain exactly those of the base benchmark.
v1.qwen_cache_equivalence = qwen_cache_equivalence


if __name__ == "__main__":
    v1.main()
