#!/usr/bin/env python3
"""Gemma 4 E2B TinyCeNN V2.

Fixes V1 loading of the multimodal google/gemma-4-E2B checkpoint into a
text-only Gemma4ForCausalLM. The checkpoint stores text weights under
model.language_model.*, so V2 remaps those keys instead of silently training a
randomly initialized text model. V2 also requires the untouched native Gemma 4
cache path to pass a sanity check before judging TinyCeNN relative to it.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import transformers
from transformers.cache_utils import DynamicCache

from scripts import benchmark_gemma4_e2b_integrated_memory as v1
from tinycenn_lm.gemma4_checkpoint import load_gemma4_text_causal
from tinycenn_lm.gemma4_integrated_memory import new_cache, text_config


class _MappedGemma4ForCausalLM:
    """Proxy used only so V1 main() resolves the corrected loader."""

    @classmethod
    def from_pretrained(cls, model_id, *args, **kwargs):
        if args:
            raise TypeError("Gemma 4 V2 loader expects keyword-only from_pretrained options")
        model, info = load_gemma4_text_causal(
            model_id,
            revision=kwargs.pop("revision", "main"),
            dtype=kwargs.pop("dtype", kwargs.pop("torch_dtype", None)),
            attn_implementation=kwargs.pop("attn_implementation", "sdpa"),
            token=kwargs.pop("token", None),
        )
        if kwargs:
            raise TypeError(f"Unsupported Gemma 4 V2 loader options: {sorted(kwargs)}")
        print(
            "gemma4_text_loader_v2:",
            json.dumps(
                {
                    "core_missing_keys": info["core_missing_keys"],
                    "unexpected_text_keys": info["unexpected_text_keys"],
                    "unexpected_multimodal_key_count": len(info.get("unexpected_keys", ())),
                    "key_mapping": info["text_key_mapping"],
                }
            ),
            flush=True,
        )
        return model


# v1.main() imports this symbol from transformers at runtime.
transformers.Gemma4ForCausalLM = _MappedGemma4ForCausalLM


@torch.no_grad()
def gemma4_cache_equivalence_v2(teacher, model, block, device, precision, block_size):
    """Native-sanity-gated cache equivalence for real Gemma 4 weights."""
    length = min(len(block) - 1, 2 * block_size + 5)
    ids = block[:length][None].to(device)
    split = min(block_size + 1, length - 1)

    tf, tc = v1._full_cached(
        teacher, ids, DynamicCache(config=text_config(teacher)), split, precision, False
    )
    cf, cc = v1._full_cached(model, ids, new_cache(model), split, precision, True)

    teacher_nmse = v1._nmse(tf, tc)
    candidate_nmse = v1._nmse(cf, cc)
    teacher_top1 = v1._top1(tf, tc)
    candidate_top1 = v1._top1(cf, cc)
    teacher_mismatch = int((tf.argmax(-1) != tc.argmax(-1)).sum().item())
    candidate_mismatch = int((cf.argmax(-1) != cc.argmax(-1)).sum().item())

    # If the untouched model cannot reproduce its own full path reasonably well,
    # the diagnostic itself is invalid and must not be used as a TinyCeNN gate.
    native_nmse_limit = 0.02
    native_top1_floor = 0.95
    if not math.isfinite(teacher_nmse) or not math.isfinite(teacher_top1):
        raise RuntimeError("Non-finite native Gemma 4 cache diagnostic")
    if teacher_nmse > native_nmse_limit or teacher_top1 < native_top1_floor:
        raise RuntimeError(
            "Native Gemma 4 cache sanity failed before TinyCeNN comparison: "
            f"NMSE={teacher_nmse:.6g} (limit {native_nmse_limit}), "
            f"top1={teacher_top1:.4f} (floor {native_top1_floor}). "
            "Check checkpoint loading/cache semantics; do not train against this baseline."
        )

    nmse_limit = max(2e-3, 4.0 * teacher_nmse + 5e-4)
    extra_allowance = max(2, math.ceil(0.05 * length))
    allowed_mismatch = min(length, teacher_mismatch + extra_allowance)
    top1_floor = max(0.90, teacher_top1 - 0.05)

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
        "native_cache_nmse_limit": native_nmse_limit,
        "native_cache_top1_floor": native_top1_floor,
        "cache_test_tokens": length,
    }
    metrics["cache_equivalence_passed"] = bool(
        candidate_nmse <= nmse_limit
        and candidate_mismatch <= allowed_mismatch
        and candidate_top1 >= top1_floor
    )
    print("cache_equivalence_gemma4_v2:", json.dumps(metrics), flush=True)

    finite = (
        "cached_logits_nmse",
        "teacher_cached_logits_nmse",
        "cached_top1_agreement",
        "teacher_cached_top1_agreement",
    )
    if not all(math.isfinite(metrics[k]) for k in finite):
        raise RuntimeError(f"Non-finite Gemma 4 V2 cache metrics: {metrics}")
    if candidate_nmse > nmse_limit:
        raise RuntimeError(f"Gemma 4 V2 cache NMSE too high: {metrics}")
    if candidate_mismatch > allowed_mismatch or candidate_top1 < top1_floor:
        raise RuntimeError(f"Gemma 4 V2 cache top-1 drift too high: {metrics}")
    return metrics


v1.gemma4_cache_equivalence = gemma4_cache_equivalence_v2


if __name__ == "__main__":
    v1.main()
