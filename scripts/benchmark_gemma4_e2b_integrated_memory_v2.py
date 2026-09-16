#!/usr/bin/env python3
"""Gemma 4 E2B TinyCeNN V2.1.

Fixes loading of the multimodal google/gemma-4-E2B checkpoint into the
text-only Gemma4ForCausalLM used by the benchmark. The checkpoint stores text
weights under model.language_model.*, so those keys are remapped to model.*.

V2.0 attempted to replace the lazy top-level transformers.Gemma4ForCausalLM
export. A later ``from transformers import Gemma4ForCausalLM`` inside V1 main()
resolved the original class and bypassed that proxy. V2.1 patches the real
Gemma4ForCausalLM.from_pretrained class method itself, while retaining the
original bound loader internally. It also requires the untouched native Gemma 4
cache path to pass a sanity check before judging TinyCeNN relative to it.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from transformers import AutoConfig, Gemma4ForCausalLM
from transformers.cache_utils import DynamicCache

from scripts import benchmark_gemma4_e2b_integrated_memory as v1
from tinycenn_lm.gemma4_checkpoint import TEXT_KEY_MAPPING
from tinycenn_lm.gemma4_integrated_memory import new_cache, text_config


# Keep the real classmethod before patching it. Calling this bound method later
# does NOT recurse through the replacement below.
_ORIGINAL_FROM_PRETRAINED = Gemma4ForCausalLM.from_pretrained


def _validate_loading_info(model, info):
    cfg = text_config(model)
    missing = set(info.get("missing_keys", ()))
    allowed_missing = {"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()
    core_missing = sorted(missing - allowed_missing)
    unexpected_text = sorted(
        key for key in info.get("unexpected_keys", ())
        if key.startswith("model.language_model.")
    )
    errors = list(info.get("error_msgs", ()))
    if core_missing or unexpected_text or errors:
        raise RuntimeError(
            "Gemma 4 V2.1 text checkpoint remap failed before training: "
            f"core_missing={core_missing[:12]}, "
            f"unexpected_text={unexpected_text[:12]}, errors={errors[:4]}"
        )
    return core_missing, unexpected_text


@classmethod
def _mapped_from_pretrained(cls, model_id, *model_args, **kwargs):
    """Load the multimodal checkpoint's language_model weights into CausalLM."""
    revision = kwargs.get("revision", "main")
    token = kwargs.get("token", None)

    # V1 passes no explicit config. Always resolve the multimodal config first
    # and use its exact text sub-config for Gemma4ForCausalLM.
    full_config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
    cfg = full_config.get_text_config(decoder=True)
    if cfg.model_type != "gemma4_text":
        raise ValueError(f"Expected gemma4_text, got {cfg.model_type}")

    kwargs = dict(kwargs)
    kwargs["config"] = cfg
    kwargs["key_mapping"] = TEXT_KEY_MAPPING
    kwargs["output_loading_info"] = True

    model, info = _ORIGINAL_FROM_PRETRAINED(model_id, *model_args, **kwargs)
    core_missing, unexpected_text = _validate_loading_info(model, info)
    model.tie_weights()
    model.eval().requires_grad_(False)

    print(
        "gemma4_text_loader_v2_1:",
        json.dumps(
            {
                "core_missing_keys": core_missing,
                "unexpected_text_keys": unexpected_text,
                "unexpected_multimodal_key_count": len(info.get("unexpected_keys", ())),
                "key_mapping": TEXT_KEY_MAPPING,
                "loader_intercept": "Gemma4ForCausalLM.from_pretrained classmethod",
            }
        ),
        flush=True,
    )
    return model


# Patch the actual class object, not the lazy top-level transformers export.
# The import inside v1.main() therefore resolves this same class with the
# corrected classmethod.
Gemma4ForCausalLM.from_pretrained = _mapped_from_pretrained


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

    # A broken native baseline must stop the experiment before TinyCeNN is
    # interpreted. With correctly loaded Gemma 4 weights this diagnostic should
    # be close to the model's own cached/full numerical path.
    native_nmse_limit = 0.02
    native_top1_floor = 0.95
    if not math.isfinite(teacher_nmse) or not math.isfinite(teacher_top1):
        raise RuntimeError("Non-finite native Gemma 4 cache diagnostic")
    if teacher_nmse > native_nmse_limit or teacher_top1 < native_top1_floor:
        raise RuntimeError(
            "Native Gemma 4 cache sanity failed before TinyCeNN comparison: "
            f"NMSE={teacher_nmse:.6g} (limit {native_nmse_limit}), "
            f"top1={teacher_top1:.4f} (floor {native_top1_floor}). "
            "Checkpoint loading succeeded, so inspect Gemma 4 cache semantics before continuing."
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
    print("cache_equivalence_gemma4_v2_1:", json.dumps(metrics), flush=True)

    finite = (
        "cached_logits_nmse",
        "teacher_cached_logits_nmse",
        "cached_top1_agreement",
        "teacher_cached_top1_agreement",
    )
    if not all(math.isfinite(metrics[k]) for k in finite):
        raise RuntimeError(f"Non-finite Gemma 4 V2.1 cache metrics: {metrics}")
    if candidate_nmse > nmse_limit:
        raise RuntimeError(f"Gemma 4 V2.1 cache NMSE too high: {metrics}")
    if candidate_mismatch > allowed_mismatch or candidate_top1 < top1_floor:
        raise RuntimeError(f"Gemma 4 V2.1 cache top-1 drift too high: {metrics}")
    return metrics


v1.gemma4_cache_equivalence = gemma4_cache_equivalence_v2


if __name__ == "__main__":
    v1.main()
