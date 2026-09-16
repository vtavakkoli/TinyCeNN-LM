"""Safe text-only loader for multimodal Gemma 4 checkpoints.

The public Gemma 4 E2B checkpoint stores text weights under
``model.language_model.*`` because its native class is
``Gemma4ForConditionalGeneration``. Loading that checkpoint directly with
``Gemma4ForCausalLM`` leaves the text model randomly initialized unless the
checkpoint keys are remapped.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, Gemma4ForCausalLM


TEXT_KEY_MAPPING = {r"^model\.language_model\.": "model."}


def load_gemma4_text_causal(
    model_id: str,
    *,
    revision: str = "main",
    dtype: torch.dtype | None = None,
    attn_implementation: str = "sdpa",
    token: str | bool | None = None,
    device=None,
):
    """Load the real Gemma 4 text weights into ``Gemma4ForCausalLM``.

    Returns ``(model, loading_info)``. Core text weights are required to load;
    the only tolerated missing key is ``lm_head.weight`` when word embeddings
    are tied. Multimodal vision/audio keys are expected to remain unexpected.
    """
    full_config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
    text_config = full_config.get_text_config(decoder=True)
    if text_config.model_type != "gemma4_text":
        raise ValueError(f"Expected gemma4_text, got {text_config.model_type}")

    model, loading_info = Gemma4ForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        config=text_config,
        dtype=dtype,
        attn_implementation=attn_implementation,
        token=token,
        key_mapping=TEXT_KEY_MAPPING,
        output_loading_info=True,
    )

    missing = set(loading_info.get("missing_keys", ()))
    allowed_missing = {"lm_head.weight"} if text_config.tie_word_embeddings else set()
    core_missing = sorted(missing - allowed_missing)
    unexpected_text = sorted(
        key for key in loading_info.get("unexpected_keys", ())
        if key.startswith("model.language_model.")
    )
    errors = list(loading_info.get("error_msgs", ()))
    if core_missing or unexpected_text or errors:
        raise RuntimeError(
            "Gemma 4 text checkpoint remap failed: "
            f"core_missing={core_missing[:12]}, "
            f"unexpected_text={unexpected_text[:12]}, errors={errors[:4]}"
        )

    model.tie_weights()
    model.eval().requires_grad_(False)
    if device is not None:
        model = model.to(device)

    loading_info = dict(loading_info)
    loading_info["core_missing_keys"] = core_missing
    loading_info["unexpected_text_keys"] = unexpected_text
    loading_info["text_key_mapping"] = dict(TEXT_KEY_MAPPING)
    return model, loading_info
