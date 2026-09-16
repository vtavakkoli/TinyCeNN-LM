#!/usr/bin/env python3
from __future__ import annotations

"""Qwen3.5 Memory Fusion sequential trainer with positional-attention capture.

Transformers 5.17 calls Qwen3.5 full attention with required positional
``position_embeddings`` and ``attention_mask`` arguments. The original shared
SmolLM2 capture helper only copied keyword arguments, so replaying the teacher
attention failed with ``attention_mask`` missing. This wrapper binds the actual
forward call signature, preserving positional arguments, then delegates to the
existing Qwen3.5 sequential trainer.
"""

import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from scripts import train_qwen35_memory_fusion_sequential as qwen
from scripts import train_smollm2_memory_fusion_sequential as seq


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(v) for v in value)
    if isinstance(value, list):
        return [_detach_tree(v) for v in value]
    return value


def _bound_forward_arguments(module, args, kwargs) -> dict[str, Any]:
    """Return named arguments for a module forward call, including positionals."""
    values: dict[str, Any] = dict(kwargs)
    try:
        signature = inspect.signature(module.forward)
        bound = signature.bind_partial(*args, **kwargs)
        values.update(bound.arguments)
    except (TypeError, ValueError):
        # The explicit fallback matches the Qwen3.5 full-attention call order in
        # Transformers 5.17 and still leaves keyword arguments authoritative.
        names = (
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_values",
            "cache_position",
        )
        for name, value in zip(names, args):
            values.setdefault(name, value)
    return values


def capture_attention_input_qwen(model, ids: torch.Tensor, layer_idx: int, amp, *, with_output: bool):
    """Capture Qwen3.5 attention inputs without losing positional mask arguments."""
    capture: dict[str, Any] = {}
    module = model.model.layers[layer_idx].self_attn

    def pre_hook(mod, args, kwargs):
        call = _bound_forward_arguments(mod, args, kwargs)
        hidden = call.get("hidden_states", args[0] if args else None)
        if hidden is None:
            raise RuntimeError("attention hidden_states not found")
        capture["hidden"] = hidden
        # Keep attention_mask even when it is None: Qwen3.5 declares it as a
        # required forward argument, so replay must pass the name explicitly.
        for key in (
            "position_embeddings",
            "position_ids",
            "attention_mask",
            "cache_position",
        ):
            if key in call:
                capture[key] = _detach_tree(call[key])

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        if with_output:
            with amp():
                out = model(
                    input_ids=ids,
                    labels=ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            with torch.no_grad(), amp():
                out = model(
                    input_ids=ids,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
    finally:
        handle.remove()
    if "hidden" not in capture:
        raise RuntimeError(f"failed to capture layer {layer_idx} attention input")
    if "position_embeddings" not in capture:
        raise RuntimeError(f"Qwen3.5 layer {layer_idx} capture missed required position_embeddings")
    if "attention_mask" not in capture:
        raise RuntimeError(f"Qwen3.5 layer {layer_idx} capture missed required attention_mask")
    return capture, out


@torch.no_grad()
def real_hidden_function_metrics_qwen(teacher, student, ids: torch.Tensor, layer_idx: int, amp):
    t_capture, _ = capture_attention_input_qwen(teacher, ids, layer_idx, amp, with_output=False)
    s_capture, _ = capture_attention_input_qwen(student, ids, layer_idx, amp, with_output=False)
    real_hidden = s_capture["hidden"].detach()
    kwargs = seq.attention_kwargs_from_capture(t_capture)
    target = seq.call_attention(teacher.model.layers[layer_idx].self_attn, real_hidden, kwargs)
    pred = seq.call_attention(student.model.layers[layer_idx].self_attn, real_hidden, kwargs)
    nmse, cosine = seq.alignment_metrics(pred, target)
    return float(nmse), float(cosine)


def install_qwen35_capture_fix() -> None:
    seq.capture_attention_input = capture_attention_input_qwen
    seq.real_hidden_function_metrics = real_hidden_function_metrics_qwen


def main() -> int:
    install_qwen35_capture_fix()
    return qwen.main()


if __name__ == "__main__":
    raise SystemExit(main())
