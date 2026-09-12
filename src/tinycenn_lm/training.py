"""Training policies shared by continuation and its offline regression tests."""
from __future__ import annotations

import hashlib
import json
import math

import torch


def training_stream_signature(args, tokenizer) -> dict:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_state = {
        "vocab": tokenizer.get_vocab(), "eos": tokenizer.eos_token_id,
        "backend": backend.to_str() if backend is not None else None,
    }
    digest = hashlib.sha256(json.dumps(tokenizer_state, sort_keys=True).encode()).hexdigest()
    return {
        "dataset": args.dataset, "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision, "split": args.split,
        "text_field": args.text_field, "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed, "tokenizer_sha256": digest,
        "partition": "text-hash-99-1-v1", "packing": "eos-v1",
    }


def optimizer_groups(model, learning_rate: float, interface_lr_scale: float, weight_decay: float):
    """Use a smaller LR for copied interfaces and no decay for norms/biases."""
    if learning_rate <= 0 or not 0 < interface_lr_scale <= 1 or weight_decay < 0:
        raise ValueError("invalid learning rate, interface LR scale, or weight decay")
    groups: dict[tuple[bool, bool], dict] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError(f"trainable parameter {name} must be float32 before creating AdamW")
        interface = ".cenn." not in name
        decay = parameter.ndim >= 2
        key = (interface, decay)
        if key not in groups:
            lr = learning_rate * (interface_lr_scale if interface else 1.0)
            groups[key] = {
                "params": [], "lr": lr, "initial_lr": lr,
                "weight_decay": weight_decay if decay else 0.0,
                "name": ("interface" if interface else "cenn") + ("_decay" if decay else "_no_decay"),
            }
        groups[key]["params"].append(parameter)
    if not groups:
        raise ValueError("no trainable parameters")
    return list(groups.values())


def lr_multiplier(
    step: int, total: int, warmup: int, *, schedule: str = "cosine",
    decay_ratio: float = 0.2, min_lr_ratio: float = 0.1,
) -> float:
    """Cosine or warmup/stable/decay, with the same explicit nonzero LR floor."""
    if total < 1 or warmup < 0 or warmup >= total:
        raise ValueError("require total >= 1 and 0 <= warmup < total")
    if schedule not in {"cosine", "wsd"} or not 0 < decay_ratio <= 1:
        raise ValueError("invalid LR schedule or decay_ratio")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if warmup and step <= warmup:
        return max(step, 1) / warmup
    decay_start = warmup
    if schedule == "wsd":
        decay_start = max(warmup, total - max(1, round(total * decay_ratio)))
    progress = min(max((step - decay_start) / max(total - decay_start, 1), 0.0), 1.0)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def annealed_weight(initial: float, final: float | None, progress: float) -> float:
    if initial < 0 or (final is not None and final < 0):
        raise ValueError("loss weights must be nonnegative")
    end = initial if final is None else final
    return initial + (end - initial) * min(max(progress, 0.0), 1.0)


def checkpoint_training_tokens(metadata: dict, report: dict | None) -> int:
    """Read the saved weights' token count, never a best run's final token count."""
    training = metadata.get("training", {})
    for key in ("cumulative_tokens", "tokens"):
        if key in training:
            return int(training[key])
    report = report or {}
    checkpoint = report.get("checkpoint", {})
    if "cumulative_training_tokens" in checkpoint:
        return int(checkpoint["cumulative_training_tokens"])
    return int(report.get("cumulative_training_tokens", report.get("seen_tokens", 0)))


def stream_resume_offset(
    metadata: dict, report: dict | None, signature: dict, *, explicit_offset: int | None = None,
) -> tuple[int, str]:
    """Recover an exact new-format cursor, or label a legacy estimate explicitly.

    A v2 run restarted its stream at zero. Its run-local token count is therefore
    the prefix to skip, not the cumulative token count across earlier runs.
    """
    if explicit_offset is not None:
        if explicit_offset < 0:
            raise ValueError("train-skip-tokens must be nonnegative")
        return explicit_offset, "explicit"
    if not metadata:
        return 0, "new_stream"
    stream = metadata.get("training", {}).get("data_stream")
    if stream:
        if stream["signature"] != signature:
            raise ValueError(
                "resume data stream differs (dataset/revision/tokenizer/shuffle). "
                "Use matching settings, or --train-skip-tokens for an intentional new stream."
            )
        return int(stream["next_token_offset"]), "exact_token_offset"
    report = report or {}
    training = metadata.get("training", {})
    # Legacy metadata identifies the actual best snapshot; a copied report may
    # describe a later final model. Favor the snapshot's run-local count.
    offset = training.get("tokens_this_run", training.get("tokens"))
    if offset is None:
        offset = report.get("seen_tokens_this_run", report.get("seen_tokens", 0))
    return int(offset), "legacy_estimate"


def plateau_summary(history: list[dict], patience: int, min_delta: float) -> dict:
    if patience < 1 or min_delta < 0:
        raise ValueError("plateau patience must be positive and min_delta nonnegative")
    if len(history) <= patience:
        return {"detected": False, "evaluations_without_progress": 0}
    prior_best = min(float(row["student_ce"]) for row in history[:-patience])
    recent_best = min(float(row["student_ce"]) for row in history[-patience:])
    improvement = prior_best - recent_best
    return {
        "detected": improvement < min_delta,
        "evaluations_without_progress": patience if improvement < min_delta else 0,
        "recent_best_ce_improvement": improvement,
    }
