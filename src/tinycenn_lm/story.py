from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor


def repetition_unlikelihood_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    window: int = 32,
    ignore_index: int = -100,
) -> Tensor:
    """Penalize probability assigned to recently seen tokens.

    For every next-token prediction, tokens that appeared in the recent history
    are treated as negative candidates unless the token is the true next target.
    This is a lightweight unlikelihood objective aimed specifically at the short
    repetition loops seen in TinyCeNN generation.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions must match")
    if window <= 0:
        return logits.new_zeros(())

    pred = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    history = labels[:, :-1]
    log_z = torch.logsumexp(pred, dim=-1)

    total = pred.new_zeros(())
    count = pred.new_zeros(())
    max_back = min(window, history.shape[1])

    for back in range(max_back):
        if back == 0:
            negatives = history
            valid = torch.ones_like(history, dtype=torch.bool)
        else:
            negatives = torch.roll(history, shifts=back, dims=1)
            valid = torch.ones_like(history, dtype=torch.bool)
            valid[:, :back] = False

        valid &= targets.ne(ignore_index)
        valid &= negatives.ne(ignore_index)
        valid &= negatives.ne(targets)

        safe_negatives = negatives.clamp_min(0)
        neg_logits = pred.gather(-1, safe_negatives.unsqueeze(-1)).squeeze(-1)
        p_negative = torch.exp(neg_logits - log_z).clamp(max=1.0 - 1e-6)
        penalties = -torch.log1p(-p_negative)

        total = total + penalties.masked_select(valid).sum()
        count = count + valid.sum().to(dtype=total.dtype)

    return total / count.clamp_min(1.0)


def repeated_ngram_fraction(text: str, n: int = 3) -> float:
    """Fraction of generated n-gram occurrences beyond their first occurrence."""
    words = text.split()
    if len(words) < n or n <= 0:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeats = sum(max(0, c - 1) for c in counts.values())
    return repeats / max(len(grams), 1)


def story_generation_kwargs(tokenizer, *, max_new_tokens: int = 120) -> dict:
    """Decoding defaults chosen to suppress loops without making text deterministic."""
    return {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min(40, max_new_tokens),
        "do_sample": True,
        "temperature": 0.78,
        "top_p": 0.90,
        "top_k": 40,
        "repetition_penalty": 1.18,
        "no_repeat_ngram_size": 4,
        "renormalize_logits": True,
        "use_cache": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
