import torch

from tinycenn_lm.story import (
    repeated_ngram_fraction,
    repetition_unlikelihood_loss,
    story_generation_kwargs,
)


def test_repetition_unlikelihood_penalizes_recent_tokens():
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    vocab = 8
    neutral = torch.zeros(1, 5, vocab)
    repetitive = neutral.clone()
    # At each next-token position, strongly prefer the immediately previous token.
    for p in range(4):
        repetitive[0, p, labels[0, p]] = 6.0

    neutral_loss = repetition_unlikelihood_loss(neutral, labels, window=3)
    repetitive_loss = repetition_unlikelihood_loss(repetitive, labels, window=3)
    assert torch.isfinite(neutral_loss)
    assert torch.isfinite(repetitive_loss)
    assert repetitive_loss > neutral_loss


def test_true_next_token_is_not_treated_as_negative():
    labels = torch.tensor([[1, 1, 1, 1]])
    logits = torch.zeros(1, 4, 4)
    logits[..., 1] = 8.0
    loss = repetition_unlikelihood_loss(logits, labels, window=3)
    assert float(loss) == 0.0


def test_repeated_ngram_fraction_detects_loops():
    clean = "a small fox walked home before sunset"
    loop = "good job good job good job good job"
    assert repeated_ngram_fraction(clean, 2) == 0.0
    assert repeated_ngram_fraction(loop, 2) > 0.0


def test_story_generation_kwargs_enable_loop_controls():
    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

    kwargs = story_generation_kwargs(Tokenizer(), max_new_tokens=80)
    assert kwargs["repetition_penalty"] > 1.0
    assert kwargs["no_repeat_ngram_size"] >= 3
    assert 0.0 < kwargs["top_p"] < 1.0
    assert kwargs["use_cache"] is False
