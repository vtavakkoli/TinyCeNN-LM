import torch

from tinycenn_lm.distill_utils import (
    buffered_shuffle,
    evaluation_fingerprint,
    holdout_bucket,
    partition_rows,
)


def test_document_hash_split_is_deterministic_and_disjoint():
    rows = [{"text": f"document {i}"} for i in range(500)]
    train = list(partition_rows(rows, "text", validation=False))
    valid = list(partition_rows(rows, "text", validation=True))

    train_text = {row["text"] for row in train}
    valid_text = {row["text"] for row in valid}
    assert train_text.isdisjoint(valid_text)
    assert train_text | valid_text == {row["text"] for row in rows}
    assert all(holdout_bucket(text) < 990 for text in train_text)
    assert all(holdout_bucket(text) >= 990 for text in valid_text)


def test_buffered_shuffle_is_seeded_and_preserves_rows():
    rows = [{"text": str(i)} for i in range(100)]
    first = list(buffered_shuffle(rows, buffer_size=16, seed=42))
    second = list(buffered_shuffle(rows, buffer_size=16, seed=42))
    third = list(buffered_shuffle(rows, buffer_size=16, seed=43))

    assert first == second
    assert sorted(row["text"] for row in first) == sorted(row["text"] for row in rows)
    assert first != third


def test_evaluation_fingerprint_is_stable_and_content_sensitive():
    batches = [
        torch.arange(16, dtype=torch.long).reshape(2, 8),
        torch.arange(16, 32, dtype=torch.long).reshape(2, 8),
    ]
    fingerprint = evaluation_fingerprint(batches)
    assert fingerprint == evaluation_fingerprint([batch.clone() for batch in batches])

    changed = [batch.clone() for batch in batches]
    changed[1][0, 0] += 1
    assert fingerprint != evaluation_fingerprint(changed)
