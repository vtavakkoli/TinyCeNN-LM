from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Iterable, Iterator
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def holdout_bucket(text: str, buckets: int = 1000) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def partition_rows(dataset: Iterable[dict], text_field: str, *, validation: bool) -> Iterator[dict]:
    """Deterministic document-level split: 99% train / 1% validation."""
    for example in dataset:
        text = example.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        is_validation = holdout_bucket(text) >= 990
        if is_validation == validation:
            yield example


def buffered_shuffle(rows: Iterable[dict], *, buffer_size: int, seed: int) -> Iterator[dict]:
    """Deterministic bounded-memory shuffle for streaming datasets."""
    if buffer_size <= 1:
        yield from rows
        return

    rng = random.Random(seed)
    buffer: list[dict] = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row

    rng.shuffle(buffer)
    yield from buffer


def token_blocks(
    rows: Iterable[dict], tokenizer, text_field: str, block_size: int, *, skip_tokens: int = 0
) -> Iterator[torch.Tensor]:
    """Pack documents, optionally advancing a deterministic stream before packing.

    Skipping is before tensor allocation and works across document boundaries and
    changes in batch/context length. Reconstructing a cursor still needs reading
    and tokenizing the prefix; it does not run the teacher or student on it.
    """
    if block_size < 2 or skip_tokens < 0:
        raise ValueError("block_size must be >= 2 and skip_tokens must be nonnegative")
    buffer: list[int] = []
    offset = 0
    eos = tokenizer.eos_token_id
    for example in rows:
        ids = tokenizer(example[text_field], add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids.append(eos)
        if skip_tokens:
            skipped = min(skip_tokens, len(ids))
            skip_tokens -= skipped
            ids = ids[skipped:]
        buffer.extend(ids)
        while len(buffer) - offset >= block_size:
            yield torch.tensor(buffer[offset : offset + block_size], dtype=torch.long)
            offset += block_size
        if offset > 1_000_000:
            buffer = buffer[offset:]
            offset = 0


def batch_blocks(blocks: Iterator[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    batch: list[torch.Tensor] = []
    for block in blocks:
        batch.append(block)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch.clear()


def collect_eval_batches(rows, tokenizer, text_field: str, block_size: int, batch_size: int, count: int):
    batches = batch_blocks(token_blocks(rows, tokenizer, text_field, block_size), batch_size)
    out: list[torch.Tensor] = []
    for _ in range(count):
        try:
            out.append(next(batches))
        except StopIteration:
            break
    if not out:
        raise RuntimeError("could not build held-out evaluation batches")
    return out


def evaluation_fingerprint(batches: list[torch.Tensor]) -> str:
    """Stable SHA256 fingerprint of the exact held-out token batches.

    Tokens are encoded explicitly as little-endian signed int64 values, avoiding
    NumPy and platform-dependent tensor byte representations.
    """
    digest = hashlib.sha256()
    for batch in batches:
        tensor = batch.detach().to(device="cpu", dtype=torch.int64).contiguous().view(-1)
        for token_id in tensor.tolist():
            digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


def chunked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    chunk_rows: int,
) -> torch.Tensor:
    if temperature <= 0 or chunk_rows < 1:
        raise ValueError("temperature and chunk_rows must be positive")
    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    total = s.new_zeros((), dtype=torch.float32)
    rows = s.shape[0]
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        s_chunk = s[start:end].float() / temperature
        t_chunk = t[start:end].float() / temperature
        total = total + F.kl_div(
            F.log_softmax(s_chunk, dim=-1),
            F.softmax(t_chunk, dim=-1),
            reduction="sum",
        )
    return total * (temperature * temperature) / max(rows, 1)


def hidden_cosine_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1)).mean()


def combined_loss(
    student_out,
    teacher_out,
    *,
    temperature: float,
    kl_chunk_rows: int,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = student_out.loss.float()
    kl = chunked_kl(student_out.logits, teacher_out.logits, temperature, kl_chunk_rows)
    hidden = hidden_cosine_loss(student_out.hidden_states[-1], teacher_out.hidden_states[-1])
    total = ce_weight * ce + kl_weight * kl + hidden_weight * hidden
    return total, {
        "ce": float(ce.detach()),
        "kl": float(kl.detach()),
        "hidden": float(hidden.detach()),
        "total": float(total.detach()),
    }


@torch.inference_mode()
def evaluate_distillation(
    teacher,
    student,
    batches: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
    kl_chunk_rows: int,
    ce_weight: float,
    kl_weight: float,
    hidden_weight: float,
) -> dict[str, float | int]:
    teacher.eval()
    student.eval()
    sums = {
        "student_ce": 0.0,
        "teacher_ce": 0.0,
        "kl": 0.0,
        "hidden": 0.0,
        "total": 0.0,
    }
    n_batches = 0
    eval_tokens = 0
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
    for cpu_ids in batches:
        ids = cpu_ids.to(device, non_blocking=True)
        with amp():
            teacher_out = teacher(
                input_ids=ids,
                labels=ids,
                output_hidden_states=True,
                use_cache=False,
            )
            student_out = student(
                input_ids=ids,
                labels=ids,
                output_hidden_states=True,
                use_cache=False,
            )
            total, parts = combined_loss(
                student_out,
                teacher_out,
                temperature=temperature,
                kl_chunk_rows=kl_chunk_rows,
                ce_weight=ce_weight,
                kl_weight=kl_weight,
                hidden_weight=hidden_weight,
            )
        sums["student_ce"] += float(student_out.loss.detach().float())
        sums["teacher_ce"] += float(teacher_out.loss.detach().float())
        sums["kl"] += parts["kl"]
        sums["hidden"] += parts["hidden"]
        sums["total"] += float(total.detach().float())
        n_batches += 1
        eval_tokens += ids.numel()

    for key in sums:
        sums[key] /= max(n_batches, 1)
    result: dict[str, float | int] = dict(sums)
    result["student_ppl"] = math.exp(min(sums["student_ce"], 30.0))
    result["teacher_ppl"] = math.exp(min(sums["teacher_ce"], 30.0))
    result["eval_batches"] = n_batches
    result["eval_tokens"] = eval_tokens
    return result
