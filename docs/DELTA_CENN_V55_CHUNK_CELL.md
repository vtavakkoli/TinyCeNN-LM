# Qwen3.5-0.8B Delta-CeNN V5.5: chunk-cell speed/fidelity experiment

## Why this change

User-provided V5.4 `qwen35_delta_cenn_all24_results (1).zip` (quick profile, T4, BF16) reports 18 accepted gated-delta replacements, six **rejected** full-attention-head-rescue candidates, audit agreement **0.985236**, audit ΔNLL **+0.0011997**, and end-to-end prefill **0.14273×** native speed (87.04 ms vs 609.82 ms). Full-attention layers 3/7/11/15/19/23 were **trained but not installed**; only 18 gated-delta layers were installed. The original fast greedy chat test errored because the tokenizer returned a BatchEncoding rather than a Tensor (`AttributeError` at `.shape`). Hugging Face upload had not succeeded (`hf_upload_status.json`: no writable runtime token).

## V5.5 computation graph

The new `Qwen35DeltaCeNNChunkLinear` preserves pretrained Qwen input projections, causal convolution, decay/write gates, normalization and output projection. Its `ChunkCellDeltaCeNNRule` **reuses the original Qwen/Transformers chunkwise gated-delta recurrence as a computational primitive**. It introduces the experimental CeNN element by applying learned, bounded nearest-neighbor spatial mixing to the persistent key/value matrix **between chunks** (defaults: 64-token cell interval, 64-token backend chunk). This removes the prior layer's Python **per-token** recurrence and replaces it with fewer Python calls to the chunk rule. It is *not* a distinct CeNN recurrence and it is not proof of speedup. With `cells=False`, the entire sequence uses one chunk call. Compressed key feature dimensions are rejected because Qwen's chunk kernel requires full-rank key features.

The learnable cell strength starts very small (`sigmoid(-12)`) for near-native parity, with per-head neighbor weights, output gain and output bias. Existing six full-attention V5.3 head-rescue candidates are still trained, screened and only installed if their quality gate passes. Rejection of a full-attention candidate is **not** a replacement, even if its checkpoint is saved. No forced installation by default.

## Measurements and safeguards

- The notebook logs an isolated 3-way backend benchmark (`engine_benchmark.json`): native Qwen, V5.4 tokenwise Delta-CeNN, V5.5 chunk-cell Delta-CeNN.
- The main progressive loop records training steps, cumulative agreement, NLL, prefill speedup, selected and rejected layer lists, and a delayed WikiText-2 train-tail diagnostic. This is **not a blind independent test**.
- Near-exact step-zero local candidates can skip unnecessary fitting if cosine ≥0.9999 and NMSE ≤0.0001; other candidates retain validation-based early stopping and best checkpoints.
- The fast greedy evaluation handles `Tensor` and `BatchEncoding`, uses short uncached generation and retains gold-token metrics. No KV-cache, streaming Qwen generation, multi-seed replication or optimized cellular CUDA kernel is claimed.
- The new adapter bundle format `tinycenn-qwen35-v55-adapter-v1` refuses to load a different engine, checks SHA-256 checksums and includes separate **selected** versus **candidate** checkpoints. It requires the original pinned Qwen weights and the TinyCeNN-LM source; it is not a standard `AutoModel.from_pretrained` directory.
- Default `HF_UPLOAD_MODE='quality_only'` blocks automatic Hub promotion when deployment gates fail; no writable user token is inferred from a connected read-only Hugging Face account.

## Test

```bash
pytest -q tests/test_delta_cenn_v55_chunk.py tests/test_qwen35_v55_notebook.py
```

Local tests run without full Transformers/Qwen, mocking the chunk backend to verify causal state, gradients, streaming, shape validation and notebook syntax. They do **not** test a live Colab CUDA kernel or demonstrate runtime speedup. Colab runs the real native/chunk/kernel comparison.

## Colab

https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_DeltaCeNN_V55_ChunkCell_Optimized_FastEval_Colab.ipynb
