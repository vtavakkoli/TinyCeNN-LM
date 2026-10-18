# CCM-v3 — positional, pretrained-initialized cellular consensus (research control)

## Problem addressed

`CCM_v2_Smol360_Train_RealFast_Results (1).zip` ended at step 80 with a DEV NLL of 3.629 (native 3.217), TEST NLL 4.151 (native 3.595), gold accuracy 32.8% (native 39.3%), five-rule chat 0/5 (native 2/5), and T4 no-cache full-model prefill ratio ~0.990x. In ARC-Easy 16-question validation, CCM-v2 predicted answer A on all 16 cases, getting 6/16 by answer-letter bias. **This fails model quality.**

CCM-v2's randomly initialized 64-dimensional key space / 256-dimensional value projection replaced a fully pretrained Llama attention mixer with **15 query heads, 5 key/value heads, 64 head-dim and rotary position encoding (RoPE)**. The CCM-v2 layer had no RoPE and no grouping of pretrained attention heads. Training 80 steps on 64 WikiText blocks cannot be taken as evidence that the positional and projection losses are recoverable.

## Correction in v3

1. Deep-copy the frozen pretrained first-layer Q, K, V and O projections (never call the original attention in the candidate forward).
2. Preserve SmolLM2's exact per-head dimensions, grouped K/V replication and passed RoPE position embeddings. Keep causal order and optional attention masks.
3. Instead of native FlashAttention / SDPA / attention module, perform local **pairwise log-mass-conserving cellular merges** over a finite causal tape, with bounded query blocks for T4 memory.
4. Initialize a zero-output small trainable low-rank correction on top of the cellular result. At initialization it changes nothing. Optional training uses WikiText gold next-token cross-entropy **only**, with frozen native projections and model backbone; choose best state using DEV gold NLL.
5. Check full-model native/logit parity **before training** at context lengths within the tape. Fail closed if the numerical agreement falls below threshold.
6. Report separate WikiText validation/test gold-label NLL and accuracy, ARC-Easy scored against correct answers, five task-rule chats, and paired CUDA full-model no-cache prefill.

## Scientific limitations

This corrected design is **algebraically attention-equivalent** while the full context fits the tape. It is **not** a novel attention-free mathematical operation, not a parameter reduction, and has no demonstrated GPU speedup or KV cache. After the capacity threshold, it becomes sliding-window attention-equivalent and may diverge. Its purpose is a controlled baseline: first establish that the insertion interface, pretrained positional information and chat quality are preserved; then test compression separately.

**CCM-v2 and CCM-v3 checkpoint formats are incompatible.** The old trained v2 checkpoint cannot become pretrained-initialized RoPE multihead attention by loading its weights; the v3 path must start by copying the native pretrained model, and only later can resume a v3 checkpoint. Do not relabel the old checkpoint.

GPU full-model parity has to be verified in Colab. Local evidence is only exact softmax/SDPA-equivalent CPU operator tests. Even exact native fidelity provides no evidence of novel CeNN speedups.

## Run

Colab: [CCM-v3 RoPE grouped-head parity, gold training, real chat and CUDA](../notebooks/CCM_v3_Smol360_RoPE_Parity_Chat_Colab.ipynb)

Source: `src/tinycenn_lm/ccm_v3_rotary_consensus.py`

Experiment: `scripts/benchmark_ccm_v3_smol360.py`

Unit tests: `tests/test_ccm_v3_rotary_consensus.py`
