# PDelta2-Flash Layer Lab

This experiment keeps the strongest prior TinyCeNN single-layer direction (`P-Delta2 F96`) and tests lightweight ingredients motivated by recent efficient-attention systems.

## Hypotheses

1. **Per-head output gating** can improve selective recurrent reads with almost no parameter or state cost.
2. **Short causal Conv4 value mixing** can recover local token interactions without a dense attention window.
3. **Compact content-indexed block summaries** can provide query-dependent long-range recall while storing one K/V summary per completed block rather than tokenwise KV.
4. **FP16 recurrent-memory storage with FP32 curvature** is the target persistent-state format for the recurrent path.
5. Compiler/fusion gains are measured separately from task quality so implementation speed does not affect architecture selection.

## Candidate screen

Balanced mode evaluates:

- `pdelta2_f96`
- `flash_gate_f96`
- `flash_conv4_f96`
- `flash_gate_conv4_f96`
- `flash_indexed_f96`
- `flash_gate_conv4_indexed_f96`
- `flash_gate_conv4_f64`

An additional `transformer_exact_trainable_control` uses exact causal softmax with a trainable per-head gain initialized to exact Transformer behavior. It is reported as a harder control but is not used to select the PDelta2 architecture winner.

## Protocol

- SmolLM2-135M, one attention layer replaced at a time.
- Frozen pretrained Q/K/V/O, RoPE, MLPs and all other layers for PDelta2 candidates.
- Document-disjoint deterministic train/validation/test partitions.
- Attention-output transfer followed by true next-token NLL refinement.
- Candidate selection locked on validation only.
- Held-out paired bootstrap NLL intervals against the original Transformer.
- Default test contexts: 256, 512, 1024 and 2048 tokens.
- Quality, persistent-state ratio, eager timing and `torch.compile` timing are reported separately.

A **strict quality win** requires the entire paired 95% confidence interval of candidate-minus-Transformer NLL to be below zero. A one-layer win is not a whole-model superiority claim.

## Research clues

The experiment borrows principles, not implementations, from current hybrid efficient-attention designs:

- Qwen3-Next combines Gated DeltaNet and Gated Attention and uses a short convolution in the linear-attention path.
- GLM-5.3-Flash uses a repeated three-linear-attention / one-sparse-attention layout and a short convolution in KDA layers.
- DeepSeek-V4.1-Flash emphasizes cache compression and reuse, motivating compact indexed summaries rather than full tokenwise long-range KV.

The implementation is an independent TinyCeNN research adaptation under the repository MIT license.
