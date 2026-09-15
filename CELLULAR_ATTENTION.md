# Cellular Attention: sparse / shifted 1-D attention for CeNN language models

## Core problem: the receptive-field bottleneck

A pure 1-D CeNN or neural cellular automaton normally applies the same small
local neighborhood repeatedly. With a causal kernel of width 3, token `i` can
only receive information from itself and a few immediately preceding cells in
one update. Reaching distant context therefore takes a number of propagation
steps that grows linearly with distance.

For an autoregressive language model the neighborhood must be **causal**:
future tokens are never visible. The diagrams below should therefore be read as
left-looking communication, not bidirectional attention.

```text
local step:
T0 <- T1 <- T2 <- T3 <- T4
               ^
               T4 can only gather a short prefix neighborhood per step
```

This is cheap, but it creates two problems:

1. **Long path length.** Distant information needs many recurrent updates.
2. **Repeated compression.** Information can be blurred by many nonlinear
   transformations before it reaches a distant token.

Dense Transformer self-attention solves the path-length problem with direct
all-to-all causal links in one layer, but materializes `O(N^2)` score pairs
during prefill.

## Fix: causal Cellular Attention

Cellular Attention keeps every individual attention neighborhood small, but
changes that neighborhood across cellular steps. The default dilated schedule is

```text
d = 1, 2, 4, 8, 16, 32, 64, 128, ...
```

For a 3-cell causal neighborhood, step `s` lets token `i` attend to

```text
i, i-d_s, i-2*d_s
```

so the maximum reachable history after several recurrent steps grows
exponentially with the number of steps. With dilations
`1,2,4,8,16,32,64,128`, eight steps have a theoretical causal receptive field
of 511 tokens:

```text
1 + 2 * (1 + 2 + 4 + ... + 128) = 511
```

Each sparse neighborhood uses ordinary learned Q/K compatibility:

$$
a_{ij}^{(s)} =
\operatorname{softmax}_{j \in \mathcal{N}_s(i)}
\left(
\frac{q_i^{(s)} \cdot k_j}{\sqrt{d_f}} + b_{s,ij}
\right)
$$

and recurrently transports the value state:

$$
h_i^{(s+1)}
=
(1-g_s)h_i^{(s)}
+
g_s \sum_{j \in \mathcal{N}_s(i)}
a_{ij}^{(s)} h_j^{(s)} .
$$

The implementation also learns a state-derived query correction, so later
cellular steps can route using information collected by earlier steps.

## Implemented candidate scenarios

| Variant | Neighborhood | Purpose |
|---|---|---|
| `cellular_local3` | `{0,1,2}` every step | control for the local receptive-field bottleneck |
| `cellular_dilated3` | `{0,d,2d}` | minimal exponential-distance cellular attention |
| `cellular_dilated5` | `{0,d,2d,3d,4d}` | more capacity per scale |
| `cellular_multiscale5` | `{0,1,d,2d,4d}` | keeps exact near context plus long-range spokes |
| `cellular_shifted8` | alternating 8-token causal block windows | shifted-window alternative |

All variants are strictly causal.

## Complexity

Let `w` be the small number of neighbors and `S` the number of cellular steps.

- Dense causal attention score pairs: `O(N^2)`.
- Cellular Attention score pairs: `O(N * w * S)`.
- If the dilation schedule is extended until it covers a sequence of length
  `N`, then `S = O(log N)`, giving `O(N log N)` score computation.
- For a fixed deployment context and a fixed step schedule, `S` is a constant
  and the score computation is effectively linear in `N`.

This does **not** by itself guarantee faster wall-clock execution. The current
PyTorch implementation is a research reference; dense SDPA is highly optimized.
The benchmark therefore reports both score-pair counts and measured kernel time.

## Experimental question

The key question is not whether sparse attention can imitate an attention map.
The benchmark asks whether replacing one pretrained SmolLM2 attention layer with
Cellular Attention can preserve or improve held-out next-token NLL while using
far fewer score pairs.

The experiment:

1. hashes whole documents into disjoint train / validation / test partitions;
2. freezes the pretrained model, including Q/K/V/O and MLPs;
3. trains only the new Cellular Attention layer;
4. first transfers the original attention output;
5. then refines on true next-token loss;
6. selects the architecture using validation NLL only;
7. locks that decision before evaluating the held-out test documents;
8. reports paired bootstrap NLL intervals, perplexity, gradient fidelity,
   measured prefill time, peak allocation and sparse-vs-dense score-pair ratios.

The included Colab downloads the complete result directory as a ZIP at the end.
