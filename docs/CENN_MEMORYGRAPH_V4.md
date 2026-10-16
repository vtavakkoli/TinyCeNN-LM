# CeNN-MemoryGraph V4 — sparse semantic cellular memory (experimental)

**Status: research implementation, not an accepted Qwen replacement.** CPU numerical tests have passed; no end-to-end Qwen GPU quality or speed results have yet been run in this release. Never claim a speedup or scientific novelty from architecture alone.

## Motivation from V3 experiment

The Qwen3.5-0.8B V3 layer-11 quick run found `adaptive_local32` local cosine **0.9261**, NMSE **0.1518**, validation ΔNLL **+0.0242** but held-out prefill speedup **0.979×** (slower). Delta-only and fixed/adaptive cell coupling without local attention clustered near **0.801–0.802 cosine**, suggesting *Local32 is the main improvement*. The V3 ablation omitted a critical **delta-only + Local32** control. Its original cell diffusion couples adjacent *feature indices*, even though feature order has no guaranteed semantic locality.

## V4 architecture

1. **Retain Qwen3.5's frozen native full-attention Q/K/V, QK norms, RoPE, per-query gate, and original output projection.** The model's residuals/MLP remain unchanged; only the full-attention mixer changes. Qwen3.5-0.8B full-attention layers: 3, 7, 11, 15, 19, 23. Default target 11.
2. Reuse the verified V3 **fp32 chunkwise delta recurrence**, a matrix memory with `Hkv` groups and `feature_dim` compressed feature cells, with a bounded causal Local32 softmax branch.
3. At **global chunk boundaries**, replace arbitrary left/center/right feature-index diffusion with a graph where each cell chooses up to `top_k` sparse semantic neighbors, based on the trained projected-key feature geometry plus learned per-cell embeddings.
4. Optionally modify each selected edge's score using **cosine similarity of current memory contents**. Edges' convex softmax coefficients and a learned per-cell rate control communication. The source cell stays in a convex mix, so finite memory transitions remain bounded by convex blending.
5. Optionally use a **per-query local-versus-memory sigmoid router**. It uses the compressed query embedding already computed for recurrent attention; unlike V3's per-head constant local gate, it can vary per token.

For feature cell `f` in chunk `b`, with semantic neighbor set `N_k(f)`:

\[
\alpha_{bfj}=\operatorname{softmax}_{j\in N_k(f)}\left[s^{\rm semantic}_{fj} + \eta\,\cos(M_{bf}, M_{bj})\right],\quad
\lambda_{bf}=\sigma(\gamma_h + w^\top M_{bf})
\]

\[
M^+_{bf}=(1-\lambda_{bf})M_{bf} + \lambda_{bf}\sum_{j\in N_k(f)}\alpha_{bfj}M_{bj}.
\]

The query readout blends bounded local attention with global associative memory:

\[
y_t=\sigma(b_h + \langle u_h,\hat q_t\rangle/\sqrt F)\,y_t^{\rm local}
+ (1-\sigma(\cdot))\,y_t^{\rm delta}.
\]

The graph is across **persistent feature-memory cells**, not token positions, making the graph update O(B·Hkv·F·k·Dv) per chunk after a small O(Hkv·F²) topology score computation. This implementation uses eager PyTorch `topk`, `gather`, Python chunk loops and triangular solves: that is a *complexity target*, **not** evidence of high GPU efficiency. The dense F×F topology scores remain a bottleneck when F is large; future implementations should cache or fuse candidate selection and benchmark honestly.

**Differentiability limitation:** `topk` chooses discrete neighbor indices. Gradients flow to selected edge values and learned rates, not through the combinatorial neighbor choices; this may impede learning meaningful topology. The notebook tests whether a trained topology matters using a **frozen-weights shuffled-neighbor intervention**. A future fully differentiable sparse-routing method should be compared.

## Controlled notebook experiment

`notebooks/Qwen35_08B_CeNN_MemoryGraph_V4_Colab.ipynb`:

| Candidate | Delta memory | Cell topology | Content-adaptive edges | Local32 | Per-token local router |
|---|---|---|---|---|---|
| `delta_local32` (V3.1) | yes | none | no | yes | no |
| `v3_adaptive_local32` | yes | index neighbors | V3 per-head router | yes | no |
| `graph_static_local32` | yes | learned sparse | no | yes | no |
| `graph_adaptive_local32` | yes | learned sparse | yes | yes | yes |
| `graph_shuffled_local32` | yes | permuted sparse | yes | yes | yes |

For a fair *incremental* ablation, a **shared Delta+Local32 model** is calibrated first and its weights are copied into each candidate; every candidate then gets the same **additional steps** and same seeded training example sequence. Train/validation/test WikiText2 splits are distinct. Model selection uses validation only. The trained `graph_adaptive_local32` or best learned graph can additionally be evaluated with graph edges shuffled **without retraining** on the validation split; this measures whether the learned neighbor topology matters.

- Default quick: Qwen3.5-0.8B layer 11; 128-token windows, 80 shared steps, 80 per candidate, 5 candidates; it is exploratory, not publication-grade.
- Strict validation gates: local teacher cosine ≥ 0.95, ΔNLL ≤ +0.10 nats/token, teacher argmax agreement ≥ 0.98.
- **Every eligible next-token prediction position** is evaluated, not sparse eighth-token subsampling. Scores include full-model NLL and teacher argmax agreement; there is no full KL (full teacher logits deliberately not cached).
- One held-out test audit is performed after validation-only model selection. Alternating *paired* native/student CUDA event timing requires > 1.05× median speedup and model-level nonregression for final acceptance.
- Optionally check prefill scaling at 128, 256, 512, 1024 tokens. It remains **cache-free prefill**, not autoregressive token generation.
- The notebook exports `final_audit.json`, `validation_candidates.csv`, `validation_ranked.csv`, checkpoint tensors, plots and exact revision fingerprints.

## Reproducible usage

```python
from tinycenn_lm.cenn_memorygraph_v4 import CeNNMemoryGraphV4
from tinycenn_lm.cenn_deltacell_v3 import Qwen35DeltaCellAttention

core = CeNNMemoryGraphV4(
    num_heads=8, num_kv_heads=2, head_dim=256,
    feature_dim=32, chunk_size=32, local_window=32,
    graph_mode="learned", top_k=4, edge_dim=12,
    content_adaptive=True, query_router=True,
)
# Do not replace the whole Transformer block: wrap original full-attention mixer
# while retaining pretrained projections and Qwen's decoder residual / FFN.
adapter = Qwen35DeltaCellAttention(original_qwen_full_attention, core)
```

Run numerical tests: `pytest -q tests/test_cenn_memorygraph_v4.py`.

## Scope and open questions

- **No cache-aware Qwen integration.** The standalone recurrent core streams, but Qwen's cache, RoPE positions, hybrid layer tracking and padded masks require separate implementation and parity tests. For notebook usage, `use_cache=False` only.
- **No existing pretrained-model weights are rewritten**; checkpoint saves only newly trained core parameters and SHA references.
- **No proven novel contribution.** Prior work includes gated delta memories, chunk parallel scans, graph/topology memory routing and dynamic retrieval. Publication needs matched parameter/compute controls, true memory retrieval tasks, multiple seeds, longer contexts, fused GPU implementations and a detailed novelty comparison.
- **Causal controls.** Always compare V3.1 delta+local, V3 adaptive+local, learned graph, content-adaptive graph and a shuffled topology intervention before attributing gains to cellular communication.