# WaveFlux-CeNN — pure cellular sequence-layer research prototype

**Status (18 October 2026):** executable CPU PyTorch reference with seven passing tests, standalone causal LM example, and a small successful delayed-recall training smoke test. **No GPU benchmarks, pretrained language-model training, Triton kernels, or verified scientific novelty.**

## No hybrid modules

The network uses only token embeddings, pointwise projection/gating, recurrent cellular-wave transport, conservative neighborhood flux for readout, output projection and residual normalization. There are **no** self-attention, cross-attention, Q/K/V, KV caches, DeltaNet/Gated DeltaNet/mLSTM/LSTM states, Transformers or external sequence mixers. A recurrent cellular **state** is its only token-history memory. Frozen pretrained models may be used later as offline teachers for distillation, but no teacher is present in the deployed architecture or current test.

## Core math

For each wave lane `r` with integer propagation velocity `v_r`, ring group `g`, and channel `c`, let input-dependent `w[t,r,g] ∈ (0,1)`, learned baseline decay `d_r ∈ (0,1)`, and token-projected excitation `u[t,g,c]` give

```
a[t,r,g] = d[r] * (1 - w[t,r,g])
b[t,r,g,c] = w[t,r,g] * u[t,g,c]
s[t,r,g,c] = a[t,r,g] * s[t-1,r,(g-v[r]) mod G,c] + b[t,r,g,c]
```

The temporal state transition is **local recurrent cellular transport** (a fixed directed neighbor per lane), not a pointwise state update with an unrelated convolution. These states have `R*G*C` floats per sequence regardless of sequence length.

For a sequence chunk with start state `s[0]`, set `n=t+1` and define a moving-frame state `z[t,r] = P_r^(-n) s[t,r]`, where `P_r` rotates the ring by `v_r`. This turns the spatially coupled recurrence into

```
z[t] = (P_r^(-n) a[t]) * z[t-1] + P_r^(-n) b[t]
```

Here the products are elementwise. Each time step is the affine map `(A,B): x ↦ A*x+B`, whose associative composition is

```
(A2, B2) ∘ (A1, B1) = (A2*A1, B2 + A2*B1)
```

The reference uses an inclusive recursive-doubling scan with `O(log T)` dependent stages and `O(T log T)` total scalar-work; an optimized work-efficient scan could reduce work to `O(T)`. This claim concerns computational depth, **not achieved GPU throughput**. Chunk continuation is exact since the moving frame resets relative to the supplied chunk boundary.

Optionally, the wave readout performs conservative local flux on the cell ring:

```
J[t,r,g,c] = k[t,r,g] * (s[t,r,g,c] - s[t,r,(g+1) mod G,c])
s_star[t,r,g,c] = s[t,r,g,c] + J[t,r,(g-1) mod G,c] - J[t,r,g,c]
```

This conserves the sum over cellular groups for each wave/channel. `0 ≤ k ≤ 0.5` gives a convex neighbor redistribution for a ring. **Flux only affects readout** in this first version; feeding state-dependent diffusion back into recurrence would generally break the straightforward associative scan. Waves are mixed with input-conditioned per-lane read gates and projected to the output.

## Installation and verification

```bash
python -m pip install torch pytest
pytest -q test_waveflux_cenn.py
python train_delayed_recall.py --steps 200 --delay 32 --batch 96 --classes 8
```

The trained delayed recall uses random first-symbol values from eight classes, 32 filler tokens, and an uninformative final query token. Training targets only the final position; no causal shortcut from that token alone exists. With seed 1234, on the available CPU, 200 updates gave **100% accuracy on 512 newly sampled examples** (chance 12.5%), 11.55 s elapsed. This is a functional synthetic test rather than a measure of generalization to realistic languages, long-range retrieval, speed, efficiency, or robustness across seeds. Independent samples use the same simple task generator, so distribution-level novelty is not implied.

Additional 120-step/8-token/4-class smoke experiment reached 100% on 512 fresh samples. Tests cover exact scan/recurrent equivalence, exact chunk streaming up to floating-point tolerance, future-token causality, flux conservation, ring velocity transport, finite gradients, and standalone LM chunk consistency. All seven pass on CPU with PyTorch 2.10.0.

## Proposed study and engineering milestones

1. Compare the proposed pure cellular wave layer with **existing pure CeNNMixer V1 and V2**, equal parameter budgets, training data and steps, and state-cache bytes. The direct attention-free CeNN reference, not a hybrid PDelta/Local32 model, is the central baseline.
2. Run practical sequence tests: associative recall, delayed multi-query recall, selective copying, length extrapolation, language modeling from scratch, and causal streaming across arbitrary chunk partitions. Report whole-network held-out CE/NLL/PPL and task accuracy on fresh splits and multiple seeds.
3. Compare optimized recurrent CPU reference and the exact scan on an actual CUDA GPU. Implement and validate Triton/CUDA **work-efficient fused scan** for wave transport; then fuse learned projections/readout where worthwhile. Never infer speed from asymptotic analysis.
4. Establish ablations: single zero-velocity bank vs traveling waves; one vs multiple velocities; constant vs token-conditioned write gates; flux readout enabled/disabled; round-robin vs fixed-varying phase; 1/2/4 blocks. Keep initial capacity and optimizer budgets controlled.
5. Before making a novelty claim, compare with **Hidden Traveling Waves bind Working Memory Variables in Recurrent Neural Networks (ICML 2024)**; **Parallelized Hierarchical Connectome (2026)**; affine parallel scan / SSM work; xLSTM (NeurIPS 2024); Mamba-3 (2026); and pure cellular/recurrent sequence architectures.

## Caveats

- Python/PyTorch version is for correctness, not production GPU performance. Sequence prefill currently materializes all wave states and uses repeated tensor operations, with `O(T*R*G*C)` activation memory plus scan intermediates; decode maintains only `O(R*G*C)` persistent state per block. No custom Triton kernel is implemented.
- Periodic cell rings are a modeling choice, not a validated architecture advantage; some tasks may benefit from other boundaries or dynamic routing.
- The scan is equivalent to a structured recurrent/state-space model after a coordinate transform, so words like "first," "unique," and "novel" require prior-art scrutiny. The contribution to test is the **combination** of cellular transport, moving-frame scan, input-dependent memory, and conservative local flux readout.
- The sample language model has no masking/padding or beam-search/cache API integration; it accepts ordinary valid unpadded causal batches and explicit layer states. It contains no external pretrained weights.
