# CCM-CeNN: feasibility-first standalone cellular competitor

## Why

The first WaveFlux-CeNN substitution for SmolLM2-360M-Instruct **failed** its
quality and speed gates. The best selected checkpoint at step 100 had validation
ΔNLL +0.46735 and teacher agreement 0.70221. The held-out ΔNLL was +0.34256.
Tesla T4 whole-model prefill speed ratio native/student was 0.97382×. Despite
0.96428 cosine on a later local-output checkpoint, downstream language quality
was much worse. The proposed next experiment is **not** another WaveFlux training
run; it tests a qualitatively different memory-access architecture first.

## Theoretical facts

A fixed-precision state of B bits cannot store N arbitrary V-ary symbols exactly
if B < N log2(V). Unlimited-context arbitrary recall is therefore not guaranteed
by fixed-state CeNN. A finite cellular spatial tape has explicit capacity:
`slots × (tag_width + cell_width)` floating point numbers, excluding parameters.

For a local cellular tournament, each adjacent pair of nodes has potentials
`r_left, r_right` and memory outputs `v_left, v_right`:

    g = sigmoid(temperature*(r_right-r_left))
    v_parent = (1-g)*v_left + g*v_right
    r_parent = max(r_left, r_right)

At a leaf `r_i = dot(normalize(f(x_query)), normalize(f(x_cell_i)))`.
Although scoring and message-passing happen locally and **no conventional
Transformer attention module is used**, content-selective routing is
**functionally attention-like**. This must not be misrepresented as a solution
without selective retrieval. It is not an original mechanism proven novel.

With a hard winner j* and normalized softmax weights p_i over bounded values
||v_i|| ≤ M, ||v_j* - sum p_i v_i|| ≤ 2M (1-p_j*).
Thus the top-one tournament approximates attention ONLY when attention is sharp.
For diffuse attention the worst-case approximation can be arbitrarily poor.

A binary tree of local convex gates can represent a **given** categorical
weighted sum exactly by passing subtree mass ratios. That constructive result
does *not* solve the hard part: learning or calculating the gates from inputs.
Computing softmax scores or exact subtree masses would effectively re-create
attention, defeating the aim of an attention-free computational advantage.

## CPU proof-of-concept results (seed 2031; all synthetic)

- Generic learned local conductance (`CellularRoutingTree`) did **not** learn
  end-to-end semantic lookup in 1,000 steps: test MSE ~0.873 vs uniform ~0.868.
- Explicit competition in `CompetitiveCeNN` on **fresh random payloads** with
  one-hot symbolic IDs learned a very sharp softmax teacher with test MSE
  1.75e-6 at 8 cells/800 updates, 7.34e-6 at 16 cells/600 updates, and
  3.67e-6 at 32 cells/600 updates. These are favorable discrete retrieval tasks,
  not continuous attention equivalence nor language-model quality.
- Position-only cellular convex routing learned a content-INdependent temporal
  exponential weighting to MSE 6.50e-7 after 400 updates (16 cells).
- Continuous random attention gives a negative result for hard routing. At
  16 tokens and attention logit scale β=1, the mean maximum weight is ~0.243
  and the hard-winner approximation has relative output MSE ~4.91.

## Go/no-go before trying SmolLM2-360M again

1. `python attention_diagnostic.py --device cuda`: use *native* SmolLM2
   attention maps at 128/256 tokens to measure top-1/top-4/top-8 mass and
   post-V oracle sparse-output error. Native attention must be retained.
2. If layer-0 attention is concentrated (e.g. top4 mean mass >0.90 **and**
   oracle top4 relative MSE <0.05 on more than one text case), prototype
   top-k *cellular* competition. Top-1 alone is likely insufficient.
3. If attention is diffuse, test parallel transport/averaging plus selective
   cellular states as a fundamentally different class. Do **not** claim full
   attention fidelity or train a 360M adapter yet.
4. Any candidate ultimately needs held-out ΔNLL ≤0.02, agreement ≥0.98,
   cosine ≥0.95 AND real GPU full-model speedup >1.0, using the prior gates.
5. Do not promote a checkpoint based solely on local-output cosine.

## Files

- `src/tinycenn_lm/competitive_cenn.py`: pure single cellular mixer using ring-memory and a
  learned shared content encoder with local, binary, winner-take-all gating.
- `tests/test_competitive_cenn.py`: streaming equivalence, causality, gradient tests.
- `src/tinycenn_lm/competitive_cenn_experiment.py`: train on discrete-memory retrieval; teacher only in
  the *loss*, not a hybrid student model.
- `run_diffuse_gap()` in the same experiment module: diffuse-attention gap.
- `scripts/attention_concentration_smol360.py`: SmolLM2-360M native layer-0 head concentration,
  oracle sparse-output error; requires GPU to evaluate the real checkpoint.
- `CeNN_Competitive_Attention_Feasibility_Colab.ipynb`: one-click diagnostics.

No actual SmolLM2-360M attention distributions or GPU student results have been
measured by this package. Research and reproducibility tool only.
