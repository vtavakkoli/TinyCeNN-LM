# Jev-CeNN: a single CeNN-assisted two-window layer replaces encoder blocks 15–17

[Open experimental Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_2Window_3Layers_To_1_Colab.ipynb)

## Precise replacement semantics

The original frozen ModernBERT encoder has 28 layers. Index 15 uses global
attention; indices 16 and 17 use sliding attention. The new experimental
`JevCeNNPairWindowSuperLayer` replaces the **entire three-layer composition**:

```text
Teacher: ... -> [L15 global attention + FFN]
             -> [L16 sliding attention + FFN]
             -> [L17 sliding attention + FFN] -> [L18] ...
Student: ... -> [L15 two sparse attentions + two CeNN injections + ONE FFN]
             -> [L16 Identity] -> [L17 Identity] -> [L18] ...
```

Do not confuse this with swapping the `attn` field in three layers:
**two complete MLP/norm/residual blocks are eliminated.** One still retains
the original 28 ModuleList positions so checkpoint-independent encoder
iteration and return types remain consistent. All other 25 encoder layers
and the typed-decision head remain original and frozen.

## Global connectivity is not exact attention equivalence

Window 1: centered local `radius 22`, 45 key positions.
Window 2: centered *dilated* `radius 22`, stride `23`, 45 key positions.

For length up to 512, the union of the first-window neighborhoods centered
at each valid second-window intermediate token includes every source token.
An exhaustive Boolean reachability test checks all 262,144 query–source
pairs at T512. Attention scores are `T*(45+45)` = 46,080 nominal pairs
at 512 positions. Full attention alone computes 262,144 pairs; the
proposed module has TWO attention stages plus CeNN/FFN and other overhead.
The reference consists of **three entire transformer blocks**, so only a
real matched CUDA benchmark gives a legitimate speedup claim.

This two-hop coverage establishes dependency paths, **not a theorem of
universal teacher attention factorization or optimizer convergence**.

## CeNN effectiveness is explicitly tested

The cellular pathway has depthwise filter receptive fields 7, 33 and
129 (dilations 1,4,16), learned channel mixing and gating. It is
inserted *before the second QKV projection*, so it can affect the
content-dependent dilated global attention, not just a tiny output residual.

The experiment first trains the cellular mixer alone to match the
original 15th encoder layer's residual, selected on separate DEV-PROBE
examples. Then the 2-attention, 1-FFN superlayer is trained end-to-end
against the frozen original 15→17 composition and an auxiliary
layer-15 intermediate target. An explicit cellular auxiliary loss
prevents the path from being completely neglected during stage B.

Critically, **after training** the same checkpoint is run with cellular
mixing enabled and completely disabled. This reports the held-out
cosine/NMSE and end-to-end teacher decision agreement/gold change.
The lab does NOT assume that a learned cellular path is beneficial.

## Training and evaluation

1. Source only the official dataset's TRAIN cases. Split cases, not
   question samples, into independent TRAIN and DEV collections.
2. Capture layer-15 input and original outputs after layer 15,16,17
   using teacher hooks. Cache TRAIN, PROBE and AUDIT separately.
3. Select Stage A CeNN weights and Stage B superlayer weights using
   the development PROBE only, never blind audit.
4. Once the model is selected, evaluate blind AUDIT output
   approximation, and independently evaluate end-to-end DEV decisions.
5. Benchmark original 3 full transformer blocks versus the 1-block
   replacement using matched inputs and actual CUDA events, plus
   end-to-end teacher/student latency at B1/B2.
6. Save an experimental checkpoint and JSON report; label checkpoint
   `ACCEPTED` only if all fidelity, gold and speed gates pass.
   Otherwise label `EXPERIMENTAL_NOT_DEPLOYABLE`.
7. Never use official TEST data, publish to Hugging Face, overwrite
   existing `vtava/Jev_CeNN` or assert a numerical outcome before GPU
   execution.

**Default GPU:** T4; batch=2, Stage A=180 optimizer steps, Stage B=400
optimizer steps, max sequence 512. The sparse gather/softmax is a
**differentiable PyTorch reference**, not yet a fused Triton production
kernel. It may be slower than dense GPU SDPA. No speedup is guaranteed.

The trained block has a smaller number of *encoder stages* than the
teacher, but not necessarily 1/3 parameters or 1/3 compute due to
its two attention paths and multi-branch CeNN. Always report
student parameter count and measured runtime explicitly.
