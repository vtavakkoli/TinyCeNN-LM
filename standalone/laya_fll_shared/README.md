# Laya V4.7 — FULL / LOCAL(128) / LOCAL(128) with shared local attention weights

This is an experimental **architectural change to V4.6**, not a new discovery of
ModernBERT's FULL/LOCAL/LOCAL pattern. ModernBERT already uses global attention
at every third layer, with two same-width sliding-window local-attention layers
between global ones. The original 28-layer model therefore already implements
10 full and 18 local attention layers.

## Architectural modification

For each of the nine adjacent local pairs (1,2), (4,5), …, (25,26), tie their
**Wqkv** and **Wo** projection parameters. Keep all other computation independent:
the full-attention layers, QKV/outputs of global layers, local RoPE, layer norms,
residual connections, every FFN and the optimized 64-context-token V4.6
decision head remain in their original locations.

Original: 28 layers, 9 × two separate local-attention projection sets.
V4.7: 28 layers, 9 × one *shared* local-attention projection set per pair.

With hidden size 1024 and bias-free attention projections, sharing nine pairs
removes **9 × (3×1024² + 1024²) = 37,748,736 unique parameters**.
Estimated saved FP32 parameter storage is **150,994,944 bytes (~151.0 MB)**.
This does not remove any attention execution; compute savings relative to all-
full attention are already present in baseline ModernBERT. There is **no
guaranteed throughput improvement** against V4.6 and quality may degrade.

## Export and validation

[Open V4.7 experiment Colab](../../notebooks/Laya_V47_Full_2SharedWindows_Colab.ipynb).

The notebook imports [published standalone V4.6](https://huggingface.co/vtava/Laya-1.2-speedup-standalone),
ties these nine local pairs, optionally teacher-distills their shared
projection weights, and checks case-disjoint PROBE and untouched AUDIT quality,
paired CUDA+wall latency, and actual deduplicated safetensors file size.

The model file excludes tensor aliases from each second local layer and the
full standalone loader **re-ties** the parameters and verifies that only the
intentional keys are omitted. Runtime and tokenizer are included in the
exported directory so it can be loaded without calling ancestor model repos:

```python
from modeling_laya_fll_shared import load_tied_model
model, tokenizer = load_tied_model("/local/new-model", device="cuda")
```

**Hugging Face publication is gated.** Only if all quality, speed-
nonregression, storage and reload gates pass will the Colab offer
publication to `vtava/Laya-v47-full-2shared-windows` using the user's own
write-authorized login. The original model repositories are not changed.

Source inspiration: [ModernBERT ACL 2025 paper](https://aclanthology.org/2025.acl-long.127/).
