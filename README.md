# TinyCeNN-LM

New experiment: **[nonlinear recurrent readout](NONLINEAR_READOUT.md)** with separate SmolLM2-135M and Qwen3.5-0.8B Colabs, shared 0/1/2-step refinement, cached decoding, and quality/speed/memory comparisons. GPU results are not yet established.

**Researching CeNN/recurrent-memory alternatives to selected Transformer attention layers without throwing away pretrained language-model capability.**

TinyCeNN-LM has grown from the original Tiny-LLM CeNN adapter into a model-replacement research lab spanning **SmolLM2, Qwen3.5, FunctionGemma and Gemma 4**. The repository intentionally keeps negative results and ablations, but they are no longer all equal entry points.

> **Start here:** [`MODEL_STATUS.md`](MODEL_STATUS.md) is the canonical model-zoo guide. [`notebooks/README.md`](notebooks/README.md) lists the notebooks that should be used first.

## Current research tracks

| Track | Role | Start here |
|---|---|---|
| **Qwen3.5 Integrated Memory V2.2** | Strongest validated quality/efficiency track; conservative replacement rather than replacing every attention layer | [`Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb`](notebooks/Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb) |
| **SmolLM2 PDelta3-GDN2-CLVR + Local32** | Strongest current attention-replacement research direction | [`SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb`](notebooks/SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb) |
| **SmolLM2 Integrated Memory V3** | Stable compact quality-preservation/cache-efficiency reference | [`SmolLM2_Integrated_Memory_V3_Colab.ipynb`](notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb) |
| **FunctionGemma Integrated Memory V2** | Specialized tool-calling experiment | [`FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb`](notebooks/FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb) |
| **Gemma 4 E2B Integrated Memory V2** | New experimental target with corrected text-checkpoint loading; not yet promoted to a validated winner | [`Gemma4_E2B_CeNN_Integrated_Memory_V2_Colab.ipynb`](notebooks/Gemma4_E2B_CeNN_Integrated_Memory_V2_Colab.ipynb) |

## Active experiments, not headline models

The Qwen3.5 MemoryFusion and PDelta3 notebooks, and the Gemma 4 Integrated Memory V2/PDelta3 notebooks, are active research. They stay in the repository because their results are useful, but an incomplete strict-gate run must not be presented as proof that the adapted model is better than the base model.

Full 30-layer SmolLM2 MemoryFusion, AMCeNN, PDelta2, the original Tiny-LLM adapter, story/anti-repeat and other early notebooks are **legacy/ablation tracks**. They remain available for reproducibility and negative-result analysis; see [`MODEL_STATUS.md`](MODEL_STATUS.md) before publishing or citing one as a recommended checkpoint.

## Research rule: selective replacement first

The strongest recent results support a more conservative principle than the original “replace everything” experiments:

1. identify attention layers that are good replacement candidates;
2. train one replacement at a time;
3. gate acceptance using representation similarity and model-level loss criteria;
4. keep the pretrained/native mechanism when a candidate fails the gate;
5. evaluate the accepted model on data or tasks that were not used to train the replacement;
6. measure quality **and** the efficiency benefit (cache, memory, latency and trainable parameters).

This makes failed replacements informative rather than allowing one bad layer to contaminate an entire model.

## Main architecture families

### Integrated Memory

Conservative selective replacement designed to preserve pretrained behavior while reducing the cost of a subset of attention layers. This is currently the strongest complete quality-preservation path in the repository.

### PDelta3-GDN2-CLVR

A recurrent/delta-memory replacement direction combining local CeNN computation with editable global memory and sequential acceptance. This is the main architecture-research track when the goal is to replace attention rather than simply augment it.

### MemoryFusion

Combines local cellular/multiscale processing with global recurrent memory. It remains scientifically useful, especially in strict sequential experiments, but the old full-replacement r48/r64 checkpoints are not the recommended model path.

### AMCeNN / Top-2 and original TinyCeNN

Earlier architecture generations. Keep them for comparisons and reproducibility, not as the default starting point for new experiments.

## Installation

```bash
git clone https://github.com/vtavakkoli/TinyCeNN-LM.git
cd TinyCeNN-LM
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Run the test suite before starting a new architecture experiment:

```bash
python -m pytest -q
```

The repository contains targeted regression tests for the SmolLM2, Qwen3.5 and Gemma 4 adapters in addition to the original TinyCeNN components.

## Repository layout

```text
src/tinycenn_lm/   architecture implementations
scripts/           training, acceptance, benchmark and evaluation runners
notebooks/         Colab experiments; see notebooks/README.md first
tests/             regression and architecture tests
MODEL_STATUS.md    canonical keep/archive/publication status
```

Older design documents such as [`DISTILLATION.md`](DISTILLATION.md), [`PDELTA2_FLASH.md`](PDELTA2_FLASH.md), [`SHARDED_MOE.md`](SHARDED_MOE.md), [`OPTIMIZED_MEMORY.md`](OPTIMIZED_MEMORY.md) and [`RESEARCH_LAYERS.md`](RESEARCH_LAYERS.md) are retained as research history.

## Hugging Face

Experimental checkpoints are published under [`vtava`](https://huggingface.co/vtava). A public checkpoint should have a model card that states:

- the exact base model and architecture variant;
- which layers were replaced;
- whether the run passed strict acceptance gates;
- whether reported metrics are training diagnostics or genuinely held-out evaluation;
- known generation or task regressions;
- the matching GitHub notebook/script and commit when possible.

Do not call a checkpoint “best” merely because it is the newest upload. Promote it only after its evaluation is stronger or its efficiency/quality trade-off is clearly better than the current reference.

## Original TinyCeNN-LM work

The first TinyCeNN-LM experiments used `arnir0/Tiny-LLM` and a causal CeNN residual state core, followed by Transformer-free distillation, sharded MoE, story/anti-repeat and PDelta experiments. Those implementations and documents remain in the repository for reproducibility, but they are now the **legacy track**, not the primary project description.

## License

MIT
