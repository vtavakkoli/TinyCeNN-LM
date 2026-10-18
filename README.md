# TinyCeNN-LM

**A research lab for compact recurrent memory and selective Transformer-layer replacement.**

[Notebook catalog](notebooks/README.md) · [Model status](MODEL_STATUS.md) · [Architecture guide](NONLINEAR_READOUT.md) · [Archived experiments](notebooks/archive/README.md)

Explore CeNN, integrated memory, delta-memory and compact feed-forward replacements on **SmolLM2, Qwen3.5, FunctionGemma, Gemma 4 and Laya / ModernBERT**. The goal is to preserve pretrained capability while measuring the quality, memory and latency trade-offs of each replacement.

**New V2.5 experiment:** [Windowed Delta + multiscale CeNN for Laya](docs/LAYA_V25.md), retaining V2.4 training stages with conditional Hugging Face export. Full GPU results are pending.

**Laya FP16 inference optimization:** [GPU benchmark notebook](notebooks/Laya_FP16_Fast_Inference_Colab.ipynb) · [Implementation and validation](docs/LAYA_FP16_FAST_INFERENCE.md). Uses the existing standalone FP16 weights; GPU speedup must be measured on your hardware.

**Laya FP16 V2:** [Compiled control and end-to-end API notebook](notebooks/Laya_FP16_Compiled_Control_API_V2_Colab.ipynb) · [Benchmark protocol](docs/LAYA_FP16_BENCHMARK_V2.md).

**New pure-cellular research prototype:** [WaveFlux-CeNN standalone mathematical reference](docs/WAVEFLUX_CENN.md) · [SmolLM2-360M-Instruct first-attention replacement experiment](docs/WAVEFLUX_SMOL360_LAYER0.md) · [GPU Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/WaveFlux_SmolLM2_360M_Layer0_Standalone_Colab.ipynb). The WaveFlux replacement module has no attention/DeltaNet/LSTM; only layer 0 is replaced in the 360M evaluation harness. Held-out quality and GPU speed remain unverified.

**Feasibility-first pure-cellular research (experimental):** [Competitive CeNN theory and synthetic convergence audit](docs/COMPETITIVE_CENN_FEASIBILITY.md) · [SmolLM2-360M native-attention diagnostic Colab](notebooks/CeNN_Competitive_Attention_Feasibility_Colab.ipynb). This work does **not** claim pretrained-model parity or CUDA speedups; its content-conditioned cellular routing is functionally attention-like and must be evaluated before another layer substitution.

**New CCM-v2 feasibility control with chat:** [Cellular consensus design and limitations](docs/CCM_CONSENSUS_V2_SMOL360_CHAT.md) · [Colab for SmolLM2-360M-Instruct first-attention replacement, DEV/test gates and five side-by-side chats](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/research/competitive-cenn-feasibility/notebooks/CCM_v2_SmolLM2_360M_Consensus_Chat_Colab.ipynb). This is an **experimental research branch**, and its soft cellular consensus is mathematically equivalent to local attention over a bounded tape; no model fidelity or GPU gains are claimed before running.\n\n**CCM-v2 real-label train/resume/eval Colab:** [Train the layer-0 CCM from scratch, resume a checkpoint ZIP, or evaluate saved weights; then check gold WikiText NLL, ARC-Easy, chat rules and CUDA speed](notebooks/CCM_v2_Smol360_RealFastEval_Only_Colab.ipynb). Only the cellular layer trains. Teacher agreement is diagnostic, not a quality gate.

## Choose a starting point

| Goal | Track | Notebook |
|---|---|---|
| Qwen quality and cache efficiency | Integrated Memory V2.2 · core reference | [Open](notebooks/Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb) |
| Selective SmolLM2 attention replacement | PDelta3-GDN2-CLVR + Local32 · core research | [Open](notebooks/SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb) |
| Compact SmolLM2 reference | Integrated Memory V3 · core reference | [Open](notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb) |
| Tool-calling behavior | FunctionGemma Integrated Memory V2 · specialized reference | [Open](notebooks/FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb) |
| Typed decisions on Laya | Integrated Memory, PDelta3 and MemoryFusion · experimental | [Choose a notebook](notebooks/README.md#laya--modernbert) |
| Direct compact-mixer training | CeNNMixer-v4 · experimental, one layer, 1024 tokens | [Open](notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb) |
| Nonlinear recurrent refinement | SmolLM2 / Qwen R0–R2 comparisons · experimental | [Read the protocol](NONLINEAR_READOUT.md) |

**[Browse all 45 current notebook entries →](notebooks/README.md)** Each entry has a direct Colab link. The [30 archived notebooks](notebooks/archive/README.md) preserve superseded models, ablations and historical runs without cluttering the active list.

Core designations reflect the existing [model-status record](MODEL_STATUS.md), not new validation from this cleanup. A partial accepted run is not a complete replacement, and a smaller model is not necessarily faster.

## How experiments are evaluated

1. Keep an untouched pretrained teacher as the reference.
2. Train a candidate replacement and measure representation fidelity.
3. Accept layers sequentially only when model-level quality gates pass.
4. Restore the native layer when a candidate fails its gate.
5. Compare held-out quality, memory, cache usage and measured latency.

The direct CeNNMixer-v4 experiment starts with one compact layer rather than an alpha curriculum; its own quality gates still control release. Laya experiments preserve the tokenizer and typed-decision head and evaluate bidirectional encoder replacements. Neither track is promoted to a validated winner merely by being included here.

## Architecture families

| Family | Purpose | Status |
|---|---|---|
| Integrated Memory | Conservative replacements that preserve pretrained behavior | Core references and model-specific experiments |
| PDelta3-GDN2-CLVR | Local cellular processing with editable recurrent memory | Selective replacement research |
| MemoryFusion | Local multiscale processing plus global recurrent memory | Active sequential experiments; old full-replacement runs archived |
| CeNNMixer / FlyFFN / FlyEmbedding / FlyCore | Compact mixer, feed-forward and embedding experiments | Experimental; see the notebook catalog |
| Original TinyCeNN / AMCeNN / PDelta2 | Earlier architectures and negative-result comparisons | Archived entry points; implementations retained for reproducibility |

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
notebooks/         all current Colab entry points and their catalog
notebooks/archive/ historical and superseded experiments
docs/              notebook migration guide and machine-readable path map
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

## Research history

Earlier notebooks are explicitly labeled as archived. Shared architecture implementations, training runners and regression tests stay available for comparisons. See the [migration guide](docs/NOTEBOOK_MIGRATION.md) for moved paths and the original revision.

## License

[MIT](LICENSE)
