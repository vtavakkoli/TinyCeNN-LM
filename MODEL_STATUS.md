# TinyCeNN-LM model status

_Catalog reviewed: 2026-09-23; core research assessments carried forward from 2026-09-16._

This file is the canonical guide to the model zoo. The repository intentionally preserves research history, but only the **Core** tracks below should be treated as current starting points.

## Core tracks

| Track | Status | Why it stays | Canonical notebook |
|---|---|---|---|
| **Qwen3.5-0.8B Integrated Memory V2.2** | **Core / strongest validated quality-efficiency track** | The validated conservative Qwen3.5 path preserves quality close to the base model while reducing attention-cache usage. V2.2 is the current implementation of that track. | `notebooks/Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb` |
| **SmolLM2 PDelta3-GDN2-CLVR + Local32** | **Core / strongest replacement research track** | Strict sequential acceptance has shown a stronger accepted prefix than MemoryFusion in the comparable SmolLM2 experiments. | `notebooks/SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb` |
| **SmolLM2 Integrated Memory V3** | **Core / stable compact reference** | The conservative partition experiment is the cleanest SmolLM2 quality-preservation reference while still reducing cache usage. | `notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb` |
| **FunctionGemma Integrated Memory V2** | **Core / tool-calling specialization** | Preserves the full-attention replacement experiment for tool-use behavior; keep separate because its evaluation target differs from general LM perplexity. | `notebooks/FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb` |

## Catalog organization

All notebook entry points are in [notebooks/](notebooks/README.md). The [archive](notebooks/archive/README.md) removes 27 historical/superseded entries from the active shortlist while preserving their source and results. Shared model implementations remain because they support comparisons and regression tests. This repository cleanup does not delete or change remote Hugging Face models.

The canonical CeNNMixer-v4 notebook is now the direct single-layer, 1024-token version that matches the current runner. Its two alpha-schedule predecessors are archived. See the [migration guide](docs/NOTEBOOK_MIGRATION.md).

## Active experiments

- **Laya / ModernBERT** — Integrated Memory V2.2, bidirectional PDelta3-GDN2-CLVR and MemoryFusion. Strict-gated encoder experiments; not complete validated replacements.
- **Qwen CeNNMixer-v4 Direct-1024** — direct layer-0 replacement, without alpha blending; still experimental.
- **FlyFFN / FlyEmbedding / FlyCore** — retained current research variants, including distinct evaluation and initialization workflows. Version numbers alone are not evidence of better quality or speed.
- **Gemma 4 all-attention MemoryFusion and FunctionGemma sequential MemoryFusion** — model-specific research, not promoted releases.


- **Nonlinear recurrent readout (SmolLM2 / Qwen3.5)** — new controlled R0/R1/R2 experiment with cached decoding. Not yet GPU-validated; see [NONLINEAR_READOUT.md](NONLINEAR_READOUT.md).

These are worth keeping for research, but they are **not current winners**:

- `Qwen3_5_0_8B_MemoryFusion_Sequential_Acceptance_Colab.ipynb` — active strict-gated MemoryFusion experiment. The current run has accepted only part of the target full-attention anchors, so do not present it as a finished replacement.
- `Qwen3_5_0_8B_PDelta3_CLVR_Sequential_Colab.ipynb` — promising Qwen-scale PDelta3 experiment; keep for comparison with Integrated Memory.
- `Gemma4_E2B_PDelta3_CLVR_Sequential_Colab.ipynb` — next-generation PDelta3 target; still experimental.
- `Gemma4_E2B_CeNN_Integrated_Memory_V2_Colab.ipynb` — corrected Gemma 4 integrated-memory experiment with the text-checkpoint remapping fix; still experimental until its held-out run is complete.

The older `Gemma4_E2B_CeNN_Integrated_Memory_Colab.ipynb` V1 path is superseded because V2 fixes the Gemma 4 multimodal-to-text checkpoint-loading issue.

## Legacy / archive tracks

The following families remain useful for reproducibility and ablations but should not be used as the main entry point:

- original Tiny-LLM / TinyCeNN residual-adapter notebooks;
- PDelta2 and early PDelta3 layer laboratories;
- full 30-layer SmolLM2 MemoryFusion r48/r64 experiments;
- AMCeNN / Top-2 experiments;
- story / anti-repeat experiments;
- superseded V1/V2 notebooks where a newer canonical version exists.

## Hugging Face publication policy

To keep the public model page understandable, publish at most one checkpoint per distinct validated architecture or one explicitly labeled active research checkpoint.

### Keep public / clearly labeled

- the current **Qwen3.5 Integrated Memory** release when published;
- the current **SmolLM2 PDelta3-CLVR** release when published;
- the current **SmolLM2 Integrated Memory V3** release when published;
- the current **FunctionGemma Integrated Memory** release when published;
- `vtava/TinyCeNN-LM-Sharded-MoE-Top2` only as a **legacy benchmark milestone**, because it has a documented held-out teacher-gap recovery result.

### Keep only as experimental if active

- `vtava/Qwen3.5-0.8B-MemoryFusion` — active research checkpoint, not a production/recommended model.

### Archive/private or remove from the public model zoo

- `vtava/TinyCeNN-LM-Base` — the published run reports `warning_no_improvement` and is superseded.
- `vtava/TinyCeNN-LM-Story-AntiRepeat` — narrow legacy branch; keep only if needed for historical reproducibility.
- `vtava/SmolLM2-135M-AMCeNN-Top2-v2` — useful experiment but superseded by stronger current tracks.
- `vtava/SmolLM2-135M-MemoryFusion-r48` and `vtava/SmolLM2-135M-MemoryFusion-r64` — full-replacement research artifacts; the notebook's intended held-out comparison did not complete successfully, so they should not clutter the public shortlist.
- any older duplicate checkpoint whose architecture and result are fully superseded by a newer version.

Do not permanently delete a checkpoint that is already cited in a paper, report, issue, or experiment log. In that case, leave it public with a clear `deprecated`/`legacy` model card or make it private if external reproducibility is not required.
