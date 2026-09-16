# Notebook guide

The `notebooks/` directory contains the full research history. To avoid picking a superseded experiment by accident, start with one of these notebooks.

## Recommended starting points

| Goal | Notebook |
|---|---|
| Best validated quality/efficiency path on Qwen3.5 | `Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb` |
| Strongest current CeNN replacement research on SmolLM2 | `SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb` |
| Stable SmolLM2 integrated-memory reference | `SmolLM2_Integrated_Memory_V3_Colab.ipynb` |
| Tool-calling experiment | `FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb` |

## Active, not yet promoted

- `Qwen3_5_0_8B_MemoryFusion_Sequential_Acceptance_Colab.ipynb`
- `Qwen3_5_0_8B_PDelta3_CLVR_Sequential_Colab.ipynb`
- `Gemma4_E2B_PDelta3_CLVR_Sequential_Colab.ipynb`
- `Gemma4_E2B_CeNN_Integrated_Memory_Colab.ipynb`

These are current research experiments, not evidence that the adapted model is better than the base model.

## Everything else

Other notebooks are retained for ablations, earlier architecture generations, reproduction, and failed/negative experiments. Versioned notebooks should not be treated as current when a newer version exists. See [`../MODEL_STATUS.md`](../MODEL_STATUS.md) before publishing a checkpoint or quoting a result.
