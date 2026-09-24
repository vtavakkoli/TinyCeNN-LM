# Notebook catalog

All **68 notebooks** live under this directory: **41 current entries** and **27 archived experiments**. Start with a core reference or choose an explicitly experimental track below.

[Project overview](../README.md) · [Model status](../MODEL_STATUS.md) · [Archived notebooks](archive/README.md) · [Path migration](../docs/NOTEBOOK_MIGRATION.md)

## Quick start

1. Choose a notebook and open its **Colab** link.
2. Select a GPU runtime and run setup before training.
3. Review strict acceptance and held-out results before exporting a checkpoint.

The core labels carry forward the existing model-status guide; this reorganization does not establish new benchmark results. Experiments and benchmarks are not validated model releases.

**CeNNMixer-v4:** the canonical notebook now uses direct layer-0 replacement and 1024-token training, matching the current runner. Older alpha-schedule variants are archived.

## Core references

| Notebook | Run |
|---|---|
| [FunctionGemma 270M CeNN Integrated Memory V2](FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb) |
| [Qwen3 5 0 8B CeNN Integrated Memory V2 2](Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb) |
| [SmolLM2 Integrated Memory V3](SmolLM2_Integrated_Memory_V3_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb) |
| [SmolLM2 PDelta3 CLVR Sequential Optimization](SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb) |

## Laya / ModernBERT

| Notebook | Run |
|---|---|
| [Laya Integrated Memory V22](Laya_Integrated_Memory_V22_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V22_Colab.ipynb) |
| [Laya Integrated Memory V23 Decision](Laya_Integrated_Memory_V23_Decision_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V23_Decision_Colab.ipynb) |
| [Laya MemoryFusion](Laya_MemoryFusion_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_MemoryFusion_Colab.ipynb) |
| [Laya PDelta3 GDN2 CLVR](Laya_PDelta3_GDN2_CLVR_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_PDelta3_GDN2_CLVR_Colab.ipynb) |
| [Laya PDelta3 GDN2 Standalone Decision](Laya_PDelta3_GDN2_Standalone_Decision_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_PDelta3_GDN2_Standalone_Decision_Colab.ipynb) |

## Qwen3.5 experiments

| Notebook | Run |
|---|---|
| [Qwen35 08B CeNNMixer v4 DeltaCell](Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb) |
| [Qwen35 08B FlyCore from v3 Standalone](Qwen35_08B_FlyCore_from_v3_Standalone_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyCore_from_v3_Standalone_Colab.ipynb) |
| [Qwen35 08B FlyCore v2 from v3](Qwen35_08B_FlyCore_v2_from_v3_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyCore_v2_from_v3_Colab.ipynb) |
| [Qwen35 08B FlyEmbedding v31 Preservation First](Qwen35_08B_FlyEmbedding_v31_Preservation_First_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyEmbedding_v31_Preservation_First_Colab.ipynb) |
| [Qwen35 08B FlyEmbedding v32 TokenGate](Qwen35_08B_FlyEmbedding_v32_TokenGate_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyEmbedding_v32_TokenGate_Colab.ipynb) |
| [Qwen35 08B FlyEmbedding v33 Progressive Replacement](Qwen35_08B_FlyEmbedding_v33_Progressive_Replacement_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyEmbedding_v33_Progressive_Replacement_Colab.ipynb) |
| [Qwen35 08B FlyFFN v2](Qwen35_08B_FlyFFN_v2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyFFN_v2_Colab.ipynb) |
| [Qwen35 08B FlyFFN v3 AllFFN](Qwen35_08B_FlyFFN_v3_AllFFN_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyFFN_v3_AllFFN_Colab.ipynb) |
| [Qwen3 5 0 8B MemoryFusion Sequential Acceptance](Qwen3_5_0_8B_MemoryFusion_Sequential_Acceptance_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_MemoryFusion_Sequential_Acceptance_Colab.ipynb) |
| [Qwen3 5 0 8B Nonlinear Recurrent Readout](Qwen3_5_0_8B_Nonlinear_Recurrent_Readout_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_Nonlinear_Recurrent_Readout_Colab.ipynb) |
| [Qwen3 5 0 8B PDelta3 CLVR Sequential](Qwen3_5_0_8B_PDelta3_CLVR_Sequential_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_PDelta3_CLVR_Sequential_Colab.ipynb) |

## SmolLM2 experiments

| Notebook | Run |
|---|---|
| [FlyFFN v2 SmolLM2 135M](FlyFFN_v2_SmolLM2_135M_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/FlyFFN_v2_SmolLM2_135M_Colab.ipynb) |
| [SmolLM2 MemoryFusion Sequential Acceptance v3](SmolLM2_MemoryFusion_Sequential_Acceptance_v3_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_MemoryFusion_Sequential_Acceptance_v3_Colab.ipynb) |
| [SmolLM2 Nonlinear Recurrent Readout](SmolLM2_Nonlinear_Recurrent_Readout_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_Nonlinear_Recurrent_Readout_Colab.ipynb) |

## Gemma / FunctionGemma experiments

| Notebook | Run |
|---|---|
| [FunctionGemma 270M MemoryFusion Sequential Acceptance](FunctionGemma_270M_MemoryFusion_Sequential_Acceptance_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/FunctionGemma_270M_MemoryFusion_Sequential_Acceptance_Colab.ipynb) |
| [Gemma4 E2B CeNN Integrated Memory V2](Gemma4_E2B_CeNN_Integrated_Memory_V2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Gemma4_E2B_CeNN_Integrated_Memory_V2_Colab.ipynb) |
| [Gemma4 E2B MemoryFusion All Attention](Gemma4_E2B_MemoryFusion_All_Attention_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Gemma4_E2B_MemoryFusion_All_Attention_Colab.ipynb) |
| [Gemma4 E2B PDelta3 CLVR Sequential](Gemma4_E2B_PDelta3_CLVR_Sequential_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Gemma4_E2B_PDelta3_CLVR_Sequential_Colab.ipynb) |

## Architecture benchmarks

| Notebook | Run |
|---|---|
| [CeNN Adaptive MaxPool Attention](CeNN_Adaptive_MaxPool_Attention_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Adaptive_MaxPool_Attention_Colab.ipynb) |
| [CeNN Attention Preservation](CeNN_Attention_Preservation_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Attention_Preservation_Colab.ipynb) |
| [CeNN Cellular Attention](CeNN_Cellular_Attention_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Cellular_Attention_Colab.ipynb) |
| [CeNN Global Memory Tournament](CeNN_Global_Memory_Tournament_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Global_Memory_Tournament_Colab.ipynb) |
| [CeNN Kernel Variants](CeNN_Kernel_Variants_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Kernel_Variants_Colab.ipynb) |
| [CeNN Learned Kernel Concepts](CeNN_Learned_Kernel_Concepts_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Learned_Kernel_Concepts_Colab.ipynb) |
| [CeNN Optimized Memory V2](CeNN_Optimized_Memory_V2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Optimized_Memory_V2_Colab.ipynb) |
| [CeNN Research Layers](CeNN_Research_Layers_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_Research_Layers_Colab.ipynb) |
| [CeNN UAMP Final Tournament](CeNN_UAMP_Final_Tournament_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_UAMP_Final_Tournament_Colab.ipynb) |
| [FlyCeNN Transformer Replacement](FlyCeNN_Transformer_Replacement_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/FlyCeNN_Transformer_Replacement_Colab.ipynb) |

## Evaluation and export

| Notebook | Run |
|---|---|
| [PrismML Ternary Bonsai2 27B FastEval50](PrismML_Ternary_Bonsai2_27B_FastEval50_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/PrismML_Ternary_Bonsai2_27B_FastEval50_Colab.ipynb) |
| [Qwen35 08B FlyEmbedding v31 FastEval HF](Qwen35_08B_FlyEmbedding_v31_FastEval_HF_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_FlyEmbedding_v31_FastEval_HF_Colab.ipynb) |
| [Qwen3 5 0 8B Three Model GGUF Export](Qwen3_5_0_8B_Three_Model_GGUF_Export_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_Three_Model_GGUF_Export_Colab.ipynb) |
| [Qwen3 5 Standalone Release Optimized](Qwen3_5_Standalone_Release_Optimized_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_Standalone_Release_Optimized_Colab.ipynb) |
| [SmolLM2 MemoryFusion Evaluation Only](SmolLM2_MemoryFusion_Evaluation_Only_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_MemoryFusion_Evaluation_Only_Colab.ipynb) |

## Legacy benchmark

| Notebook | Run |
|---|---|
| [TinyCeNN MoE Top2](TinyCeNN_MoE_Top2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/TinyCeNN_MoE_Top2_Colab.ipynb) |
