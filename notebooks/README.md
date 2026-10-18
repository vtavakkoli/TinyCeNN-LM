# Notebook catalog

All **118 notebooks** live under this directory: **88 current entries** and **30 archived experiments**. Start with a core reference or choose an explicitly experimental track below.

[Project overview](../README.md) · [Model status](../MODEL_STATUS.md) · [Archived notebooks](archive/README.md) · [Path migration](../docs/NOTEBOOK_MIGRATION.md)

## Quick start

1. Choose a notebook and open its **Colab** link.
2. Select a GPU runtime and run setup before training.
3. Review strict acceptance and held-out results before exporting a checkpoint.

The core labels carry forward the existing model-status guide; this reorganization does not establish new benchmark results. Experiments and benchmarks are not validated model releases.

**CeNNMixer-v4:** the canonical notebook now uses direct layer-0 replacement and 1024-token training, matching the current runner. Older alpha-schedule variants are archived.

## Pure cellular WaveFlux-CeNN (experimental)

WaveFlux is a standalone traveling-cell recurrent layer with exact parallel-scan reference (no attention/QKV/DeltaNet/LSTM). The SmolLM2-360M-Instruct experiment **replaces only the first attention module** as a controlled first-step evaluation; the remainder of the language model is still native. Passing CPU unit tests is **not** a GPU speed or quality result. See [architecture](../docs/WAVEFLUX_CENN.md) and [first-layer protocol](../docs/WAVEFLUX_SMOL360_LAYER0.md).

| Notebook | Run |
|---|---|
| [WaveFlux-CeNN Standalone Reference](WaveFlux_CeNN_Standalone_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/WaveFlux_CeNN_Standalone_Colab.ipynb) |
| [WaveFlux × SmolLM2-360M-Instruct Layer-0 Pure Cellular Replacement](WaveFlux_SmolLM2_360M_Layer0_Standalone_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/WaveFlux_SmolLM2_360M_Layer0_Standalone_Colab.ipynb) |

## Core references

| Notebook | Run |
|---|---|
| [FunctionGemma 270M CeNN Integrated Memory V2](FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/FunctionGemma_270M_CeNN_Integrated_Memory_V2_Colab.ipynb) |
| [Qwen3 5 0 8B CeNN Integrated Memory V2 2](Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen3_5_0_8B_CeNN_Integrated_Memory_V2_2_Colab.ipynb) |
| [SmolLM2 Integrated Memory V3](SmolLM2_Integrated_Memory_V3_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_Integrated_Memory_V3_Colab.ipynb) |
| [SmolLM2 PDelta3 CLVR Sequential Optimization](SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/SmolLM2_PDelta3_CLVR_Sequential_Optimization_Colab.ipynb) |

## Laya / ModernBERT

**FP16 benchmark V2:** [Compiled-original control + complete API](Laya_FP16_Compiled_Control_API_V2_Colab.ipynb) · [Run in Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/laya-fp16-fast-inference/notebooks/Laya_FP16_Compiled_Control_API_V2_Colab.ipynb). Separates compilation gains from implementation gains; supports custom JSONL cases.

**Existing FP16 model, faster inference:** [Notebook](Laya_FP16_Fast_Inference_Colab.ipynb) · [Run in Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/laya-fp16-fast-inference/notebooks/Laya_FP16_Fast_Inference_Colab.ipynb). Parity-gated SDPA, compilation and CUDA-graph benchmark; no retraining.

**V2.5:** all encoder attention is replaced with local Delta/global CeNN, followed by a pooled MLP head. Training stages match V2.4; passing models can be uploaded to Hugging Face. [Architecture, evaluation gate and limitations](../docs/LAYA_V25.md).

| Notebook | Run |
|---|---|
| [Laya V2.4 Bi-Gated Delta Decision](Laya_Integrated_Memory_V24_BiGatedDeltaLite_Decision_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V24_BiGatedDeltaLite_Decision_Colab.ipynb) |
| [Laya V2.5 Sliding CeNN + Multiscale CeNN Decision](Laya_Integrated_Memory_V25_WindowDelta_CeNN_Decision_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V25_WindowDelta_CeNN_Decision_Colab.ipynb) |
| [Laya V2.6 Hard Full-Attention Tournament](Laya_Integrated_Memory_V26_FullAttention_Tournament_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V26_FullAttention_Tournament_Colab.ipynb) |
| [Laya V2.7 PDelta3 Local32 + AttentionCeNN](Laya_V27_PDelta3_Local32_AttentionCeNN_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V27_PDelta3_Local32_AttentionCeNN_Colab.ipynb) |
| [Laya V2.8 Fast Full-Attention-First PDelta3 + Local32](Laya_V28_Fast_FullAttention_First_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V28_Fast_FullAttention_First_Colab.ipynb) |
| [Laya V2.6 Layer15 Full Attention Tournament](Laya_V26_Layer15_FullAttention_Tournament_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V26_Layer15_FullAttention_Tournament_Colab.ipynb) |
| [Laya V2.8 Layer15 FastWindowMemory](Laya_V28_Layer15_FastWindowMemory_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V28_Layer15_FastWindowMemory_Colab.ipynb) |
| [Laya V2.9.1 All-Global FastWindowMemory + Hugging Face Standalone](Laya_V29_AllGlobal_FastWindowMemory_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V29_AllGlobal_FastWindowMemory_Colab.ipynb) |
| [Laya V3.0 SlidingWindowFusion: W33/W65/W97/W129, 0.99 teacher cosine target](Laya_V30_SlidingWindowFusion_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V30_SlidingWindowFusion_Colab.ipynb) |
| [Laya V3.1 All-Attention Conversion + Joint Fine-tuning + HF Publish](Laya_V31_AllAttention_FineTune_HF_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V31_AllAttention_FineTune_HF_Colab.ipynb) |
| [Jev_CeNN — CUDA Optimized All-30-Attention + HF Export](Jev_CeNN_CUDA_Fast_AllAttention_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_CUDA_Fast_AllAttention_Colab.ipynb) |
| [Jev_CeNN — Single Global Layer-15 Branch Ablation + Recovery](Jev_CeNN_Layer15_Branch_Ablation_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_Layer15_Branch_Ablation_Colab.ipynb) |
| [Jev_CeNN 2-Window Global Coverage: True Encoder Layers 15–17 to One](Jev_CeNN_2Window_3Layers_To_1_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_2Window_3Layers_To_1_Colab.ipynb) |
| [Jev_CeNN Full Encoder 28→10 Progressive 3-to-1 Two-Window CeNN](Jev_CeNN_AllLayers_3to1_Progressive_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_AllLayers_3to1_Progressive_Colab.ipynb) |
| [Jev_CeNN W512 Selective 3→1 with Independent Holdout](Jev_CeNN_W512_Selective_3to1_Heldout_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_W512_Selective_3to1_Heldout_Colab.ipynb) |
| [Jev_CeNN Global→W129→Global→W129 CeNN 600-Step 3→1](Jev_CeNN_GlobalLocal_x2_600Step_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_GlobalLocal_x2_600Step_Colab.ipynb) |
| [Laya V40 Group18-20 Full Window CeNN Sweep](Laya_V40_Group18_20_FullWindow_CeNN1025_128_Sweep_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V40_Group18_20_FullWindow_CeNN1025_128_Sweep_Colab.ipynb) |
| [Laya V41 Residual-Faithful CeNN Group 18–20](Laya_V41_ResidualFaithful_CeNN_Group18_20_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V41_ResidualFaithful_CeNN_Group18_20_Colab.ipynb) |
| [Laya V42 Progressive All-Full Then All-Sliding CeNN](Laya_V42_Progressive_AllFull_Then_AllSliding_CeNN_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V42_Progressive_AllFull_Then_AllSliding_CeNN_Colab.ipynb) |
| [Laya V39 Global CeNN Two-Sliding 3→1 Colab](Laya_V39_Global_CeNN_TwoSliding_3to1_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V39_Global_CeNN_TwoSliding_3to1_Colab.ipynb) |
| [Jev_CeNN Bonsai G128 Layerwise Ablation](Jev_CeNN_Bonsai_G128_Layerwise_Ablation_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_Bonsai_G128_Layerwise_Ablation_Colab.ipynb) |
| [Laya V3.7 Attention Necessity Fast SDPA Ablation](Laya_V37_Attention_Necessity_Fast_SDPA_Ablation_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V37_Attention_Necessity_Fast_SDPA_Ablation_Colab.ipynb) |
| [Laya V3.8 Greedy Layerwise Fast Accurate](Laya_V38_Greedy_Layerwise_Fast_Accurate_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V38_Greedy_Layerwise_Fast_Accurate_Colab.ipynb) |
| [Laya V3.6 All-Attention Routed and Sliding Native SDPA](Laya_V36_AllAttention_RoutedAndSliding_NativeSDPA_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V36_AllAttention_RoutedAndSliding_NativeSDPA_Colab.ipynb) |
| [Laya V3.5 All-Full Fused Triton CUDA](Laya_V35_AllFullAttention_FusedTriton_CUDA_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V35_AllFullAttention_FusedTriton_CUDA_Colab.ipynb) |
| [Jev-CeNN Attention-Free Feasibility: Global + Sliding (0.99 Goal)](Jev_CeNN_Attention_Free_Global_Sliding_Feasibility_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Jev_CeNN_Attention_Free_Global_Sliding_Feasibility_Colab.ipynb) |
| [Laya V3.3 Retrieval Oracle Gaussian VAE Audit](Laya_V33_Retrieval_Oracle_Gaussian_VAE_Audit_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V33_Retrieval_Oracle_Gaussian_VAE_Audit_Colab.ipynb) |
| [Laya V3.4 All-Full Attention Query-Routed Sparse](Laya_V34_AllFullAttention_QueryRoutedSparse_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V34_AllFullAttention_QueryRoutedSparse_Colab.ipynb) |
| [Laya V3.2 Layer15 U-Net MaxPool + Landmark Memory](Laya_V32_Layer15_UNet_MaxPool_LandmarkMemory_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V32_Layer15_UNet_MaxPool_LandmarkMemory_Colab.ipynb) |
| [Laya V3.1 Layer 15 Cellular-Dilated Fusion](Laya_V31_Layer15_Cellular_Dilated_Fusion_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V31_Layer15_Cellular_Dilated_Fusion_Colab.ipynb) |
| [Laya V2.9 Mathematical Audit](Laya_V29_Mathematical_Audit_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V29_Mathematical_Audit_Colab.ipynb) |
| [Laya V2.9 MultiscaleCeNN Full Attention](Laya_V29_MultiscaleCeNN_FullAttention_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V29_MultiscaleCeNN_FullAttention_Colab.ipynb) |
| [Laya V3.0 Fourier Spectral Replacement Audit](Laya_V30_Fourier_Spectral_Replacement_Audit_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_V30_Fourier_Spectral_Replacement_Audit_Colab.ipynb) |
| [Laya Integrated Memory V22](Laya_Integrated_Memory_V22_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V22_Colab.ipynb) |
| [Laya Integrated Memory V23 Decision](Laya_Integrated_Memory_V23_Decision_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_Integrated_Memory_V23_Decision_Colab.ipynb) |
| [Laya MemoryFusion](Laya_MemoryFusion_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_MemoryFusion_Colab.ipynb) |
| [Laya PDelta3 GDN2 CLVR](Laya_PDelta3_GDN2_CLVR_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_PDelta3_GDN2_CLVR_Colab.ipynb) |
| [Laya PDelta3 GDN2 Standalone Decision](Laya_PDelta3_GDN2_Standalone_Decision_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Laya_PDelta3_GDN2_Standalone_Decision_Colab.ipynb) |

## Qwen3.5 experiments

| Notebook | Run |
|---|---|
| [Qwen3.5 0.8B Delta-CeNN + V5.3 All 24 Progressive (experimental)](Qwen35_08B_DeltaCeNN_V53_All24_Progressive_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_DeltaCeNN_V53_All24_Progressive_Colab.ipynb) |
| [Qwen35 08B CeNNMixer v4 DeltaCell](Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN-X2 Middle-Layer Ablation (experimental)](Qwen35_08B_CeNN_X2_Middle_Layer_Ablation_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_X2_Middle_Layer_Ablation_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN-DeltaCell V3 Cellular Memory (experimental)](Qwen35_08B_CeNN_DeltaCell_V3_Cellular_Memory_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_DeltaCell_V3_Cellular_Memory_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN-MemoryGraph V4 (experimental)](Qwen35_08B_CeNN_MemoryGraph_V4_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_MemoryGraph_V4_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN V5 Head Rescue (experimental)](Qwen35_08B_CeNN_HeadRescue_V5_Quality_First_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V5_Quality_First_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN V5.2 Fidelity Recovery (experimental)](Qwen35_08B_CeNN_HeadRescue_V52_FidelityRecovery_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V52_FidelityRecovery_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN V5.3 Cross-Calibrated Head Rescue (experimental)](Qwen35_08B_CeNN_HeadRescue_V53_CrossCalibrated_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V53_CrossCalibrated_Colab.ipynb) |
| [Qwen3.5-0.8B CeNN V5.3 all six full-attention layers (experimental)](Qwen35_08B_CeNN_V53_All6_FullAttention_Sequential_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_V53_All6_FullAttention_Sequential_Colab.ipynb) |
| [Qwen3.5 0.8B CeNN V5.1 Checkpoint-Guided Head Rescue (experimental)](Qwen35_08B_CeNN_HeadRescue_V51_Warmstart_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/Qwen35_08B_CeNN_HeadRescue_V51_Warmstart_Colab.ipynb) |
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
| [CeNN-X2 Dual-Timescale GPU Layer (experimental)](CeNN_X2_DualTimescale_GPU_Colab.ipynb) | [Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/main/notebooks/CeNN_X2_DualTimescale_GPU_Colab.ipynb) |
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


## Laya FP16 V3: all-28-layer brute-force search

[Open V3 in Colab](https://colab.research.google.com/github/vtavakkoli/TinyCeNN-LM/blob/codex/laya-fp16-fast-inference/notebooks/Laya_FP16_All28_BruteForce_V3_Colab.ipynb) · [Method and V2 findings](../docs/LAYA_FP16_LAYER_SEARCH_V3.md)

Search attention/FFN/block bypasses and activation-ranked FFN channel pruning at every layer, then bounded combinations. Includes labelled case-disjoint validation, compiled speed controls, complete API timing, a final official-test audit and a reconstructable recipe. GPU results are required before claiming further improvement.
