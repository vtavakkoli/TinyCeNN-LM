# Notebook migration — 2026-09-23

All notebooks now live under `notebooks/`. Superseded and legacy entries are under `notebooks/archive/`; shared implementation modules remain in place because current runners and regression tests still use them. No Hugging Face checkpoint was changed.

## CeNNMixer-v4 correction

The nested Direct-1024 notebook now occupies the canonical `notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb` path. It calls the current runner with `--layer`, `--seq-len 1024` and context-length probes. The former canonical notebook used removed alpha-schedule arguments; it and the runtime-branch version are archived separately.

## Reproducing old experiments

Original files are available at commit [`1a2b371b6a70`](https://github.com/vtavakkoli/TinyCeNN-LM/tree/1a2b371b6a70e8b73dce18cbc8c4f8996b05f459). Check out that revision for the pre-cleanup layout. An older branch-pinned notebook may additionally require the branch named in its setup cell.

Old external notebook/Colab URLs cannot redirect automatically. Update bookmarks using the table below. Current canonical URLs are retained wherever possible.

| Previous path | New path | Reason |
|---|---|---|
| `notebooks/TinyCeNN_LM_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_LM_Colab.ipynb](../notebooks/archive/original/TinyCeNN_LM_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/TinyCeNN_Distill_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_Distill_Colab.ipynb](../notebooks/archive/original/TinyCeNN_Distill_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/TinyCeNN_Optimized_Continue_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_Optimized_Continue_Colab.ipynb](../notebooks/archive/original/TinyCeNN_Optimized_Continue_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/TinyCeNN_Rigorous_Continue_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_Rigorous_Continue_Colab.ipynb](../notebooks/archive/original/TinyCeNN_Rigorous_Continue_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/TinyCeNN_SharedFFN_Top2_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_SharedFFN_Top2_Colab.ipynb](../notebooks/archive/original/TinyCeNN_SharedFFN_Top2_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/TinyCeNN_Story_v2_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_Story_v2_Colab.ipynb](../notebooks/archive/original/TinyCeNN_Story_v2_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/TinyCeNN_Story_AntiRepeat_Colab.ipynb` | [notebooks/archive/original/TinyCeNN_Story_AntiRepeat_Colab.ipynb](../notebooks/archive/original/TinyCeNN_Story_AntiRepeat_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/SmolLM2_AMCeNN_Top2_v2_Colab.ipynb` | [notebooks/archive/amcenn/SmolLM2_AMCeNN_Top2_v2_Colab.ipynb](../notebooks/archive/amcenn/SmolLM2_AMCeNN_Top2_v2_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/SmolLM2_AMCeNN_Adaptive_v4_Colab.ipynb` | [notebooks/archive/amcenn/SmolLM2_AMCeNN_Adaptive_v4_Colab.ipynb](../notebooks/archive/amcenn/SmolLM2_AMCeNN_Adaptive_v4_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_PDelta2_Error_Residual_Layer_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_PDelta2_Error_Residual_Layer_Colab.ipynb](../notebooks/archive/pdelta/CeNN_PDelta2_Error_Residual_Layer_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_PDelta2_Flash_Layer_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_PDelta2_Flash_Layer_Colab.ipynb](../notebooks/archive/pdelta/CeNN_PDelta2_Flash_Layer_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_PDelta2_ER2_Long_Context_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_PDelta2_ER2_Long_Context_Colab.ipynb](../notebooks/archive/pdelta/CeNN_PDelta2_ER2_Long_Context_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_PDelta2_Feature_Lab_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_PDelta2_Feature_Lab_Colab.ipynb](../notebooks/archive/pdelta/CeNN_PDelta2_Feature_Lab_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_Preconditioned_Delta2_Beat_Transformer_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_Preconditioned_Delta2_Beat_Transformer_Colab.ipynb](../notebooks/archive/pdelta/CeNN_Preconditioned_Delta2_Beat_Transformer_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_PDelta3_Frontier_Layer_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_PDelta3_Frontier_Layer_Colab.ipynb](../notebooks/archive/pdelta/CeNN_PDelta3_Frontier_Layer_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/CeNN_PDelta3_Tiny_LLM_Compatibility_Colab.ipynb` | [notebooks/archive/pdelta/CeNN_PDelta3_Tiny_LLM_Compatibility_Colab.ipynb](../notebooks/archive/pdelta/CeNN_PDelta3_Tiny_LLM_Compatibility_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/SmolLM2_MemoryFusion_r48_r64_Colab.ipynb` | [notebooks/archive/memoryfusion/SmolLM2_MemoryFusion_r48_r64_Colab.ipynb](../notebooks/archive/memoryfusion/SmolLM2_MemoryFusion_r48_r64_Colab.ipynb) | Legacy/ablation family in MODEL_STATUS.md |
| `notebooks/Qwen35_08B_CeNNMixer_v1_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v1_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v1_Colab.ipynb) | Earlier generation; newer notebook retained in active catalog |
| `notebooks/Qwen35_08B_CeNNMixer_v2_Progressive_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v2_Progressive_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v2_Progressive_Colab.ipynb) | Earlier generation; newer notebook retained in active catalog |
| `notebooks/Qwen35_08B_CeNNMixer_v3_GlobalMemory_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v3_GlobalMemory_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v3_GlobalMemory_Colab.ipynb) | Earlier generation; newer notebook retained in active catalog |
| `notebooks/FlyFFN_SmolLM2_135M_Colab.ipynb` | [notebooks/archive/superseded/FlyFFN_SmolLM2_135M_Colab.ipynb](../notebooks/archive/superseded/FlyFFN_SmolLM2_135M_Colab.ipynb) | Earlier generation; newer notebook retained in active catalog |
| `notebooks/Qwen35_08B_FlyCore_v1_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_FlyCore_v1_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_FlyCore_v1_Colab.ipynb) | Earlier generation; newer notebook retained in active catalog |
| `notebooks/Qwen35_08B_FlyEmbedding_v3_Residual_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_FlyEmbedding_v3_Residual_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_FlyEmbedding_v3_Residual_Colab.ipynb) | Earlier generation; newer notebook retained in active catalog |
| `SmolLM2_AMCeNN_Hybrid_v3_Colab.ipynb` | [notebooks/archive/amcenn/SmolLM2_AMCeNN_Hybrid_v3_Colab.ipynb](../notebooks/archive/amcenn/SmolLM2_AMCeNN_Hybrid_v3_Colab.ipynb) | Legacy AMCeNN notebook moved out of repository root |
| `notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v4_Alpha_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v4_Alpha_Colab.ipynb) | Old alpha-schedule notebook mismatches the current direct-training runner |
| `cennmixer-v4-runtime-training/notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb` | [notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v4_Runtime_Branch_Colab.ipynb](../notebooks/archive/superseded/Qwen35_08B_CeNNMixer_v4_Runtime_Branch_Colab.ipynb) | Distinct older branch-pinned alpha experiment |
| `cenn-v4-context-quality/cennmixer-v4-runtime-training/notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb` | [notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb](../notebooks/Qwen35_08B_CeNNMixer_v4_DeltaCell_Colab.ipynb) | Canonical direct single-layer 1024-token notebook matching current runner |
