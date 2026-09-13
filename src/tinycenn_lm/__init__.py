from .live_console import configure_live_console

# Configure the current process before importing model modules. Every train_*.py
# script imports tinycenn_lm, so notebook-launched trainers inherit immediate,
# line-buffered stdout/stderr even when the notebook uses subprocess.run(...).
configure_live_console()

from .cenn import CeNNConfig, FastCeNNCore
from .modeling import (
    DEFAULT_BASE_MODEL,
    HybridDecoderLayer,
    build_from_adapter,
    freeze_for_adapter_training,
    inject_cenn,
    load_adapter,
    save_adapter,
    trainable_parameter_summary,
)
from .student import (
    CeNNReplacementLayer,
    build_cenn_student,
    freeze_student_interfaces,
    load_cenn_student_weights,
    replace_transformer_with_cenn,
    save_cenn_student,
    student_parameter_summary,
)
from .moe import (
    FastMoECeNNCore,
    MoECeNNConfig,
    MoECeNNReplacementLayer,
    build_moe_cenn_student,
    freeze_moe_student_interfaces,
    load_moe_cenn_student_weights,
    moe_router_stats,
    replace_transformer_with_moe_cenn,
    save_moe_cenn_student,
    warmstart_moe_from_plain_cenn,
)
from .sharded_moe import (
    FastShardedMoECeNNCore,
    ShardedMoECeNNConfig,
    ShardedMoECeNNReplacementLayer,
    build_sharded_moe_student,
    freeze_sharded_moe_interfaces,
    load_sharded_moe_student_weights,
    replace_transformer_with_sharded_moe_cenn,
    save_sharded_moe_student,
    sharded_router_stats,
    warmstart_sharded_moe_from_plain_cenn,
)
from .story_v2 import (
    CausalStoryMemory,
    LowRankLMHeadAdapter,
    StoryV2Config,
    StoryV2ReplacementLayer,
    build_story_v2_from_story_v1,
    build_story_v2_student,
    freeze_story_v2_interfaces,
    load_story_v2_weights,
    save_story_v2_student,
    story_v2_parameter_summary,
    story_v2_router_stats,
    upgrade_sharded_model_to_story_v2,
)
from .smollm2_amcenn import (
    DEFAULT_SMOLLM2,
    AMCeNNAttention,
    PositiveSoftmaxFeatures,
    ShardedTop2LlamaMLP,
    SmolAMCeNNConfig,
    amcenn_parameter_summary,
    amcenn_router_stats,
    build_smollm2_amcenn,
    freeze_smollm2_for_amcenn_training,
    load_smollm2_amcenn_weights,
    replace_smollm2_core,
    save_smollm2_amcenn,
)
from .smollm2_amcenn_v2 import (
    AMCeNNAttentionV2,
    AdaptivePositiveSoftmaxFeatures,
    SmolAMCeNNV2Config,
    build_smollm2_amcenn_v2,
    convert_all_ffns_to_sharded_top2,
    freeze_for_global_training,
    freeze_for_group_calibration,
    load_smollm2_amcenn_v2_weights,
    replace_all_smollm2_attention,
    replace_attention_layers,
    save_smollm2_amcenn_v2,
    v2_parameter_summary,
)
from .hf_persistence import (
    build_model_card,
    collect_reports,
    install_colab_hf_upload_enhancer,
    persist_hf_run,
    redact_secrets,
    utc_run_id,
)
from .colab_live_backup import (
    install_colab_training_backup,
    is_tinycenn_training_command,
    output_dir_from_command,
)
from .direct_colab_backup import install_direct_training_backup

# In Colab, make Hugging Face backup mandatory. The parent notebook wrapper handles
# normal subprocess-launched trainers. A trainer-side fallback covers notebooks that
# launch train_*.py before the notebook kernel imports tinycenn_lm.
install_colab_training_backup()
install_direct_training_backup()
install_colab_hf_upload_enhancer()

__all__ = [
    "configure_live_console",
    "CeNNConfig", "FastCeNNCore", "DEFAULT_BASE_MODEL", "HybridDecoderLayer",
    "build_from_adapter", "freeze_for_adapter_training", "inject_cenn", "load_adapter",
    "save_adapter", "trainable_parameter_summary", "CeNNReplacementLayer", "build_cenn_student",
    "freeze_student_interfaces", "load_cenn_student_weights", "replace_transformer_with_cenn",
    "save_cenn_student", "student_parameter_summary", "MoECeNNConfig", "FastMoECeNNCore",
    "MoECeNNReplacementLayer", "build_moe_cenn_student", "freeze_moe_student_interfaces",
    "load_moe_cenn_student_weights", "moe_router_stats", "replace_transformer_with_moe_cenn",
    "save_moe_cenn_student", "warmstart_moe_from_plain_cenn", "ShardedMoECeNNConfig",
    "FastShardedMoECeNNCore", "ShardedMoECeNNReplacementLayer", "build_sharded_moe_student",
    "freeze_sharded_moe_interfaces", "load_sharded_moe_student_weights",
    "replace_transformer_with_sharded_moe_cenn", "save_sharded_moe_student", "sharded_router_stats",
    "warmstart_sharded_moe_from_plain_cenn", "StoryV2Config", "CausalStoryMemory",
    "StoryV2ReplacementLayer", "LowRankLMHeadAdapter", "upgrade_sharded_model_to_story_v2",
    "freeze_story_v2_interfaces", "story_v2_router_stats", "story_v2_parameter_summary",
    "save_story_v2_student", "load_story_v2_weights", "build_story_v2_student",
    "build_story_v2_from_story_v1", "DEFAULT_SMOLLM2", "SmolAMCeNNConfig",
    "PositiveSoftmaxFeatures", "AMCeNNAttention", "ShardedTop2LlamaMLP", "replace_smollm2_core",
    "freeze_smollm2_for_amcenn_training", "amcenn_router_stats", "amcenn_parameter_summary",
    "save_smollm2_amcenn", "load_smollm2_amcenn_weights", "build_smollm2_amcenn",
    "SmolAMCeNNV2Config", "AdaptivePositiveSoftmaxFeatures", "AMCeNNAttentionV2",
    "convert_all_ffns_to_sharded_top2", "replace_attention_layers", "replace_all_smollm2_attention",
    "freeze_for_group_calibration", "freeze_for_global_training", "v2_parameter_summary",
    "save_smollm2_amcenn_v2", "load_smollm2_amcenn_v2_weights", "build_smollm2_amcenn_v2",
    "build_model_card", "collect_reports", "persist_hf_run", "install_colab_hf_upload_enhancer",
    "redact_secrets", "utc_run_id", "install_colab_training_backup", "install_direct_training_backup",
    "is_tinycenn_training_command", "output_dir_from_command",
]
