"""Jev_CeNN one-layer L15 ablation regression and notebook coverage."""
import json
import copy
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.jev_cenn_ablation import (
    BRANCHES,ABLATION_MODES,JevGlobalBranchAblation,
    active_branches,mean_routing_weights,
)

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"Jev_CeNN_Layer15_Branch_Ablation_Colab.ipynb"


def _tiny_reference():
    from transformers import AutoModel, ModernBertConfig, __version__
    if __version__!="4.57.6":
        pytest.skip("Jev-CeNN strict ModernBERT V3.1 attention uses Transformers 4.57.6")
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    from tinycenn_lm.laya_lab.pdelta import PDelta3GDN2CLVRAttention
    from tinycenn_lm.laya_lab.core import _depthwise_sequence_conv,_valid_tokens
    torch.manual_seed(2028)
    config=ModernBertConfig(
        vocab_size=64,hidden_size=32,intermediate_size=48,
        num_hidden_layers=4,num_attention_heads=2,
        global_attn_every_n_layers=3,local_attention=8,
        max_position_embeddings=128,reference_compile=False,pad_token_id=0,
        attention_dropout=0.0,embedding_dropout=0.0,mlp_dropout=0.0
    )
    teacher=AutoModel.from_config(config,attn_implementation="sdpa").eval()
    source=json.loads((ROOT/"notebooks"/
                       "Laya_V31_AllAttention_FineTune_HF_Colab.ipynb").read_text())
    text="".join(source["cells"][4]["source"])
    ctx={"torch":torch,"nn":nn,"F":F,"copy":copy,"math":math,
         "PDelta3GDN2CLVRAttention":PDelta3GDN2CLVRAttention,
         "_depthwise_sequence_conv":_depthwise_sequence_conv,
         "_valid_tokens":_valid_tokens,
         "apply_rotary_pos_emb":apply_rotary_pos_emb,
         "MAX_LEN":64}
    exec(compile(text,"<real_v31_class>","exec"),ctx)
    real=ctx["PDeltaV28FastWindowMemory"](
        teacher.layers[0].attn,local_window=63,
        landmark_stride=4,max_landmarks=8,feature_dim=8).eval()
    with torch.no_grad():
        # The V3.1 initialized CeNN kernel is a perfect identity; perturb it
        # to ensure drop-CeNN actually removes a nontrivial output contribution.
        real.local_weight[:,0,2]+=0.1
    return real


def test_mode_definitions_are_exhaustive():
    assert active_branches("full")==BRANCHES
    assert active_branches("minus_local")==("landmarks","linear","value")
    assert active_branches("only_linear")==("linear",)
    assert active_branches("only_cenn")==()
    for mode in ABLATION_MODES:
        assert isinstance(active_branches(mode),tuple)
    with pytest.raises(ValueError):
        active_branches("imaginary")


def test_exact_manual_replay_and_all_modes_on_real_modernbert():
    model=_tiny_reference()
    x=torch.randn(2,21,32)
    mask=torch.ones(2,21,dtype=torch.long)
    mask[1,-3:]=0
    kwargs={"attention_mask":mask,
            "position_ids":torch.arange(21)[None,:].expand(2,-1)}
    original=JevGlobalBranchAblation(model,"full")
    manual=JevGlobalBranchAblation(model,"manual_full")
    with torch.no_grad():
        expected=original(x,**kwargs)[0]
        found=manual(x,**kwargs)[0]
    torch.testing.assert_close(expected,found,rtol=1e-5,atol=1e-5)
    outputs={}
    with torch.no_grad():
        for mode in ABLATION_MODES:
            out=JevGlobalBranchAblation(model,mode)(x,**kwargs)[0]
            assert out.shape==expected.shape
            assert torch.isfinite(out).all()
            outputs[mode]=out
    assert (outputs["minus_cenn"]-expected).abs().max().item()>1e-6
    assert (outputs["minus_local"]-expected).abs().max().item()>1e-6
    assert (outputs["static_routing"]-expected).abs().max().item()<0.1


def test_disabled_paths_are_not_executed():
    real=_tiny_reference()
    x=torch.randn(1,13,32)
    kw={"attention_mask":torch.ones(1,13,dtype=torch.long),
        "position_ids":torch.arange(13)[None,:]}
    with patch.object(real,"_local_attention",side_effect=AssertionError(
        "Disabled full local SDPA path must never execute"
    )):
        with torch.no_grad():
            JevGlobalBranchAblation(real,"minus_local")(x,**kw)
    with patch.object(real,"_landmarks",side_effect=AssertionError(
        "Disabled landmark SDPA path must never execute"
    )):
        with torch.no_grad():
            JevGlobalBranchAblation(real,"only_local")(x,**kw)
    with patch.object(real,"_local_attention",side_effect=AssertionError("attention used")), \
         patch.object(real,"_landmarks",side_effect=AssertionError("attention used")), \
         patch.object(real,"_global_linear",side_effect=AssertionError("memory used")):
        with torch.no_grad():
            JevGlobalBranchAblation(real,"only_cenn")(x,**kw)


def test_route_statistics_are_normalized():
    layer=_tiny_reference()
    x=torch.randn(2,11,32)
    mask=torch.ones(2,11,dtype=torch.bool)
    mask[1,-2:]=False
    cache=[{"x":x,"valid":mask,"call_kwargs":{
        "position_ids":torch.arange(11)[None,:].expand(2,-1)
    }}]
    weights=mean_routing_weights(layer,cache)
    assert set(weights)==set(BRANCHES)
    assert abs(sum(weights.values())-1.0)<1e-6


def test_colab_is_only_layer15_blind_dev_ablation():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"]==4
    assert len(nb["cells"])>=12
    blocks=[c for c in nb["cells"] if c["cell_type"]=="code"]
    text="\n".join("".join(c["source"]) for c in blocks)
    for key in (
        "LAYER = 15", "global_W1024", "load_dataset(",
        "JevGlobalBranchAblation(", "ABLATION_MODES",
        "manual_full", "mean_routing_weights(",
        "cache_teacher_io(", "score_local(", "local_metrics_from_cache(",
        "teacher_logits", "decisions_and_accuracy()",
        "delta_agreement_pp_vs_full", "whole_model_median_ms",
        "RECOVERY_STEPS", "l15_recovery_", "probe_cache", "audit_cache",
        "official_test_split_used",
    ):
        assert key in text,key
    assert 'split="test"' not in text
    assert "upload_standalone_to_hub" not in text
    assert "student_eval.encoder.layers[LAYER].attn=module" in text
    assert "student_eval.encoder.layers[LAYER].attn=original_layer15" in text
    assert all(c.get("execution_count") is None and not c.get("outputs")
               for c in blocks)
