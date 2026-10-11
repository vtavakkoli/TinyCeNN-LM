"""Full-encoder 28->10 Jev-CeNN group compression + Colab invariants."""
import ast
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tinycenn_lm.jev_cenn_full_encoder import (
    group_plan,preserved_tail,JevCeNNGroupSuperLayer,
    install_group,restore_group,installed_summary,
    exact_centered_window_offsets,stable_window_attention,
)
from tinycenn_lm.jev_cenn_pairwindow_superlayer import (
    JevCeNNPairWindowSuperLayer,BypassModernBertLayer,
    pair_reachable,
)

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"Jev_CeNN_AllLayers_3to1_Progressive_Colab.ipynb"


@pytest.mark.parametrize("window", [1,2,5,8,32,512])
def test_exact_even_attention_window_indices(window):
    positions=exact_centered_window_offsets(window)
    assert positions.numel()==window
    assert int(positions[0])== -((window-1)//2)
    assert int(positions[-1])==window//2


@pytest.mark.parametrize("window", [5,8,17])
def test_cuda_native_window_sdpa_matches_explicit_reference_cpu(window):
    torch.manual_seed(27)
    b,h,t,d=2,2,19,8
    q=torch.randn(b,h,t,d,requires_grad=True)
    k=torch.randn(b,h,t,d,requires_grad=True)
    v=torch.randn(b,h,t,d,requires_grad=True)
    valid=torch.ones((b,t),dtype=torch.bool)
    valid[1,-4:]=False
    result=stable_window_attention(q,k,v,valid,window)
    pos=torch.arange(t)
    delta=pos[None,:]-pos[:,None]
    left=(window-1)//2
    right=window-1-left
    band=(delta>=-left)&(delta<=right)
    mask=band[None,None,:,:]&valid[:,None,None,:]
    scores=(q.float()@k.float().transpose(-2,-1))*d**-0.5
    probs=torch.softmax(scores.masked_fill(~mask,-1e4),dim=-1)
    expected=(probs@v.float())*valid[:,None,:,None]
    torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-5)
    result.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_group_geometry_complete_no_overlaps_and_native_tail():
    expected=tuple((i,i+1,i+2) for i in range(0,27,3))
    assert group_plan(28)==expected
    assert preserved_tail(28)==(27,)
    assert len(set(i for g in expected for i in g))==27
    assert group_plan(10)==((0,1,2),(3,4,5),(6,7,8))
    assert preserved_tail(10)==(9,)
    assert group_plan(3)==((0,1,2),)
    assert preserved_tail(3)==()
    with pytest.raises(ValueError):
        group_plan(2)
    assert pair_reachable(512,22,23,22)


def _tiny_teacher():
    from transformers import AutoModel,ModernBertConfig,__version__
    if __version__!="4.57.6":
        pytest.skip("Pinned ModernBERT Transformers 4.57.6 required")
    c=ModernBertConfig(
        vocab_size=64,hidden_size=32,intermediate_size=48,
        num_hidden_layers=10,num_attention_heads=2,
        global_attn_every_n_layers=3,local_attention=8,
        max_position_embeddings=128,reference_compile=False,
        pad_token_id=0,attention_dropout=0.,embedding_dropout=0.,mlp_dropout=0.,
    )
    return AutoModel.from_config(c,attn_implementation="sdpa").eval()


class _Wrap(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder


def test_zero_correction_reproduces_exact_previous_two_window_layer():
    torch.manual_seed(2028)
    teacher=_tiny_teacher()
    a,b=teacher.layers[3],teacher.layers[4]
    basic=JevCeNNPairWindowSuperLayer(
        a,b,local_radius=3,stride=4,dilated_radius=4,max_len=18).eval()
    upgraded=JevCeNNGroupSuperLayer(
        a,b,local_radius=3,stride=4,dilated_radius=4,max_len=18,
        correction_rank=8).eval()
    upgraded.load_state_dict(basic.state_dict(),strict=False)
    x=torch.randn(2,17,32)
    mask=torch.ones(2,17,dtype=torch.bool)
    mask[1,-3:]=False
    positions=torch.arange(17)[None,:]
    with torch.no_grad():
        plain=basic(x,attention_mask=mask,position_ids=positions)[0]
        corrected=upgraded(x,attention_mask=mask,position_ids=positions)[0]
    torch.testing.assert_close(plain,corrected,rtol=0.,atol=0.)
    assert upgraded.correction_up.weight.abs().max()==0.
    upgraded.requires_grad_(True)
    # The parity outputs above were computed under no_grad. Run a fresh
    # differentiable forward to verify low-rank correction receives gradients.
    with torch.enable_grad():
        train_output=upgraded(
            x,attention_mask=mask,position_ids=positions)[0]
        train_output.square().mean().backward()
    assert upgraded.correction_up.weight.grad is not None
    assert upgraded.correction_up.weight.grad.abs().sum()>0
    # Unlike the original W45 prototype, the new version supports a
    # user-selected explicit window without altering the old path.
    widened=JevCeNNGroupSuperLayer(
        a,b,local_radius=3,stride=4,dilated_radius=4,max_len=18,
        local_window=18,correction_rank=8).eval()
    assert widened.local_window==18
    with torch.no_grad():
        out=widened(x,attention_mask=mask,position_ids=positions)[0]
    assert out.shape==plain.shape and torch.isfinite(out).all()


def test_multiple_groups_are_installed_and_restored_correctly():
    torch.manual_seed(2028)
    native=_tiny_teacher()
    student=_Wrap(copy.deepcopy(native))
    ids=torch.randint(1,64,(2,14))
    masks=torch.ones(2,14,dtype=torch.long)
    masks[1,-3:]=0
    with torch.no_grad():
        baseline=native(input_ids=ids,attention_mask=masks).last_hidden_state
    originals={}
    for g in group_plan(10):
        block=JevCeNNGroupSuperLayer(
            native.layers[g[0]],native.layers[g[1]],
            local_radius=3,stride=4,dilated_radius=4,max_len=18,
            correction_rank=8).eval()
        originals[g]=install_group(student,g,block)
    counts=installed_summary(student,group_plan(10))
    assert counts["active_encoder_blocks"]==4
    assert counts["kept_original_layers"]==[9]
    assert counts["bypassed_complete_layers"]==6
    assert all(isinstance(student.encoder.layers[i],BypassModernBertLayer)
               for i in (1,2,4,5,7,8))
    with pytest.raises(ValueError):
        install_group(student,(0,1,2),student.encoder.layers[0])
    with torch.no_grad():
        actual=student.encoder(input_ids=ids,attention_mask=masks).last_hidden_state
    assert torch.isfinite(actual).all()
    assert actual.shape==baseline.shape
    for g in reversed(group_plan(10)):
        restore_group(student,g,originals[g])
    with torch.no_grad():
        restored=student.encoder(input_ids=ids,attention_mask=masks).last_hidden_state
    torch.testing.assert_close(restored,baseline,rtol=1e-5,atol=1e-5)


def test_real_colab_is_clean_and_full_encoder_probes_are_isolated():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf8"))
    assert nb["nbformat"]==4 and len(nb["cells"])>=15
    code="\n".join("".join(c["source"]) for c in nb["cells"]
                   if c["cell_type"]=="code")
    for required in (
        "GROUPS=group_plan", "TAIL=preserved_tail",
        "TEACHER_CACHE", "capture_all_groups",
        "add_rollout_view", "ROLL_OUT_WEIGHT", "CENN_STEPS_PER_GROUP",
        "STEPS_PER_GROUP", "CORRECTION_RANK", "REUSE_GROUP_CHECKPOINTS",
        "install_group(", "installed_summary(",
        "COMPRESSED", "JOINT_STEPS",
        "FINAL_CENN_ABLATION", "decision_quality(",
        "cuda_latency(", "LOCAL_FIDELITY_ALL",
        "ACCEPTED", "EXPERIMENTAL_NOT_DEPLOYABLE",
        "official_test_used", "hf_published",
        "jev_cenn_all_groups_report.json",
        "LOCAL_WINDOW=512", "FEEDBACK_STEPS_PER_GROUP",
        "train_group_on_changed_model(", "student_eval(**inp)",
        "stable_fp32", "prefix_fingerprint",
    ):
        assert required in code,required
    assert 'split="test"' not in code
    assert "upload_standalone_to_hub" not in code
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]!="code":
            continue
        assert c.get("execution_count") is None and not c.get("outputs")
        if i!=1:
            ast.parse("".join(c["source"]),f"<allgroups cell{i}>")
