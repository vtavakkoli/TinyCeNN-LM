"""Regression tests for the 4-pass global/local repeat 600-step CeNN lab."""
import ast
import copy
import json
import io
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.jev_cenn_full_local_repeat import (
    FullLocalRepeatCeNNSuperLayer,sdpa_full_or_window,cpu_state_snapshot,
)
from tinycenn_lm.jev_cenn_full_encoder import (
    install_group,restore_group,
)
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"Jev_CeNN_GlobalLocal_x2_600Step_Colab.ipynb"


def test_true_global_attention_can_reach_opposite_sequence_end():
    # Full attention at position 0 must see token 511, unlike centered W512.
    t=512
    q=torch.zeros((1,1,t,1))
    k=torch.zeros_like(q)
    v=torch.zeros_like(q)
    v[:,:,511,0]=1.0
    valid=torch.ones((1,t),dtype=torch.bool)
    full=sdpa_full_or_window(q,k,v,valid,window=None)
    short=sdpa_full_or_window(q,k,v,valid,window=129)
    assert full[0,0,0,0].item()>0
    assert short[0,0,0,0].item()==0
    assert torch.isfinite(full).all()


def test_window_129_and_global_match_explicit_reference_and_autograd():
    torch.manual_seed(2028)
    b,h,t,d=2,2,19,8
    q=torch.randn(b,h,t,d,requires_grad=True)
    k=torch.randn(b,h,t,d,requires_grad=True)
    v=torch.randn(b,h,t,d,requires_grad=True)
    valid=torch.ones((b,t),dtype=torch.bool)
    valid[1,-4:]=False
    for window in (None,5,17,129):
        y=sdpa_full_or_window(q,k,v,valid,window=window)
        pos=torch.arange(t)
        band=torch.ones((t,t),dtype=torch.bool) if window is None else (
            (pos[None,:]-pos[:,None]).abs()<=((window-1)//2))
        allowed=band[None,None,:,:]&valid[:,None,None,:]
        logits=(q@k.transpose(-2,-1))*d**-0.5
        prob=torch.softmax(logits.float().masked_fill(~allowed,-1e4),-1)
        ref=(prob@v)*valid[:,None,:,None]
        torch.testing.assert_close(y,ref,atol=1e-5,rtol=1e-5)
        y.square().mean().backward(retain_graph=True)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def _tiny():
    from transformers import AutoModel,ModernBertConfig,__version__
    if __version__!="4.57.6":
        pytest.skip("Requires exact original ModernBERT runtime 4.57.6")
    config=ModernBertConfig(
        vocab_size=64,hidden_size=32,intermediate_size=48,
        num_hidden_layers=18,num_attention_heads=2,
        global_attn_every_n_layers=3,local_attention=8,
        max_position_embeddings=128,reference_compile=False,
        pad_token_id=0,attention_dropout=0.,embedding_dropout=0.,mlp_dropout=0.)
    return AutoModel.from_config(
        config,attn_implementation="sdpa").eval()


def test_four_stage_model_is_differentiable_and_replaces_exactly_three_full_layers():
    torch.manual_seed(2028)
    original=_tiny()
    class Model(nn.Module):
        def __init__(self,encoder):
            super().__init__()
            self.encoder=encoder
    student=Model(copy.deepcopy(original))
    layer=FullLocalRepeatCeNNSuperLayer(
        original.layers[3],original.layers[4],original.layers[5],
        window=9,max_len=24,correction_rank=8)
    assert [m.window for m in layer.stages]==[None,9,None,9]
    ids=torch.randint(1,64,(2,17))
    attn=torch.ones((2,17),dtype=torch.long)
    attn[1,-3:]=0
    with torch.no_grad():
        old=original(input_ids=ids,attention_mask=attn).last_hidden_state
    modules=install_group(student,(3,4,5),layer)
    with torch.no_grad():
        converted=student.encoder(
            input_ids=ids,attention_mask=attn).last_hidden_state
    assert converted.shape==old.shape
    assert torch.isfinite(converted).all()
    student.encoder.eval()
    layer.train()
    output=student.encoder(input_ids=ids,attention_mask=attn).last_hidden_state
    output.square().mean().backward()
    assert layer.stages[0].qkv.weight.grad is not None
    assert layer.stages[2].qkv.weight.grad is not None
    assert any(p.grad is not None for p in layer.cenn.parameters())
    restore_group(student,(3,4,5),modules)
    with torch.no_grad():
        restored=student.encoder(
            input_ids=ids,attention_mask=attn).last_hidden_state
    torch.testing.assert_close(restored,old,atol=1e-5,rtol=1e-5)


def test_attention_curriculum_phases_are_weight_preserving_and_restorable():
    torch.manual_seed(2028)
    model=_tiny()
    block=FullLocalRepeatCeNNSuperLayer(
        model.layers[3],model.layers[4],model.layers[5],
        window=5,max_len=22,correction_rank=8).eval()
    assert block.curriculum_phase==2
    assert block.active_attention_topology==(
        "full","window5","full","window5")
    with pytest.raises(ValueError):
        block.set_curriculum_phase(3)
    with pytest.raises(ValueError):
        block.set_curriculum_phase(True)
    with pytest.raises(ValueError):
        block.set_curriculum_phase(-1)
    x=torch.randn(1,17,32)
    mask=torch.ones(1,17,dtype=torch.bool)
    pos=torch.arange(17)[None,:]
    qkv_before=[block.stages[i].qkv.weight.detach().clone()
                for i in range(4)]
    outputs=[]
    for phase,expected in (
        (0,("full","full","full","full")),
        (1,("full","window5","full","full")),
        (2,("full","window5","full","window5"))):
        block.set_curriculum_phase(phase)
        assert block.active_attention_topology==expected
        with torch.no_grad():
            y=block(x,attention_mask=mask,position_ids=pos)[0]
        assert y.shape==x.shape and torch.isfinite(y).all()
        outputs.append(y)
    assert not torch.allclose(outputs[0],outputs[1],atol=1e-7,rtol=0)
    assert not torch.allclose(outputs[1],outputs[2],atol=1e-7,rtol=0)
    for i,weight in enumerate(qkv_before):
        torch.testing.assert_close(block.stages[i].qkv.weight,weight)
    block.set_curriculum_phase(1)
    saved=copy.deepcopy(block.state_dict())
    assert saved["_extra_state"]=={"curriculum_phase":1}
    fresh=FullLocalRepeatCeNNSuperLayer(
        model.layers[3],model.layers[4],model.layers[5],
        window=5,max_len=22,correction_rank=8).eval()
    fresh.load_state_dict(saved,strict=True)
    assert fresh.active_attention_topology==(
        "full","window5","full","full")
    fresh.set_curriculum_phase(2)
    assert fresh.state_dict()["_extra_state"]["curriculum_phase"]==2


def test_curriculum_stage0_matches_full_four_pass_and_final_is_restored():
    torch.manual_seed(27)
    teacher=_tiny()
    candidate=FullLocalRepeatCeNNSuperLayer(
        teacher.layers[0],teacher.layers[1],teacher.layers[2],
        window=7,max_len=24,correction_rank=0).eval()
    assert [layer.window for layer in candidate.stages]==[None,7,None,7]
    candidate.set_curriculum_phase(0)
    assert candidate.active_attention_topology==("full",)*4
    snapshot=copy.deepcopy(candidate.state_dict())
    candidate.set_curriculum_phase(2)
    assert candidate.active_attention_topology==(
        "full","window7","full","window7")
    candidate.load_state_dict(snapshot,strict=True)
    assert candidate.active_attention_topology==("full",)*4



@pytest.mark.parametrize("phase", [0,1,2])
def test_real_curriculum_checkpoint_mixed_state_cpu_clone_and_strict_reload(phase):
    # Reproduces the user's first phase-2 checkpoint crash at update 241.
    torch.manual_seed(2028)
    teacher=_tiny()
    block=FullLocalRepeatCeNNSuperLayer(
        teacher.layers[3],teacher.layers[4],teacher.layers[5],
        window=9,max_len=24,correction_rank=8).eval()
    block.set_curriculum_phase(phase)
    snapshot=cpu_state_snapshot(block)
    assert snapshot["_extra_state"]=={"curriculum_phase":phase}
    assert isinstance(snapshot["_extra_state"],dict)
    assert isinstance(snapshot["stages.0.qkv.weight"],torch.Tensor)
    assert snapshot["stages.0.qkv.weight"].device.type=="cpu"
    assert not snapshot["stages.0.qkv.weight"].requires_grad
    snapshot["_extra_state"]["curriculum_phase"]=(phase+1)%3
    assert block.curriculum_phase==phase
    snapshot["_extra_state"]["curriculum_phase"]=phase
    previous=snapshot["stages.0.qkv.weight"].clone()
    with torch.no_grad():
        block.stages[0].qkv.weight.add_(0.01)
    torch.testing.assert_close(snapshot["stages.0.qkv.weight"],previous,
                               rtol=0,atol=0)
    fresh=FullLocalRepeatCeNNSuperLayer(
        teacher.layers[3],teacher.layers[4],teacher.layers[5],
        window=9,max_len=24,correction_rank=8).eval()
    fresh.load_state_dict(snapshot,strict=True)
    assert fresh.curriculum_phase==phase
    torch.testing.assert_close(fresh.stages[0].qkv.weight,previous,
                               rtol=0,atol=0)
    # The generated checkpoint also supports torch.load(weights_only=True).
    buf=io.BytesIO()
    torch.save(snapshot,buf)
    buf.seek(0)
    disk=torch.load(buf,map_location="cpu",weights_only=True)
    assert disk["_extra_state"]=={"curriculum_phase":phase}
    restored=FullLocalRepeatCeNNSuperLayer(
        teacher.layers[3],teacher.layers[4],teacher.layers[5],
        window=9,max_len=24,correction_rank=8).eval()
    restored.load_state_dict(disk,strict=True)
    assert restored.active_attention_topology==fresh.active_attention_topology


def test_notebook_has_exact_total_600_budget_and_sealed_final_holdout():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf8"))
    codes=[(i,"".join(c["source"])) for i,c in enumerate(nb["cells"])
           if c["cell_type"]=="code"]
    assert nb["nbformat"]==4 and len(codes)>=12
    full="\n".join(s for _,s in codes)
    for token in (
        "TOTAL_OPTIMIZER_STEPS=600",
        "LOCAL_DISTILL_STEPS=520",
        "FULL_ATTENTION_STEPS=100",
        "FIRST_SLIDING_STEPS=140",
        "BOTH_SLIDING_STEPS=280",
        "FULL_ATTENTION_STEPS+FIRST_SLIDING_STEPS+BOTH_SLIDING_STEPS==LOCAL_DISTILL_STEPS",
        "block.set_curriculum_phase(phase)",
        "phase==2 and rank<best_rank",
        "assert block.curriculum_phase==2",
        "RECOVERY_MAX_AGREEMENT_DROP",
        "return cpu_state_snapshot(model)",
        "cpu_state_snapshot",
        "FULL_MODEL_RECOVERY_STEPS=80",
        "TOTAL_OPTIMIZER_STEPS==LOCAL_DISTILL_STEPS+FULL_MODEL_RECOVERY_STEPS==600",
        "FullLocalRepeatCeNNSuperLayer(",
        "def capture_group_example(",
        "def prepare_group_cache(",
        "def train_cached_group(",
        "def task_recover(",
        "def decision_quality(",
        "restore_group(student_eval,",
        "judge_replacement(",
        "HOLDOUT_ITEMS=make_items_from_rows(",
        "paired_cuda(", "EXPERIMENTAL_NOT_DEPLOYABLE",
    ):
        assert token in full,token
    assert "global_local_x2" in nb["cells"][0]["source"][0].lower() or "Global + Local" in "".join(nb["cells"][0]["source"])
    assert "PROBE_CACHE_BATCHES" not in full
    assert "AUDIT_CACHE_BATCHES" not in full
    assert "dev_items" not in full
    assert 'split="test"' not in full
    assert "upload_standalone_to_hub" not in full
    assert all(c.get("execution_count") is None and not c.get("outputs")
               for c in nb["cells"] if c["cell_type"]=="code")
    assert [i for i,s in codes if "HOLDOUT_ITEMS=make_items_from_rows(" in s]==[10]
    for i,s in codes:
        if i==1:continue # Colab %pip magic
        ast.parse(s,filename=f"<GlobalLocal×2 Colab cell {i}>")
