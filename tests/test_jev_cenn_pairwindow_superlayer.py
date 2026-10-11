"""Real architecture + mathematical contracts for Jev-CeNN 15/16/17->1."""
import ast
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from tinycenn_lm.jev_cenn_pairwindow_superlayer import (
    JevCeNNPairWindowSuperLayer,BypassModernBertLayer,
    install_superlayer,restore_original_layers,
    pair_reachable,pair_coverage_fraction,sparse_attention
)

ROOT=Path(__file__).resolve().parents[1]
COLAB=ROOT/"notebooks"/"Jev_CeNN_2Window_3Layers_To_1_Colab.ipynb"


def test_math_full_512_reachability_and_incomplete_ordinary_windows():
    for n in (17,23,45,77,128,256,512):
        assert pair_reachable(n,22,23,22)
        assert pair_coverage_fraction(n,22,23,22)==1.
    # Two local windows R=22 each only reach a range +/-44.
    assert not pair_reachable(512,22,1,22)
    with pytest.raises(ValueError):
        pair_reachable(0,22,23,22)
    with pytest.raises(ValueError):
        pair_reachable(512,22,0,22)


def test_sparse_softmax_matches_dense_masked_reference_and_gradients():
    torch.manual_seed(2028)
    b,h,t,d=2,2,17,8
    q=torch.randn(b,h,t,d,requires_grad=True)
    k=torch.randn(b,h,t,d,requires_grad=True)
    v=torch.randn(b,h,t,d,requires_grad=True)
    valid=torch.ones((b,t),dtype=torch.bool)
    valid[1,-3:]=False
    for radius,stride in ((2,1),(3,4)):
        out=sparse_attention(q,k,v,valid,radius=radius,stride=stride)
        scores=torch.einsum("bhtd,bhsd->bhts",q,k)*d**-0.5
        positions=torch.arange(t)
        legal=(positions[None,:]-positions[:,None])
        allowed=((legal%stride)==0)&(legal.abs()<=radius*stride)
        allowed=allowed[None,:,:]&valid[:,None,:]
        probs=torch.softmax(scores.float().masked_fill(~allowed[:,None],-1e4),-1)
        ref=torch.einsum("bhts,bhsd->bhtd",probs,v)*valid[:,None,:,None]
        torch.testing.assert_close(out,ref,atol=2e-6,rtol=2e-6)
        out.square().mean().backward(retain_graph=True)
        for x in (q,k,v):
            assert x.grad is not None and torch.isfinite(x.grad).all()


def _toy_model():
    from transformers import AutoModel,ModernBertConfig,__version__
    if __version__!="4.57.6":
        pytest.skip("Pinned exact teacher ModernBERT 4.57.6")
    config=ModernBertConfig(
        vocab_size=64,hidden_size=32,intermediate_size=48,
        num_hidden_layers=18,num_attention_heads=2,
        global_attn_every_n_layers=3,local_attention=8,
        max_position_embeddings=128,reference_compile=False,
        pad_token_id=0,attention_dropout=0.0,
        embedding_dropout=0.0,mlp_dropout=0.0)
    return AutoModel.from_config(config,attn_implementation="sdpa").eval()


def test_whole_modernbert_three_to_one_bypasses_complete_ffn_blocks():
    torch.manual_seed(2028)
    native=_toy_model()
    # Normal student whose encoder.layers contains complete transformer layers.
    class Student(nn.Module):
        def __init__(self,encoder):
            super().__init__()
            self.encoder=encoder
    student=Student(copy.deepcopy(native))
    a,b=student.encoder.layers[15],student.encoder.layers[16]
    superlayer=JevCeNNPairWindowSuperLayer(
        a,b,local_radius=3,stride=4,dilated_radius=5,max_len=24
    )
    ids=torch.randint(1,64,(2,17))
    masks=torch.ones(2,17,dtype=torch.long)
    masks[1,-2:]=0
    originals=install_superlayer(student,superlayer)
    assert isinstance(student.encoder.layers[16],BypassModernBertLayer)
    assert isinstance(student.encoder.layers[17],BypassModernBertLayer)
    assert len(list(student.encoder.layers[16].parameters()))==0
    assert len(list(student.encoder.layers[17].parameters()))==0
    with torch.no_grad():
        baseline=native(input_ids=ids,attention_mask=masks).last_hidden_state
        output=student.encoder(input_ids=ids,attention_mask=masks).last_hidden_state
    assert output.shape==baseline.shape
    assert torch.isfinite(output).all()
    assert (output-baseline).abs().max().item()>0
    # CeNN must affect outputs (it feeds second QKV, not merely a zero gate).
    original_output=output
    superlayer.set_cenn_enabled(False)
    with torch.no_grad():
        without=student.encoder(input_ids=ids,attention_mask=masks).last_hidden_state
    assert (without-original_output).abs().max().item()>1e-7
    restore_original_layers(student,originals)
    with torch.no_grad():
        restored=student.encoder(input_ids=ids,attention_mask=masks).last_hidden_state
    torch.testing.assert_close(restored,baseline,rtol=1e-5,atol=1e-5)


def test_cellular_gradient_path_is_present_in_both_attention_stages():
    torch.manual_seed(9)
    native=_toy_model()
    model=JevCeNNPairWindowSuperLayer(
        native.layers[15],native.layers[16],
        local_radius=3,stride=4,dilated_radius=5,max_len=24)
    model.train()
    states=torch.randn(1,19,32)
    pred=model(states,attention_mask=torch.ones(1,19,dtype=torch.bool),
               position_ids=torch.arange(19)[None,:])[0]
    pred.square().mean().backward()
    grads=[p.grad for p in model.cenn.parameters()]
    assert any(x is not None and float(x.abs().sum())>0 for x in grads)
    assert model.last_mid is not None and model.last_mid.requires_grad


def test_notebook_is_experiment_and_never_publishes():
    doc=json.loads(COLAB.read_text(encoding="utf8"))
    assert doc["nbformat"]==4
    assert len(doc["cells"])>=13
    code="\n".join("".join(x["source"]) for x in doc["cells"] if x["cell_type"]=="code")
    for fragment in [
        "capture_three(", "CENN_STEPS", "SUPERLAYER_STEPS",
        "TRAIN_CACHE", "PROBE_CACHE", "AUDIT_CACHE",
        "DEV_QUALITY", "LAST_LAYER=17", "FIRST_LAYER=15",
        "pair_reachable(", "student_eval", "quality_with",
        "audit_with", "audit_without", "speed_accepted",
        "official_test_used", "install_superlayer(", "cpu_snapshot(",
    ]:
        assert fragment in code,fragment
    assert 'split="test"' not in code
    assert "upload_standalone_to_hub" not in code
    # Historical experiment outputs are valuable research evidence:
    # allow them when they belong to executed cells. The NEW full-encoder
    # Colab remains a clean, never-executed template.
    assert all(not c.get("outputs") or c.get("execution_count") is not None
               for c in doc["cells"] if c["cell_type"]=="code")
    for i,c in enumerate(doc["cells"]):
        if c["cell_type"]!="code" or i==1:continue
        ast.parse("".join(c["source"]),filename=f"<Jev_CeNN cell{i}>")
