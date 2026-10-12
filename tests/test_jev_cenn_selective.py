"""Scientific controls and notebook syntax for selective W512 Jev-CeNN."""
import ast
import json
from pathlib import Path

import pytest

from tinycenn_lm.jev_cenn_selective import (
    verify_case_partition,judge_replacement,plan_active_layers,
)

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks"/"Jev_CeNN_W512_Selective_3to1_Heldout_Colab.ipynb"


def _row(id, text):
    return {"id":id,"state":text}


def _quality(agreement=1.0,gold=0.88,decisions=240):
    return {
        "decisions":decisions,
        "agreement":agreement,
        "student_gold_accuracy":gold,
        "teacher_gold_accuracy":0.88,
        "mean_js":0.005,
    }


def test_true_train_probe_gate_holdout_case_isolation():
    training=[_row("a","state A"),_row("b","state B")]
    dev=[_row("c","state C")]
    held=[_row("d","state D")]
    report=verify_case_partition(training,dev,held)
    assert report["all_partitions_disjoint"]
    assert report["holdout_cases"]==1
    with pytest.raises(ValueError,match="overlap"):
        verify_case_partition(training,dev,[_row("c","state E")])
    with pytest.raises(ValueError,match="overlap"):
        verify_case_partition(training,dev,[_row("d","state A")])
    with pytest.raises(ValueError,match="duplicate"):
        verify_case_partition(training+training,dev,held)
    with pytest.raises(ValueError,match="nonempty"):
        verify_case_partition(training,dev,[])


def test_candidate_cannot_be_accepted_on_cosine_alone():
    baseline=_quality()
    bad=_quality(agreement=.95,gold=.84)
    answer=judge_replacement(
        baseline,bad,local_cosine=.999,local_nmse=.001)
    assert not answer["accepted"]
    assert "teacher_agreement_below_gate" in answer["reasons"]
    assert "teacher_gold_accuracy_drop" in answer["reasons"]


def test_true_prefix_fidelity_and_nmse_are_gates():
    candidate=_quality(agreement=.997,gold=.88)
    for cosine,nmse in ((.98,.01),(.996,.03)):
        v=judge_replacement(
            _quality(),candidate,local_cosine=cosine,local_nmse=nmse)
        assert not v["accepted"]
    v=judge_replacement(
        _quality(),candidate,local_cosine=.995,local_nmse=.010)
    assert v["accepted"]
    assert v["reasons"]==[]


def test_incremental_agreement_and_gold_constraints():
    baseline=_quality(agreement=.999,gold=.91)
    candidate=_quality(agreement=.99,gold=.90)
    accepted=judge_replacement(
        baseline,candidate,local_cosine=.993,local_nmse=.015)
    assert accepted["accepted"]
    # A 0.99 global agreement floor alone cannot protect from
    # repeated >0.5 percentage point incremental drops.
    bad=judge_replacement(
        baseline,candidate,local_cosine=.993,local_nmse=.015,
        max_incremental_agreement_drop=.005)
    assert "incremental_teacher_agreement_drop" in bad["reasons"]
    with pytest.raises(ValueError,match="same"):
        judge_replacement(
            baseline,_quality(decisions=25),local_cosine=.999,local_nmse=.01)


def test_optional_speed_gate_must_have_actual_measurement():
    base=_quality()
    good=_quality()
    kw=dict(local_cosine=.995,local_nmse=.01,require_speedup=True)
    with pytest.raises(ValueError,match="speedup"):
        judge_replacement(base,good,**kw)
    slow=judge_replacement(base,good,speedup=.8,**kw)
    assert not slow["accepted"]
    assert "candidate_not_faster" in slow["reasons"]
    fast=judge_replacement(base,good,speedup=1.1,**kw)
    assert fast["accepted"]


def test_active_layer_accounting_with_partial_selection():
    summary=plan_active_layers(28,[(3,4,5),(24,25,26)])
    assert summary["active_encoder_blocks"]==24
    assert summary["bypassed_blocks"]==4
    assert summary["native_layers"][0]==0
    assert summary["native_layers"][-1]==27
    assert plan_active_layers(28,[])["active_encoder_blocks"]==28
    assert plan_active_layers(28,[(i,i+1,i+2) for i in range(0,27,3)])[ 
        "active_encoder_blocks"]==10
    with pytest.raises(ValueError,match="overlapping"):
        plan_active_layers(28,[(0,1,2),(1,2,3)])


def test_notebook_syntax_and_independent_holdout_seal():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"]==4
    assert len(nb["cells"])>=13
    codes=[(i,"".join(c["source"])) for i,c in enumerate(nb["cells"])
           if c["cell_type"]=="code"]
    full="\n".join(text for _,text in codes)
    for required in (
        "LOCAL_WINDOW=512","TRAIN_CASES=700",
        "HOLDOUT_CASES=200","PROBE_CASES=80",
        "verify_case_partition(","capture_triplet(",
        "fresh_cycle(","teacher.encoder.layers[start]",
        "student_eval.encoder.layers[start]",
        "local_probe(","distill_stream(",
        "recover_in_changed_model(","encoder_fidelity_on_holdout(",
        "judge_replacement(","restore_group(","paired_models_cuda(",
        "FINAL_HOLDOUT","one_time_independent_holdout_quality",
        "EXPERIMENTAL_NOT_DEPLOYABLE",
    ):
        assert required in full,required
    # Materialize holdout examples exactly once, after candidate selection.
    sealing=[i for i,s in codes if "HOLDOUT_ITEMS=make_items_from_rows" in s]
    assert sealing==[10]
    assert "HOLDOUT_RECORDS" not in "\n".join(
        s for i,s in codes if i<10
    )
    # The selective notebook has its own case partitions. Older Laya
    # notebook trailing assertions once referenced undefined dev_items and
    # PROBE_CACHE_BATCHES, aborting immediately after successful dataset load.
    data_cell="".join(nb["cells"][3]["source"])
    assert "len(train_items)" in data_cell
    assert "len(probe_items)" in data_cell
    assert "len(gate_items)" in data_cell
    for obsolete in (
        "dev_items", "PROBE_CACHE_BATCHES", "AUDIT_CACHE_BATCHES",
        "cache_teacher_io(", "attn_io(", "local_metrics_from_cache(",
    ):
        assert obsolete not in full, f"Unexpected legacy reference: {obsolete}"
    assert 'split="test"' not in full
    assert "upload_standalone_to_hub" not in full
    for i,s in codes:
        c=nb["cells"][i]
        assert c.get("execution_count") is None and not c.get("outputs")
        if i==1:continue  # Colab %pip magic commands
        ast.parse(s,filename=f"<selective W512 Colab cell {i}>")
