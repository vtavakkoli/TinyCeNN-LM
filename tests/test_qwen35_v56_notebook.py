"""Contract/syntax tests of the V5.6 Colab without Qwen downloads."""
import sys
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/'notebooks/Qwen35_08B_DeltaCeNN_V56_FourWay_QualityGated_Colab.ipynb'


def read_code():
    n=nbformat.read(NB,as_version=4)
    nbformat.validate(n)
    for i,c in enumerate(n.cells):
        if c.cell_type=='code' and not c.source.strip().startswith('%'):
            compile(c.source, f'cell_{i}', 'exec')
    return '\n'.join(c.source for c in n.cells if c.cell_type=='code')


def test_valid_notebook():
    assert len(nbformat.read(NB,as_version=4).cells) >= 20
    assert 'select_validation_arm' in read_code()


def test_four_arm_evaluation_and_rollback():
    s=read_code()
    for phrase in ('A_native','B_delta18','C_gated_hybrid','D_forced24',
                   'temporary_full_attention', "MANAGER.restore_all()",
                   "SELECTED_ARM!='D_forced24'", "set(FULL_CANDIDATES)==set(FULL_TARGETS)"):
        assert phrase in s


def test_real_model_logs_and_no_upload():
    s=read_code()
    for label in ('training_steps.jsonl','training.log','four_way_validation.csv',
                  'four_way_diagnostic_audit.csv','validation_selection.json',
                  'final_audit.json', 'files.download(ARCHIVE)',
                  'FAST_EVAL_RESULT', 'slow_greedy'):
        assert label in s
    assert 'drive.mount(' not in s and 'files.upload(' not in s


def test_four_way_only_selected_from_validation():
    s=read_code()
    assert 'SELECTION=select_validation_arm(ARM_VALIDATION' in s
    assert s.index('SELECTION=select_validation_arm(ARM_VALIDATION') < s.index("AUDIT=token_windows('train',AUDIT_BLOCKS")
    assert "HF_UPLOAD_MODE='disabled'" in s


def test_native_controls_and_distinct_full_delta_cores():
    s=read_code()
    assert 'with native_everywhere()' in s
    assert 'Qwen35DeltaCeNNChunkLinear' in s
    assert 'CeNNHeadRescueV53' in s
    assert 'MANAGER.candidate(layer,adapter)' in s
