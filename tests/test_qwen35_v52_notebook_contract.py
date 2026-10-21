"""Static smoke tests for the no-upload V5.2 Colab contract."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/Qwen35_08B_CeNN_HeadRescue_V52_FidelityRecovery_Colab.ipynb'


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert nb['nbformat'] == 4
    return nb['cells']


def _code():
    return '\n'.join(''.join(c['source']) if isinstance(c['source'], list) else c['source']
                     for c in _cells() if c['cell_type'] == 'code')


def test_notebook_python_syntax():
    for i, cell in enumerate(_cells()):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if source.lstrip().startswith(('%', '!')):
                continue
            compile(source, f'cell{i}', 'exec')


def test_no_upload_or_implicit_checkpoint_download():
    s = _code()
    assert 'files.upload(' not in s
    assert 'drive.mount(' not in s
    assert 'V5_CORE_STATE=None' in s  # automatic baseline is supported
    assert 'BOOTSTRAP_PATH' in s


def test_distillation_checkpoint_can_be_restored():
    s = _code()
    assert "'v52_joint_w128_plus_kl'" in s
    assert 'base_variant(name)' in s
    assert "f'v52_L{TARGET_LAYER}_{SELECTED}.pt'" in s


def test_independent_validation_and_late_audit():
    s = _code()
    assert "capture('train',TRAIN)" in s
    assert "capture('dev',DEV)" in s
    assert 'AUDIT_START_TOKEN=1_000_000' in s
    assert "AUDIT=build_windows('train'" in s
    assert 'MIN_COSINE=.95' in s
    assert 'MIN_AGREEMENT=.98' in s


def test_code_resides_in_repository():
    s = _code()
    assert 'from tinycenn_lm.cenn_head_rescue_v52 import' in s
    assert "str(V52_TESTS)" in s
    assert 'warmstart_v51_to_v52' in s