"""Static protocol checks for the Qwen3.5 V5.3 Colab."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/Qwen35_08B_CeNN_HeadRescue_V53_CrossCalibrated_Colab.ipynb'


def cells():
    data=json.loads(NOTEBOOK.read_text())
    assert data['nbformat']==4
    assert len(data['cells'])>=25
    return data['cells']


def code():
    return '\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source']
                     for c in cells() if c['cell_type']=='code')


def test_all_code_cells_compile():
    for i,c in enumerate(cells()):
        if c['cell_type']=='code':
            src=''.join(c['source']) if isinstance(c['source'],list) else c['source']
            if not src.lstrip().startswith(('%','!')):compile(src,f'cell{i}','exec')


def test_no_upload_requirements():
    src=code()
    assert 'files.upload(' not in src and 'drive.mount(' not in src
    assert 'BOOTSTRAP_PATH' in src and 'REFERENCE_PATH' in src
    assert "V52_CORE_STATE=None" in src


def test_reusable_model_import_and_tests():
    src=code()
    assert 'from tinycenn_lm.cenn_head_rescue_v53 import' in src
    assert 'warmstart_v52_to_v53' in src
    assert 'str(V53_TESTS)' in src


def test_validation_before_audit():
    src=code()
    assert "capture('train',TRAIN)" in src
    assert "capture('dev',DEV)" in src
    assert 'AUDIT_START_TOKEN=1_250_000' in src
    assert "AUDIT=build_windows('train'" in src
    assert 'MIN_AGREEMENT=.98' in src


def test_kl_candidate_restoration():
    src=code()
    assert "'v53_joint_w128_plus_kl'" in src
    assert "return 'v53_joint_w128' if name=='v53_joint_w128_plus_kl'" in src
    assert 'student.core.load_state_dict(payload[\'core_state\'],strict=True)' in src
    assert 'if not FINAL_PASS:' in src


def test_output_artifacts():
    src=code()
    assert "final_audit.json" in src
    assert "validation_ranked.csv" in src
    assert "shutil.make_archive(" in src
