"""Notebook regression checks requiring neither HuggingFace downloads nor CUDA."""
from pathlib import Path
import ast
import nbformat

NOTEBOOK=Path(__file__).parents[1]/'notebooks/Qwen35_08B_CeNN_V53_All6_FullAttention_Sequential_Colab.ipynb'


def notebook():
    nb=nbformat.read(NOTEBOOK,as_version=4)
    nbformat.validate(nb)
    return nb


def executable_cells():
    return [cell.source for cell in notebook().cells if cell.cell_type=='code']


def test_structural_and_python_syntax():
    cells=executable_cells()
    assert len(cells)>=8
    for cell in cells:
        if cell.lstrip().startswith('%pip'): continue
        ast.parse(cell)


def test_all_six_and_full_head_option():
    code='\n'.join(executable_cells())
    assert 'FULL_ATTENTION_LAYERS' in code
    assert "REPLACEMENT_MODE='one_group'" in code
    assert "'both_groups'" in code
    assert 'MANAGER.install(layer,adapter)' in code
    assert 'with MANAGER.candidate(layer,adapter):' in code
    assert 'capture_target(layer' in code
    assert 'native_probe' in code and 'exact_probe' in code


def test_training_step_logs_and_automatic_download():
    code='\n'.join(executable_cells())
    assert 'training_steps.jsonl' in code
    assert 'training.log' in code
    assert 'append_jsonl(TRAIN_EVENTS,event)' in code
    assert 'cumulative_layer_results.csv' in code
    assert "OUT/'final_audit.json'" in code
    assert 'files.download(ARCHIVE)' in code
    assert 'finally:' in code
    assert 'manifest' in code.lower()


def test_no_upload_and_no_cached_generation():
    code='\n'.join(executable_cells())
    assert 'files.upload(' not in code
    assert 'drive.mount(' not in code
    assert 'use_cache=False' in code
    assert 'RUN_FINGERPRINT' in code
    assert 'REPLACE_KV_GROUPS' not in code


def test_no_claim_to_replace_linear_layers():
    prose='\n'.join(cell.source for cell in notebook().cells if cell.cell_type=='markdown')
    assert '18 gated-delta' in prose or '18 gated' in prose
    assert 'six full-attention' in prose
