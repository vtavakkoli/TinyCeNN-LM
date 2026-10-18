"""V5.4 manifest integrity and notebook behavior checks; CPU-only."""
from pathlib import Path
import json
import sys
import tempfile
import pytest
import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tinycenn_lm.delta_cenn_v54_io import (
    write_adapter_package, prepare_hf_model_card, load_delta_cenn_bundle)
NOTEBOOK = ROOT/'notebooks/Qwen35_08B_DeltaCeNN_V54_All24_QualityGated_FastEval_Colab.ipynb'


def test_export_selected_and_failed_distinct():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check = tmp/'run'/'checkpoints'
        check.mkdir(parents=True)
        for i, family in [(0,'linear_attention'), (3,'full_attention')]:
            torch.save({'layer':i,'core_state':{'x':torch.ones(2)}},
                check/f'L{i:02d}_{family}_core.pt')
        manifest = write_adapter_package(tmp/'hf',tmp/'run',model_id='Qwen/Qwen3.5-0.8B',
            model_revision='abc',git_sha='def',selected_layers=[3],full_layers=[3],
            replacement_mode='one_group',source_sha256='cde',run_fingerprint='ghi',
            final_audit={'final_pass':False},fast_eval={},experimental=True)
        assert [x['layer'] for x in manifest['candidate_checkpoints'] if x['selected']] == [3]
        assert (tmp/'hf'/'selected'/'L03_full_attention_core.pt').exists()
        assert (tmp/'hf'/'candidates'/'L00_linear_attention_core.pt').exists()
        assert manifest['research_only']
        assert 'not directly loadable' in prepare_hf_model_card(manifest,'vtava/test')
        with pytest.raises(ValueError):
            load_delta_cenn_bundle(tmp/'hf',allow_research=False)


def test_manifest_rejects_missing_installed_weights():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        with pytest.raises(FileNotFoundError):
            write_adapter_package(tmp/'hf',tmp/'run',model_id='x',model_revision='abc',
                git_sha='def',selected_layers=[7],full_layers=[7],replacement_mode='both_groups',
                source_sha256='x',run_fingerprint='x',final_audit=None,fast_eval=None,
                experimental=True)


def test_v54_notebook_contract():
    n=nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(n)
    code='\n'.join(c.source for c in n.cells if c.cell_type=='code')
    for j,c in enumerate(n.cells):
        if c.cell_type=='code' and not c.source.lstrip().startswith('%'):
            compile(c.source,f'cell_{j}','exec')
    for required in ('DELTA_STEPS=80','FORCE_ALL_24=False','best_step=0',
        'RUN_FAST_EVAL=True','slow_greedy','fast_eval.json',
        'HF_UPLOAD_MODE=', 'hf_upload_status.json', 'files.download(ARCHIVE)',
        'write_adapter_package','load_delta_cenn_bundle'):
        assert required in code or required in '\n'.join(c.source for c in n.cells)
    assert 'files.upload(' not in code
