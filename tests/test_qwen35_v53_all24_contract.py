"""Pure notebook/source contract checks (no Qwen download / CUDA required)."""
import sys
from pathlib import Path
import nbformat
import pytest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from tinycenn_lm.cenn_v53_multilayer import FULL_ATTENTION_LAYERS
from tinycenn_lm.delta_cenn_qwen35 import Qwen35DeltaCeNNLinear,DeltaCeNNRule
NB=root/'notebooks/Qwen35_08B_DeltaCeNN_V53_All24_Progressive_Colab.ipynb'


def notebook():
    n=nbformat.read(NB,as_version=4)
    nbformat.validate(n)
    return n


def code():
    return '\n'.join(c.source for c in notebook().cells if c.cell_type=='code')


def test_full_layer_contract():
    assert FULL_ATTENTION_LAYERS==(3,7,11,15,19,23)
    assert len(set(range(24))-set(FULL_ATTENTION_LAYERS))==18


def test_notebook_syntax():
    for i,c in enumerate(notebook().cells):
        if c.cell_type=='code' and not c.source.lstrip().startswith('%'):
            compile(c.source,f'cell_{i}','exec')


def test_native_baseline_restores_both_families():
    s=code()
    assert 'def native_everywhere()' in s
    assert 'MANAGER.native_context()' in s
    assert 'LAYERS[i].linear_attn=original' in s
    assert 'with native_everywhere() if native else nullcontext()' in s


def test_download_on_exception_and_complete():
    s=code()
    assert 'finally:' in s and 'files.download(ARCHIVE)' in s
    for file in ('training.log','training_steps.jsonl','cumulative_layer_results.csv',
                 'final_audit.json','artifact_manifest.json','run_status.json'):
        assert file in s


def test_all_24_progressive_and_forced_diagnostic():
    s=code()
    for item in ('FORCE_ALL_24=True','TARGET_LAYERS=tuple(range(cfg.num_hidden_layers))',
                 'family==\'linear_attention\'','family==\'full_attention\'',
                 'fit_local(layer','RUN_FINGERPRINT','forced=bool(install and not gate.quality_pass)'):
        assert item in s


def test_no_external_checkpoint_upload():
    s=code()
    assert 'files.upload(' not in s and 'drive.mount(' not in s


def test_native_delta_recurrence_is_substituted():
    import inspect
    s=inspect.getsource(Qwen35DeltaCeNNLinear.forward)
    assert 'self.core(q,k,v,g,beta)' in s
    assert 'original(hidden_states' not in s
    assert 'if self.native_backend_oracle:' in s
    assert 'native_backend_oracle=True' in code()
    assert 'parity_diagnostics.json' in code()


def test_quality_requires_real_speed():
    s=code()
    assert "FINAL_TIMING['speedup']>MIN_SPEEDUP" in s
    assert "STUDENT_AUDIT['teacher_agreement']>=MIN_AGREEMENT" in s
