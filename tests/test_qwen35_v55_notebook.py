"""V5.5 notebook contract and fast-eval regression tests; no Qwen downloads."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/'notebooks/Qwen35_08B_DeltaCeNN_V55_ChunkCell_Optimized_FastEval_Colab.ipynb'

def content():
    n=nbformat.read(NB,as_version=4)
    nbformat.validate(n)
    for i,c in enumerate(n.cells):
        if c.cell_type=='code' and not c.source.lstrip().startswith('%'):
            compile(c.source,f'cell_{i}','exec')
    return '\n'.join(c.source for c in n.cells if c.cell_type=='code')

def test_upload_free_and_export():
    s=content()
    assert 'files.upload(' not in s
    assert 'drive.mount(' not in s
    assert 'files.download(ARCHIVE)' in s
    assert 'training_steps.jsonl' in s and 'final_audit.json' in s

def test_fast_eval_batchencoding_fixed():
    s=content()
    assert "encoded['input_ids'] if isinstance(encoded,dict)" in s
    assert 'ids.ndim==1' in s
    assert 'fast_eval.json' in s

def test_chunk_engine_and_full_quality_gate():
    s=content()
    assert 'Qwen35DeltaCeNNChunkLinear' in s
    assert 'BACKEND_CHUNK_SIZE=64' in s
    assert "SKIP_NEAR_EXACT_FIT=True" in s
    assert "family=='full_attention'" in s
    assert 'if install:' in s
    assert "quality_pass=gate.quality_pass" in s
    assert "HF_UPLOAD_MODE='quality_only'" in s

def test_v55_io_roundtrip_code_contract():
    s=(ROOT/'src/tinycenn_lm/delta_cenn_v55_io.py').read_text()
    assert 'Qwen35DeltaCeNNChunkLinear' in s
    assert 'tinycenn-qwen35-v55-adapter-v1' in s
    assert 'chunk_cell_v55' in s
