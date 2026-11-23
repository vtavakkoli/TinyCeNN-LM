"""CPU-only contract tests for V5.6 four-way research design."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tinycenn_lm.cenn_v56_fourway import (ARMS,select_validation_arm,
    validate_arm_rows,temporary_full_attention)


def rows(b_agreement=.985, b_speed=.96, c_agreement=.986, c_speed=.91):
    return [dict(arm='A_native', nll=3.0, agreement=1., speedup=1.),
            dict(arm='B_delta18',nll=3.005,agreement=b_agreement,speedup=b_speed),
            dict(arm='C_gated_hybrid',nll=3.008,agreement=c_agreement,speedup=c_speed),
            dict(arm='D_forced24',nll=3.007,agreement=1.,speedup=4.)]


def test_four_arms_fixed():
    assert len(ARMS)==4 and len(validate_arm_rows(rows()))==4


def test_never_select_forced_even_if_fast():
    assert select_validation_arm(rows())['selected']=='B_delta18'


def test_fallback_to_native():
    assert select_validation_arm(rows(b_agreement=.93))['selected']=='A_native'
    assert select_validation_arm(rows(b_speed=.75))['selected']=='A_native'


def test_select_quality_improving_hybrid():
    r=rows(c_agreement=.992,c_speed=.955)
    assert select_validation_arm(r)['selected']=='C_gated_hybrid'


def test_no_hybrid_if_too_slow():
    assert select_validation_arm(rows(c_agreement=.995,c_speed=.80))['selected']=='B_delta18'


def test_nonfinite_and_missing_rejected():
    r=rows();r[1]['nll']=float('nan')
    with pytest.raises(ValueError):validate_arm_rows(r)
    with pytest.raises(ValueError):validate_arm_rows(rows()[:-1])


def test_temporary_full_restores_on_error():
    n0=object();n1=object();d0=object();d1=object()
    layers=[SimpleNamespace(self_attn=n0),SimpleNamespace(self_attn=n1)]
    a0=SimpleNamespace(original=n0);a1=SimpleNamespace(original=n1)
    with pytest.raises(RuntimeError):
        with temporary_full_attention(layers,{0:n0,1:n1},{0:a0,1:a1}):
            assert layers[0].self_attn is a0
            raise RuntimeError('test')
    assert layers[0].self_attn is n0 and layers[1].self_attn is n1


def test_invalid_adapter_also_restores():
    n0=object();n1=object()
    layers=[SimpleNamespace(self_attn=n0),SimpleNamespace(self_attn=n1)]
    with pytest.raises(ValueError):
        with temporary_full_attention(layers,{0:n0,1:n1},
                                      {0:SimpleNamespace(original=n0),1:SimpleNamespace(original=object())}):
            pass
    assert layers[0].self_attn is n0
