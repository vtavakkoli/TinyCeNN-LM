"""CPU-only correctness of 6-layer grafts, gate decisions and log artifacts."""
import json
from pathlib import Path
import pytest
from tinycenn_lm.cenn_v53_multilayer import (
    FULL_ATTENTION_LAYERS, FullAttentionGraftManager,
    append_jsonl, check_quality, experiment_fingerprint,
    validate_full_attention_targets,
)


class Layer:
    def __init__(self): self.self_attn = object()


class Adapter:
    def __init__(self, original): self.original = original


def fake_layers():
    return [Layer() for _ in range(24)]


def test_six_targets():
    kinds = ['linear_attention'] * 24
    for i in FULL_ATTENTION_LAYERS: kinds[i] = 'full_attention'
    assert validate_full_attention_targets(kinds) == FULL_ATTENTION_LAYERS
    assert validate_full_attention_targets(kinds, (3, 11)) == (3, 11)
    with pytest.raises(ValueError): validate_full_attention_targets(kinds, (3, 4))
    with pytest.raises(ValueError): validate_full_attention_targets(kinds, (7, 3))
    with pytest.raises(ValueError): validate_full_attention_targets(kinds, (3, 3))
    with pytest.raises(ValueError): validate_full_attention_targets(kinds, ())


def test_install_and_native_restoration():
    layers = fake_layers()
    mgr = FullAttentionGraftManager(layers)
    orig = [x.self_attn for x in layers]
    a3, a7 = Adapter(orig[3]), Adapter(orig[7])
    mgr.install(3, a3)
    mgr.install(7, a7)
    assert mgr.installed_layers == (3, 7)
    with mgr.native_context():
        assert all(x.self_attn is orig[i] for i, x in enumerate(layers))
    assert layers[3].self_attn is a3 and layers[7].self_attn is a7
    mgr.restore_all()
    assert mgr.installed_layers == ()
    assert all(x.self_attn is orig[i] for i, x in enumerate(layers))


def test_native_restoration_on_exception():
    layers = fake_layers()
    mgr = FullAttentionGraftManager(layers)
    a3 = Adapter(layers[3].self_attn)
    mgr.install(3, a3)
    with pytest.raises(RuntimeError), mgr.native_context():
        raise RuntimeError('test failure')
    assert layers[3].self_attn is a3


def test_candidate_restores_even_on_exception():
    layers=fake_layers();mgr=FullAttentionGraftManager(layers)
    orig=layers[11].self_attn;adapter=Adapter(orig)
    with pytest.raises(RuntimeError),mgr.candidate(11,adapter):
        assert layers[11].self_attn is adapter
        raise RuntimeError('failure')
    assert layers[11].self_attn is orig
    with pytest.raises(ValueError): mgr.candidate(11,Adapter(object())).__enter__()


def test_reject_invalid_grafts():
    layers=fake_layers();mgr=FullAttentionGraftManager(layers)
    with pytest.raises(ValueError):mgr.install(4,Adapter(layers[4].self_attn))
    with pytest.raises(ValueError):mgr.install(3,Adapter(object()))


def test_gates_separate_quality_and_speed():
    assert check_quality(.96,.99,.001,1.06).deploy_pass
    assert check_quality(.96,.99,.001,.90).quality_pass
    assert not check_quality(.96,.99,.001,.90).deploy_pass
    assert not check_quality(.94,.99,.001,2).quality_pass
    assert not check_quality(.96,.97,.001,2).quality_pass
    assert not check_quality(.96,.99,.12,2).quality_pass
    with pytest.raises(ValueError):check_quality(float('nan'),.98,.01,1.1)


def test_fingerprint_and_training_log(tmp_path):
    a=experiment_fingerprint({'seed':42,'layers':[3,7]})
    b=experiment_fingerprint({'layers':[3,7],'seed':42})
    assert a==b and len(a)==64
    assert a!=experiment_fingerprint({'seed':43,'layers':[3,7]})
    log=tmp_path/'nested'/'train_steps.jsonl'
    append_jsonl(log,{'layer':3,'step':1,'loss':.2})
    append_jsonl(log,{'layer':3,'step':2,'loss':.1})
    rows=[json.loads(s) for s in log.read_text().splitlines()]
    assert len(rows)==2 and rows[1]['step']==2
