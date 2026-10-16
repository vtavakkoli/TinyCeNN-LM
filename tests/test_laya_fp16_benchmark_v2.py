import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_laya_fp16_v2 as v2
from test_laya_fast_inference import tiny_model, Tokenizer, runtime


def test_attribution_uses_compiled_control_and_excludes_failures():
    rows = {name: {'passed': True, 'median_ms': time} for name, time in
            [('original_eager', 30), ('optimized_eager', 29),
             ('original_compiled', 10), ('optimized_compiled', 8)]}
    result = v2.attribution(rows)
    assert result['ratios']['optimization_beyond_compile'] == 1.25
    assert result['ratios']['combined_vs_original_eager'] == 3.75
    rows['optimized_compiled']['passed'] = False
    result = v2.attribution(rows)
    assert result['winner'] == 'original_compiled'
    assert 'optimization_beyond_compile' not in result['ratios']


def test_answer_parity_rejects_changed_decision_and_nan():
    ref = {'q': {'choice': 'a', 'probabilities': {'a': .51, 'b': .49}}}
    changed = {'q': {'choice': 'b', 'probabilities': {'a': .49, 'b': .51}}}
    assert not v2.answer_parity(ref, changed, .1)['passed']
    changed['q'] = {'choice': 'a', 'probabilities': {'a': float('nan'), 'b': .49}}
    assert not v2.answer_parity(ref, changed, .1)['passed']


def test_interleaved_timing_counts_and_excludes_errors():
    calls = {'a': 0, 'b': 0}
    def count(name):
        calls[name] += 1
    def fail():
        raise RuntimeError('intentional')
    results, order = v2.interleaved_times({'a': lambda: count('a'), 'b': lambda: count('b'), 'bad': fail},
                                        device=torch.device('cpu'), rounds=5, repeats=2, seed=29)
    assert calls == {'a': 10, 'b': 10}
    assert not results['bad']['passed']
    assert len(results['a']['round_ms']) == 5
    assert all('bad' not in row for row in order[1:])
    assert len({tuple(row) for row in order[1:]}) > 1


def test_jsonl_validation(tmp_path):
    p = tmp_path / 'cases.jsonl'
    p.write_text('')
    with pytest.raises(ValueError, match='at least one'):
        v2.read_cases(p)
    p.write_text(json.dumps({'questions': {}}))
    with pytest.raises(ValueError, match='state'):
        v2.read_cases(p)


@torch.inference_mode()
def test_fullgraph_frontend_and_compiled_api_route():
    # backend=eager tests Dynamo fullgraph capture/routing, NOT Inductor speed.
    torch.manual_seed(3)
    original = tiny_model(torch.float16)
    model_set = v2.candidates(original, backend='eager')
    q = {'q': {'criteria': ['left', 'right']}}
    prepared = v2.PreparedChoices(runtime, Tokenizer(), q, max_len=96, head_max_len=64)
    for state in ['left 7', 'right 8']:
        ref = prepared.predict(original, state)
        for name, model in model_set.items():
            actual = prepared.predict(model, state)
            assert v2.answer_parity(ref, actual, .01)['passed'], name


@torch.inference_mode()
def test_complete_runner_smoke(tmp_path):
    torch.manual_seed(4)
    model = tiny_model(torch.float16)
    casefile = tmp_path / 'cases.jsonl'
    casefile.write_text(json.dumps({'state': 'road left', 'questions': {'q': {'criteria': ['left', 'right']}}}) + '\n')
    args = SimpleNamespace(compile_mode='reduce-overhead', dynamic=False, no_compile=True,
                           batch_sizes=[1], lengths=[128], tolerance=.01,
                           warmup=1, rounds=1, repeats=1, api_repeats=1, seed=29,
                           cases_jsonl=str(casefile), api_max_len=96)
    report = {}
    saved = []
    v2.benchmark(model, Tokenizer(), runtime, args, report, lambda: saved.append(True))
    assert len(report['forward']) == 1
    assert report['forward'][0]['candidates']['optimized_eager']['passed']
    assert report['end_to_end']['candidates']['original_eager']['passed']
    assert report['end_to_end']['candidates']['optimized_eager']['passed']
    assert report['api_input_source'] == 'custom_jsonl'
    assert saved
