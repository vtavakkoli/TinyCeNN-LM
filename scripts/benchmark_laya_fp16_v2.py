#!/usr/bin/env python3
"""Laya FP16 V2: compiled control, interleaved timing, and complete prediction API."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import benchmark_laya_fp16 as v1
from fast_inference import PreparedChoices, inference_context, optimize_model

ROOT = Path(__file__).resolve().parents[1]


def candidates(original, *, compile_mode='reduce-overhead', dynamic=False, compile_enabled=True,
               backend='inductor'):
    models = {'original_eager': original, 'optimized_eager': optimize_model(original)}
    if compile_enabled:
        options = {'backend': backend, 'dynamic': dynamic, 'fullgraph': True}
        if backend == 'inductor':
            options['mode'] = compile_mode
        # Compile both complete forward paths under the same settings. Calling
        # .predict on OptimizedModule can bypass compiled forward; PreparedChoices
        # explicitly invokes model(**batch), so the API benchmark really compiles.
        models['original_compiled'] = torch.compile(original, **options)
        models['optimized_compiled'] = torch.compile(optimize_model(original), **options)
    return models


def answer_parity(reference, actual, tolerance):
    if reference.keys() != actual.keys():
        return {'passed': False, 'reason': 'question keys differ'}
    drift = 0.
    same = True
    for name, ref in reference.items():
        got = actual[name]
        if ref['probabilities'].keys() != got['probabilities'].keys():
            return {'passed': False, 'reason': 'option keys differ'}
        same &= ref['choice'] == got['choice']
        for label, expected in ref['probabilities'].items():
            observed = got['probabilities'][label]
            if not math.isfinite(observed) or not math.isfinite(expected):
                return {'passed': False, 'reason': 'nonfinite probability'}
            drift = max(drift, abs(expected - observed))
    return {'passed': same and drift <= tolerance, 'identical_decisions': same,
            'max_probability_drift': drift}


def interleaved_times(functions, *, device, rounds, repeats, seed, units=1):
    """Alternate candidate order each round; report ms per requested work unit.

    Each round measures a block of repeats, synchronizing before/after the block.
    Round statistics are means per call, not individual-request latency percentiles.
    All setup and warmup must have completed before calling this function.
    """
    samples = {name: [] for name in functions}
    errors, orders = {}, []
    rng = random.Random(seed)
    for _ in range(rounds):
        order = [name for name in functions if name not in errors]
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            try:
                v1.synchronize(device)
                start = time.perf_counter()
                for _ in range(repeats):
                    functions[name]()
                v1.synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000 / (repeats * units))
            except Exception as error:
                if v1.fatal_cuda_error(error):
                    raise
                errors[name] = f'{type(error).__name__}: {error}'
    results = {}
    for name, values in samples.items():
        if name in errors:
            results[name] = {'passed': False, 'error': errors[name], 'partial_round_ms': values}
        else:
            results[name] = {'passed': True, 'median_ms': statistics.median(values),
                             'round_ms': values, 'min_round_ms': min(values),
                             'max_round_ms': max(values)}
    return results, orders


def attribution(rows):
    valid = {k: v for k, v in rows.items() if v.get('passed') and 'median_ms' in v}
    pairs = [('compilation_on_original', 'original_eager', 'original_compiled'),
             ('optimization_without_compile', 'original_eager', 'optimized_eager'),
             ('optimization_beyond_compile', 'original_compiled', 'optimized_compiled'),
             ('combined_vs_original_eager', 'original_eager', 'optimized_compiled')]
    ratios = {label: valid[a]['median_ms'] / valid[b]['median_ms']
              for label, a, b in pairs if a in valid and b in valid}
    return {'ratios': ratios, 'winner': min(valid, key=lambda k: valid[k]['median_ms']) if valid else None}


def default_cases():
    # Illustrative application-like inputs, explicitly synthetic and unlabelled.
    return [
        {'state': {'road_direction': 'left', 'speed': 7.0, 'visibility': .95}, 'questions': v1.QUESTIONS},
        {'state': {'road_direction': 'right', 'speed': 35.0, 'visibility': .25,
                   'obstacle_ahead': True}, 'questions': v1.QUESTIONS},
        {'state': {'road_direction': 'straight', 'speed': 18.0, 'visibility': .7,
                   'traffic': ['pedestrian near crossing', 'vehicle ahead slowing'] * 16},
         'questions': v1.QUESTIONS},
    ]


def read_cases(path):
    if path is None:
        return default_cases()
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Audit JSONL must contain at least one case')
    for i, case in enumerate(rows):
        if not isinstance(case, dict) or 'state' not in case:
            raise ValueError(f'Case {i} needs state and optional questions')
        case.setdefault('questions', v1.QUESTIONS)
    return rows


def original_batch(runtime, tok, case, max_len):
    items = []
    for raw in case['questions'].values():
        q = runtime.normalize_question(raw)
        ids, marks = runtime.build_sequence(tok, case['state'], q, max_len=max_len)
        items.append(dict(ids=ids, markers=marks, qtype=runtime.QTYPES[q['t']]))
    return runtime.collate_items(items, tok.pad_token_id)


def audit_case_metadata(batch, answers):
    margins = []
    for answer in answers.values():
        probs = sorted(answer['probabilities'].values(), reverse=True)
        margins.append(probs[0] - probs[1])
    return {'batch_size': batch['input_ids'].shape[0], 'padded_tokens': batch['input_ids'].shape[1],
            'option_columns': batch['marker_pos'].shape[1], 'minimum_top2_probability_margin': min(margins)}


@torch.inference_mode()
def benchmark(original, tok, runtime, args, report, save):
    device = next(original.parameters()).device
    models = candidates(original, compile_mode=args.compile_mode, dynamic=args.dynamic,
                        compile_enabled=not args.no_compile)
    report['forward'] = []
    for size in args.batch_sizes:
        for length in args.lengths:
            batches = v1.synthetic_batches(runtime, tok, size, length, device)
            with inference_context(original):
                references = [original(**batch)[0].clone() for batch in batches]
            rows, functions = {}, {}
            for name, model in models.items():
                print(f'Forward B={size} T={length}: {name}', flush=True)
                start = time.perf_counter()
                try:
                    with inference_context(original):
                        audits = [v1.parity(ref, model(**batch)[0].clone(), batch['marker_mask'], args.tolerance)
                                  for ref, batch in zip(references, batches)]
                        passed = all(a['passed'] for a in audits)
                        if passed:
                            for _ in range(args.warmup):
                                model(**batches[0])
                        v1.synchronize(device)
                    rows[name] = {'passed': passed, 'audits': audits,
                                  'setup_validation_warmup_seconds': time.perf_counter() - start}
                    if passed:
                        functions[name] = lambda m=model: m(**batches[0])
                except Exception as error:
                    if v1.fatal_cuda_error(error):
                        raise
                    rows[name] = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
            if not rows['original_eager']['passed']:
                raise RuntimeError('Original reference repeatability failed')
            with inference_context(original):
                timed, order = interleaved_times(functions, device=device, rounds=args.rounds,
                                                 repeats=args.repeats, seed=args.seed)
            for name, metrics in timed.items():
                rows[name].update(metrics)
            result = {'batch_size': size, 'sequence_length': length, 'candidates': rows,
                      'timing_order': order, **attribution(rows)}
            report['forward'].append(result)
            save()
            print('Attribution:', result['ratios'], flush=True)

    cases = read_cases(args.cases_jsonl)
    prepared = [PreparedChoices(runtime, tok, c['questions'], max_len=args.api_max_len) for c in cases]
    refs, metadata = [], []
    # Independently ensure cached tokenization preserves original tensors.
    for case, cache in zip(cases, prepared):
        expected = original_batch(runtime, tok, case, args.api_max_len)
        actual = cache.batch(case['state'])
        if any(not torch.equal(expected[k], actual[k]) for k in expected):
            raise RuntimeError('Prepared input differs from original sequence builder')
        ref = original.predict(tok, case['state'], case['questions'], max_len=args.api_max_len)
        refs.append(ref)
        metadata.append(audit_case_metadata(expected, ref))
    report['api_case_metadata'] = metadata
    report['api_input_source'] = 'custom_jsonl' if args.cases_jsonl else 'synthetic_examples'
    # Includes the legacy API to quantify tokenization savings separately.
    def legacy_sweep():
        for case in cases:
            original.predict(tok, case['state'], case['questions'], max_len=args.api_max_len)

    def sweep(model):
        for case, cache in zip(cases, prepared):
            cache.predict(model, case['state'])

    functions, rows = {}, {}
    for name, model in [('legacy_predict', None), *models.items()]:
        print(f'Complete prediction API: {name}', flush=True)
        start = time.perf_counter()
        try:
            if model is None:
                outputs = [original.predict(tok, c['state'], c['questions'], max_len=args.api_max_len) for c in cases]
                fn = legacy_sweep
            else:
                outputs = [p.predict(model, c['state']) for p, c in zip(prepared, cases)]
                fn = lambda m=model: sweep(m)
            audits = [answer_parity(ref, out, args.tolerance) for ref, out in zip(refs, outputs)]
            passed = all(a['passed'] for a in audits)
            if passed:
                for _ in range(args.warmup):
                    fn()
            v1.synchronize(device)
            rows[name] = {'passed': passed, 'audits': audits,
                          'setup_validation_warmup_seconds': time.perf_counter() - start}
            if passed:
                functions[name] = fn
        except Exception as error:
            if v1.fatal_cuda_error(error):
                raise
            rows[name] = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
    if not rows['legacy_predict']['passed'] or not rows['original_eager']['passed']:
        report['end_to_end'] = {'candidates': rows}
        save()
        raise RuntimeError('Original API reference repeatability failed')
    timed, order = interleaved_times(functions, device=device, rounds=args.rounds,
                                     repeats=args.api_repeats, seed=args.seed, units=len(cases))
    for name, metrics in timed.items():
        rows[name].update(metrics)
    report['end_to_end'] = {'unit': 'milliseconds per state request (all its questions)',
                          'includes': 'state tokenization, collation, H2D, forward, D2H, softmax, decode',
                          'excludes': 'one-time question preparation and reported setup/warmup',
                          'candidates': rows, 'timing_order': order, **attribution(rows)}
    save()
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 512])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4])
    p.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune'], default='reduce-overhead')
    p.add_argument('--dynamic', action='store_true', help='Dynamic-shape compile for diverse custom inputs')
    p.add_argument('--no-compile', action='store_true', help='CPU smoke test only; attribution will be incomplete')
    p.add_argument('--cases-jsonl', help='Each line: {"state": ..., "questions": {...}}; questions optional')
    p.add_argument('--api-max-len', type=int, default=512)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--api-repeats', type=int, default=5)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--tolerance', type=float, default=.01)
    p.add_argument('--output', default='results/laya_fp16_v2.json')
    return p


def main():
    args = parser().parse_args()
    if (min(args.lengths) < 128 or min(args.batch_sizes) < 1 or
            min(args.api_max_len, args.warmup, args.rounds, args.repeats, args.api_repeats, args.threads) < 1
            or not math.isfinite(args.tolerance) or args.tolerance < 0):
        raise SystemExit('Invalid lengths, counts or tolerance')
    import transformers
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('Install transformers==4.57.6')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime; CPU smoke testing needs --device cpu --no-compile')
    torch.set_num_threads(args.threads)
    if args.model_dir:
        folder = Path(args.model_dir)
    else:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(v1.MODEL_ID, revision=v1.MODEL_REVISION,
                      allow_patterns=['*.py', '*.safetensors', '*.json', 'tokenizer/*', 'encoder/*']))
    runtime = v1.load_runtime(folder)
    original, tok = runtime.load_model(folder, device=args.device, compile_head=False)
    if any(p.dtype != torch.float16 for p in original.parameters()):
        raise RuntimeError('Expected actual FP16 model weights')
    sources = [Path(__file__), Path(v1.__file__), ROOT / 'standalone/laya_1p2_speedup/fast_inference.py']
    report = {'format': 'laya-fp16-benchmark-v2', 'model_id': v1.MODEL_ID,
              'model_revision': v1.MODEL_REVISION if not args.model_dir else None,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'hardware': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
              'arguments': vars(args), 'fullgraph_required': True,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'limitations': 'Synthetic unless custom cases supplied; agreement is not task accuracy. '
                             'Timings exclude cold setup. Interleaved rounds reduce but do not eliminate clock/thermal bias. '
                             'Fullgraph compilation is required: failures are recorded, not silently relabeled eager results. '
                             'Ratios >1 favor the second path in each comparison.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        benchmark(original, tok, runtime, args, report, save)
    except Exception as error:
        report['fatal_error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    print('Complete prediction API attribution:', report['end_to_end']['ratios'])
    print('Report:', output.resolve())


if __name__ == '__main__':
    main()
