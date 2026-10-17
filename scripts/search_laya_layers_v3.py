#!/usr/bin/env python3
"""Bounded brute-force layer search with case-disjoint selection and one final audit.

The official benchmark is synthetic labelled data, not a production workload.
No test labels or timing from test cases are used to select the model.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
import math
import random
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path
import torch
import benchmark_laya_fp16 as v1
from benchmark_laya_fp16_v2 import interleaved_times
from fast_inference import inference_context, optimize_model
from layer_search import MODES, apply_plan, channel_specs


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def jsonish(value):
    return json.loads(value) if isinstance(value, str) else value


def canonical_label(value):
    s = str(value).strip().lower()
    try:
        number = float(s)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return s


def prepare_case(row, runtime, tok, max_len):
    state, questions, gold = (jsonish(row[k]) for k in ('state', 'questions', 'gold'))
    items, targets, types = [], [], []
    # Keep exact runtime serialization, option order and dynamic padding.
    for qid, raw in questions.items():
        q = runtime.normalize_question(raw)
        labels = list(q['crit']) if q['t'] == 'choice' else (
            [str(i) for i in range(len(q['crit']))] if q['t'] == 'score' else ['false', 'true'])
        labels = [canonical_label(x) for x in labels]
        target = canonical_label(gold[qid]['label'])
        if len(set(labels)) != len(labels) or target not in labels:
            raise ValueError(f"Invalid gold option order in {row['id']}/{qid}")
        probs = {canonical_label(k): float(v) for k, v in jsonish(gold[qid]['probabilities']).items()}
        if set(probs) != set(labels) or not all(math.isfinite(x) and x >= 0 for x in probs.values()):
            raise ValueError(f"Invalid gold probabilities in {row['id']}/{qid}")
        ids, markers = runtime.build_sequence(tok, state, q, max_len=max_len)
        if len(markers) != len(labels):
            raise ValueError(f"Truncated options in {row['id']}/{qid}; increase max_len")
        items.append(dict(ids=ids, markers=markers, qtype=runtime.QTYPES[q['t']]))
        targets.append(labels.index(target))
        types.append(q['t'])
    if not items:
        raise ValueError(f"No questions in case {row['id']}")
    return dict(id=str(row['id']), workflow=str(row.get('workflow', 'unknown')),
                batch=runtime.collate_items(items, int(tok.pad_token_id)),
                target=torch.tensor(targets), types=types)


def split_cases(rows, sizes, seed):
    """Workflow-balanced deterministic case partitions, with ID disjointness."""
    if sum(sizes) > len(rows) or min(sizes) < 1:
        raise ValueError('Insufficient cases for requested partitions')
    ids = [str(r['id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case IDs')
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get('workflow', 'unknown'))].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    while any(groups.values()):
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    result, offset = [], 0
    for size in sizes:
        result.append(ordered[offset:offset + size])
        offset += size
    return result


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.inference_mode()
def predict_cases(model, cases):
    device = next(model.parameters()).device
    out = []
    with inference_context(model):
        for case in cases:
            logits = model(**to_device(case['batch'], device))[0].float()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite candidate logits')
            # Own the tensor before another CUDA graph replay can reuse storage.
            out.append(logits.cpu().clone())
    return out


def metrics(cases, outputs, references):
    if not (len(cases) == len(outputs) == len(references)) or not cases:
        raise ValueError('Empty/misaligned evaluation')
    buckets = defaultdict(lambda: [0, 0, 0, 0])
    n = good = ref_good = same = 0
    js_sum = nll_sum = drift = 0.
    case_deltas = []
    for case, logits, ref in zip(cases, outputs, references):
        mask = case['batch']['marker_mask']
        if logits.shape != mask.shape or ref.shape != mask.shape or not torch.isfinite(logits).all() or not torch.isfinite(ref).all():
            raise ValueError('Invalid logits')
        p = logits.float().masked_fill(~mask, -1e9).softmax(-1)
        q = ref.float().masked_fill(~mask, -1e9).softmax(-1)
        mix = (p + q) * .5
        js = .5 * (p * (p.clamp_min(1e-9).log() - mix.clamp_min(1e-9).log()) +
                   q * (q.clamp_min(1e-9).log() - mix.clamp_min(1e-9).log())).sum(-1)
        pred, ref_pred, gold = p.argmax(-1), q.argmax(-1), case['target']
        hits, ref_hits, agreements = pred == gold, ref_pred == gold, pred == ref_pred
        count = len(gold)
        n += count
        good += int(hits.sum()); ref_good += int(ref_hits.sum()); same += int(agreements.sum())
        js_sum += float(js.sum())
        nll_sum -= float(p[torch.arange(count), gold].clamp_min(1e-9).log().sum())
        drift = max(drift, float((p - q).abs().max()))
        case_deltas.append([int(hits.sum()) - int(ref_hits.sum()), count])
        for j, qtype in enumerate(case['types']):
            for key in ('type:' + qtype, 'workflow:' + case['workflow']):
                b = buckets[key]
                for k, value in enumerate((1, int(hits[j]), int(ref_hits[j]), int(agreements[j]))):
                    b[k] += value
    groups = {k: dict(n=v[0], accuracy=v[1]/v[0], reference_accuracy=v[2]/v[0],
                     accuracy_delta=(v[1]-v[2])/v[0], agreement=v[3]/v[0]) for k, v in buckets.items()}
    return dict(cases=len(cases), decisions=n, accuracy=good/n, reference_accuracy=ref_good/n,
                accuracy_delta=(good-ref_good)/n, agreement=same/n, mean_js=js_sum/n,
                gold_nll=nll_sum/n, max_probability_drift=drift, groups=groups,
                case_accuracy_deltas=case_deltas)


def quality_ok(row, args):
    return (row['agreement'] >= args.min_agreement and
            row['accuracy_delta'] >= -args.max_accuracy_drop and row['mean_js'] <= args.max_js and
            all(v['accuracy_delta'] >= -args.max_group_drop for v in row['groups'].values()))


def paired_case_interval(row, seed, repeats=2000):
    """Descriptive paired case bootstrap; accounts for correlated questions in a case."""
    values = row['case_accuracy_deltas']
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        picked = rng.choices(values, k=len(values))
        samples.append(sum(x[0] for x in picked) / sum(x[1] for x in picked))
    samples.sort()
    return [samples[int(.025 * repeats)], samples[min(repeats - 1, int(.975 * repeats))]]


@torch.inference_mode()
def calibrate(original, cases):
    sums = [torch.zeros(layer.mlp.Wo.in_features, device=next(original.parameters()).device)
            for layer in original.encoder.layers]
    counts = [0] * len(sums)
    handles, active = [], {}
    def hook(index):
        def collect(module, inputs):
            x = inputs[0].float()
            mask = active['mask'].bool()
            sums[index].add_((x.abs() * mask[..., None]).sum((0, 1)))
            counts[index] += int(mask.sum())
        return collect
    for i, layer in enumerate(original.encoder.layers):
        handles.append(layer.mlp.Wo.register_forward_pre_hook(hook(i)))
    try:
        with inference_context(original):
            for case in cases:
                batch = to_device(case['batch'], next(original.parameters()).device)
                active['mask'] = batch['attention_mask']
                original(**batch)
    finally:
        for handle in handles:
            handle.remove()
    return [(s / max(1, n) * layer.mlp.Wo.weight.float().norm(dim=0)).cpu()
            for s, n, layer in zip(sums, counts, original.encoder.layers)]


class Search:
    def __init__(self, original, cases, references, args, specs):
        self.original, self.cases, self.refs = original, cases, references
        self.args, self.specs = args, specs
        self.path = args.output / 'trials.json'
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else {}

    def evaluate(self, plan, phase):
        key = digest(plan)
        if key in self.rows:
            return self.rows[key]
        start = time.perf_counter()
        model = None
        row = dict(key=key, phase=phase, plan=plan)
        try:
            model = apply_plan(self.original, plan)
            row['metrics'] = metrics(self.cases, predict_cases(model, self.cases), self.refs)
            row['quality_pass'] = quality_ok(row['metrics'], self.args)
            row['parameters'] = sum(p.numel() for p in model.parameters())
            row['status'] = 'ok'
        except Exception as error:
            if v1.fatal_cuda_error(error):
                raise
            row.update(status='error', quality_pass=False, error=f'{type(error).__name__}: {error}')
        finally:
            del model
        row['seconds'] = time.perf_counter() - start
        self.rows[key] = row
        write_json(self.path, self.rows)
        m = row.get('metrics', {})
        print(f"{len(self.rows):4} {phase:12} changed={len(plan):2} "
              f"pass={row['quality_pass']} agree={m.get('agreement',0):.4f} "
              f"gold_delta={m.get('accuracy_delta',0):+.4f}", flush=True)
        return row

    def run(self):
        baseline = self.evaluate({}, 'baseline')
        singles = []
        for i in self.specs:
            for mode in MODES:
                singles.append(self.evaluate({str(i): self.specs[i][mode]}, 'single'))
        # Explicit all-layer trials, even if individual substitutions failed.
        for mode in ('mlp75', 'mlp50', 'drop_attn', 'drop_mlp'):
            self.evaluate({str(i): self.specs[i][mode] for i in self.specs}, 'all_layers')
        usable = [r for r in singles if r['status'] == 'ok']
        # Include near misses so compensating pairs can still be discovered.
        usable.sort(key=lambda r: (not r['quality_pass'], -r['metrics']['agreement'], r['metrics']['mean_js']))
        pool = usable[:self.args.pair_top]
        for a, b in itertools.combinations(pool, 2):
            if set(a['plan']).isdisjoint(b['plan']):
                self.evaluate({**a['plan'], **b['plan']}, 'pair')
        # Bounded beam search; all scores are recomputed in the accumulated model.
        frontier = [baseline]
        for depth in range(self.args.depth):
            next_rows = []
            for parent in frontier:
                for single in usable[:self.args.combine_top]:
                    if set(parent['plan']).isdisjoint(single['plan']):
                        row = self.evaluate({**parent['plan'], **single['plan']}, f'combine_{depth+1}')
                        if row['quality_pass']:
                            next_rows.append(row)
            unique = {r['key']: r for r in next_rows}
            frontier = sorted(unique.values(), key=lambda r: (r['parameters'], -r['metrics']['agreement']))[:self.args.beam]
            if not frontier:
                break
        return list(self.rows.values())


def timing_cases(cases, count):
    ordered = sorted(cases, key=lambda c: c['batch']['input_ids'].shape[1])
    if count >= len(ordered):
        return ordered
    return [ordered[round(i * (len(ordered)-1) / (count-1))] for i in range(count)] if count > 1 else [ordered[len(ordered)//2]]


@torch.inference_mode()
def timed_models(models, cases, args):
    device = next(next(iter(models.values())).parameters()).device
    # Forward timing includes H2D, forward, softmax and D2H on real cases.
    def sweep(model):
        with inference_context(model):
            for case in cases:
                batch = to_device(case['batch'], device)
                model(**batch)[0].float().softmax(-1).cpu()
    functions = {name: lambda m=model: sweep(m) for name, model in models.items()}
    setup = {}
    for name, fn in functions.items():
        start = time.perf_counter()
        for _ in range(args.warmup):
            fn()
        v1.synchronize(device)
        setup[name] = time.perf_counter() - start
    timed, orders = interleaved_times(functions, device=device, rounds=args.rounds,
        repeats=args.repeats, seed=args.seed, units=len(cases))
    return dict(results=timed, orders=orders, setup_seconds=setup,
                unit='ms per pretokenized case, all questions; H2D + forward + softmax + D2H',
                case_ids=[c['id'] for c in cases])


def compile_model(model, args):
    # Variable lengths preserve the original landmark positions; do not repad.
    return torch.compile(model, dynamic=True, fullgraph=True, mode=args.compile_mode)


@torch.inference_mode()
def predict_request(model, row, runtime, tok, max_len):
    """Complete uncached typed-decision API, including tokenization and decoding."""
    state, questions = jsonish(row['state']), jsonish(row['questions'])
    items, names, labels = [], [], []
    for name, raw in questions.items():
        q = runtime.normalize_question(raw)
        options = list(q['crit']) if q['t'] == 'choice' else (
            [str(i) for i in range(len(q['crit']))] if q['t'] == 'score' else ['false', 'true'])
        ids, marks = runtime.build_sequence(tok, state, q, max_len=max_len)
        if len(marks) != len(options):
            raise ValueError('Truncated API options')
        items.append(dict(ids=ids, markers=marks, qtype=runtime.QTYPES[q['t']]))
        names.append(name); labels.append(options)
    batch = to_device(runtime.collate_items(items, tok.pad_token_id), next(model.parameters()).device)
    with inference_context(model):
        logits = model(**batch)[0]
        probabilities = logits.float().masked_fill(~batch['marker_mask'], -1e9).softmax(-1).cpu()
    result = {}
    for i, name in enumerate(names):
        p = probabilities[i, :len(labels[i])]
        result[name] = dict(label=labels[i][int(p.argmax())],
                           probabilities=dict(zip(labels[i], map(float, p))))
    return result


@torch.inference_mode()
def final_api_benchmark(original, plan, rows, runtime, tok, args):
    """Descriptive post-selection timing on validation, never on test cases."""
    clear_compile()
    models = dict(original_eager=original,
        original_compiled=compile_model(original, args),
        optimized_compiled=compile_model(optimize_model(original), args),
        selected_compiled=compile_model(apply_plan(original, plan), args))
    device = next(original.parameters()).device
    def sweep(model):
        for row in rows:
            predict_request(model, row, runtime, tok, args.max_len)
    functions, setup, errors = {}, {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        try:
            # Check the API's tensor inputs/decoded probabilities against the
            # same architecture's forward path, including every question type.
            eager = apply_plan(original, plan) if name == 'selected_compiled' else original
            for row in rows:
                case = prepare_case(row, runtime, tok, args.max_len)
                logits = predict_cases(eager, [case])[0]
                expected = logits.masked_fill(~case['batch']['marker_mask'], -1e9).softmax(-1)
                actual = predict_request(model, row, runtime, tok, args.max_len)
                for i, answer in enumerate(actual.values()):
                    p = torch.tensor(list(answer['probabilities'].values()))
                    ref = expected[i, :len(p)]
                    if p.argmax() != ref.argmax() or (p-ref).abs().max() > .01:
                        raise ValueError('Full API parity failed')
            del eager
            for _ in range(args.warmup):
                sweep(model)
            v1.synchronize(device)
            setup[name] = time.perf_counter()-start
            functions[name] = lambda m=model: sweep(m)
        except Exception as error:
            if v1.fatal_cuda_error(error):
                raise
            errors[name] = f'{type(error).__name__}: {error}'
    timed, orders = interleaved_times(functions, device=device, rounds=args.rounds,
        repeats=args.repeats, seed=args.seed, units=len(rows)) if functions else ({}, [])
    del models, functions
    clear_compile()
    return dict(results=timed, errors=errors, setup_seconds=setup, orders=orders,
        unit='ms per complete uncached state request, all questions',
        includes='state/question tokenization, collation, H2D, model, softmax, D2H, Python decode',
        excludes='model loading, compilation and warmup', case_ids=[str(r['id']) for r in rows])


def clear_compile():
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args):
    import transformers
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('Use transformers==4.57.6')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime')
    if min(args.calibration, args.screen, args.validation, args.finalists, args.rounds,
           args.repeats, args.warmup, args.timing_cases, args.pair_top, args.combine_top, args.beam, args.depth) < 1:
        raise ValueError('All search budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    # Many different architecture guards are intentional, but clear between finalists.
    torch._dynamo.config.cache_size_limit = 64
    revision = args.dataset_revision or HfApi().dataset_info('LocalLLaMA/typed-decisions').sha
    train = list(load_dataset('LocalLLaMA/typed-decisions', 'all', split='train', revision=revision))
    partitions = split_cases(train, [args.calibration, args.screen, args.validation], args.seed)
    source_paths = [Path(__file__), v1.ROOT/'standalone/laya_1p2_speedup/layer_search.py',
                    v1.ROOT/'standalone/laya_1p2_speedup/fast_inference.py',
                    Path(__file__).with_name('benchmark_laya_fp16.py'),
                    Path(__file__).with_name('benchmark_laya_fp16_v2.py')]
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        model_id=v1.MODEL_ID, model_revision=v1.MODEL_REVISION, dataset_revision=revision,
        torch=torch.__version__, transformers=transformers.__version__,
        hardware=torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU',
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        partitions=[[str(r['id']) for r in group] for group in partitions])
    config_path = args.output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration, code, dataset or hardware changed. Choose a new output directory.')
    write_json(config_path, config)
    if (args.output/'final_report.json').exists():
        print('Completed report exists; reusing it. Choose a new output directory for a new experiment.')
        return
    folder = Path(snapshot_download(v1.MODEL_ID, revision=v1.MODEL_REVISION,
        allow_patterns=['*.json', '*.py', '*.safetensors', 'encoder/*', 'tokenizer/*']))
    runtime = v1.load_runtime(folder)
    original, tok = runtime.load_model(folder, device=args.device, compile_head=False)
    assert len(original.encoder.layers) == 28, 'Unexpected checkpoint architecture'
    assert all(p.dtype == torch.float16 for p in original.parameters()), 'Expected actual FP16 weights'
    prepared = [[prepare_case(r, runtime, tok, args.max_len) for r in group] for group in partitions]
    calibration, screen, validation = prepared
    print('Caching reference outputs and activation calibration...', flush=True)
    screen_ref, val_ref = predict_cases(original, screen), predict_cases(original, validation)
    # First verify the no-change optimized path on real inputs.
    unchanged = metrics(validation, predict_cases(optimize_model(original), validation), val_ref)
    if unchanged['agreement'] != 1 or unchanged['max_probability_drift'] > .01:
        raise RuntimeError('Unchanged FastLaya failed real-case parity; do not search on a broken baseline')
    scores = calibrate(original, calibration)
    specs = channel_specs(original, scores)
    write_json(args.output/'channel_specs.json', specs)
    search = Search(original, screen, screen_ref, args, specs)
    rows = search.run()
    eligible = [r for r in rows if r['quality_pass'] and r['plan']]
    # Shortlist by physical parameter reduction; actual latency decides later.
    eligible.sort(key=lambda r: (r['parameters'], -r['metrics']['agreement']))
    checked, shortlist = [], []
    for row in eligible[:args.validation_top]:
        model = apply_plan(original, row['plan'])
        m = metrics(validation, predict_cases(model, validation), val_ref)
        checked.append(dict(key=row['key'], plan=row['plan'], metrics=m, quality_pass=quality_ok(m, args)))
        if quality_ok(m, args):
            shortlist.append(row)
        del model
    write_json(args.output/'validation.json', checked)
    shortlist = shortlist[:args.finalists]
    # Include identity as a valid outcome; avoid choosing a slower changed model.
    benchmarks = []
    timed_cases = timing_cases(validation, args.timing_cases)
    for row in [dict(key='unchanged', plan={})] + shortlist:
        clear_compile()
        base = candidate = None
        entry = dict(key=row['key'], plan=row['plan'])
        try:
            base = compile_model(optimize_model(original), args)
            candidate = compile_model(apply_plan(original, row['plan']), args)
            candidate_out = predict_cases(candidate, validation)
            eager_out = predict_cases(apply_plan(original, row['plan']), validation)
            parity = metrics(validation, candidate_out, eager_out)
            quality = metrics(validation, candidate_out, val_ref)
            base_parity = metrics(validation, predict_cases(base, validation), val_ref)
            entry.update(compile_parity=parity, quality=quality, baseline_parity=base_parity)
            if parity['agreement'] != 1 or parity['max_probability_drift'] > .01 or not quality_ok(quality, args):
                raise ValueError('Compiled candidate failed quality/parity gate')
            if base_parity['agreement'] != 1 or base_parity['max_probability_drift'] > .01:
                raise ValueError('Compiled baseline failed parity gate')
            timing = timed_models({'baseline': base, 'candidate': candidate}, timed_cases, args)
            entry['timing'] = timing
            result = timing['results']
            if not all(x['passed'] for x in result.values()):
                raise RuntimeError('Timing failed')
            ratios = [a / b for a,b in zip(result['baseline']['round_ms'], result['candidate']['round_ms'])]
            entry.update(status='ok', speedup=statistics.median(ratios), round_speedups=ratios,
                         majority_faster=sum(r > 1 for r in ratios) >= math.ceil(.8 * len(ratios)))
        except Exception as error:
            if v1.fatal_cuda_error(error):
                raise
            entry.update(status='error', error=f'{type(error).__name__}: {error}')
        finally:
            del base, candidate
            clear_compile()
        benchmarks.append(entry)
        write_json(args.output/'compiled_validation.json', benchmarks)
        print('Compiled finalist:', row['key'][:12], entry.get('speedup', entry.get('error')), flush=True)
    winners = [r for r in benchmarks if r['status'] == 'ok' and r['plan'] and
               r['speedup'] >= args.min_speedup and r['majority_faster']]
    selected = max(winners, key=lambda r: r['speedup']) if winners else dict(key='unchanged', plan={})
    # Freeze selection BEFORE fetching the official test split. A crash after
    # this point resumes this same plan, never a holdout-selected alternative.
    selected_path = args.output/'selected_before_test.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        write_json(selected_path, selected)
    timed_ids = {c['id'] for c in timed_cases}
    api = final_api_benchmark(original, selected['plan'],
        [r for r in partitions[2] if str(r['id']) in timed_ids], runtime, tok, args)
    write_json(args.output/'end_to_end_api.json', api)
    print('Selection frozen. Loading official test for final audit only.', flush=True)
    test_rows = list(load_dataset('LocalLLaMA/typed-decisions', 'all', split='test', revision=revision))
    train_ids, test_ids = {str(r['id']) for r in train}, [str(r['id']) for r in test_rows]
    if len(set(test_ids)) != len(test_ids) or train_ids.intersection(test_ids):
        raise ValueError('Train/test case overlap or duplicate test IDs')
    test = [prepare_case(r, runtime, tok, args.max_len) for r in test_rows]
    test_ref = predict_cases(original, test)
    final_model = apply_plan(original, selected['plan'])
    eager_test = predict_cases(final_model, test)
    audit = metrics(test, eager_test, test_ref)
    audit['paired_case_accuracy_delta_95pct_interval'] = paired_case_interval(audit, args.seed)
    final_compile = {'passed': False}
    compiled = None
    try:
        compiled = compile_model(final_model, args)
        out = predict_cases(compiled, test)
        parity = metrics(test, out, eager_test)
        final_compile = dict(passed=parity['agreement'] == 1 and parity['max_probability_drift'] <= .01,
                             parity=parity, quality=metrics(test, out, test_ref))
    except Exception as error:
        if v1.fatal_cuda_error(error):
            raise
        final_compile['error'] = f'{type(error).__name__}: {error}'
    del compiled
    clear_compile()
    # Verify reconstruction before exporting its recipe (same pinned weights).
    reloaded = apply_plan(original, selected['plan'])
    replay = metrics(test[:4], predict_cases(reloaded, test[:4]), eager_test[:4])
    assert replay['agreement'] == 1 and replay['max_probability_drift'] <= 1e-6
    del reloaded
    recommendation = bool(selected['plan'] and quality_ok(audit, args) and final_compile['passed'] and
                          quality_ok(final_compile['quality'], args))
    recipe = dict(format='laya-layer-search-v3-recipe', model_id=v1.MODEL_ID,
        model_revision=v1.MODEL_REVISION, plan=selected['plan'], max_len=args.max_len,
        recommended= recommendation, dataset_revision=revision,
        note='Architecture recipe, not standalone model weights. Requires the exact pinned source checkpoint.')
    write_json(args.output/'selected_recipe.json', recipe)
    for name in ('layer_search.py', 'fast_inference.py'):
        shutil.copy2(v1.ROOT/'standalone/laya_1p2_speedup'/name, args.output/name)
    report = dict(format='laya-layer-search-v3', config=config,
        search_trials=len(rows), changed_layers=len(selected['plan']), selected=selected,
        final_audit=audit, final_compile=final_compile, export_roundtrip=replay, end_to_end_api=api,
        recommended= recommendation,
        outcome='measured_candidate_passed' if recommendation else (
            'no_qualifying_layer_change' if not selected['plan'] else 'candidate_failed_final_audit'),
        limitations=[
            'Bounded search: exhaustive single-layer grid, bounded pairs and beam combinations; not all 6^28 configurations.',
            'Labelled benchmark data is synthetic. Prior checkpoint training overlap is not established.',
            'Point-estimate gates do not establish statistical non-inferiority. See paired case confidence interval.',
            'Selection speed uses pretokenized validation cases; separate API timings include tokenization. Compilation excluded from steady-state latency.',
            'No fine-tuning or recovery training; test results do not trigger re-selection.'])
    write_json(args.output/'final_report.json', report)
    print(json.dumps({k: report[k] for k in ('outcome','changed_layers','search_trials','recommended')}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('/content/laya_layer_search_v3'))
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--dataset-revision', default=None)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--calibration', type=int, default=24)
    p.add_argument('--screen', type=int, default=32)
    p.add_argument('--validation', type=int, default=160)
    p.add_argument('--pair-top', type=int, default=12)
    p.add_argument('--combine-top', type=int, default=24)
    p.add_argument('--beam', type=int, default=2)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--validation-top', type=int, default=12)
    p.add_argument('--finalists', type=int, default=3)
    p.add_argument('--timing-cases', type=int, default=8)
    p.add_argument('--min-agreement', type=float, default=.98)
    p.add_argument('--max-accuracy-drop', type=float, default=.005)
    p.add_argument('--max-group-drop', type=float, default=.02)
    p.add_argument('--max-js', type=float, default=.005)
    p.add_argument('--min-speedup', type=float, default=1.03)
    p.add_argument('--compile-mode', choices=['default','reduce-overhead','max-autotune'], default='reduce-overhead')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--repeats', type=int, default=3)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
