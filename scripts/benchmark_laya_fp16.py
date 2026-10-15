#!/usr/bin/env python3
"""Measure parity and speed of unchanged Laya FP16 weights. GPU needed for GPU claims."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'standalone' / 'laya_1p2_speedup'))
from fast_inference import (CUDAGraphRunner, PreparedChoices, inference_context,
                            make_flash_attention_model, optimize_model)
MODEL_ID = 'vtava/Laya-1.2-speedup-standalone-fp16'
MODEL_REVISION = '2891f7312d8e7c65b43a6e04c29cc226b06b4fac'
QUESTIONS = {
    'steering': {'type': 'choice', 'instructions': 'Choose road direction.',
                 'criteria': {'left': 'road curves left', 'straight': 'road is straight', 'right': 'road curves right'}},
    'speed': {'type': 'choice', 'instructions': 'Choose safe longitudinal intent.',
              'criteria': {'accelerate': 'gain speed', 'hold': 'maintain speed', 'brake': 'reduce speed'}},
}


def load_runtime(folder):
    spec = importlib.util.spec_from_file_location('laya_fp16_reference_runtime',
                                                 Path(folder) / 'modeling_laya_standalone.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def latency(fn, device, warmup, repeats, rounds):
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        synchronize(device)
        samples.append(1000 * (time.perf_counter() - start) / repeats)
    return dict(median_ms=statistics.median(samples), round_ms=samples)


def parity(reference, actual, marker_mask, tolerance=.01):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        return dict(passed=False, reason='nonfinite logits')
    rp = reference.float().masked_fill(~marker_mask, -1e9).softmax(-1)
    ap = actual.float().masked_fill(~marker_mask, -1e9).softmax(-1)
    same = bool(torch.equal(rp.argmax(-1), ap.argmax(-1)))
    drift = float((rp - ap).abs().max())
    return dict(passed=same and drift <= tolerance, identical_decisions=same,
                max_probability_drift=drift,
                max_valid_logit_drift=float((reference[marker_mask] - actual[marker_mask]).abs().max()))


def synthetic_batches(runtime, tok, size, length, device):
    batches = []
    for phase in range(3):
        items = []
        for i in range(size):
            q = runtime.normalize_question(list(QUESTIONS.values())[(i + phase) % 2])
            state = json.dumps({'road_direction': ['left', 'straight', 'right'][(i + phase) % 3],
                                'speed': 7 + 13 * phase + i, 'visibility': .9 - .3 * phase})
            state += ' Observed road context and surrounding traffic.' * length
            ids, marks = runtime.build_sequence(tok, state, q, max_len=max(96, length - (i % 3) * 13))
            if len(marks) != len(runtime.render_options(q)):
                raise ValueError('Benchmark length truncates options')
            items.append(dict(ids=ids, markers=marks, qtype=0))
        batches.append({k: v.to(device) for k, v in runtime.collate_items(items, tok.pad_token_id).items()})
    return batches


def fatal_cuda_error(error):
    return any(s in str(error).lower() for s in
               ('illegal memory access', 'device-side assert', 'unspecified launch failure'))


@torch.inference_mode()
def run(args):
    import transformers
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('Use transformers==4.57.6, matching the checkpoint runtime')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime or use --device cpu')
    if args.model_dir:
        folder = Path(args.model_dir).resolve()
    else:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                       allow_patterns=['*.json', '*.py', '*.safetensors', 'tokenizer/*', 'encoder/*']))
    runtime = load_runtime(folder)
    print('Loading actual FP16 checkpoint ...', flush=True)
    start = time.perf_counter()
    baseline, tok = runtime.load_model(folder, device=str(device), compile_head=False)
    synchronize(device)
    load_seconds = time.perf_counter() - start
    if any(p.dtype != torch.float16 for p in baseline.parameters()):
        raise RuntimeError('Expected the actual-FP16 checkpoint loader')
    candidates = [('original_fp16', baseline),
                  ('device_masks_gather', optimize_model(baseline, prune_head=False)),
                  ('marker_sdpa', optimize_model(baseline))]
    if args.compile:
        candidates.append(('marker_compile', optimize_model(baseline, compile_mode=args.compile_mode)))
    skipped = []
    if args.flash_attention:
        try:
            fa = make_flash_attention_model(runtime, baseline)
            candidates.append(('flash2_marker', optimize_model(fa)))
        except Exception as error:
            if fatal_cuda_error(error):
                raise
            skipped.append(dict(candidate='flash2_marker', error=str(error)))
            print('FlashAttention-2 unavailable:', error, flush=True)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    sources = [Path(__file__), ROOT / 'standalone/laya_1p2_speedup/fast_inference.py',
               folder / 'modeling_laya_standalone.py']
    report = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION if not args.model_dir else None,
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  code_revision=revision, torch=torch.__version__, transformers=transformers.__version__,
                  python=platform.python_version(), device=str(device), cuda=torch.version.cuda,
                  hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine(),
                  cpu_threads=args.threads, load_seconds=load_seconds,
                  max_probability_drift=args.max_probability_drift,
                  limitations='Synthetic decision parity, not held-out accuracy. GPU speed requires GPU measurements. '
                              'Winners apply only to tested shapes/hardware, excluding compilation/capture/startup.',
                  skipped=skipped, results=[], winners=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    for size in args.batch_sizes:
        for length in args.lengths:
            batches = synthetic_batches(runtime, tok, size, length, device)
            with inference_context(baseline):
                refs = [baseline(**b)[0].clone() for b in batches]
            rows = []
            shape_candidates = list(candidates)
            if args.cuda_graph and device.type == 'cuda':
                shape_candidates.append(('marker_cuda_graph', None))
            for name, candidate in shape_candidates:
                print(f'B={size} T={length} {name} ...', flush=True)
                row = dict(batch_size=size, sequence_length=length, candidate=name)
                try:
                    synchronize(device)
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    if name == 'marker_cuda_graph':
                        candidate = CUDAGraphRunner(optimize_model(baseline), batches[0])
                    with inference_context(baseline):
                        audits = [parity(ref, candidate(**batch)[0].clone(), batch['marker_mask'],
                                         args.max_probability_drift) for ref, batch in zip(refs, batches)]
                        synchronize(device)
                        row.update(setup_and_validation_seconds=time.perf_counter() - start,
                                   audits=audits, passed=all(a['passed'] for a in audits))
                        if row['passed']:
                            row.update(latency(lambda: candidate(**batches[0]), device, args.warmup,
                                               args.repeats, args.rounds))
                        if device.type == 'cuda':
                            row['peak_process_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                    print(json.dumps(row), flush=True)
                except Exception as error:
                    row.update(passed=False, error=f'{type(error).__name__}: {error}')
                    print(row['error'], flush=True)
                    if name == 'original_fp16' or fatal_cuda_error(error):
                        report['results'].append(row)
                        save()
                        raise
                rows.append(row)
                report['results'].append(row)
                save()
                if name == 'marker_cuda_graph':
                    del candidate
                    gc.collect()
            original = next(r for r in rows if r['candidate'] == 'original_fp16')
            if not original.get('passed'):
                raise RuntimeError('Reference repeatability gate failed')
            eligible = [r for r in rows if r.get('passed') and 'median_ms' in r]
            for row in eligible:
                row['speedup_vs_original'] = original['median_ms'] / row['median_ms']
            winner = min(eligible, key=lambda r: r['median_ms'])
            report['winners'].append({k: winner[k] for k in
                ('batch_size', 'sequence_length', 'candidate', 'median_ms', 'speedup_vs_original')})
            save()
    # Same short input/order/padding; includes tokenization, copies, softmax/decode.
    prepared = PreparedChoices(runtime, tok, QUESTIONS)
    fast = optimize_model(baseline)
    state = {'road_direction': 'left', 'speed': 7.0}
    ref_answer = baseline.predict(tok, state, QUESTIONS)
    answer = prepared.predict(fast, state)
    same = all(answer[k]['choice'] == ref_answer[k]['choice'] for k in QUESTIONS)
    drift = max(abs(answer[k]['probabilities'][c] - ref_answer[k]['probabilities'][c])
                for k in QUESTIONS for c in answer[k]['probabilities'])
    report['end_to_end'] = dict(identical_decisions=same, max_probability_drift=drift)
    if same and drift <= args.max_probability_drift:
        for name, fn in [('original_predict', lambda: baseline.predict(tok, state, QUESTIONS)),
                         ('prepared_marker_predict', lambda: prepared.predict(fast, state))]:
            report['end_to_end'][name] = latency(fn, device, args.warmup, args.repeats, args.rounds)
    save()
    print('Measured winners:', json.dumps(report['winners'], indent=2))
    print('Report:', output.resolve())
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', help='Existing offline FP16 snapshot')
    p.add_argument('--device', default='cuda')
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 512])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune'], default='reduce-overhead')
    p.add_argument('--cuda-graph', action='store_true')
    p.add_argument('--flash-attention', action='store_true')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--max-probability-drift', type=float, default=.01)
    p.add_argument('--output', default='results/laya_fp16_benchmark.json')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.lengths) < 128 or min(args.batch_sizes) < 1:
        raise SystemExit('Use lengths >=128 and batch sizes >=1')
    if min(args.warmup, args.repeats, args.rounds, args.threads) < 1 or args.max_probability_drift < 0:
        raise SystemExit('Invalid benchmark counts or tolerance')
    run(args)
