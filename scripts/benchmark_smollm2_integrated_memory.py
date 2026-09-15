#!/usr/bin/env python3
"""V3: joint multi-context SmolLM2 training and complete-model cached evaluation."""
import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
import torch.nn.functional as F
from tinycenn_lm.integrated_memory import (
    native_dtype, build_student, wrappers, inference_mode, adapter_payload,
    restore_student, new_cache, greedy_generate,
)
from tinycenn_lm.optimized_memory import ridge_calibrate
from scripts.benchmark_cenn_research_layers import (
    capture_samples, fit_transfer, evaluate_nll, write_json, write_csv, paired_interval,
)


def collect_documents(rows, tokenizer, counts, length, excluded=(), max_documents=200000):
    excluded, seen = set(excluded), set()
    blocks, hashes = {k: [] for k in counts}, {k: [] for k in counts}
    for i, row in enumerate(rows):
        if i >= max_documents:
            break
        text = ' '.join(str(row.get('text', '')).split())
        digest = hashlib.sha256(text.encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if not text or bucket < 40 or digest in excluded or digest in seen:
            continue
        split = 'test' if bucket < 50 else 'validation' if bucket < 60 else 'train'
        if len(blocks[split]) >= counts[split]:
            continue
        seen.add(digest)
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=length + 1)['input_ids']
        if len(ids) < length + 1:
            continue
        blocks[split].append(torch.tensor(ids, dtype=torch.long))
        hashes[split].append(digest)
        if all(len(blocks[k]) == v for k, v in counts.items()):
            return blocks, hashes
    raise RuntimeError(f'Insufficient documents: { {k: len(v) for k,v in blocks.items()} }')


def read_exclusions(paths):
    result = set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        groups = data.get('document_hashes')
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f'{path}: no document_hashes mapping')
        for values in groups.values():
            if not isinstance(values, list) or any(not isinstance(x, str) or len(x) != 64 for x in values):
                raise ValueError(f'{path}: invalid document hashes')
            result.update(values)
    return result


def configurations(args):
    return [
        ('attention_conservative', 'transformer_readout', args.conservative_layers, None),
        ('attention_expanded', 'transformer_readout', args.expanded_layers, None),
        ('partition_conservative', 'cenn_partition', args.conservative_layers, 'attention_conservative'),
        ('partition_expanded', 'cenn_partition', args.expanded_layers, 'attention_expanded'),
        ('sink_conservative', 'sink_window', args.conservative_layers, 'attention_conservative'),
    ]


def joint_loss(student_logits, teacher_logits, targets, kl_weight=1., temperature=1., chunk=64):
    s, t, labels = student_logits.reshape(-1, student_logits.shape[-1]), teacher_logits.reshape(
        -1, teacher_logits.shape[-1]), targets.reshape(-1)
    ce, kl = s.new_zeros((), dtype=torch.float32), s.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), chunk):
        end = min(start + chunk, len(labels))
        logits = s[start:end].float()
        ce = ce + F.cross_entropy(logits, labels[start:end], reduction='sum') / len(labels)
        logp = F.log_softmax(logits / temperature, -1)
        with torch.no_grad():
            teacher_logp = F.log_softmax(t[start:end].float() / temperature, -1)
        kl = kl + F.kl_div(logp, teacher_logp, log_target=True, reduction='sum') * (
            temperature ** 2 / len(labels))
    return ce + kl_weight * kl, ce.detach(), kl.detach()


def validation_score(model, blocks, contexts, device, precision):
    with inference_mode(model, precision):
        scores = {str(c): statistics.fmean(evaluate_nll(model, blocks, c, device)) for c in contexts}
    if not all(math.isfinite(x) for x in scores.values()):
        raise RuntimeError('Nonfinite validation loss')
    return statistics.fmean(scores.values()), scores


def save_adapter(path, model, metadata):
    temporary = path.with_suffix('.tmp')
    torch.save(adapter_payload(model, metadata), temporary)
    temporary.replace(path)


def train_joint(teacher, model, blocks, args, out, name):
    history = []
    params = [p for p in model.parameters() if p.requires_grad]
    score, by_context = validation_score(model, blocks['validation'], args.train_contexts,
                                          args.device, args.precision)
    best, path = score, out / 'checkpoints' / f'{name}.pt'
    save_adapter(path, model, {'candidate': name, 'step': 0, 'validation_nll': score,
                               'model_revision': args.model_sha})
    history.append({'candidate': name, 'step': 0, 'validation_nll': score, **by_context})
    if not params:
        return history, best
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    order = list(range(len(blocks['train'])))
    random.Random(args.seed).shuffle(order)
    for step in range(args.joint_steps):
        context = args.train_contexts[step % len(args.train_contexts)]
        block = blocks['train'][order[step % len(order)]]
        ids, labels = block[:context][None].to(args.device), block[1:context+1][None].to(args.device)
        with torch.no_grad():
            target = teacher(input_ids=ids, use_cache=False).logits
        prediction = model(input_ids=ids, use_cache=False).logits
        loss, ce, kl = joint_loss(prediction, target, labels, args.kl_weight, args.temperature)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'{name}: nonfinite training objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        del prediction, target, loss
        if (step + 1) % args.eval_every == 0 or step + 1 == args.joint_steps:
            score, by_context = validation_score(model, blocks['validation'], args.train_contexts,
                                                  args.device, args.precision)
            if score < best:
                best = score
                save_adapter(path, model, {'candidate': name, 'step': step+1, 'validation_nll': best,
                                           'model_revision': args.model_sha})
            record = {'candidate': name, 'step': step+1, 'context': context,
                      'train_ce': float(ce), 'train_kl': float(kl), 'validation_nll': score, **by_context}
            history.append(record)
            write_csv(out / f'{name}_history.csv', history)
            write_json(out / 'progress.json', {'status': 'joint_training', **record})
            print(json.dumps(record), flush=True)
    return history, best


@torch.no_grad()
def cache_equivalence(model, block, device, precision, block_size):
    length = min(len(block)-1, 2 * block_size + 5)
    ids = block[:length][None].to(device)
    with inference_mode(model, precision):
        full = model(input_ids=ids, use_cache=False).logits
        split = min(block_size + 1, length-1)
        cache = new_cache(model)
        first = model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True).logits
        pieces = [first]
        for index in range(split, length):
            pieces.append(model(input_ids=ids[:, index:index+1], past_key_values=cache, use_cache=True).logits)
        cached = torch.cat(pieces, 1)
        error = float((cached.float()-full.float()).square().mean() / full.float().square().mean().clamp_min(1e-8))
        if not math.isfinite(error) or error > .001:
            raise RuntimeError(f'Full-model cache equivalence failed: NMSE={error}')
        return {'cached_logits_nmse': error, 'cache_test_tokens': length}


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_model(model, blocks, context, args):
    """Complete forward paths, identical teacher-forced decode inputs across models."""
    prefill_times, decode_times, states, end_states = [], [], [], []
    with inference_mode(model, args.precision):
        for block in blocks[:args.timing_documents]:
            ids = block[:context][None].to(args.device)
            continuation = block[context:context+args.decode_tokens][None].to(args.device)
            for repeat in range(args.timing_repeats + 1):
                cache = new_cache(model)
                sync(args.device)
                start = time.perf_counter()
                model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                sync(args.device)
                prefill = (time.perf_counter()-start)*1000
                initial_bytes = cache.nbytes
                start = time.perf_counter()
                for i in range(args.decode_tokens):
                    model(input_ids=continuation[:, i:i+1], past_key_values=cache,
                          use_cache=True, logits_to_keep=1)
                sync(args.device)
                decode = (time.perf_counter()-start)*1000 / args.decode_tokens
                if repeat:
                    prefill_times.append(prefill)
                    decode_times.append(decode)
                    states.append(initial_bytes)
                    end_states.append(cache.nbytes)
                del cache
    p, d = statistics.median(prefill_times), statistics.median(decode_times)
    return {'model_prefill_ms': p, 'model_decode_ms_per_token': d, 'decode_tokens_per_second': 1000/d,
            'total_cache_bytes': statistics.median(states), 'cache_bytes_after_decode': statistics.median(end_states),
            'timing_documents': min(args.timing_documents, len(blocks)), 'timing_repeats': args.timing_repeats,
            'decode_tokens': args.decode_tokens, 'prefill_samples_ms': prefill_times,
            'decode_samples_ms': decode_times}


def paired_metrics(values, reference, seed, prefix=''):
    delta, lo, hi = paired_interval(values, reference, seed)
    return {prefix+'delta_nll': delta, prefix+'ci_low': lo, prefix+'ci_high': hi,
            prefix+'ppl_ratio': math.exp(delta)}


def evaluate_models(teacher, candidates, blocks, hashes, args, out, selected):
    rows, documents, generation = [], [], []
    refs, timings = {}, {}
    all_records = [{'candidate': 'original', 'checkpoint': None, 'matched_control': None}] + candidates
    for record in all_records:
        name = record['candidate']
        model = teacher if name == 'original' else restore_student(teacher, torch.load(
            out / record['checkpoint'], map_location='cpu', weights_only=True))
        before_values = {}
        if name != 'original':
            before_model = restore_student(teacher, torch.load(
                out / 'checkpoints' / f'{name}_before_joint.pt', map_location='cpu', weights_only=True))
            with inference_mode(before_model, args.precision):
                before_values = {c: evaluate_nll(before_model, blocks['test'], c, args.device)
                                 for c in args.test_contexts}
            del before_model
        for context in args.test_contexts:
            with inference_mode(model, args.precision):
                values = evaluate_nll(model, blocks['test'], context, args.device)
            if not all(math.isfinite(x) for x in values):
                raise RuntimeError(f'{name}: nonfinite test NLL')
            timing = benchmark_model(model, blocks['test'], context, args)
            timings[name, context] = timing
            refs[name, context] = values
            base = refs['original', context]
            row = {'candidate': name, 'context': context, 'test_documents': len(values),
                   'test_nll': statistics.fmean(values), 'test_perplexity': math.exp(statistics.fmean(values)),
                   'selected_on_validation': name == selected, **record,
                   **paired_metrics(values, base, args.seed),
                   **{k:v for k,v in timing.items() if not isinstance(v, list)}}
            if before_values:
                row.update(paired_metrics(values, before_values[context], args.seed, 'before_joint_'))
                documents.extend({'candidate':name+'_before_joint','context':context,
                                  'document_hash':digest,'nll':nll}
                                 for digest,nll in zip(hashes['test'],before_values[context]))
            ref_timing = timings['original', context]
            row.update(prefill_speedup=ref_timing['model_prefill_ms']/timing['model_prefill_ms'],
                       decode_speedup=ref_timing['model_decode_ms_per_token']/timing['model_decode_ms_per_token'],
                       total_cache_ratio=timing['total_cache_bytes']/ref_timing['total_cache_bytes'])
            if record.get('matched_control'):
                row.update(paired_metrics(values, refs[record['matched_control'], context], args.seed, 'adapted_'))
                row['beats_both_quality'] = len(values) >= 8 and max(row['ci_high'], row['adapted_ci_high']) < 0
                row['quality_preserving_efficiency_win'] = (len(values) >= 8 and
                    max(row['ci_high'], row['adapted_ci_high']) <= args.nll_margin and
                    min(row['prefill_speedup'], row['decode_speedup']) > 1 and row['total_cache_ratio'] < 1)
            rows.append(row)
            documents.extend({'candidate':name, 'context':context, 'document_hash':digest, 'nll':nll}
                             for digest,nll in zip(hashes['test'],values))
            write_json(out / f'{name}_timing_T{context}.json', timing)
            write_csv(out / 'integrated_summary.csv', rows)
            write_csv(out / 'test_document_nll.csv', documents)
            print(json.dumps(row), flush=True)
        for index, block in enumerate(blocks['test'][:2]):
            ids = block[:min(64,args.test_contexts[0])][None].to(args.device)
            with inference_mode(model,args.precision):
                continuation, _ = greedy_generate(model, ids, args.decode_tokens)
            generation.append({'candidate':name, 'prompt_index':index, 'prompt_ids':ids[0].tolist(),
                               'generated_ids':continuation[0].tolist()})
        if model is not teacher:
            del model
    return rows, documents, generation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',default='HuggingFaceTB/SmolLM2-135M')
    p.add_argument('--model-revision',default='main')
    p.add_argument('--dataset',default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config',default='sample-10BT')
    p.add_argument('--dataset-revision',default='main')
    p.add_argument('--conservative-layers',default='0,29')
    p.add_argument('--expanded-layers',default='0,18,29')
    p.add_argument('--train-contexts',default='256,512,1024')
    p.add_argument('--test-contexts',default='256,512,1024,2048')
    for name,value in [('train-documents',128),('validation-documents',16),('test-documents',32),
                       ('warm-documents',16),('warm-steps',100),('joint-steps',300),('eval-every',50),
                       ('features',64),('block-size',32),('sinks',4),('seed',2028),('dataset-seed',9317),
                       ('decode-tokens',32),('timing-documents',3),('timing-repeats',3)]:
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--kl-weight',type=float,default=1.)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--nll-margin',type=float,default=.02)
    p.add_argument('--exclude-manifest',action='append',default=[])
    p.add_argument('--output-dir',default='result/smollm2-integrated-v3')
    a=p.parse_args()
    for key in ('conservative_layers','expanded_layers','train_contexts','test_contexts'):
        setattr(a,key,list(dict.fromkeys(int(x) for x in getattr(a,key).split(','))))
    if (min(a.train_documents,a.validation_documents,a.test_documents,a.warm_documents,a.timing_documents,
            a.timing_repeats,a.decode_tokens,a.eval_every,a.block_size) < 1 or min(a.warm_steps,a.joint_steps,a.sinks)<0
            or a.features<2 or a.features%2 or min(a.lr,a.temperature,a.nll_margin)<=0 or a.kl_weight<0
            or min(a.conservative_layers+a.expanded_layers)<0
            or min(a.train_contexts+a.test_contexts)<=2*a.block_size+a.sinks):
        p.error('Invalid dimensions, budget, or contexts; exercise compressed history in every context')
    return a


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi
    import transformers
    args=parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory; completed checkpoints can be evaluated separately')
    (out/'checkpoints').mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    args.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=native_dtype(args.device)
    args.precision=str(dtype).removeprefix('torch.')
    if args.device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
    api=HfApi()
    args.model_sha=api.model_info(args.base_model,revision=args.model_revision).sha
    data_sha=api.dataset_info(args.dataset,revision=args.dataset_revision).sha
    teacher=AutoModelForCausalLM.from_pretrained(args.base_model,revision=args.model_sha,
        torch_dtype=dtype,attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
    if teacher.config.model_type!='llama' or max(args.expanded_layers+args.conservative_layers)>=len(teacher.model.layers):
        raise ValueError('Select valid Llama/SmolLM2 layer indices')
    if max(args.train_contexts+args.test_contexts)+args.decode_tokens > teacher.config.max_position_embeddings:
        raise ValueError('Requested context plus decode exceeds the base model position limit')
    tokenizer=AutoTokenizer.from_pretrained(args.base_model,revision=args.model_sha)
    exclusions=read_exclusions([ROOT/'configs/smollm2_v3_prior_exclusions.json',*args.exclude_manifest])
    stream=load_dataset(args.dataset,args.dataset_config,revision=data_sha,split='train',streaming=True).shuffle(
        seed=args.dataset_seed,buffer_size=2048)
    blocks,hashes=collect_documents(stream,tokenizer,{'train':args.train_documents,'validation':args.validation_documents,
        'test':args.test_documents},max(args.train_contexts+args.test_contexts)+args.decode_tokens,exclusions)
    manifest={'status':'running','experiment':'smollm2-integrated-v3',
        'args':{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items()},
        'model_revision':args.model_sha,'dataset_revision':data_sha,'document_hashes':hashes,
        'prior_document_hashes_excluded':len(exclusions),'source_commit':subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python':platform.python_version(),
        'torch':torch.__version__,'transformers':transformers.__version__,'native_dtype':args.precision,
        'gpu':torch.cuda.get_device_name(0) if args.device.type=='cuda' else None,
        'compute_capability':list(torch.cuda.get_device_capability()) if args.device.type=='cuda' else None,
        'split_protocol':'exclude buckets0..39 and supplied prior hashes; test40..49; val50..59; train60..99',
        'unique_train_tokens_at_max_context':len(blocks['train'])*max(args.train_contexts)}
    write_json(out/'manifest.json',manifest)
    torch.save(blocks,out/'token_blocks.pt')
    indices=sorted(set(args.conservative_layers+args.expanded_layers))
    warm_context=min(args.train_contexts)
    train=capture_samples(teacher,blocks['train'][:args.warm_documents],indices,warm_context,args.device)
    valid=capture_samples(teacher,blocks['validation'],indices,warm_context,args.device)
    records,history=[],[]
    for name,variant,layers,control in configurations(args):
        torch.manual_seed(args.seed)
        model=build_student(teacher,layers,variant,args.features,args.block_size,args.sinks)
        started=time.perf_counter()
        if variant!='sink_window':
            for a in wrappers(model):
                # Warmup operates on teacher states; joint training below uses actual student states.
                ridge_calibrate(a.core,train[a.layer_idx],args.device)
                warmargs=SimpleNamespace(device=args.device,steps=args.warm_steps,eval_every=args.eval_every,
                                         lr=.002,seed=args.seed)
                fit_transfer(a.core,train[a.layer_idx],valid[a.layer_idx],warmargs,
                             out/'checkpoints'/f'{name}_warm_{a.layer_idx}.pt',f'{name}_layer{a.layer_idx}')
        before,_=validation_score(model,blocks['validation'],args.train_contexts,args.device,args.precision)
        path=out/'checkpoints'/f'{name}_before_joint.pt'
        save_adapter(path,model,{'candidate':name,'stage':'before_joint','model_revision':args.model_sha})
        fit_history,best=train_joint(teacher,model,blocks,args,out,name)
        history.extend(fit_history)
        checkpoint=f'checkpoints/{name}.pt'
        del model
        model=restore_student(teacher,torch.load(out/checkpoint,map_location='cpu',weights_only=True))
        equivalence=cache_equivalence(model,blocks['validation'][0],args.device,args.precision,args.block_size)
        record={'candidate':name,'variant':variant,'layers':layers,'matched_control':control,
            'checkpoint':checkpoint,'before_joint_validation_nll':before,'validation_nll':best,
            'training_seconds':time.perf_counter()-started,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters()),
            'replacement_count':len(layers),'full_attention_layers':len(model.model.layers) if variant=='transformer_readout' else len(model.model.layers)-len(layers),
            'joint_updates':args.joint_steps if variant!='sink_window' else 0,
            'joint_token_presentations':sum(args.train_contexts[i%len(args.train_contexts)] for i in range(args.joint_steps))
                if variant!='sink_window' else 0,**equivalence}
        records.append(record)
        write_csv(out/'validation_summary.csv',records)
        write_csv(out/'training_history.csv',history)
        write_json(out/'progress.json',{'status':'training','completed_candidates':records})
        del model
    selected=min((r for r in records if r['matched_control']),key=lambda r:r['validation_nll'])['candidate']
    write_json(out/'selection.json',{'selected':selected,'criterion':'mean validation NLL across training contexts',
                                   'test_not_used_for_selection':True})
    rows,_,examples=evaluate_models(teacher,records,blocks,hashes,args,out,selected)
    for example in examples:
        example['prompt']=tokenizer.decode(example['prompt_ids'])
        example['continuation']=tokenizer.decode(example['generated_ids'])
    write_json(out/'generation_examples.json',examples)
    manifest['status']='completed'
    write_json(out/'manifest.json',manifest)
    write_json(out/'progress.json',{'status':'completed','selected':selected})
    write_json(out/'integrated_report.json',{**manifest,'candidates':records,'rows':rows,'selected':selected,
        'interpretation':'Jointly trained partial replacements, not a fully attention-free model. '
        'Quality CI is over documents, not seeds. End-to-end timings use the same teacher-forced tokens, '
        'batch one, real mixed caches, full projections and one-token LM-head output. '
        'Warmups and model loading excluded; no throughput confidence interval or universal-win claim.'})
    print(f'Completed: {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
