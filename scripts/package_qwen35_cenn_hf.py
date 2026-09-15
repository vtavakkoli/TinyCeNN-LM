#!/usr/bin/env python3
"""Package and optionally upload the validation-selected Qwen3.5 TinyCeNN adapter."""
import argparse, json, os, shutil, textwrap
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--repo-id',default='vtava/Qwen3.5-0.8B-CeNN-Integrated-V1')
    p.add_argument('--token',default=None)
    p.add_argument('--private',action='store_true')
    p.add_argument('--no-upload',action='store_true')
    a=p.parse_args()
    run=Path(a.run_dir); root=Path(__file__).resolve().parents[1]
    manifest=json.loads((run/'manifest.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    report=json.loads((run/'integrated_report.json').read_text())
    selected=selection['selected']
    record=next(r for r in report['candidates'] if r['candidate']==selected)
    if record.get('variant')!='cenn_partition':
        raise RuntimeError(f'Selected model is not CeNN partition: {record}')
    checkpoint=run/record['checkpoint']
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)

    export=run/'huggingface_export'; shutil.rmtree(export,ignore_errors=True); export.mkdir(parents=True)
    (export/'tinycenn_lm').mkdir()
    shutil.copy2(checkpoint,export/'cenn_adapter.pt')
    for name in ('qwen35_integrated_memory.py','optimized_memory.py'):
        shutil.copy2(root/'src'/'tinycenn_lm'/name,export/'tinycenn_lm'/name)
    (export/'tinycenn_lm'/'__init__.py').write_text('')
    lic=root/'LICENSE'
    if lic.exists(): shutil.copy2(lic,export/'TINY_CENN_LICENSE')
    for name in ('manifest.json','selection.json','integrated_report.json','validation_summary.csv','integrated_summary.csv','test_document_nll.csv','generation_examples.json'):
        src=run/name
        if src.exists(): shutil.copy2(src,export/name)

    cfg={
        'format':'qwen35-cenn-integrated-v1',
        'base_model':manifest['args']['base_model'],
        'base_revision':manifest['model_revision'],
        'selected_candidate':selected,
        'variant':record['variant'],
        'layers':record['layers'],
        'checkpoint':'cenn_adapter.pt',
        'source_commit':manifest['source_commit'],
        'transformers_version':manifest['transformers'],
        'architecture':manifest['architecture'],
    }
    (export/'adapter_config.json').write_text(json.dumps(cfg,indent=2))
    (export/'requirements.txt').write_text(f"torch\ntransformers=={manifest['transformers']}\n")
    (export/'load_model.py').write_text(textwrap.dedent('''
        import json
        from pathlib import Path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tinycenn_lm.qwen35_integrated_memory import restore_student, native_dtype

        def load_model(repo_dir='.', device=None):
            repo=Path(repo_dir)
            cfg=json.loads((repo/'adapter_config.json').read_text())
            device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
            dtype=native_dtype(device)
            base=AutoModelForCausalLM.from_pretrained(
                cfg['base_model'], revision=cfg['base_revision'], dtype=dtype, attn_implementation='sdpa'
            ).to(device).eval()
            payload=torch.load(repo/cfg['checkpoint'],map_location='cpu',weights_only=True)
            model=restore_student(base,payload).to(device).eval()
            tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],revision=cfg['base_revision'])
            return model, tokenizer
    ''').strip()+"\n")

    rows=[r for r in report['rows'] if r['candidate']==selected]
    rows=sorted(rows,key=lambda x:x['context'])
    table='\n'.join(
        f"| {r['context']} | {r['test_nll']:.4f} | {r['test_perplexity']:.3f} | {r['ppl_ratio']:.4f} | {r.get('adapted_ppl_ratio',float('nan')):.4f} | {r['total_cache_ratio']:.4f} | {r['prefill_speedup']:.3f} | {r['decode_speedup']:.3f} |"
        for r in rows
    )
    remaining=record['remaining_full_attention_layers']
    readme=f'''---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- qwen3_5
- cenn
- linear-attention
- efficient-attention
- custom-code
---

# Qwen3.5-0.8B + TinyCeNN Integrated Memory V1

Experimental **text-backbone adapter** for `{cfg['base_model']}`. The original Qwen3.5 model has 24 text decoder layers: 18 native Gated DeltaNet linear-attention layers and 6 full-attention layers. This experiment leaves all native linear-attention layers untouched and replaces selected full-attention layers with TinyCeNN bounded memory.

**Validation-selected candidate:** `{selected}`  
**Replaced full-attention layers:** `{record['layers']}`  
**Remaining original full-attention layers:** `{remaining}`  
**Base revision:** `{cfg['base_revision']}`  
**TinyCeNN source commit:** `{cfg['source_commit']}`

This repository stores an adapter checkpoint plus the exact custom loader/source required to reconstruct the model. It is **not** a standalone `save_pretrained()` checkpoint and does not include the original Qwen weights.

## Architecture

Qwen3.5-0.8B uses a 3:1 hybrid text stack (Gated DeltaNet linear attention plus periodic full attention). TinyCeNN is applied only to the original full-attention positions. Qwen3.5's Q/K normalization, partial MRoPE, and post-attention output gate are preserved. The TinyCeNN readout remains explicit because it cannot be folded through Qwen3.5's elementwise output gate without changing the computation.

The `cenn_partition` memory keeps sink/current/previous-block information exact and compresses older history into a bounded recurrent state. Therefore this is not "attention-free" in the strict sense: local exact attention remains inside the replacement, while unbounded global full attention is replaced.

## Held-out benchmark

Selection used validation NLL only; held-out test documents were not used for model selection.

| Context | Test NLL | PPL | PPL / original | PPL / matched adapted control | Cache / original | Prefill speedup | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

The current implementation is research PyTorch, not a fused production kernel. Cache reduction and quality should be interpreted separately from wall-clock speed.

## Load

```python
from load_model import load_model
model, tokenizer = load_model('.')
```

For generation, use batch size 1 with the custom `greedy_generate` helper in `tinycenn_lm.qwen35_integrated_memory` until a standard Transformers cache/generation integration is packaged.

## Reproducibility files

`manifest.json`, `selection.json`, `integrated_report.json`, validation/test CSVs, and generation examples are included. The manifest pins the exact base-model revision, dataset revision, source commit, package versions, split hashes, and experiment settings.

## Limitations

- Experimental research adapter; not a production model.
- Text backbone only. The original Qwen3.5 vision tower is not modified or packaged here.
- Benchmark confidence intervals are over held-out documents, not multiple independent training seeds.
- Custom TinyCeNN cache currently targets batch-one greedy decoding; beam/batch cache reordering is unsupported.
- Replacing all six full-attention layers removes original quadratic global attention from the text backbone, but native Gated DeltaNet layers and TinyCeNN local exact attention remain.

## Licenses

The base Qwen3.5 checkpoint is Apache-2.0. TinyCeNN-LM source is distributed under its repository license; a copy is included when available.
'''
    (export/'README.md').write_text(readme)
    print(f'Packaged: {export}')
    print(f'Repo: https://huggingface.co/{a.repo_id}')
    if a.no_upload: return
    token=a.token or os.environ.get('HF_TOKEN')
    if not token: raise RuntimeError('HF_TOKEN is required for upload')
    from huggingface_hub import HfApi
    api=HfApi(token=token)
    api.create_repo(a.repo_id,repo_type='model',private=a.private,exist_ok=True)
    api.upload_folder(repo_id=a.repo_id,repo_type='model',folder_path=str(export),commit_message=f'Publish {selected} TinyCeNN adapter')
    print(f'Uploaded: https://huggingface.co/{a.repo_id}')

if __name__=='__main__': main()
