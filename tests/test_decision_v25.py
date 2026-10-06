import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import AutoModel, ModernBertConfig

from tinycenn_lm.standalone_decision import StandaloneDecisionModel
from tinycenn_lm.decision_v25 import (
    WindowedBiDelta, MultiscaleCeNN, LinearMaskEncoder, PooledMarkerDecisionModel,
    attention_audit, core_parameters, layer_kinds, replacement_for,
    save_v25, load_v25, export_gate,
)

torch.set_num_threads(1)
ARCH = dict(window=8, cell_dim=8, cell_steps=2, scales=[1, 2, 4], backend="reference")


def encoder():
    cfg = ModernBertConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=2, global_attn_every_n_layers=3,
        local_attention=8, max_position_embeddings=128, reference_compile=False,
        pad_token_id=0, attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0)
    return AutoModel.from_config(cfg, attn_implementation="sdpa").eval()


def converted_encoder():
    enc = encoder()
    for layer, kind in zip(enc.layers, layer_kinds(enc)):
        layer.attn = replacement_for(layer.attn, kind, ARCH)
    return enc


def final_model():
    source = StandaloneDecisionModel(LinearMaskEncoder(converted_encoder()), head_layers=0, n_act=2)
    return PooledMarkerDecisionModel(source, compact_dim=8, layers=2).eval()


def inputs(length=17):
    return dict(input_ids=torch.randint(1, 64, (2, length)),
        attention_mask=torch.ones(2, length, dtype=torch.long),
        marker_pos=torch.tensor([[1, 3, 5], [1, 3, 0]]),
        marker_mask=torch.tensor([[True, True, True], [True, True, False]]),
        qtype=torch.tensor([0, 2]))


@pytest.mark.parametrize("kind", ["full_attention", "sliding_attention"])
def test_padding_invariance_finite_gradients_and_masking(kind):
    enc = encoder()
    block = replacement_for(enc.layers[0].attn, kind, ARCH)
    x = torch.randn(2, 13, 16, requires_grad=True)
    mask = torch.ones(2, 13, dtype=torch.long)
    y = block(x, attention_mask=mask)[0]
    padded = torch.cat([x.detach(), torch.randn(2, 11, 16)*100], dim=1)
    padded_mask = torch.nn.functional.pad(mask, (0, 11))
    yp = block(padded, attention_mask=padded_mask)[0]
    torch.testing.assert_close(y, yp[:, :13], atol=2e-6, rtol=1e-5)
    assert torch.count_nonzero(yp[:, 13:]) == 0
    loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in block.trainable_core_parameters())


def test_window_has_local_bidirectional_support_and_no_wraparound():
    block = WindowedBiDelta(encoder().layers[1].attn, window=8, backend="reference")
    x = torch.randn(1, 32, 16)
    mask = torch.ones(1, 32, dtype=torch.long)
    y = block(x, attention_mask=mask)[0]
    near = x.clone(); near[:, 3] += 5  # future token in the same window
    far = x.clone(); far[:, 28] += 5
    assert not torch.allclose(y[:, 1], block(near, attention_mask=mask)[0][:, 1])
    torch.testing.assert_close(y[:, 1], block(far, attention_mask=mask)[0][:, 1])


def test_global_cenn_has_distant_context_and_recurrent_template_gradient():
    block = MultiscaleCeNN(encoder().layers[0].attn, cell_dim=8, steps=2, scales=(1, 2, 4))
    x = torch.randn(1, 32, 16)
    y = block(x)[0]
    far = x.clone(); far[:, 30] = torch.randn(16)*10
    assert (y[:, 0] - block(far)[0][:, 0]).abs().max() > 1e-6
    y.square().mean().backward()
    assert block.template.grad.abs().sum() > 0


def test_linear_mask_encoder_preserves_converted_native_computation_and_ffns():
    enc = converted_encoder()
    ffn = enc.layers[0].mlp
    wrapped = LinearMaskEncoder(enc)
    assert wrapped.layers[0].mlp is ffn
    b = inputs()
    b["attention_mask"][1, -4:] = 0
    a = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    z = wrapped(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    valid = b["attention_mask"].bool()
    torch.testing.assert_close(a[valid], z[valid], rtol=1e-5, atol=1e-6)
    # No native mask builder or self-attention can be reached by the final path.
    enc._update_attention_mask = lambda *a, **k: (_ for _ in ()).throw(AssertionError("T x T mask"))
    wrapped(input_ids=b["input_ids"], attention_mask=b["attention_mask"])


def test_final_head_no_attention_option_permutation_and_training_step():
    model = final_model()
    assert attention_audit(model)["passed"]
    b = inputs()
    y = model(**b)
    perm = torch.tensor([2, 0, 1])
    bp = {**b, "marker_pos": b["marker_pos"][:, perm], "marker_mask": b["marker_mask"][:, perm]}
    yp = model(**bp)
    torch.testing.assert_close(yp[0], y[0][:, perm])
    torch.testing.assert_close(yp[1], y[1])
    model.requires_grad_(False)
    params = core_parameters(model)
    assert params
    for p in params: p.requires_grad_(True)
    before = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=0.01)
    loss = torch.nn.functional.cross_entropy(model(**b)[0], torch.tensor([1, 0]))
    loss.backward(); opt.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert any(not torch.equal(a, p) for a, p in zip(before, params))


class DummyTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path)/"tokenizer_config.json").write_text('{}')


def test_checkpoint_roundtrip_without_teacher_or_network(tmp_path):
    model = final_model()
    b = inputs()
    expected = model(**b)
    save_v25(model, tmp_path, DummyTokenizer(), ARCH, {"source": "test"}, {"test": True})
    loaded = load_v25(tmp_path, backend="reference")
    assert attention_audit(loaded)["passed"]
    for a, z in zip(expected, loaded(**b)):
        torch.testing.assert_close(a, z, rtol=0, atol=0)
    assert (tmp_path/"tinycenn_lm"/"decision_v25.py").exists()
    assert (tmp_path/"tinycenn_lm"/"standalone_decision.py").exists()
    # Verify the exported package in a fresh interpreter, not the repository import.
    import os, subprocess, sys
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c",
        "from tinycenn_lm.decision_v25 import load_v25, attention_audit; "
        "assert attention_audit(load_v25('.', backend='reference'))['passed']"],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_export_gate_fails_closed():
    gold = dict(accuracy=.77, cases=400, decisions=2000, kl_from_gold=.1, brier=.05)
    teacher = {"gold_test": gold}
    student = {"gold_test": {**gold, "accuracy": .765}, "speedup_vs_teacher": 1.2,
               "quality_vs_teacher": {"accuracy_gap_95pct_case_bootstrap": [-.015, .005]}}
    audit = {"passed": True}
    assert export_gate(teacher, student, audit)["passed"]
    for changed in ({"speedup_vs_teacher": 1.0}, {"speedup_vs_teacher": float("nan")},
                    {"gold_test": {**gold, "accuracy": .60}},
                    {"quality_vs_teacher": {"accuracy_gap_95pct_case_bootstrap": [-.05, .01]}}):
        assert not export_gate(teacher, {**student, **changed}, audit)["passed"]
    assert not export_gate(teacher, student, {"passed": False})["passed"]
    assert not export_gate({}, {}, audit)["passed"]


def test_notebook_preserves_schedules_and_has_no_saved_results():
    import ast
    root = Path(__file__).resolve().parents[1]
    names = ["Laya_Integrated_Memory_V24_BiGatedDeltaLite_Decision_Colab.ipynb",
             "Laya_Integrated_Memory_V25_WindowDelta_CeNN_Decision_Colab.ipynb"]
    notebooks = [json.loads((root/"notebooks"/n).read_text()) for n in names]
    def schedule(nb):
        tree = ast.parse(''.join(nb['cells'][2]['source']))
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if any(x in name for x in ('STEPS', '_LR', 'CASES')) or name in ('BATCH', 'MAX_LEN', 'COMPACT_DIM', 'COMPACT_LAYERS'):
                    out[name] = ast.literal_eval(node.value)
        return out
    assert schedule(notebooks[0]) == schedule(notebooks[1])
    for i, cell in enumerate(notebooks[1]['cells']):
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            if i != 1: ast.parse(''.join(cell['source']))


@pytest.mark.parametrize("version", ["V24", "V25"])
def test_notebook_all_training_stages_on_tiny_cpu_model(tmp_path, version):
    """Execute the actual training cells with tiny data/steps and the reference scan."""
    import ast
    import math
    import random
    import numpy as np
    from contextlib import nullcontext
    from tinycenn_lm.standalone_decision import collate_items
    root = Path(__file__).resolve().parents[1]
    suffix = 'BiGatedDeltaLite' if version == 'V24' else 'WindowDelta_CeNN'
    nb = json.loads((root/f'notebooks/Laya_Integrated_Memory_{version}_{suffix}_Decision_Colab.ipynb').read_text())
    ns = dict(torch=torch, nn=nn, F=torch.nn.functional, copy=copy, math=math,
              np=np, random=random, nullcontext=nullcontext, device=torch.device('cpu'),
              replacement_for=replacement_for, LinearMaskEncoder=LinearMaskEncoder,
              WindowedBiDelta=WindowedBiDelta, MultiscaleCeNN=MultiscaleCeNN,
              PooledMarkerDecisionModel=PooledMarkerDecisionModel,
              v25_core_parameters=core_parameters, collate_items=collate_items,
              ARCHITECTURE=ARCH, BATCH=2, amp_dtype=torch.float32,
              scaler=torch.amp.GradScaler('cuda', enabled=False))
    tree = ast.parse(''.join(nb['cells'][2]['source']))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if '_LR' in name or name.endswith('STEPS'):
                ns[name] = 2 if name.endswith('STEPS') else ast.literal_eval(node.value)
    from tinycenn_lm.decision_training import (require_finite, guarded_step, gold_checkpoint_key,
        trainable_snapshot, recovery_lr, save_stage, load_stage, forward_with_markers, marker_alignment_loss)
    ns.update(require_finite=require_finite, guarded_step=guarded_step,
        gold_checkpoint_key=gold_checkpoint_key, trainable_snapshot=trainable_snapshot,
        recovery_lr=recovery_lr, save_stage=save_stage, load_stage=load_stage,
        forward_with_markers=forward_with_markers, marker_alignment_loss=marker_alignment_loss,
        MARKER_ALIGNMENT_WEIGHT=.05, RESUME_STAGE_A=False, STAGE_A_DIR=tmp_path/'stage_A', SOURCE_MODEL='test',
        source_dir=Path('test-revision'), SEED=42, MAX_LEN=32, TRAIN_CASES=16, DEV_CASES=6)
    ns.update(COMPACT_DIM=8, COMPACT_LAYERS=2, RUN_COMPACT_HEAD=True)
    ns['teacher'] = StandaloneDecisionModel(encoder(), head_layers=1).eval().requires_grad_(False)
    ns['student_full'] = copy.deepcopy(ns['teacher'])
    ns['kinds'] = layer_kinds(ns['student_full'].encoder)
    ns['full_layers'] = [0]
    ns['replacement_layers'] = [0, 1, 2]
    ns['v25_indices'] = lambda m: [i for i,l in enumerate(m.encoder.layers)
        if isinstance(l.attn, (WindowedBiDelta, MultiscaleCeNN))]
    ns['tokenizer'] = type('Tokenizer', (), {'pad_token_id': 0})()
    rng = random.Random(42)
    def items(n):
        return [dict(ids=[rng.randrange(1,64) for _ in range(11)], markers=[1,3,5],
                     qtype=0, gold_probs=[.1,.7,.2], gold_index=1) for _ in range(n)]
    ns['train_items'], ns['dev_items'] = items(16), items(6)
    cell4 = ast.parse(''.join(nb['cells'][4]['source']))
    helpers = [node for node in cell4.body if isinstance(node, ast.FunctionDef)
               and node.name in ('batch', 'attn_io', 'local_loss')]
    def run(source):
        source = source.replace('torch.autocast(device_type="cuda", dtype=amp_dtype)', 'nullcontext()')
        exec(compile(source, '<v25-notebook-test>', 'exec'), ns)
    if version == 'V24':
        from tinycenn_lm.standalone_decision import _valid_tokens, _apply_modernbert_rope
        from tinycenn_lm.decision_v25 import reference_delta
        ns.update(_valid_tokens=_valid_tokens, _apply_modernbert_rope=_apply_modernbert_rope,
                  ALLOW_NEG_EIGVAL=False, USE_BIDIRECTIONAL=True)
        ns['chunk_gated_delta_rule'] = lambda q,k,v,g,beta,**kwargs: (reference_delta(q,k,v,g,beta), None)
        run(''.join(nb['cells'][3]['source']))
    run(ast.unparse(ast.Module(body=helpers, type_ignores=[])))
    ffn_before = [copy.deepcopy(l.mlp.state_dict()) for l in ns['student_full'].encoder.layers]
    for i in (5, 6, 7, 9):
        run(''.join(nb['cells'][i]['source']))
    if version == 'V25':
        assert attention_audit(ns['compact_student'])['passed']
    for before, layer in zip(ffn_before, ns['compact_student'].encoder.layers):
        for key, value in layer.mlp.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    # Rerunning the completed Stage A cell restores its checkpoint even when the
    # current in-memory encoder has already been converted/trained.
    ns['RESUME_STAGE_A'] = True
    run(''.join(nb['cells'][5]['source']))
