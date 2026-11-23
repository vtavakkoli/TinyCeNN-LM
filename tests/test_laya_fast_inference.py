"""Numerical regression tests against the unmodified standalone runtime."""
import sys
from pathlib import Path
import pytest
import torch
from torch import nn
from transformers import ModernBertConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'standalone/laya_1p2_speedup'))
import modeling_laya_standalone as runtime
from fast_inference import PreparedChoices, CUDAGraphRunner, marker_head, optimize_model


def tiny_model(dtype=torch.float32):
    cfg = ModernBertConfig(vocab_size=128, hidden_size=64, intermediate_size=96,
                          num_hidden_layers=3, num_attention_heads=4,
                          local_attention=8, global_attn_every_n_layers=3,
                          max_position_embeddings=128, pad_token_id=0, reference_compile=False)
    return runtime.LayaStandalone(cfg).to(dtype=dtype).eval().requires_grad_(False)


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@torch.inference_mode()
def test_marker_head_matches_full_head(depth, dtype):
    torch.manual_seed(11)
    layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, norm_first=True)
    head = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(64),
                                 enable_nested_tensor=False).to(dtype=dtype).eval()
    x = torch.randn(3, 29, 64, dtype=dtype)
    keep = torch.rand(3, 29) > .3
    keep[:, 0] = True
    ref = head(x, src_key_padding_mask=~keep)[:, 1:5]
    actual = marker_head(head, x, keep, 4)
    tolerance = 3e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, ref, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('length', [1, 7, 37, 85])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@torch.inference_mode()
def test_end_to_end_tiny_modernbert(length, dtype):
    torch.manual_seed(22)
    model = tiny_model(dtype)
    ids = torch.randint(4, 127, (2, length))
    mask = torch.ones_like(ids)
    if length > 7:
        mask[1, -4:] = 0
        ids[1, -4:] = 0
    pos = torch.tensor([[0, min(2, length - 1), min(4, length - 1)],
                        [0, min(2, length - 1), 0]])
    mm = torch.tensor([[True, True, True], [True, True, False]])
    if length == 1:
        # Only one unique token exists. Repeating it as multiple valid options
        # creates exact ties whose argmax depends on floating-point rounding.
        # Keep the singleton boundary test and mask the duplicate slots.
        mm[:, 1:] = False
    batch = dict(input_ids=ids, attention_mask=mask, marker_pos=pos,
                 marker_mask=mm, qtype=torch.tensor([0, 2]))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ref = model(**batch)[0]
    fast = optimize_model(model)
    actual = fast(**batch)[0]
    tolerance = 3e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, ref, atol=tolerance, rtol=tolerance)
    assert torch.equal(ref.argmax(-1), actual.argmax(-1))
    assert actual[1, 2] == -1e4
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    with pytest.raises(RuntimeError, match='inference-only'):
        fast.train()
    with pytest.raises(ValueError, match='CUDA'):
        CUDAGraphRunner(fast, batch)


class Tokenizer:
    mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = '[MASK]', 3, 1, 2, 0

    def __init__(self):
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        ids = [4 + ord(c) % 100 for c in text]
        if kwargs.get('truncation'):
            ids = ids[:kwargs['max_length']]
        return {'input_ids': ids}


@pytest.mark.parametrize('state', ['', {'road': 'left', 'speed': 7}, 'abc ' * 500, '[MASK] hi'])
def test_prepared_matches_original_and_tokenizes_state_once(state):
    tok = Tokenizer()
    questions = {'a': {'type': 'choice', 'instructions': 'turn', 'criteria': ['left', 'right']},
                 'b': {'type': 'choice', 'instructions': 'safe [MASK]',
                       'criteria': {'yes': 'drive', 'no': 'stop', 'wait': ''}}}
    prepared = PreparedChoices(runtime, tok, questions, max_len=128, head_max_len=70)
    tok.calls = 0
    actual = prepared.batch(state)
    assert tok.calls == 1
    items = []
    for q in questions.values():
        ids, marks = runtime.build_sequence(tok, state, runtime.normalize_question(q),
                                            max_len=128, head_max_len=70)
        items.append(dict(ids=ids, markers=marks, qtype=0))
    expected = runtime.collate_items(items, tok.pad_token_id)
    for k in actual:
        assert torch.equal(actual[k], expected[k]), k


def test_prepared_rejects_truncated_options_and_empty_questions():
    with pytest.raises(ValueError, match='empty'):
        PreparedChoices(runtime, Tokenizer(), {})
    with pytest.raises(ValueError, match='truncates'):
        PreparedChoices(runtime, Tokenizer(), {'a': {'criteria': ['left', 'right']}}, max_len=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_cuda_graph_changed_inputs_and_owned_output():
    model = optimize_model(tiny_model(torch.float16).cuda())
    batch = dict(input_ids=torch.randint(4, 127, (2, 37), device='cuda'),
                 attention_mask=torch.ones(2, 37, device='cuda', dtype=torch.long),
                 marker_pos=torch.tensor([[2, 4], [2, 4]], device='cuda'),
                 marker_mask=torch.ones(2, 2, dtype=torch.bool, device='cuda'),
                 qtype=torch.zeros(2, dtype=torch.long, device='cuda'))
    runner = CUDAGraphRunner(model, batch)
    first = runner(**batch)[0]
    saved = first.clone()
    batch['input_ids'] = torch.randint(4, 127, (2, 37), device='cuda')
    batch['attention_mask'][1, -3:] = 0
    with torch.autocast('cuda', dtype=torch.float16):
        expected = model(**batch)[0]
    torch.testing.assert_close(runner(**batch)[0], expected, atol=.003, rtol=.003)
    assert torch.equal(first, saved)
    with pytest.raises(ValueError, match='mismatch'):
        runner(**dict(batch, input_ids=batch['input_ids'][:, :-1]))
