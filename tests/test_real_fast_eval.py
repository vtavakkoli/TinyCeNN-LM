import pytest,torch
from types import SimpleNamespace
from scripts.real_fast_eval import (prepare_arc_example, score_chat_response, score_continuations,
                           quality_gate, arc_easy_eval, render_chat_input, greedy_no_cache,
                           real_chat_eval)
class Tok:
    def encode(self,text,add_special_tokens=False):
        return [ord(x)%120 for x in text]
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.ones(1))
    def forward(self,input_ids,use_cache,return_dict):
        ids=input_ids
        logits=torch.zeros((ids.shape[0],ids.shape[1],128),device=ids.device)
        # next character always A (65), so predict A
        logits[...,ord('A')]=10.
        return SimpleNamespace(logits=logits)
def test_ground_truth_arc():
    ex={"id":"x","question":"What?","choices":{"label":["1","2"],"text":["yes","no"]},"answerKey":"1"}
    x=prepare_arc_example(ex);assert x['answer']=='A';assert x['options']==['A','B']
    r=arc_easy_eval(Model(),Tok(),limit=1,dataset=[ex],seed=14)
    assert r['accuracy']==1 and r['rows'][0]['pred']=='A'
def test_rubric_distinguishes_semantics():
    assert score_chat_response({'kind':'contains','answer':'vienna'},'Vienna is the capital.')
    assert score_chat_response({'kind':'number_only','answer':'42'},'42')
    assert not score_chat_response({'kind':'number_only','answer':'42'},'42 is the answer')
    assert score_chat_response({'kind':'three_words'},'Bright lunar surface')
    assert not score_chat_response({'kind':'three_words'},'The Moon is a celestial body.')
def test_gold_gate_ignores_teacher_agreement():
    results={'teacher_nll':3.0,'student_nll':3.01,'teacher_gold_accuracy':0.4,'student_gold_accuracy':0.395,'teacher_agreement':0.15}
    got=quality_gate(None,results,{'available':True,'speedup':1.1});assert got['quality_pass'] and got['speed_pass']
    results['student_nll']=3.6;assert not quality_gate(None,results,{'available':True,'speedup':1.1})['quality_pass']


class MockBatchEncoding:
    """Mimics Hugging Face BatchEncoding without being an actual dict."""
    def __init__(self, ids):
        self.data = {"input_ids": ids}
    def __getitem__(self, key):
        return self.data[key]
    def to(self, device):
        return self   # This is why .to(...) does not make .shape usable!

class ChatTokenizer:
    eos_token_id = 2
    chat_template = "user/assistant"
    def __init__(self, form):
        self.form = form
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['return_tensors'] == 'pt'
        raw = torch.tensor([[42, 43]], dtype=torch.long)
        return {'tensor':raw, 'mapping':{'input_ids':raw},
                'batchencoding':MockBatchEncoding(raw),
                'list':[42,43]}[self.form]
    def decode(self, token_ids, skip_special_tokens=True):
        return ''.join(map(str, token_ids))

class ChatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p=torch.nn.Parameter(torch.ones(()))
    def forward(self, input_ids, use_cache, return_dict):
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.ndim==2
        logits=torch.zeros(input_ids.shape[0],input_ids.shape[1],64)
        logits[...,2]=20  # EOS immediately; no additional prompt attention cache
        return SimpleNamespace(logits=logits)

@pytest.mark.parametrize('form',['tensor','mapping','batchencoding','list'])
def test_template_returns_tensor_not_batchencoding(form):
    ids = render_chat_input(ChatTokenizer(form), 'Hi')
    assert isinstance(ids,torch.Tensor)
    torch.testing.assert_close(ids,torch.tensor([[42,43]]))

@pytest.mark.parametrize('form',['tensor','mapping','batchencoding','list'])
def test_chat_inference_works_with_template_forms(form):
    result=greedy_no_cache(ChatModel(), ChatTokenizer(form), 'Hello', max_new_tokens=2)
    assert result['prompt_tokens']==2
    assert result['generated_tokens']==0

def test_template_missing_ids_is_actionable():
    with pytest.raises(ValueError,match='missing input_ids'):
        from scripts.real_fast_eval import _as_input_ids
        _as_input_ids({'attention_mask':torch.ones((1,2))})

class FallbackTokenizer:
    def __call__(self, text, return_tensors):
        return MockBatchEncoding(torch.tensor([[9,10]]))

def test_no_chat_template_uses_batchencoding_as_well():
    ids=render_chat_input(FallbackTokenizer(),'Hello')
    torch.testing.assert_close(ids,torch.tensor([[9,10]]))
