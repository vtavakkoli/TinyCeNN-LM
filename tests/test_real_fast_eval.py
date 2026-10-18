import pytest,torch
from types import SimpleNamespace
from scripts.real_fast_eval import prepare_arc_example,score_chat_response,score_continuations,quality_gate,arc_easy_eval
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
