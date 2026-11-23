"""Reproducible CPU learnability and diffuse-attention stress tests.

All data are synthetic, all records seeded. No model downloads or GPU claims.
"""
from __future__ import annotations
import random
import math
import torch
import torch.nn.functional as F
from torch import nn
from .competitive_cenn import CCMConfig, CompetitiveCeNN


def _sample(batch,slots,ids,valdim):
    key_ids=torch.rand(batch,ids).argsort(-1)[:,:slots]
    keys=F.one_hot(key_ids,ids).float()
    values=torch.randn(batch,slots,valdim)
    picked=torch.randint(slots,(batch,))
    query_key=keys[torch.arange(batch),picked]
    memory=torch.cat((keys,values),-1)
    query=torch.cat((query_key,torch.zeros(batch,valdim)),-1)
    p=torch.softmax(32*torch.einsum('bd,bnd->bn',query_key,keys),-1)
    target=(p[...,None]*values).sum(1)
    return memory,query,target


def run_symbolic_retrieval(*,steps=400,batch=128,slots=8,ids=16,valdim=4,seed=2031):
    """End-to-end learn to select independently varied values from symbolic IDs.

    Attention teacher appears only in target construction. Experimental model
    uses a shared feature encoder and local binary-cell sigmoid competitions.
    """
    if slots>ids:raise ValueError('need a distinct symbolic ID for each memory cell')
    torch.manual_seed(seed);random.seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    model=CompetitiveCeNN(CCMConfig(hidden_size=ids+valdim,slots=slots,tag_width=24,cell_width=24,
                                   conductance_temperature=20.0))
    head=nn.Linear(ids+valdim,valdim,bias=False)
    opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=0.004)
    history=[]
    points=set((1,50,100,200,400,600,steps))
    for step in range(1,steps+1):
        memory,query,y=_sample(batch,slots,ids,valdim)
        guess=head(model.out_proj(model.read(query,model.tag_encoder(memory),model.content_encoder(memory))))
        loss=F.mse_loss(guess,y)
        opt.zero_grad();loss.backward();opt.step()
        if step in points:
            with torch.no_grad():
                mem,q,true=_sample(2048,slots,ids,valdim)
                out=head(model.out_proj(model.read(q,model.tag_encoder(mem),model.content_encoder(mem))))
                mse=F.mse_loss(out,true).item()
                uniform=F.mse_loss(mem[:,:,-valdim:].mean(1),true).item()
                history.append({'step':step,'test_mse':mse,'uniform_mse':uniform,
                                'relative_error':math.sqrt(mse/(true.square().mean().item()+1e-12))})
    return {'task':'sharp softmax teacher of 8-32 random symbolic associations',
            'slots':slots,'ids':ids,'steps':steps,'seed':seed,'history':history,
            'limitations':'One-hot symbolic keys, random values, small training task; no natural-language teacher, no model-level NLL or GPU validation.'}


@torch.no_grad()
def run_diffuse_gap(seed=2031,batch=2048,dim=32,valdim=8):
    """Compare ideal hard winner to softmax on continuous random embeddings."""
    torch.manual_seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    rows=[]
    for slots in (8,16,64):
        q=F.normalize(torch.randn(batch,dim),dim=-1)
        k=F.normalize(torch.randn(batch,slots,dim),dim=-1)
        v=torch.randn(batch,slots,valdim)
        scores=(q[:,None]*k).sum(-1)*(dim**0.5)
        for beta in (0.5,1,2,4,8):
            probs=torch.softmax(beta*scores,-1)
            exact=(probs[...,None]*v).sum(1)
            winner=v[torch.arange(batch),scores.argmax(-1)]
            mse=F.mse_loss(winner,exact).item()
            rows.append({'slots':slots,'temperature_multiplier':beta,
                         'average_max_probability':probs.max(-1).values.mean().item(),
                         'hard_routing_mse':mse,
                         'hard_routing_relative_mse':mse/(exact.square().mean().item()+1e-12)})
    return rows
