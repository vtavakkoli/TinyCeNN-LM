"""Competitive Cellular Memory (CCM): a single, standalone local-routing cell.

This is a binary nearest-neighbor tournament with content-based conductance,
not an LSTM/DeltaNet/Transformer attention module. BUT it performs content-
selective retrieval and is therefore ATTENTION-LIKE IN FUNCTION. It is *not*
a proof of novel attention-equivalent efficiency and uses a finite memory tape.
"""
from dataclasses import dataclass
import torch
from torch import nn, Tensor
import torch.nn.functional as F

@dataclass(frozen=True)
class CCMConfig:
    hidden_size: int=64
    slots: int=16
    tag_width: int=32
    cell_width: int=64
    conductance_temperature: float=20.0

    def validate(self):
        if self.slots<2 or self.slots & (self.slots-1):
            raise ValueError('slots must be a power of two')
        if min(self.hidden_size,self.tag_width,self.cell_width)<=0 or self.conductance_temperature<=0:
            raise ValueError('invalid sizes')

@dataclass
class CCMState:
    tags: Tensor
    values: Tensor
    position: int

class CompetitiveCeNN(nn.Module):
    def __init__(self,config:CCMConfig):
        super().__init__();config.validate();self.config=config
        d=config.hidden_size
        self.tag_encoder=nn.Linear(d,config.tag_width,bias=False) # SHARED for write/query
        self.content_encoder=nn.Linear(d,config.cell_width,bias=False)
        self.out_proj=nn.Linear(config.cell_width,d,bias=False)

    def init_state(self,b,device=None):
        t=self.tag_encoder.weight
        return CCMState(tags=torch.zeros(b,self.config.slots,self.config.tag_width,device=device or t.device,dtype=t.dtype),
                        values=torch.zeros(b,self.config.slots,self.config.cell_width,device=device or t.device,dtype=t.dtype),position=0)

    def read(self,query_x:Tensor,stored_tags:Tensor,stored_values:Tensor, valid:Tensor|None=None)->Tensor:
        """Logarithmic cell-interaction depth; O(slots) parallel work per query.
        Each node compares only its two immediate neighbors' conductances.
        No token-to-token attention matrix, softmax normalization or K/V cache.
        The argmax limiting case is a hard content-based selector.
        """
        b,n,c=stored_values.shape
        if n!=self.config.slots or stored_tags.shape!=(b,n,self.config.tag_width):
            raise ValueError('wrong memory shape')
        q=F.normalize(self.tag_encoder(query_x.float()),dim=-1)
        k=F.normalize(stored_tags.float(),dim=-1)
        # Local tag compatibility. The same feature extractor is used for all
        # cells, and a cellular tournament propagates the winning potential.
        score=(q[:,None,:]*k).sum(-1)
        if valid is not None:score=score.masked_fill(~valid,-1e4)
        value=stored_values
        temp=self.config.conductance_temperature
        while value.shape[1]>1:
            left,right=score[:,0::2],score[:,1::2]
            conductance=torch.sigmoid(temp*(right-left)).unsqueeze(-1)
            value=value[:,0::2]*(1-conductance)+value[:,1::2]*conductance
            score=torch.maximum(left,right)
        return value[:,0]

    def forward(self,hidden_states:Tensor,state:CCMState|None=None,*,return_state:bool=False):
        if hidden_states.ndim!=3 or hidden_states.shape[-1]!=self.config.hidden_size:
            raise ValueError('expected [B,T,hidden]')
        batch,T,_=hidden_states.shape
        if state is None:state=self.init_state(batch,hidden_states.device)
        out=[]
        for t in range(T):
            x=hidden_states[:,t].float()
            i=state.position%self.config.slots
            new_tag=self.tag_encoder(x)
            new_val=self.content_encoder(x)
            tags=torch.cat((state.tags[:,:i],new_tag[:,None],state.tags[:,i+1:]),1)
            values=torch.cat((state.values[:,:i],new_val[:,None],state.values[:,i+1:]),1)
            # Explicitly exclude never-written cells in fresh-cache startup.
            valid=torch.arange(self.config.slots,device=x.device)<min(state.position+1,self.config.slots)
            # After wraparound all slots are valid; before wraparound they're written in order.
            result=self.out_proj(self.read(x,tags,values,valid[None].expand(batch,-1)))
            out.append(result.to(hidden_states.dtype))
            state=CCMState(tags,values,state.position+1)
        y=torch.stack(out,1)
        return (y,state) if return_state else y
