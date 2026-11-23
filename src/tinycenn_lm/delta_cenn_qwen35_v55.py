"""V5.5 chunkwise Delta-CeNN for Qwen3.5 linear-attention layers.

The chunked gated-delta recurrence is Qwen/Transformers' existing backend;
only the trainable cellular STATE coupling between chunks is novel here.
This is NOT an independent replacement for the gated-delta recurrence, and
any speedup must be benchmarked. Cache-based generation is unsupported.
"""
from __future__ import annotations

import torch
from torch import Tensor
from .delta_cenn_qwen35 import DeltaCeNNRule, DeltaCeNNState, Qwen35DeltaCeNNLinear


def qwen_chunk_backend(q: Tensor,k: Tensor,v: Tensor,g: Tensor,beta: Tensor,
                       initial_state: Tensor | None, chunk_size: int):
    """Use Qwen3.5's pinned backend with final recurrent state output."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    return torch_chunk_gated_delta_rule(q,k,v,g=g,beta=beta,
       initial_state=initial_state,output_final_state=True,
       use_qk_l2norm_in_kernel=True,chunk_size=chunk_size)


class ChunkCellDeltaCeNNRule(DeltaCeNNRule):
    """Chunk-level cellular diffusion on the Qwen gated-delta state.

    Uses ONE native chunk recurrence for a full sequence when cells=False;
    otherwise each cell_interval is a recurrent segment and cellular mixing
    runs after its last token. The saved state follows token alignment.
    Trainable output calibration is inherited from DeltaCeNNRule.
    """
    engine = 'chunk_cell_v55'
    def __init__(self,heads: int,key_dim: int,value_dim: int,
                 feature_dim: int | None=None,cells: bool=True,
                 cell_interval: int=64,backend_chunk_size: int=64):
        super().__init__(heads,key_dim,value_dim,feature_dim,cells,cell_interval)
        if self.feature_dim != self.key_dim:
            raise ValueError('V5.5 chunk backend requires full-rank key dimension; compressed mode needs a custom kernel')
        if not 1<=backend_chunk_size<=64 or backend_chunk_size&(backend_chunk_size-1):
            raise ValueError('backend_chunk_size must be a power of two in [1,64]')
        self.backend_chunk_size=int(backend_chunk_size)

    @property
    def config(self):
        return dict(heads=self.heads,key_dim=self.key_dim,value_dim=self.value_dim,
                    feature_dim=self.feature_dim,cells=self.cells,
                    cell_interval=self.cell_interval,backend_chunk_size=self.backend_chunk_size,
                    engine=self.engine)

    def forward(self,q:Tensor,k:Tensor,v:Tensor,g:Tensor,beta:Tensor,
                state:DeltaCeNNState|None=None,return_state:bool=False):
        if (q.ndim!=4 or q.shape!=k.shape or v.shape[:3]!=q.shape[:3]
            or q.shape[2:]!=(self.heads,self.key_dim)
            or v.shape[-1]!=self.value_dim or q.shape[1]<1
            or g.shape!=q.shape[:3] or beta.shape!=q.shape[:3]):
            raise ValueError('V5.5 q/k/v/g/beta shape mismatch')
        if state is None:state=self.initial_state(q.shape[0],q.device)
        if (state.memory.shape!=(q.shape[0],self.heads,self.key_dim,self.value_dim)
            or state.memory.dtype!=torch.float32 or state.memory.device!=q.device
            or state.position<0):
            raise ValueError('V5.5 invalid state')
        result=[];memory=state.memory;pos=state.position;start=0
        with torch.autocast(device_type=q.device.type,enabled=False):
            while start<q.shape[1]:
                # Avoid needless per-segment launches if cells are disabled.
                length=(self.cell_interval-pos%self.cell_interval) if self.cells else (q.shape[1]-start)
                end=min(q.shape[1],start+length)
                output,memory=qwen_chunk_backend(q[:,start:end],k[:,start:end],
                    v[:,start:end],g[:,start:end],beta[:,start:end],memory,
                    self.backend_chunk_size)
                if memory is None:
                    raise RuntimeError('Chunk backend must return its recurrent state')
                memory=memory.float()
                result.append(output.float())
                pos+=end-start
                if self.cells and pos%self.cell_interval==0:
                    memory=self._diffuse(memory)
                start=end
            y=torch.cat(result,dim=1)
            y=(y*self.output_log_gain.float().clamp(-.3,.3).exp()[None,None,:,None]
                +self.output_bias.float()[None,None])
        next_state=DeltaCeNNState(memory,pos)
        return (y,next_state) if return_state else y


class Qwen35DeltaCeNNChunkLinear(Qwen35DeltaCeNNLinear):
    """Original Qwen projections + chunked gated-delta + learned CeNN coupling."""
    def __init__(self,original,feature_dim=None,cells=True,cell_interval=64,
                 backend_chunk_size=64):
        super().__init__(original,feature_dim=feature_dim,cells=cells,
                         cell_interval=cell_interval)
        self.core=ChunkCellDeltaCeNNRule(original.num_v_heads,
            original.head_k_dim,original.head_v_dim,feature_dim,cells,
            cell_interval,backend_chunk_size)
