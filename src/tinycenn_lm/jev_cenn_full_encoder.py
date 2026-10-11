"""Full-encoder Jev-CeNN progressive 3->1 compression helpers.

For an original 28-layer ModernBERT encoder the nonoverlapping plan is
(0,1,2), (3,4,5), ..., (24,25,26), with original layer 27 retained.
Nine independent superlayers replace 27 complete transformer blocks.
The intermediate two slots of each group become parameter-free bypasses.

This file extends the original two-window/CeNN superlayer, with an OPTIONAL
zero-initialized, low-rank output correction for three-block distillation.
It deliberately does not modify the original 15-17 experiment or model.
"""
from __future__ import annotations

import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tinycenn_lm.jev_cenn_pairwindow_superlayer import (
    JevCeNNPairWindowSuperLayer, BypassModernBertLayer, pair_reachable,
)


def group_plan(num_encoder_layers: int) -> tuple[tuple[int, int, int], ...]:
    """Group full original global/sliding/sliding triplets without overlap."""
    if num_encoder_layers < 3:
        raise ValueError("Encoder must have >=3 layers")
    return tuple((start, start + 1, start + 2)
                 for start in range(0, num_encoder_layers - 2, 3))


def preserved_tail(num_encoder_layers: int) -> tuple[int, ...]:
    covered={i for group in group_plan(num_encoder_layers) for i in group}
    return tuple(i for i in range(num_encoder_layers) if i not in covered)


class JevCeNNGroupSuperLayer(JevCeNNPairWindowSuperLayer):
    """Two-window+CeNN block, optionally augmented with low-rank correction.

    The correction is zero-initialized so the initial full forward exactly
    matches JevCeNNPairWindowSuperLayer; it can learn the residual of three
    complete teacher blocks without adding a third attention operation.
    """

    architecture="jev_cenn_full_encoder_group_v1"

    def __init__(self, teacher_global: nn.Module, teacher_sliding: nn.Module,
                 *, correction_rank: int = 96,
                 local_radius: int = 22, stride: int = 23,
                 dilated_radius: int = 22, max_len: int = 512):
        super().__init__(
            teacher_global,teacher_sliding,
            local_radius=local_radius,stride=stride,
            dilated_radius=dilated_radius,max_len=max_len,
        )
        hidden=self.num_heads*self.head_dim
        self.correction_rank=int(correction_rank)
        if self.correction_rank < 0:
            raise ValueError("correction_rank must be nonnegative")
        if correction_rank:
            self.correction_norm=nn.LayerNorm(hidden)
            self.correction_down=nn.Linear(hidden,correction_rank,bias=False)
            self.correction_up=nn.Linear(correction_rank,hidden,bias=False)
            # Preserve original two-window function at construction.
            nn.init.zeros_(self.correction_up.weight)

    def forward(self,hidden_states:Tensor,attention_mask=None,
                sliding_window_mask=None,position_ids=None,
                cu_seqlens=None,max_seqlen=None,
                output_attentions=False,**kwargs):
        result=super().forward(
            hidden_states,attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,output_attentions=output_attentions,
            **kwargs,
        )
        if not self.correction_rank:
            return result
        y=result[0]
        delta=self.correction_up(F.gelu(
            self.correction_down(self.correction_norm(y))))
        return (y+delta,)+result[1:]


def install_group(student: nn.Module, group: tuple[int,int,int],
                  superlayer: nn.Module) -> tuple[nn.Module,...]:
    """Insert ONE block and two complete-layer bypasses; return rollback state."""
    group=tuple(group)
    if len(group)!=3 or group!=(group[0],group[0]+1,group[0]+2):
        raise ValueError("Expected consecutive 3-layer group")
    layers=student.encoder.layers
    if min(group)<0 or max(group)>=len(layers):
        raise IndexError(f"Invalid group {group}")
    if any(isinstance(layers[i],BypassModernBertLayer) for i in group):
        raise ValueError(f"Layer group {group} already compressed")
    saved=tuple(layers[i] for i in group)
    layers[group[0]]=superlayer
    layers[group[1]]=BypassModernBertLayer()
    layers[group[2]]=BypassModernBertLayer()
    return saved


def restore_group(student:nn.Module,group:tuple[int,int,int],original):
    if len(original)!=3:
        raise ValueError("Need exactly 3 original modules")
    for index,module in zip(group,original):
        student.encoder.layers[index]=module


def installed_summary(student:nn.Module,groups):
    installed=tuple(tuple(g) for g in groups)
    start={g[0] for g in installed}
    bypass={g[1] for g in installed}|{g[2] for g in installed}
    layers=student.encoder.layers
    if any(not isinstance(layers[i],JevCeNNGroupSuperLayer) for i in start):
        raise RuntimeError("Missing compressed superlayer")
    if any(not isinstance(layers[i],BypassModernBertLayer) for i in bypass):
        raise RuntimeError("Missing expected bypass")
    return {
        "original_slots":len(layers),
        "compressed_groups":len(installed),
        "active_encoder_blocks":len(layers)-2*len(installed),
        "replaced_complete_layers":3*len(installed),
        "bypassed_complete_layers":2*len(installed),
        "kept_original_layers":[i for i in range(len(layers))
                                if i not in start and i not in bypass],
    }
