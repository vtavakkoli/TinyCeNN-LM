"""PDelta2-Flash research ingredients for a stronger/faster single-layer replacement.

The module keeps the proven P-Delta2 recurrent core and tests lightweight ideas
suggested by recent hybrid efficient-attention models:
- function-preserving per-head output gating;
- a short causal value convolution (kernel 4 by default);
- compact content-indexed block summaries for long-range recall;
- FP16 persistent recurrent-memory storage with FP32 curvature.

The indexed path stores one K/V summary per completed block, not every token.
It is therefore a compact growing memory, while the P-Delta2 recurrent state
remains bounded. This is an independent TinyCeNN experiment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.pdelta2_features import PDelta2Core, PDeltaState


@dataclass
class FlashState:
    recurrent: PDeltaState
    conv_tail: Tensor | None = None
    block_keys: Tensor | None = None
    block_values: Tensor | None = None
    pending_keys: Tensor | None = None
    pending_values: Tensor | None = None


def causal_depthwise_value_conv(v: Tensor, weight: Tensor) -> Tensor:
    """Causal depthwise 1-D convolution over KV-head/value channels."""
    if weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError("weight must be [channels, 1, kernel]")
    b, h, t, d = v.shape
    channels = h * d
    if weight.shape[0] != channels:
        raise ValueError("weight channel count does not match values")
    x = v.transpose(1, 2).reshape(b, t, channels).transpose(1, 2)
    kernel = weight.shape[-1]
    x = F.pad(x, (kernel - 1, 0))
    y = F.conv1d(x, weight, groups=channels)
    return y.transpose(1, 2).reshape(b, t, h, d).transpose(1, 2)


def indexed_block_attention(q: Tensor, k: Tensor, v: Tensor, groups: int,
                            block_size: int = 16, topk: int = 4):
    """Attend to compact summaries of completed causal blocks.

    A query at token ``t`` may only see blocks whose final token is < ``t``.
    Each block contributes one mean-key and one mean-value summary, reducing
    long-range storage by roughly ``block_size`` versus tokenwise KV storage.
    """
    if block_size < 2 or topk < 1:
        raise ValueError("block_size must be >=2 and topk positive")
    b, h, t, d = q.shape
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    blocks = t // block_size
    if blocks == 0:
        return q.new_zeros(q.shape), torch.zeros((b, h, t, 1), dtype=torch.bool, device=q.device)

    usable = blocks * block_size
    bk = k[:, :, :usable].reshape(b, h, blocks, block_size, d).mean(dim=3)
    bv = v[:, :, :usable].reshape(b, h, blocks, block_size, d).mean(dim=3)
    bk = F.normalize(bk, dim=-1)
    qn = F.normalize(q, dim=-1)
    scores = torch.einsum("bhtd,bhnd->bhtn", qn, bk) / math.sqrt(d)

    positions = torch.arange(t, device=q.device)
    block_ends = torch.arange(blocks, device=q.device) * block_size + (block_size - 1)
    valid = block_ends[None, :] < positions[:, None]
    scores = scores.masked_fill(~valid[None, None], float("-inf"))

    ksel = min(topk, blocks)
    top_scores, index = torch.topk(scores, k=ksel, dim=-1)
    top_valid = torch.isfinite(top_scores)
    safe = top_scores.masked_fill(~top_valid, -1e4)
    weights = safe.softmax(dim=-1) * top_valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    bank = bv[:, :, None].expand(-1, -1, t, -1, -1)
    selected = torch.gather(bank, 3, index.unsqueeze(-1).expand(-1, -1, -1, -1, d))
    output = (weights.unsqueeze(-1) * selected).sum(dim=3)
    available = top_valid.any(dim=-1, keepdim=True)
    return output, available


def indexed_summary_attention(q: Tensor, block_keys: Tensor | None,
                              block_values: Tensor | None, groups: int, topk: int):
    """One-step indexed lookup against already-completed block summaries."""
    if block_keys is None or block_keys.shape[2] == 0:
        shape = (q.shape[0], q.shape[1], q.shape[2], q.shape[3])
        return q.new_zeros(shape), torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], 1), dtype=torch.bool, device=q.device
        )
    k = block_keys.float().repeat_interleave(groups, dim=1)
    v = block_values.float().repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhtd,bhnd->bhtn", F.normalize(q.float(), dim=-1),
                          F.normalize(k, dim=-1)) / math.sqrt(q.shape[-1])
    ksel = min(topk, scores.shape[-1])
    top_scores, index = torch.topk(scores, k=ksel, dim=-1)
    bank = v[:, :, None].expand(-1, -1, q.shape[2], -1, -1)
    selected = torch.gather(
        bank, 3, index.unsqueeze(-1).expand(-1, -1, -1, -1, q.shape[-1])
    )
    weights = top_scores.softmax(dim=-1)
    return (weights.unsqueeze(-1) * selected).sum(dim=3), torch.ones(
        (q.shape[0], q.shape[1], q.shape[2], 1), dtype=torch.bool, device=q.device
    )


class FlashPDelta2Layer(nn.Module):
    """P-Delta2 plus cheap gating, short causal convolution and indexed recall."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 96, chunk_size: int = 32,
                 output_gate: bool = False, conv_kernel: int = 1,
                 indexed_retrieval: bool = False, block_size: int = 16,
                 index_topk: int = 4, state_dtype: str = "fp16"):
        super().__init__()
        if state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.groups = num_heads // num_kv_heads
        self.chunk_size = chunk_size
        self.output_gate = bool(output_gate)
        self.conv_kernel = int(conv_kernel)
        self.indexed_retrieval = bool(indexed_retrieval)
        self.block_size = int(block_size)
        self.index_topk = int(index_topk)
        self.state_dtype = state_dtype

        self.core = PDelta2Core(
            num_heads, num_kv_heads, head_dim, feature_dim=feature_dim, chunk_size=chunk_size
        )
        if self.conv_kernel > 1:
            channels = num_kv_heads * head_dim
            kernel = torch.zeros(channels, 1, self.conv_kernel)
            kernel[:, 0, -1] = 1.0
            self.conv_weight = nn.Parameter(kernel)
        else:
            self.register_parameter("conv_weight", None)

        if self.output_gate:
            self.output_gate_w = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.output_gate_b = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_parameter("output_gate_w", None)
            self.register_parameter("output_gate_b", None)

        if self.indexed_retrieval:
            self.index_mix_w = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.index_mix_b = nn.Parameter(torch.full((num_heads,), -2.0))
        else:
            self.register_parameter("index_mix_w", None)
            self.register_parameter("index_mix_b", None)

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "chunk_size": self.chunk_size,
            "output_gate": self.output_gate,
            "conv_kernel": self.conv_kernel,
            "indexed_retrieval": self.indexed_retrieval,
            "block_size": self.block_size,
            "index_topk": self.index_topk,
            "state_dtype": self.state_dtype,
        }

    def _convolve_values(self, v: Tensor, tail: Tensor | None = None):
        v = v.float()
        if self.conv_weight is None:
            return v
        if tail is None:
            return causal_depthwise_value_conv(v, self.conv_weight)
        joined = torch.cat((tail.float(), v), dim=2)
        return causal_depthwise_value_conv(joined, self.conv_weight)[:, :, -v.shape[2]:]

    def _pack_recurrent_state(self, state: PDeltaState):
        memory = state.memory
        if self.state_dtype == "fp16":
            memory = memory.to(torch.float16)
        else:
            memory = memory.float()
        return PDeltaState(memory=memory, curvature=state.curvature.float())

    def _unpack_recurrent_state(self, state: PDeltaState | None):
        if state is None:
            return None
        return PDeltaState(memory=state.memory.float(), curvature=state.curvature.float())

    def _build_index_state(self, k: Tensor, v: Tensor):
        if not self.indexed_retrieval:
            return None, None, None, None
        complete = (k.shape[2] // self.block_size) * self.block_size
        # A completed block is only available to queries after its last token.
        bk = bv = None
        if complete:
            blocks = complete // self.block_size
            bk = k[:, :, :complete].reshape(
                k.shape[0], k.shape[1], blocks, self.block_size, k.shape[-1]
            ).mean(dim=3).to(torch.float16)
            bv = v[:, :, :complete].reshape(
                v.shape[0], v.shape[1], blocks, self.block_size, v.shape[-1]
            ).mean(dim=3).to(torch.float16)
        pk = k[:, :, complete:].float()
        pv = v[:, :, complete:].float()
        return bk, bv, pk, pv

    def _advance_index_state(self, state: FlashState, k: Tensor, v: Tensor):
        pk = k.float() if state.pending_keys is None else torch.cat((state.pending_keys.float(), k.float()), dim=2)
        pv = v.float() if state.pending_values is None else torch.cat((state.pending_values.float(), v.float()), dim=2)
        bk, bv = state.block_keys, state.block_values
        while pk.shape[2] >= self.block_size:
            new_k = pk[:, :, :self.block_size].mean(dim=2, keepdim=True).to(torch.float16)
            new_v = pv[:, :, :self.block_size].mean(dim=2, keepdim=True).to(torch.float16)
            bk = new_k if bk is None else torch.cat((bk, new_k), dim=2)
            bv = new_v if bv is None else torch.cat((bv, new_v), dim=2)
            pk, pv = pk[:, :, self.block_size:], pv[:, :, self.block_size:]
        return bk, bv, pk, pv

    def forward(self, q: Tensor, k: Tensor, v: Tensor, state: FlashState | None = None,
                return_state: bool = False, implementation: str = "chunk"):
        if implementation != "chunk":
            raise ValueError("PDelta2-Flash uses the chunk implementation")
        if state is not None and self.indexed_retrieval and q.shape[2] != 1:
            raise ValueError("stateful indexed retrieval currently supports one decode token at a time")

        tail = None if state is None else state.conv_tail
        conv_v = self._convolve_values(v, tail)
        recurrent_state = None if state is None else self._unpack_recurrent_state(state.recurrent)
        recurrent, new_recurrent = self.core(
            q, k, conv_v, state=recurrent_state, return_state=True
        )

        output = recurrent
        if self.output_gate:
            raw = (
                torch.einsum("bhtd,hd->bht", F.normalize(q.float(), dim=-1), self.output_gate_w)
                + self.output_gate_b[None, :, None]
            )
            # Exactly 1.0 at initialization; bounded to [0.75, 1.25].
            gain = 1.0 + 0.25 * torch.tanh(raw)
            output = output * gain.unsqueeze(-1)

        if self.indexed_retrieval:
            if state is None:
                indexed, available = indexed_block_attention(
                    q.float(), k.float(), conv_v, self.groups, self.block_size, self.index_topk
                )
            else:
                indexed, available = indexed_summary_attention(
                    q.float(), state.block_keys, state.block_values, self.groups, self.index_topk
                )
            mix = (
                torch.einsum("bhtd,hd->bht", F.normalize(q.float(), dim=-1), self.index_mix_w)
                + self.index_mix_b[None, :, None]
            ).sigmoid().unsqueeze(-1)
            mix = mix * available.to(mix.dtype)
            output = (1.0 - mix) * output + mix * indexed

        if not return_state:
            return output

        keep = self.conv_kernel - 1
        if keep:
            raw = v.float() if tail is None else torch.cat((tail.float(), v.float()), dim=2)
            new_tail = raw[:, :, -keep:].clone()
        else:
            new_tail = None

        if self.indexed_retrieval:
            if state is None:
                bk, bv, pk, pv = self._build_index_state(k.float(), conv_v)
            else:
                bk, bv, pk, pv = self._advance_index_state(state, k.float(), conv_v)
        else:
            bk = bv = pk = pv = None

        new_state = FlashState(
            recurrent=self._pack_recurrent_state(new_recurrent),
            conv_tail=new_tail,
            block_keys=bk,
            block_values=bv,
            pending_keys=pk,
            pending_values=pv,
        )
        return output, new_state

    def recurrent_state_bytes(self, batch_size: int = 1, context: int | None = None):
        memory_elements = self.num_kv_heads * self.feature_dim * self.head_dim
        curvature_elements = self.num_kv_heads * self.feature_dim
        memory_bytes = 2 if self.state_dtype == "fp16" else 4
        total = batch_size * (memory_elements * memory_bytes + curvature_elements * 4)
        if self.conv_kernel > 1:
            total += batch_size * (self.conv_kernel - 1) * self.num_kv_heads * self.head_dim * 4
        if self.indexed_retrieval and context is not None:
            blocks = context // self.block_size
            total += batch_size * 2 * blocks * self.num_kv_heads * self.head_dim * 2
            pending = context % self.block_size
            total += batch_size * 2 * pending * self.num_kv_heads * self.head_dim * 4
        return total

    def index_pairs_per_token(self, context: int):
        if not self.indexed_retrieval:
            return 0
        blocks = context // self.block_size
        return min(self.index_topk, blocks)
