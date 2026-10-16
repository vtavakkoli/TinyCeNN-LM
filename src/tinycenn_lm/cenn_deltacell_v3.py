"""CeNN-DeltaCell V3: chunkwise gated-delta memory with cellular state coupling.

Experimental research reference (NOT a fused GPU kernel or a pretrained model).
Native Qwen3.5 attention projections may be reused via Qwen35DeltaCellAttention.
The core supports strict causal streaming, including arbitrary call boundaries.

The delta update is inherited from ``research_layers.delta_recurrence``; the
new operator applies a convex, content-adaptive spatial transition to the
persistent feature-memory array after every GLOBAL chunk boundary.  This is
not a claim of mathematical equivalence to Qwen's softmax attention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .research_layers import delta_recurrence, local_window_attention

CELL_MODES = ("none", "fixed", "adaptive")


@dataclass
class DeltaCellV3State:
    memory: Tensor
    position: int = 0
    keys: Tensor | None = None
    values: Tensor | None = None

    @property
    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in
                   (self.memory, self.keys, self.values) if v is not None)


class CeNNDeltaCellV3(nn.Module):
    """GPU-oriented, causal, GQA-compatible delta memory with cellular diffusion.

    q: [B,Hq,T,D], k/v: [B,Hkv,T,D], output: [B,Hq,T,D].
    The convolution is across the **F memory-feature cells**, NOT token time.
    Exactly one memory transition is applied after every ``chunk_size`` global
    tokens; streaming on arbitrary subsegments gives identical results.

    ``mode='none'`` is the exact delta-only ablation; ``fixed`` and
    ``adaptive`` add respectively fixed and content-adaptive neighborhood
    coupling while reusing all other parameters and the same delta rule.
    ``local_window=0`` removes local retrieval; positive values enable a
    bounded causal softmax window.  There is NO support for padded batches.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 32, chunk_size: int = 32,
                 mode: str = "adaptive", local_window: int = 32):
        super().__init__()
        if mode not in CELL_MODES:
            raise ValueError(f"Invalid cellular mode {mode!r}; choose {CELL_MODES}")
        if min(num_heads, num_kv_heads, head_dim, feature_dim, chunk_size) < 1:
            raise ValueError("Heads, dimensions and chunk_size must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("Query heads must be a multiple of KV heads")
        if not 1 <= chunk_size <= 32:
            raise ValueError("Reference triangular solve requires chunk_size <= 32")
        if local_window < 0:
            raise ValueError("local_window cannot be negative")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim, self.feature_dim = head_dim, feature_dim
        self.chunk_size, self.mode = chunk_size, mode
        self.local_window = local_window
        self.groups = num_heads // num_kv_heads
        self.wq = nn.Parameter(torch.empty(num_heads, feature_dim, head_dim))
        self.wk = nn.Parameter(torch.empty(num_kv_heads, feature_dim, head_dim))
        for p in (self.wq, self.wk):
            for head in p:
                nn.init.orthogonal_(head)
        self.forget_logits = nn.Parameter(torch.full((num_kv_heads, feature_dim), -3.2))
        self.erase_logits = nn.Parameter(torch.full((num_kv_heads, feature_dim), -1.0))
        self.write_logits = nn.Parameter(torch.full((num_kv_heads, head_dim), 0.5))
        self.log_gain = nn.Parameter(torch.zeros(num_heads))
        self.diffusion_logits = nn.Parameter(torch.full((num_kv_heads,), -2.5))
        self.cell_logits = nn.Parameter(torch.tensor([[0., 1., 0.]]).repeat(num_kv_heads, 1))
        # Zero-initialized readout makes the adaptive model initially equivalent
        # to a static cellular operator, improving reproducibility of ablations.
        self.cell_router = nn.Linear(feature_dim, 3, bias=False)
        nn.init.zeros_(self.cell_router.weight)
        if local_window > 0:
            self.local_logits = nn.Parameter(torch.full((num_heads,), -1.5))

    @property
    def config(self):
        return dict(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, feature_dim=self.feature_dim,
                    chunk_size=self.chunk_size, mode=self.mode,
                    local_window=self.local_window)

    def initial_state(self, b: int, device: torch.device) -> DeltaCellV3State:
        return DeltaCellV3State(torch.zeros(
            b, self.num_kv_heads, self.feature_dim, self.head_dim,
            device=device, dtype=torch.float32,
        ))

    def diffuse(self, memory: Tensor) -> Tensor:
        """Apply bounded, positive neighborhood mixing across memory cells."""
        if self.mode == "none":
            return memory
        if self.mode == "fixed":
            weights = memory.new_tensor((.25, .5, .25))[None, None, :]
        else:
            descriptor = memory.mean(dim=-1)  # [B,Hkv,F], content dependent
            weights = (self.cell_logits[None] + self.cell_router(descriptor)).softmax(-1)
        zero = torch.zeros_like(memory[:, :, :1])
        left = torch.cat((zero, memory[:, :, :-1]), dim=2)
        right = torch.cat((memory[:, :, 1:], zero), dim=2)
        mixed = (weights[..., 0, None, None] * left
                 + weights[..., 1, None, None] * memory
                 + weights[..., 2, None, None] * right)
        rate = self.diffusion_logits.sigmoid()[None, :, None, None]
        return memory + rate * (mixed - memory)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: DeltaCellV3State | None = None,
                return_state: bool = False):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Q,K,V must be rank-4 [batch,heads,time,dim]")
        b, h, t, d = q.shape
        if (h != self.num_heads or k.shape != (b, self.num_kv_heads, t, d)
                or v.shape != k.shape or d != self.head_dim or t < 1):
            raise ValueError("Q,K,V dimensions mismatch")
        if state is None:
            state = self.initial_state(b, q.device)
        if (state.memory.shape != (b, self.num_kv_heads, self.feature_dim, self.head_dim)
                or state.memory.device != q.device or state.memory.dtype != torch.float32
                or state.position < 0):
            raise ValueError("Invalid state shape/device/dtype/position")
        # Keep triangular solves and recurrent state in fp32 under CUDA autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            qf = F.normalize(torch.einsum("bhtd,hfd->bhtf", F.normalize(q, dim=-1),
                                           self.wq.float()), dim=-1)
            kf = F.normalize(torch.einsum("bhtd,hfd->bhtf", F.normalize(k, dim=-1),
                                           self.wk.float()), dim=-1)
            decay = (-0.25 * self.forget_logits.sigmoid())[None, :, None, :].expand(b, -1, t, -1)
            erase = kf * self.erase_logits.sigmoid()[None, :, None, :]
            z = v * self.write_logits.sigmoid()[None, :, None, :]
            outputs = []
            memory = state.memory
            position = state.position
            start = 0
            while start < t:
                until_boundary = self.chunk_size - position % self.chunk_size
                stop = min(t, start + until_boundary)
                out, memory = delta_recurrence(
                    qf[:, :, start:stop], kf[:, :, start:stop], z[:, :, start:stop],
                    erase[:, :, start:stop], decay[:, :, start:stop],
                    memory, self.groups, chunk_size=self.chunk_size,
                )
                outputs.append(out)
                position += stop - start
                if position % self.chunk_size == 0:
                    memory = self.diffuse(memory)
                start = stop
            y = torch.cat(outputs, dim=2)
            y = y * self.log_gain.clamp(-4., 4.).exp()[None, :, None, None]
            next_keys = next_values = None
            if self.local_window:
                keys = k if state.keys is None else torch.cat((state.keys, k), dim=2)
                values = v if state.values is None else torch.cat((state.values, v), dim=2)
                if (keys.shape != values.shape or keys.shape[2] < t
                        or keys.shape[2] - t > self.local_window - 1):
                    raise ValueError("Invalid local cache")
                local = local_window_attention(q, keys, values,
                                               self.local_window, self.groups)
                mix = self.local_logits.sigmoid()[None, :, None, None]
                y = (1. - mix) * y + mix * local
                keep = self.local_window - 1
                next_keys = keys[:, :, -keep:].clone() if keep else keys[:, :, :0].clone()
                next_values = values[:, :, -keep:].clone() if keep else values[:, :, :0].clone()
            next_state = DeltaCellV3State(memory, position, next_keys, next_values)
            return (y, next_state) if return_state else y

    def recurrent_state_bytes(self, batch_size: int = 1):
        elements = self.num_kv_heads * self.feature_dim * self.head_dim
        elements += 2 * (self.local_window - 1) * self.num_kv_heads * self.head_dim if self.local_window else 0
        return int(4 * batch_size * elements)


class Qwen35DeltaCellAttention(nn.Module):
    """Drop-in replacement for a Qwen3.5 FULL-attention mixer (prefill only).

    Reuses frozen native Q/K/V/output projections, QK norms, RoPE, and gate.
    The original decoder's MLP, residuals and norms are left untouched.
    Does not support cache or padding. Supply the native full-attention module.
    """

    def __init__(self, original: nn.Module, core: CeNNDeltaCellV3):
        super().__init__()
        self.original = original
        self.core = core
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.layer_idx = original.layer_idx
        self.config = original.config
        self.scaling = original.scaling
        self.is_causal = True
        self.attention_dropout = 0.
        if core.num_heads != original.q_proj.out_features // (2 * self.head_dim):
            raise ValueError("core query heads must match pretrained projections")
        if core.num_kv_heads != original.k_proj.out_features // self.head_dim:
            raise ValueError("core KV heads must match pretrained projections")

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise RuntimeError("Experimental DeltaCell V3 supports use_cache=False only")
        if position_embeddings is None:
            raise ValueError("Qwen rotary position embeddings are required")
        if attention_mask is not None:
            # The notebook uses strictly unpadded, contiguous text. Do not
            # silently reinterpret arbitrary padding / sliding masks.
            if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (
                    hidden_states.shape[1], hidden_states.shape[1]):
                raise ValueError("Only unpadded full-prefix causal masks are supported")
            t = hidden_states.shape[1]
            below = torch.ones((t, t), dtype=torch.bool, device=hidden_states.device).tril()
            if not bool((attention_mask.masked_select(below[None, None]) == 0).all()):
                raise ValueError("Padded/modified attention masks unsupported")
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        b, t, _ = hidden_states.shape
        heads_shape = (b, t, -1, self.head_dim)
        q, gate = torch.chunk(self.original.q_proj(hidden_states).reshape(
            b, t, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(b, t, -1)
        q = self.original.q_norm(q.reshape(*heads_shape)).transpose(1, 2)
        k = self.original.k_norm(self.original.k_proj(hidden_states).reshape(
            *heads_shape)).transpose(1, 2)
        v = self.original.v_proj(hidden_states).reshape(*heads_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        y = self.core(q, k, v).to(hidden_states.dtype)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.original.o_proj(y * gate.sigmoid()), None
