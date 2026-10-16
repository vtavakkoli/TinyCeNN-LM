"""CeNN-MemoryGraph V4: learned sparse memory-cell coupling (research reference).

Subclass CeNNDeltaCellV3 to reuse exact Qwen projection compatibility and the
validated fp32 chunkwise delta recurrence. V4 changes only global chunk-boundary
cell-to-cell memory transitions and optional tokenwise local/global gating.

This is an eager PyTorch REFERENCE (not a fused kernel). Top-k edge *choices*
are discrete; gradient flows through selected edge scores/weights. Compare
against BOTH V3 mode='none' + Local32 and mode='adaptive' + Local32.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cenn_deltacell_v3 import CeNNDeltaCellV3, DeltaCellV3State
from .research_layers import delta_recurrence, local_window_attention

GRAPH_MODES = ('none', 'learned', 'shuffled')


class CeNNMemoryGraphV4(CeNNDeltaCellV3):
    """Top-k semantic graph between persistent feature-memory cells.

    Inputs Q/K/V [B,Hq,T,D], [B,Hkv,T,D], [B,Hkv,T,D].
    Every ``chunk_size`` *global* tokens, cell f receives a convex mixture
    of its own old state and top-k selected source cells. The score combines
    learned node embeddings and the semantic geometry of pretrained-adapted
    key feature maps. Content-dependent edge weights are optional.

    The ``shuffled`` variant applies a fixed random permutation to learned
    neighbor *sources* (with identical parameters/compute) as a topology-null
    control. These are feature memory cells, NOT token positions.

    Local32 is optionally combined with a per-token query-dependent router.
    Uses explicit state.position so streaming segmentation is invariant.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 feature_dim: int = 32, chunk_size: int = 32,
                 local_window: int = 32, graph_mode: str = 'learned',
                 top_k: int = 4, edge_dim: int = 12,
                 content_adaptive: bool = True,
                 query_router: bool = True,
                 shuffle_seed: int = 2031):
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f'graph_mode must be one of {GRAPH_MODES}')
        if not 1 <= top_k < feature_dim:
            raise ValueError('top_k must lie in [1, feature_dim - 1]')
        if edge_dim < 1:
            raise ValueError('edge_dim must be positive')
        if query_router and local_window == 0:
            raise ValueError('query_router requires local_window > 0')
        super().__init__(num_heads, num_kv_heads, head_dim, feature_dim,
                         chunk_size, mode='none', local_window=local_window)
        self.graph_mode = graph_mode
        self.top_k = top_k
        self.edge_dim = edge_dim
        self.content_adaptive = bool(content_adaptive)
        self.query_router = bool(query_router)
        self.edge_q = nn.Parameter(torch.randn(num_kv_heads, feature_dim, edge_dim) * .08)
        self.edge_k = nn.Parameter(torch.randn(num_kv_heads, feature_dim, edge_dim) * .08)
        # Positive prior on learned key-feature similarity avoids arbitrary
        # 'neighboring' indices in a semantically unordered feature basis.
        self.log_basis_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_content_scale = nn.Parameter(torch.tensor(-.7))
        self.graph_strength_logits = nn.Parameter(torch.full((num_kv_heads,), -1.75))
        self.cell_rate = nn.Linear(head_dim, 1, bias=False)
        nn.init.zeros_(self.cell_rate.weight)
        # Token-specific local-vs-memory routing, initialized to same behavior
        # as the original fixed-per-head V3 local_logits.
        self.query_local_router = nn.Parameter(torch.zeros(num_heads, feature_dim))
        perms = []
        generator = torch.Generator(device='cpu').manual_seed(int(shuffle_seed))
        for _ in range(num_kv_heads):
            perm = torch.randperm(feature_dim, generator=generator)
            if bool(torch.equal(perm, torch.arange(feature_dim))):
                perm = perm.roll(1)
            perms.append(perm)
        self.register_buffer('shuffle_perm', torch.stack(perms), persistent=True)

    @property
    def config(self):
        return dict(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, feature_dim=self.feature_dim,
                    chunk_size=self.chunk_size, local_window=self.local_window,
                    graph_mode=self.graph_mode, top_k=self.top_k,
                    edge_dim=self.edge_dim, content_adaptive=self.content_adaptive,
                    query_router=self.query_router)

    def _semantic_scores(self) -> Tensor:
        basis = F.normalize(self.wk.float(), dim=-1)  # [Hkv,F,D]
        geometry = basis @ basis.transpose(-1, -2)
        embeddings = (self.edge_q.float() @ self.edge_k.float().transpose(-1, -2))
        embeddings = embeddings / math.sqrt(self.edge_dim)
        score = self.log_basis_scale.float().exp().clamp(max=10) * geometry + embeddings
        # Never select self as a 'remote' neighbor.
        eye = torch.eye(self.feature_dim, device=score.device, dtype=torch.bool)
        return score.masked_fill(eye.unsqueeze(0), torch.finfo(score.dtype).min)

    def selected_neighbors(self) -> Tensor:
        index = self._semantic_scores().topk(self.top_k, dim=-1).indices
        if self.graph_mode == 'shuffled':
            index = self.shuffle_perm.to(index.device).gather(-1, index.flatten(1)).reshape_as(index)
        return index

    def diffuse(self, memory: Tensor) -> Tensor:
        if self.graph_mode == 'none':
            return memory
        b, heads, cells, d = memory.shape
        assert heads == self.num_kv_heads and cells == self.feature_dim
        scores = self._semantic_scores()
        selected_score, chosen = scores.topk(self.top_k, dim=-1)  # [H,F,K]
        if self.graph_mode == 'shuffled':
            chosen = self.shuffle_perm.gather(1, chosen.reshape(heads, -1)).reshape_as(chosen)
        index = chosen[None, :, :, :, None].expand(b, -1, -1, -1, d)
        # expand() is a zero-stride VIEW: only the selected [B,H,F,K,D]
        # neighborhood is materialized, not dense all-to-all memory traffic.
        neighbors = torch.gather(memory.unsqueeze(2).expand(-1, -1, cells, -1, -1),
                                 dim=3, index=index)
        logits = selected_score.unsqueeze(0).expand(b, -1, -1, -1)
        if self.content_adaptive:
            norm_self = F.normalize(memory, dim=-1).unsqueeze(3)
            norm_neighbors = F.normalize(neighbors, dim=-1)
            content = (norm_self * norm_neighbors).sum(-1)
            logits = logits + self.log_content_scale.float().exp().clamp(max=5.) * content
        weights = logits.softmax(-1)
        mixed = (weights[..., None] * neighbors).sum(dim=3)
        # Rate is bounded in (0,1) for a convex memory transition.
        rate = (self.graph_strength_logits[None, :, None, None]
                + self.cell_rate(memory).float()).sigmoid()
        return memory + rate * (mixed - memory)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                state: DeltaCellV3State | None = None,
                return_state: bool = False):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError('Q,K,V must be [batch,heads,time,dim]')
        b, h, t, d = q.shape
        if (h != self.num_heads or k.shape != (b, self.num_kv_heads, t, d)
                or v.shape != k.shape or d != self.head_dim or t < 1):
            raise ValueError('Q/K/V shape mismatch')
        if state is None:
            state = self.initial_state(b, q.device)
        if (state.memory.shape != (b, self.num_kv_heads, self.feature_dim, self.head_dim)
                or state.memory.device != q.device or state.memory.dtype != torch.float32
                or state.position < 0):
            raise ValueError('Invalid recurrent state')
        # The projected Q/K/V can originate from BF16/FP16 pretrained Qwen.
        # Keep the recurrent triangular solves, graph topology and gates FP32.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            qf = F.normalize(torch.einsum('bhtd,hfd->bhtf', F.normalize(q, dim=-1),
                                           self.wq.float()), dim=-1)
            kf = F.normalize(torch.einsum('bhtd,hfd->bhtf', F.normalize(k, dim=-1),
                                           self.wk.float()), dim=-1)
            decay = (-.25 * self.forget_logits.sigmoid())[None, :, None, :].expand(b, -1, t, -1)
            erase = kf * self.erase_logits.sigmoid()[None, :, None, :]
            z = v * self.write_logits.sigmoid()[None, :, None, :]
            memory, position = state.memory, int(state.position)
            pieces, start = [], 0
            while start < t:
                stop = min(t, start + self.chunk_size - position % self.chunk_size)
                out, memory = delta_recurrence(
                    qf[:, :, start:stop], kf[:, :, start:stop], z[:, :, start:stop],
                    erase[:, :, start:stop], decay[:, :, start:stop], memory,
                    self.groups, chunk_size=self.chunk_size)
                pieces.append(out)
                position += stop - start
                if position % self.chunk_size == 0:
                    memory = self.diffuse(memory)
                start = stop
            y = torch.cat(pieces, dim=2)
            y = y * self.log_gain.clamp(-4., 4.).exp()[None, :, None, None]
            next_keys = next_values = None
            if self.local_window:
                keys = k if state.keys is None else torch.cat((state.keys, k), dim=2)
                values = v if state.values is None else torch.cat((state.values, v), dim=2)
                if (keys.shape != values.shape or keys.shape[2] < t
                        or keys.shape[2] - t > self.local_window - 1):
                    raise ValueError('Invalid bounded local-cache state')
                local = local_window_attention(q, keys, values, self.local_window,
                                               self.groups)
                gate_logits = self.local_logits[None, :, None]
                if self.query_router:
                    gate_logits = (gate_logits + torch.einsum('bhtf,hf->bht', qf,
                                                             self.query_local_router.float())
                                   / math.sqrt(self.feature_dim))
                local_weight = gate_logits.sigmoid().unsqueeze(-1)
                y = (1.0 - local_weight) * y + local_weight * local
                retain = self.local_window - 1
                next_keys = keys[:, :, -retain:].clone() if retain else keys[:, :, :0].clone()
                next_values = values[:, :, -retain:].clone() if retain else values[:, :, :0].clone()
            new_state = DeltaCellV3State(memory, position, next_keys, next_values)
            return (y, new_state) if return_state else y