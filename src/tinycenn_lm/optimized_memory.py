"""Parallel normalized memory with disjoint exact sink/local attention.

Independent research implementation. See OPTIMIZED_MEMORY.md for derivation.
The block partition is terraced: current and previous block are exact, older
non-sink blocks are compressed. No T-by-T matrix or triangular solve is used
by the new memory candidates.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

VARIANTS = ("cenn_linear", "cenn_partition", "sink_window", "transformer_readout")


@dataclass
class MemoryState:
    numerator: torch.Tensor
    denominator: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    sinks_k: torch.Tensor
    sinks_v: torch.Tensor
    position: int

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.numerator, self.denominator, self.keys, self.values,
            self.sinks_k, self.sinks_v))


class OptimizedMemory(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, feature_dim=64,
                 variant="cenn_partition", block_size=32, sink_tokens=4,
                 compute_dtype="float32"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        if min(num_heads, num_kv_heads, head_dim, feature_dim, block_size) < 1:
            raise ValueError("Dimensions must be positive")
        if num_heads % num_kv_heads or feature_dim % 2 or sink_tokens < 0:
            raise ValueError("Require valid GQA, even feature_dim, nonnegative sink count")
        if compute_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unknown compute dtype")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim, self.feature_dim = head_dim, feature_dim
        self.groups = num_heads // num_kv_heads
        self.variant, self.block_size, self.sink_tokens = variant, block_size, sink_tokens
        self.compute_dtype = compute_dtype
        self.has_memory = variant in ("cenn_linear", "cenn_partition")
        if self.has_memory:
            weight = torch.randn(num_kv_heads, feature_dim // 2, head_dim) / math.sqrt(head_dim)
            self.wk = nn.Parameter(weight.clone())
            self.wq = nn.Parameter(weight.repeat_interleave(self.groups, dim=0).clone())
        if variant == "cenn_partition":
            self.log_mass = nn.Parameter(torch.full((num_heads,), math.log(feature_dim)))
            self.mass_w = nn.Parameter(torch.zeros(num_heads, head_dim))
        readout = torch.eye(head_dim).repeat(num_heads, 1, 1)
        if variant == "sink_window":
            self.register_buffer("readout", readout)
        else:
            self.readout = nn.Parameter(readout)

    @property
    def config(self):
        return dict(num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, feature_dim=self.feature_dim,
                    variant=self.variant, block_size=self.block_size,
                    sink_tokens=self.sink_tokens, compute_dtype=self.compute_dtype)

    def mm(self, a, b):
        dtype = getattr(torch, self.compute_dtype) if a.is_cuda else torch.float32
        return torch.matmul(a.to(dtype), b.to(dtype)).float()

    def attend(self, q, k, v, mask=None, causal=False):
        dtype = getattr(torch, self.compute_dtype) if q.is_cuda else torch.float32
        shape = q.shape
        if q.ndim == 5:
            b, h, n, c, d = shape
            length = k.shape[-2]
            q, k, v = (x.reshape(b * h * n, 1, x.shape[-2], d) for x in (q, k, v))
            if mask is not None:
                mask = mask.expand(b, h, n, c, length).reshape(b * h * n, 1, c, length)
        out = F.scaled_dot_product_attention(
            q.to(dtype), k.to(dtype), v.to(dtype), attn_mask=mask, is_causal=causal
        ).float()
        return out.reshape(shape)

    def features(self, x, query=False):
        weight = self.wq if query else self.wk
        # Retain input norms: do not impose the previous experiment's L2 normalization.
        logits = self.mm(x.float() / self.head_dim ** 0.25,
                         weight.view(1, weight.shape[0], *([1] * (x.ndim - 4)),
                                     weight.shape[1], weight.shape[2]).transpose(-1, -2))
        return torch.cat((logits, -logits), dim=-1).softmax(dim=-1)

    def calibrate(self, output):
        return self.mm(output, self.readout[None])

    def empty_state(self, k):
        b = k.shape[0]
        empty = k.new_empty(b, self.num_kv_heads, 0, self.head_dim)
        n = k.new_zeros(b, self.num_kv_heads, self.feature_dim, self.head_dim
                       ) if self.has_memory else k.new_empty(0)
        z = k.new_zeros(b, self.num_kv_heads, self.feature_dim
                       ) if self.has_memory else k.new_empty(0)
        return MemoryState(n, z, empty.clone(), empty.clone(), empty.clone(),
                           empty.clone(), 0)

    def combine(self, q, local_k, local_v, valid, numerator=None, denominator=None):
        """Shared log-stabilized denominator for exact and compressed contributions."""
        if self.variant == "sink_window":
            return self.attend(q, local_k.repeat_interleave(self.groups, 1),
                               local_v.repeat_interleave(self.groups, 1), mask=valid)
        scores = self.mm(q / math.sqrt(self.head_dim),
                         local_k.repeat_interleave(self.groups, dim=1).transpose(-1, -2))
        scores = scores.masked_fill(~valid, float("-inf"))
        maximum = scores.amax(dim=-1, keepdim=True)
        if self.variant == "cenn_partition":
            qp = self.features(q, query=True)
            num = self.mm(qp, numerator.repeat_interleave(self.groups, dim=1))
            den = self.mm(qp, denominator.repeat_interleave(self.groups, dim=1).unsqueeze(-1))
            raw_q = F.normalize(q.float(), dim=-1)
            # Works for [B,H,T,D] and [B,H,N,C,D].
            mass_shape = [1, self.num_heads] + [1] * (q.ndim - 3)
            mass_w_shape = [1, self.num_heads] + [1] * (q.ndim - 3) + [self.head_dim]
            mass = self.log_mass.view(mass_shape) + (
                raw_q * self.mass_w.view(mass_w_shape)).sum(-1)
            log_global = mass.clamp(-12, 12).unsqueeze(-1) + den.clamp_min(1e-30).log()
            log_global = torch.where(den > 0, log_global, float("-inf"))
            maximum = torch.maximum(maximum, log_global)
            global_weight = (log_global - maximum).exp()
            global_value = num / den.clamp_min(1e-20)
        weights = (scores - maximum).exp()
        local_num = self.mm(weights, local_v.repeat_interleave(self.groups, dim=1))
        local_den = weights.sum(dim=-1, keepdim=True)
        if self.variant == "cenn_partition":
            return (local_num + global_weight * global_value) / (
                local_den + global_weight).clamp_min(1e-20)
        return local_num / local_den.clamp_min(1e-20)

    @staticmethod
    def prefix(blocks, delay):
        cumulative = blocks.cumsum(dim=2)
        zeros = torch.zeros_like(blocks[:, :, :1]).expand(
            *blocks.shape[:2], delay, *blocks.shape[3:])
        return torch.cat((zeros, cumulative), dim=2)[:, :, :blocks.shape[2]]

    def prefill(self, q, k, v, need_state=True):
        b, _, t, d = q.shape
        state = self.empty_state(k) if need_state else None
        if self.variant == "transformer_readout":
            output = self.attend(q, k.repeat_interleave(self.groups, 1),
                                 v.repeat_interleave(self.groups, 1), causal=True)
            if need_state:
                state.keys, state.values, state.position = k.clone(), v.clone(), t
            return output, state
        c = self.block_size
        n = (t + c - 1) // c
        pad = n * c - t
        kb = F.pad(k, (0, 0, 0, pad)).reshape(b, self.num_kv_heads, n, c, d)
        vb = F.pad(v, (0, 0, 0, pad)).reshape(b, self.num_kv_heads, n, c, d)
        if self.has_memory:
            phi_k = self.features(k)
            if self.variant == "cenn_partition":
                phi_k = phi_k * (torch.arange(t, device=q.device) >= self.sink_tokens)[None, None, :, None]
            pk = F.pad(phi_k, (0, 0, 0, pad)).reshape(
                b, self.num_kv_heads, n, c, self.feature_dim)
            writes = self.mm(pk.transpose(-1, -2), vb)
            masses = pk.sum(dim=-2)
            delay = 1 if self.variant == "cenn_linear" else 2
            past_n, past_z = self.prefix(writes, delay), self.prefix(masses, delay)
        if self.variant == "cenn_linear":
            pq = F.pad(self.features(q, query=True), (0, 0, 0, pad)).reshape(
                b, self.num_heads, n, c, self.feature_dim)
            within = self.mm(pq, pk.repeat_interleave(self.groups, 1).transpose(-1, -2)).tril()
            numerator = self.mm(pq, past_n.repeat_interleave(self.groups, 1)) + self.mm(
                within, vb.repeat_interleave(self.groups, 1))
            denominator = self.mm(
                pq, past_z.repeat_interleave(self.groups, 1).unsqueeze(-1)
            ) + within.sum(-1, keepdim=True)
            output = (numerator / denominator.clamp_min(1e-20)).reshape(b, self.num_heads, n * c, d)[:, :, :t]
            if need_state:
                state.numerator, state.denominator = writes.sum(2), masses.sum(2)
        else:
            qb = F.pad(q, (0, 0, 0, pad)).reshape(b, self.num_heads, n, c, d)
            previous_k = torch.cat((torch.zeros_like(kb[:, :, :1]), kb[:, :, :-1]), dim=2)
            previous_v = torch.cat((torch.zeros_like(vb[:, :, :1]), vb[:, :, :-1]), dim=2)
            s = min(t, self.sink_tokens)
            sinks_k = k[:, :, :s].unsqueeze(2).expand(b, self.num_kv_heads, n, s, d)
            sinks_v = v[:, :, :s].unsqueeze(2).expand(b, self.num_kv_heads, n, s, d)
            local_k = torch.cat((sinks_k, previous_k, kb), dim=3)
            local_v = torch.cat((sinks_v, previous_v, vb), dim=3)
            block = torch.arange(n, device=q.device)[:, None]
            offset = torch.arange(c, device=q.device)[None]
            positions = block * c + offset
            local_positions = torch.cat((
                torch.arange(s, device=q.device)[None].expand(n, s),
                positions - c, positions), dim=1)
            query_positions = positions[:, :, None]
            valid = (local_positions[:, None, :] <= query_positions) & (
                local_positions[:, None, :] >= 0) & (local_positions[:, None, :] < t)
            # Every sink occurs only in the sink columns, never twice in exact local attention.
            valid[:, :, s:] &= local_positions[:, None, s:] >= self.sink_tokens
            output = self.combine(qb, local_k, local_v, valid[None, None],
                                  past_n if self.has_memory else None,
                                  past_z if self.has_memory else None)
            output = output.reshape(b, self.num_heads, n * c, d)[:, :, :t]
            if need_state:
                if self.has_memory:
                    state.numerator, state.denominator = past_n[:, :, -1].clone(), past_z[:, :, -1].clone()
                keep = min(t, c + (t - 1) % c + 1)
                state.keys, state.values = k[:, :, -keep:].clone(), v[:, :, -keep:].clone()
                state.sinks_k, state.sinks_v = k[:, :, :s].clone(), v[:, :, :s].clone()
        if need_state:
            state.position = t
        return output, state

    def step(self, q, k, v, state):
        """One token; the cache and compressed state are bounded in context length."""
        t, c = state.position, self.block_size
        num, den = state.numerator, state.denominator
        keys, values = state.keys, state.values
        sinks_k, sinks_v = state.sinks_k, state.sinks_v
        if self.variant == "transformer_readout":
            keys, values = torch.cat((keys, k), 2), torch.cat((values, v), 2)
            output = self.attend(q, keys.repeat_interleave(self.groups, 1),
                                 values.repeat_interleave(self.groups, 1), causal=False)
        elif self.variant == "cenn_linear":
            pk = self.features(k)
            num = num + self.mm(pk.transpose(-1, -2), v)
            den = den + pk[:, :, 0]
            qp = self.features(q, query=True)
            output = self.mm(qp, num.repeat_interleave(self.groups, 1)) / self.mm(
                qp, den.repeat_interleave(self.groups, 1).unsqueeze(-1)).clamp_min(1e-20)
        else:
            if t % c == 0 and keys.shape[2] > c:
                retired = keys.shape[2] - c
                if self.has_memory:
                    pk = self.features(keys[:, :, :retired])
                    positions = torch.arange(t - keys.shape[2], t - c, device=q.device)
                    pk = pk * (positions >= self.sink_tokens)[None, None, :, None]
                    num = num + self.mm(pk.transpose(-1, -2), values[:, :, :retired])
                    den = den + pk.sum(2)
                keys, values = keys[:, :, -c:].clone(), values[:, :, -c:].clone()
            keys, values = torch.cat((keys, k), 2), torch.cat((values, v), 2)
            if t < self.sink_tokens:
                sinks_k, sinks_v = torch.cat((sinks_k, k), 2), torch.cat((sinks_v, v), 2)
            lk, lv = torch.cat((sinks_k, keys), 2), torch.cat((sinks_v, values), 2)
            positions = torch.arange(t + 1 - keys.shape[2], t + 1, device=q.device)
            valid = torch.cat((torch.ones(sinks_k.shape[2], dtype=torch.bool, device=q.device),
                               positions >= self.sink_tokens))[None, None, None]
            output = self.combine(q, lk, lv, valid, num, den)
        return output, MemoryState(num, den, keys, values, sinks_k, sinks_v, t + 1)

    def forward(self, q, k, v, state=None, return_state=False, apply_readout=True):
        if (q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]
                or q.shape[2:] != k.shape[2:] or q.shape[1] != self.num_heads
                or k.shape[1] != self.num_kv_heads or q.shape[-1] != self.head_dim
                or q.shape[2] < 1):
            raise ValueError("Incompatible Q/K/V")
        q, k, v = q.float(), k.float(), v.float()
        if state is None:
            output, state = self.prefill(q, k, v, need_state=return_state)
        else:
            outputs = []
            for i in range(q.shape[2]):
                value, state = self.step(q[:, :, i:i+1], k[:, :, i:i+1], v[:, :, i:i+1], state)
                outputs.append(value)
            output = torch.cat(outputs, 2)
        if apply_readout and self.variant != "sink_window":
            output = self.calibrate(output)
        return (output, state) if return_state else output


@torch.no_grad()
def ridge_calibrate(core, samples, device, relative_ridge=0.01):
    """Identity-prior ridge solution, fitted only on training attention outputs.

    Solve (Y^T Y + lambda I) R = Y^T T + lambda I. Diagonal symmetric
    preconditioning preserves the exact solution while improving conditioning.
    CPU float64 is used for the small D-by-D systems.
    """
    h, d = core.num_heads, core.head_dim
    gram = torch.zeros(h, d, d, dtype=torch.float64)
    cross = torch.zeros_like(gram)
    before, count = 0.0, 0
    for q, k, v, target in samples:
        y = core(q.to(device), k.to(device), v.to(device), apply_readout=False).cpu().double()
        target = target.double()
        y = y.permute(1, 0, 2, 3).reshape(h, -1, d)
        target = target.permute(1, 0, 2, 3).reshape(h, -1, d)
        gram += y.transpose(-1, -2) @ y
        cross += y.transpose(-1, -2) @ target
        before += (y - target).square().sum().item()
        count += y.numel()
    identity = torch.eye(d, dtype=torch.float64).expand(h, d, d)
    ridge = relative_ridge * gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
    a, rhs = gram + ridge[:, None, None] * identity, cross + ridge[:, None, None] * identity
    scale = a.diagonal(dim1=-2, dim2=-1).rsqrt()
    conditioned = scale[:, :, None] * a * scale[:, None, :]
    solution = scale[:, :, None] * torch.linalg.solve(conditioned, scale[:, :, None] * rhs)
    core.readout.copy_(solution.to(core.readout))
    residual = (a @ solution - rhs).norm() / rhs.norm().clamp_min(1e-20)
    return {"ridge_relative_residual": float(residual), "uncalibrated_train_mse": before / count}
