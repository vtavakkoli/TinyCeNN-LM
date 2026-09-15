"""PDelta2-ER2: selective compressed error memory for long-context replacement.

The layer keeps the strongest Conv4 + PDelta2 F96 core and focuses the secondary
Residual16 state on a low-rank teacher-error subspace.  A learned query gate
predicts which head/token positions need the correction, so the residual branch
is not trained to imitate every attention output equally.

This is a research reference implementation.  Persistent recurrent matrices may
be stored in FP16 between streaming calls while curvature remains FP32.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.pdelta2_er import (
    causal_value_conv_stream,
    initialize_retention_spectrum,
    retention_half_lives,
)
from tinycenn_lm.pdelta2_features import PDelta2Core, PDeltaState


@dataclass
class ER2State:
    base: PDeltaState
    residual: PDeltaState | None = None
    conv_tail: Tensor | None = None


def hard_error_mask(base: Tensor, target: Tensor, fraction: float = 0.25) -> tuple[Tensor, Tensor]:
    """Return per-head hard mask and squared error, both [B,H,T]."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1)")
    error = (target.detach() - base.detach()).square().mean(dim=-1)
    threshold = torch.quantile(error, 1.0 - fraction, dim=-1, keepdim=True)
    return error >= threshold, error


def normalized_hard_weights(mask: Tensor, boost: float = 3.0) -> Tensor:
    """Weights with mean one, preserving the overall loss scale."""
    if boost < 0:
        raise ValueError("boost must be non-negative")
    weights = 1.0 + boost * mask.to(torch.float32)
    return weights / weights.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-8)


class SelectiveCompressedPDelta2Layer(nn.Module):
    """Conv4 PDelta2 with optional retention and selective low-rank Residual16.

    ``residual_mode="compressed"`` constrains the correction to a learned
    per-query-head rank-R subspace.  ``residual_mode="raw"`` is included as a
    same-budget control.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        feature_dim: int = 96,
        residual_dim: int = 16,
        code_rank: int = 8,
        chunk_size: int = 32,
        conv_kernel: int = 4,
        retention_spectrum: bool = True,
        retention_min: float = 8.0,
        retention_max: float = 4096.0,
        residual_mode: str = "compressed",
        state_dtype: str = "fp16",
    ):
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if residual_mode not in {"none", "raw", "compressed"}:
            raise ValueError("residual_mode must be none, raw, or compressed")
        if state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if conv_kernel < 1 or residual_dim < 0:
            raise ValueError("conv_kernel must be positive and residual_dim non-negative")
        if residual_mode == "compressed" and not 1 <= code_rank <= head_dim:
            raise ValueError("compressed mode needs 1 <= code_rank <= head_dim")
        if residual_mode == "none":
            residual_dim = 0
            code_rank = 0

        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.feature_dim = int(feature_dim)
        self.residual_dim = int(residual_dim)
        self.code_rank = int(code_rank)
        self.chunk_size = int(chunk_size)
        self.conv_kernel = int(conv_kernel)
        self.retention_spectrum = bool(retention_spectrum)
        self.retention_min = float(retention_min)
        self.retention_max = float(retention_max)
        self.residual_mode = residual_mode
        self.state_dtype = state_dtype
        self.groups = self.num_heads // self.num_kv_heads

        self.base = PDelta2Core(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            feature_dim=self.feature_dim,
            chunk_size=self.chunk_size,
        )
        if self.retention_spectrum:
            initialize_retention_spectrum(self.base, self.retention_min, self.retention_max)

        if self.conv_kernel > 1:
            channels = self.num_kv_heads * self.head_dim
            kernel = torch.zeros(channels, 1, self.conv_kernel)
            kernel[:, 0, -1] = 1.0
            self.conv_weight = nn.Parameter(kernel)
        else:
            self.register_parameter("conv_weight", None)

        if self.residual_dim:
            self.residual = PDelta2Core(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                feature_dim=self.residual_dim,
                chunk_size=self.chunk_size,
            )
            if self.retention_spectrum:
                initialize_retention_spectrum(
                    self.residual,
                    max(4.0, self.retention_min / 2.0),
                    self.retention_max * 2.0,
                )
            self.residual_gain = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.residual = None
            self.register_parameter("residual_gain", None)

        if self.residual_mode == "compressed":
            basis = torch.empty(self.num_heads, self.code_rank, self.head_dim)
            for head in range(self.num_heads):
                nn.init.orthogonal_(basis[head])
            self.residual_basis = nn.Parameter(basis)
            self.error_gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.error_gate_b = nn.Parameter(torch.full((self.num_heads,), math.log(0.2 / 0.8)))
        else:
            self.register_parameter("residual_basis", None)
            self.register_parameter("error_gate_w", None)
            self.register_parameter("error_gate_b", None)

    @property
    def config(self):
        return {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "feature_dim": self.feature_dim,
            "residual_dim": self.residual_dim,
            "code_rank": self.code_rank,
            "chunk_size": self.chunk_size,
            "conv_kernel": self.conv_kernel,
            "retention_spectrum": self.retention_spectrum,
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
            "residual_mode": self.residual_mode,
            "state_dtype": self.state_dtype,
        }

    @property
    def storage_dtype(self):
        return torch.float16 if self.state_dtype == "fp16" else torch.float32

    def normalized_basis(self) -> Tensor | None:
        if self.residual_basis is None:
            return None
        return F.normalize(self.residual_basis.float(), dim=-1)

    def project_to_code(self, x: Tensor) -> Tensor:
        basis = self.normalized_basis()
        if basis is None:
            raise RuntimeError("code projection requires compressed residual mode")
        return torch.einsum("bhtd,hrd->bhtr", x.float(), basis)

    def reconstruct_code(self, code: Tensor) -> Tensor:
        basis = self.normalized_basis()
        if basis is None:
            raise RuntimeError("code reconstruction requires compressed residual mode")
        return torch.einsum("bhtr,hrd->bhtd", code.float(), basis)

    def project_vector(self, x: Tensor) -> Tensor:
        return self.reconstruct_code(self.project_to_code(x))

    def orthogonality_penalty(self) -> Tensor:
        basis = self.normalized_basis()
        if basis is None:
            return torch.zeros((), device=self.base.wq.device)
        gram = torch.einsum("hrd,hsd->hrs", basis, basis)
        eye = torch.eye(self.code_rank, device=gram.device, dtype=gram.dtype)[None]
        return (gram - eye).square().mean()

    def predicted_error_gate(self, q: Tensor) -> Tensor:
        if self.residual_mode != "compressed":
            return q.new_ones(q.shape[0], q.shape[1], q.shape[2])
        qn = F.normalize(q.float(), dim=-1)
        return (
            torch.einsum("bhtd,hd->bht", qn, self.error_gate_w.float())
            + self.error_gate_b.float()[None, :, None]
        ).sigmoid()

    def _working_state(self, state: PDeltaState | None, core: PDelta2Core):
        if state is None:
            return None
        dtype = core.wq.dtype
        return PDeltaState(state.memory.to(dtype), state.curvature.to(dtype))

    def _stored_state(self, state: PDeltaState):
        return PDeltaState(state.memory.to(self.storage_dtype), state.curvature.float())

    def _run(self, q: Tensor, k: Tensor, v: Tensor, state: ER2State | None):
        tail = None if state is None else state.conv_tail
        conv_v, new_tail = causal_value_conv_stream(v.float(), self.conv_weight, tail)

        base_state = None if state is None else state.base
        base_out, new_base = self.base(
            q, k, conv_v,
            state=self._working_state(base_state, self.base),
            return_state=True,
        )

        residual_raw = None
        residual_projected = None
        gate = None
        new_residual = None
        output = base_out

        if self.residual is not None:
            residual_state = None if state is None else state.residual
            residual_raw, residual_state_out = self.residual(
                q, k, v.float(),
                state=self._working_state(residual_state, self.residual),
                return_state=True,
            )
            new_residual = self._stored_state(residual_state_out)
            gain = self.residual_gain.clamp(-1.5, 1.5)[None, :, None, None]

            if self.residual_mode == "compressed":
                residual_projected = self.project_vector(residual_raw)
                gate = self.predicted_error_gate(q).unsqueeze(-1)
                correction = gain * gate * residual_projected
            elif self.residual_mode == "raw":
                residual_projected = residual_raw
                gate = residual_raw.new_ones(residual_raw.shape[:-1] + (1,))
                correction = gain * residual_raw
            else:
                correction = 0.0
            output = base_out + correction

        new_state = ER2State(
            base=self._stored_state(new_base),
            residual=new_residual,
            conv_tail=None if new_tail is None else new_tail.to(self.storage_dtype),
        )
        return output, new_state, {
            "base": base_out,
            "residual_raw": residual_raw,
            "residual_projected": residual_projected,
            "gate": gate,
            "output": output,
        }

    def components(self, q: Tensor, k: Tensor, v: Tensor, state: ER2State | None = None):
        return self._run(q, k, v, state)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: ER2State | None = None,
        return_state: bool = False,
        implementation: str = "chunk",
    ):
        if implementation != "chunk":
            raise ValueError("SelectiveCompressedPDelta2Layer supports chunk implementation")
        output, new_state, _ = self._run(q, k, v, state)
        return (output, new_state) if return_state else output

    def auxiliary_losses(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        teacher: Tensor,
        hard_fraction: float = 0.25,
        hard_boost: float = 3.0,
    ):
        """Teacher-error-directed losses used only during training."""
        output, _, parts = self._run(q, k, v, None)
        base = parts["base"]
        mask, error = hard_error_mask(base, teacher, hard_fraction)
        weights = normalized_hard_weights(mask, hard_boost).to(output.device)
        weights4 = weights.unsqueeze(-1)

        teacher_den = (teacher.detach().square() * weights4).mean().clamp_min(1e-8)
        hard_teacher_nmse = ((output - teacher.detach()).square() * weights4).mean() / teacher_den
        result = {
            "hard_teacher_nmse": hard_teacher_nmse,
            "hard_fraction_observed": mask.float().mean(),
            "teacher_error_mean": error.mean(),
            "hard_error_mean": error[mask].mean(),
            "easy_error_mean": error[~mask].mean(),
        }

        if self.residual is None:
            zero = hard_teacher_nmse.new_zeros(())
            result.update(
                residual_code_nmse=zero,
                residual_reconstruction_nmse=zero,
                gate_bce=zero,
                orthogonality=zero,
            )
            return result

        residual_target = teacher.detach() - base.detach()
        hard4 = mask.unsqueeze(-1).to(residual_target.dtype)

        if self.residual_mode == "compressed":
            target_code = self.project_to_code(residual_target)
            predicted_code = self.project_to_code(parts["residual_raw"])
            hard_code = mask.unsqueeze(-1).to(target_code.dtype)
            code_den = (target_code.square() * hard_code).sum().clamp_min(1e-8)
            code_nmse = ((predicted_code - target_code).square() * hard_code).sum() / code_den

            target_projection = self.reconstruct_code(target_code)
            recon_den = (residual_target.square() * hard4).sum().clamp_min(1e-8)
            recon_nmse = ((target_projection - residual_target).square() * hard4).sum() / recon_den

            gate = parts["gate"].squeeze(-1).clamp(1e-5, 1 - 1e-5)
            gate_bce = F.binary_cross_entropy(gate, mask.to(gate.dtype))
            orth = self.orthogonality_penalty()
        else:
            pred = parts["residual_raw"]
            den = (residual_target.square() * hard4).sum().clamp_min(1e-8)
            code_nmse = ((pred - residual_target).square() * hard4).sum() / den
            recon_nmse = code_nmse.new_zeros(())
            gate_bce = code_nmse.new_zeros(())
            orth = code_nmse.new_zeros(())

        result.update(
            residual_code_nmse=code_nmse,
            residual_reconstruction_nmse=recon_nmse,
            gate_bce=gate_bce,
            orthogonality=orth,
        )
        return result

    def recurrent_state_bytes(self, batch_size: int = 1, context: int | None = None):
        del context
        memory_bytes = 2 if self.state_dtype == "fp16" else 4
        total = self.num_kv_heads * self.feature_dim * self.head_dim * memory_bytes
        total += self.num_kv_heads * self.feature_dim * 4
        if self.residual_dim:
            total += self.num_kv_heads * self.residual_dim * self.head_dim * memory_bytes
            total += self.num_kv_heads * self.residual_dim * 4
        if self.conv_kernel > 1:
            total += (self.conv_kernel - 1) * self.num_kv_heads * self.head_dim * memory_bytes
        return batch_size * total

    def retention_statistics(self):
        values = retention_half_lives(self.base).float().reshape(-1)
        result = {
            "base_half_life_min": float(values.min()),
            "base_half_life_median": float(values.median()),
            "base_half_life_max": float(values.max()),
        }
        if self.residual is not None:
            rv = retention_half_lives(self.residual).float().reshape(-1)
            result.update(
                residual_half_life_min=float(rv.min()),
                residual_half_life_median=float(rv.median()),
                residual_half_life_max=float(rv.max()),
            )
        if self.residual_gain is not None:
            result["residual_gain_abs_mean"] = float(self.residual_gain.detach().abs().mean())
        if self.residual_mode == "compressed":
            result["predicted_gate_mean"] = float(self.error_gate_b.detach().sigmoid().mean())
        return result
