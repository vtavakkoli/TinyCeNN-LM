#!/usr/bin/env python3
"""Sequential SmolLM2 optimization with PDelta3/GDN2 + true one-hop CLVR.

This experiment ports the strongest Tiny-LLM recurrent candidate
(conv4_gdn2_inputroute_f96) to multi-layer SmolLM2 and makes the routing
actually cross-layer for layers >= 1.  It also adds a bounded exact local
attention window, logit-first distillation, previous-layer warm starts, and
quality-first checkpoint rollback.

The goal is scientific: replace one Transformer attention layer at a time and
accept a replacement only when held-out probe quality stays inside strict gates.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make src-layout imports work even when the repository is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_smollm2_memory_fusion_sequential as seq
from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer


@dataclass(frozen=True)
class PDelta3CLVRConfig:
    feature_dim: int = 96
    local_window: int = 32
    chunk_size: int = 32
    conv_kernel: int = 4
    state_dtype: str = "fp16"
    variant: str = "conv4_gdn2_clvr_f96"
    local_gate_init: float = 0.72
    warm_start_previous_core: bool = True

    def validate(self, model_config) -> None:
        if self.feature_dim < 16:
            raise ValueError("feature_dim must be >= 16")
        if self.local_window < 1:
            raise ValueError("local_window must be positive")
        if not 1 <= self.chunk_size <= 32:
            raise ValueError("chunk_size must be in [1, 32]")
        if self.conv_kernel < 1:
            raise ValueError("conv_kernel must be positive")
        if self.state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if int(model_config.num_attention_heads) % int(model_config.num_key_value_heads):
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "PDelta3CLVRConfig":
        return cls(**dict(value))


class SmolPDelta3CLVRAttention(nn.Module):
    """Drop-in Llama attention: Local-W exact attention + PDelta3/GDN2 CLVR.

    Local attention is bounded to ``local_window`` tokens.  The global path is
    recurrent PDelta3/GDN2.  For layer 0 the route uses current V (the Tiny-LLM
    proxy); for layer >= 1 it uses the previous accepted replacement's V, which
    is a true one-hop cross-layer value route.

    The reference implementation uses a dense local mask for correctness during
    research training.  A production kernel should implement the same W-token
    window with O(T*W) work/storage.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        model_config,
        config: PDelta3CLVRConfig,
        layer_idx: int,
        previous_attention: "SmolPDelta3CLVRAttention | None" = None,
    ) -> None:
        super().__init__()
        config.validate(model_config)
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_kv_heads = int(model_config.num_key_value_heads)
        self.head_dim = int(getattr(model_config, "head_dim", self.hidden_size // self.num_heads))
        self.groups = self.num_heads // self.num_kv_heads
        self.local_window = int(config.local_window)
        # Keep a non-owning link to the preceding replacement.  Assigning an
        # nn.Module directly here would register it as a child and duplicate the
        # previous layer inside this layer's state_dict.
        object.__setattr__(
            self,
            "_previous_attention_ref",
            weakref.ref(previous_attention) if previous_attention is not None else None,
        )

        self.q_proj = copy.deepcopy(original_attn.q_proj)
        self.k_proj = copy.deepcopy(original_attn.k_proj)
        self.v_proj = copy.deepcopy(original_attn.v_proj)
        self.o_proj = copy.deepcopy(original_attn.o_proj)

        self.core = FrontierPDelta3Layer(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            feature_dim=config.feature_dim,
            variant=config.variant,
            chunk_size=config.chunk_size,
            conv_kernel=config.conv_kernel,
            state_dtype=config.state_dtype,
        )

        # Input-dependent local/global gate.  Start local-heavy to protect
        # short-range syntax while the recurrent path learns the teacher.
        init = min(max(float(config.local_gate_init), 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init / (1.0 - init))
        self.local_gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.local_gate_b = nn.Parameter(torch.full((self.num_heads,), init_logit))

        self.last_value: Tensor | None = None
        self.last_projected: Tensor | None = None
        self.last_local_gate: Tensor | None = None
        self.last_global: Tensor | None = None
        self.last_local: Tensor | None = None

    @staticmethod
    def _apply_rope(q: Tensor, k: Tensor, position_embeddings) -> tuple[Tensor, Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required")
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(q, k, cos, sin)

    def _local_exact(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        kh = k.repeat_interleave(self.groups, dim=1)
        vh = v.repeat_interleave(self.groups, dim=1)
        t = q.shape[-2]
        pos = torch.arange(t, device=q.device)
        distance = pos[:, None] - pos[None, :]
        allowed = (distance >= 0) & (distance < self.local_window)
        mask = torch.zeros((t, t), device=q.device, dtype=q.dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(q.dtype).min)
        return F.scaled_dot_product_attention(
            q, kh, vh, attn_mask=mask[None, None], dropout_p=0.0, is_causal=False
        )

    def _route_value(self, current_v: Tensor) -> Tensor:
        if self.layer_idx == 0:
            return current_v
        ref = getattr(self, "_previous_attention_ref", None)
        prev = ref() if ref is not None else None
        if isinstance(prev, SmolPDelta3CLVRAttention) and prev.last_value is not None:
            routed = prev.last_value
            if routed.shape == current_v.shape:
                return routed.to(device=current_v.device, dtype=current_v.dtype)
        # Safe fallback if a standalone attention call bypasses the previous layer.
        return current_v

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        past_key_value=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if use_cache or past_key_values is not None or past_key_value is not None:
            raise RuntimeError("PDelta3-CLVR reference attention requires use_cache=False")
        bsz, seq_len, _ = hidden_states.shape
        if attention_mask is not None and attention_mask.ndim == 4:
            if attention_mask.shape[-1] != seq_len:
                raise ValueError("attention mask length mismatch")
            # Last query has no causal-future entries; negatives there imply padding.
            if bool((attention_mask[..., -1, :] < -1e4).any()):
                raise ValueError("padded batches are not supported in this research trainer")

        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        routed_v = self._route_value(v)
        global_out = self.core(
            q.float(), k.float(), v.float(), routed_v=routed_v.detach().float()
        )
        local_out = self._local_exact(q, k, v).float()
        gate_logits = torch.einsum(
            "bhtd,hd->bht", q.float(), self.local_gate_w.float()
        ) + self.local_gate_b.float()[None, :, None]
        local_gate = gate_logits.sigmoid()
        heads = local_gate[..., None] * local_out + (1.0 - local_gate[..., None]) * global_out

        flat = heads.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        projected = self.o_proj(flat.to(dtype=hidden_states.dtype))

        # previous layers are frozen, so detach their route value.  Keep current
        # projected/gate tensors attached for the current-layer training loss.
        self.last_value = v.detach()
        self.last_projected = projected
        self.last_local_gate = local_gate
        self.last_global = global_out
        self.last_local = local_out
        return projected, None


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _atomic_json_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _attention_prefix(idx: int) -> str:
    return f"model.layers.{idx}.self_attn."


def _selected_attention_state(model, layers: list[int]) -> dict[str, Tensor]:
    prefixes = tuple(_attention_prefix(i) for i in layers)
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if prefixes and k.startswith(prefixes)
    }


def _module_state_cpu(module: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_module_state(module: nn.Module, state: dict[str, Tensor]) -> None:
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"module state mismatch: missing={incompatible.missing_keys[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )


def install_replacement(
    model,
    layer_idx: int,
    config: PDelta3CLVRConfig,
    *,
    warm_start_previous_core: bool,
) -> SmolPDelta3CLVRAttention:
    layer = model.model.layers[layer_idx]
    if isinstance(layer.self_attn, SmolPDelta3CLVRAttention):
        return layer.self_attn

    old = layer.self_attn
    prev = model.model.layers[layer_idx - 1].self_attn if layer_idx > 0 else None
    prev_custom = prev if isinstance(prev, SmolPDelta3CLVRAttention) else None
    device = old.q_proj.weight.device
    projection_dtype = old.q_proj.weight.dtype

    new = SmolPDelta3CLVRAttention(
        old, model.config, config, layer_idx, previous_attention=prev_custom
    )
    for projection in (new.q_proj, new.k_proj, new.v_proj, new.o_proj):
        projection.to(device=device, dtype=projection_dtype)
    new.core.to(device=device, dtype=torch.float32)
    new.local_gate_w.data = new.local_gate_w.data.to(device=device, dtype=torch.float32)
    new.local_gate_b.data = new.local_gate_b.data.to(device=device, dtype=torch.float32)

    if warm_start_previous_core and prev_custom is not None:
        new.core.load_state_dict(prev_custom.core.state_dict(), strict=True)
        new.local_gate_w.data.copy_(prev_custom.local_gate_w.detach())
        new.local_gate_b.data.copy_(prev_custom.local_gate_b.detach())
        print(f"  warm-started layer {layer_idx} recurrent core from layer {layer_idx - 1}", flush=True)

    layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return new


def structural_summary(model) -> dict[str, int]:
    recurrent = sum(
        isinstance(layer.self_attn, SmolPDelta3CLVRAttention)
        for layer in model.model.layers
    )
    return {
        "pdelta3_clvr_layers": recurrent,
        "transformer_attention_layers": int(model.config.num_hidden_layers) - recurrent,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential SmolLM2 PDelta3/GDN2-CLVR + Local-W optimization"
    )
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--shuffle-buffer", type=int, default=2048)

    p.add_argument("--feature-dim", type=int, default=96)
    p.add_argument("--local-window", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--conv-kernel", type=int, default=4)
    p.add_argument("--state-dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--local-gate-init", type=float, default=0.72)
    p.add_argument(
        "--warm-start-previous-core",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--min-layer-steps", type=int, default=75)
    p.add_argument("--max-layer-steps", type=int, default=300)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--layer-lr", type=float, default=4e-4)
    p.add_argument("--qkv-lr-scale", type=float, default=0.20)
    p.add_argument("--train-qkv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=1.5)

    p.add_argument("--functional-weight", type=float, default=0.25)
    p.add_argument("--kl-weight", type=float, default=1.00)
    p.add_argument("--ce-weight", type=float, default=0.10)
    p.add_argument("--cosine-weight", type=float, default=0.20)
    p.add_argument("--local-gate-penalty", type=float, default=0.002)

    # Rescue rounds become more logit-first after the first calibration round.
    p.add_argument("--rescue-lr-scale", type=float, default=0.45)
    p.add_argument("--rescue-functional-weight", type=float, default=0.10)
    p.add_argument("--rescue-kl-weight", type=float, default=1.50)
    p.add_argument("--rescue-ce-weight", type=float, default=0.18)

    p.add_argument("--accept-nmse", type=float, default=0.15)
    p.add_argument("--accept-cosine", type=float, default=0.94)
    p.add_argument("--accept-incremental-delta-nll", type=float, default=0.015)
    p.add_argument("--accept-cumulative-delta-nll", type=float, default=0.05)
    p.add_argument("--probe-blocks", type=int, default=8)
    p.add_argument("--probe-context", type=int, default=256)
    p.add_argument("--strict-acceptance", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--target-layers", type=int, default=6)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-runtime-minutes", type=float, default=240.0)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_factory(device: torch.device, dtype: torch.dtype):
    from contextlib import nullcontext
    if device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=dtype)
    return nullcontext


def capture_student_forward(model, ids: Tensor, layer_idx: int, amp, *, labels: bool):
    capture: dict[str, Any] = {}
    module = model.model.layers[layer_idx].self_attn
    if not isinstance(module, SmolPDelta3CLVRAttention):
        raise TypeError(f"layer {layer_idx} is not a PDelta3-CLVR replacement")

    def pre_hook(mod, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("attention hidden_states not found")
        capture["hidden"] = hidden
        for key in ("position_embeddings", "position_ids", "attention_mask", "cache_position"):
            if key in kwargs and kwargs[key] is not None:
                capture[key] = seq._detach_tree(kwargs[key])

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with amp():
            out = model(
                input_ids=ids,
                labels=ids if labels else None,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        handle.remove()

    if "hidden" not in capture or module.last_projected is None:
        raise RuntimeError("failed to capture student replacement output")
    return capture, out, module


def teacher_target_on_student_hidden(teacher, capture: dict[str, Any], layer_idx: int, amp) -> Tensor:
    kwargs = seq.attention_kwargs_from_capture(capture)
    with torch.no_grad(), amp():
        return seq.call_attention(
            teacher.model.layers[layer_idx].self_attn,
            capture["hidden"].detach(),
            kwargs,
        )


def make_layer_optimizer(model, layer_idx: int, args, round_number: int, device):
    for p in model.parameters():
        p.requires_grad = False
    module = model.model.layers[layer_idx].self_attn
    assert isinstance(module, SmolPDelta3CLVRAttention)

    if round_number == 1:
        base_lr = args.layer_lr
    else:
        base_lr = args.layer_lr * args.rescue_lr_scale

    main_params: list[nn.Parameter] = []
    for p in module.core.parameters():
        p.requires_grad = True
        main_params.append(p)
    for p in module.o_proj.parameters():
        p.requires_grad = True
        main_params.append(p)
    for p in (module.local_gate_w, module.local_gate_b):
        p.requires_grad = True
        main_params.append(p)

    groups = [{"params": main_params, "lr": base_lr, "weight_decay": 1e-4}]
    if args.train_qkv:
        qkv = []
        for projection in (module.q_proj, module.k_proj, module.v_proj):
            for p in projection.parameters():
                p.requires_grad = True
                qkv.append(p)
        groups.append(
            {"params": qkv, "lr": base_lr * args.qkv_lr_scale, "weight_decay": 1e-4}
        )

    try:
        return torch.optim.AdamW(groups, fused=(device.type == "cuda"))
    except Exception:
        return torch.optim.AdamW(groups)


def round_weights(args, round_number: int) -> tuple[float, float, float]:
    if round_number == 1:
        return args.functional_weight, args.kl_weight, args.ce_weight
    return (
        args.rescue_functional_weight,
        args.rescue_kl_weight,
        args.rescue_ce_weight,
    )


@torch.no_grad()
def evaluate_layer(
    teacher,
    student,
    layer_idx: int,
    probe_blocks: list[Tensor],
    teacher_probe_nll: float,
    pre_probe_nll: float,
    device,
    amp,
) -> dict:
    student.eval()
    nlls, nmses, cosines, gates = [], [], [], []
    for block in probe_blocks:
        ids = block.unsqueeze(0).to(device)
        capture, out, module = capture_student_forward(
            student, ids, layer_idx, amp, labels=True
        )
        target = teacher_target_on_student_hidden(teacher, capture, layer_idx, amp)
        pred = module.last_projected
        nmse, cosine = seq.alignment_metrics(pred, target)
        nlls.append(float(out.loss.detach().float()))
        nmses.append(float(nmse))
        cosines.append(float(cosine))
        if module.last_local_gate is not None:
            gates.append(float(module.last_local_gate.detach().mean()))

    nll = sum(nlls) / len(nlls)
    return {
        "nmse": sum(nmses) / len(nmses),
        "cosine": sum(cosines) / len(cosines),
        "probe_nll": nll,
        "incremental_delta_nll": nll - pre_probe_nll,
        "cumulative_delta_nll": nll - teacher_probe_nll,
        "local_gate_mean": (sum(gates) / len(gates)) if gates else float("nan"),
    }


def acceptance_passes(metrics: dict, args) -> bool:
    quality = (
        metrics["incremental_delta_nll"] <= args.accept_incremental_delta_nll
        and metrics["cumulative_delta_nll"] <= args.accept_cumulative_delta_nll
    )
    if not args.strict_acceptance:
        return quality
    return (
        quality
        and metrics["nmse"] <= args.accept_nmse
        and metrics["cosine"] >= args.accept_cosine
    )


def candidate_better(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    # Quality first.  Functional similarity breaks near-ties.
    c = (
        candidate["incremental_delta_nll"],
        candidate["cumulative_delta_nll"],
        candidate["nmse"],
        -candidate["cosine"],
    )
    b = (
        best["incremental_delta_nll"],
        best["cumulative_delta_nll"],
        best["nmse"],
        -best["cosine"],
    )
    return c < b


def train_one_round(
    *,
    teacher,
    student,
    layer_idx: int,
    batch_iter,
    probe_blocks,
    teacher_probe_nll: float,
    pre_probe_nll: float,
    args,
    round_number: int,
    device,
    amp,
) -> tuple[dict, dict[str, Tensor]]:
    module = student.model.layers[layer_idx].self_attn
    assert isinstance(module, SmolPDelta3CLVRAttention)
    optimizer = make_layer_optimizer(student, layer_idx, args, round_number, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and choose_dtype(device) == torch.float16)
    )
    fw, kw, cw = round_weights(args, round_number)
    best_metrics: dict | None = None
    best_state = _module_state_cpu(module)

    student.train()
    for step in range(1, args.max_layer_steps + 1):
        ids = next(batch_iter).to(device, non_blocking=True)
        with torch.no_grad(), amp():
            teacher_out = teacher(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

        capture, student_out, module = capture_student_forward(
            student, ids, layer_idx, amp, labels=True
        )
        target = teacher_target_on_student_hidden(teacher, capture, layer_idx, amp)
        pred = module.last_projected
        functional = seq.alignment_loss(pred, target, args.cosine_weight)
        kl = seq.distillation_kl(student_out.logits, teacher_out.logits, args.temperature)
        ce = student_out.loss
        gate_penalty = (
            module.last_local_gate.mean()
            if module.last_local_gate is not None
            else pred.new_zeros(())
        )
        loss = (
            fw * functional
            + kw * kl
            + cw * ce
            + args.local_gate_penalty * gate_penalty
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at layer {layer_idx}, round {round_number}, step {step}"
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [p for group in optimizer.param_groups for p in group["params"]]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            nmse_batch, cos_batch = seq.alignment_metrics(pred.detach(), target.detach())
            print(
                f"layer={layer_idx:02d} round={round_number:02d} "
                f"step={step:03d}/{args.max_layer_steps} "
                f"loss={float(loss.detach()):.4f} "
                f"func={float(functional.detach()):.4f} "
                f"nmse={float(nmse_batch):.4f} cos={float(cos_batch):.4f} "
                f"kl={float(kl.detach()):.4f} ce={float(ce.detach()):.4f} "
                f"local={float(gate_penalty.detach()):.3f} grad={grad_norm:.3f}",
                flush=True,
            )

        should_check = (
            step >= args.min_layer_steps
            and (step % args.check_every == 0 or step == args.max_layer_steps)
        )
        if should_check:
            metrics = evaluate_layer(
                teacher,
                student,
                layer_idx,
                probe_blocks,
                teacher_probe_nll,
                pre_probe_nll,
                device,
                amp,
            )
            metrics["step"] = step
            metrics["round"] = round_number
            metrics["accepted"] = acceptance_passes(metrics, args)
            if metrics["accepted"]:
                # Never roll a passing checkpoint back to an earlier non-passing
                # checkpoint merely because one scalar metric was slightly lower.
                best_metrics = dict(metrics)
                best_state = _module_state_cpu(module)
            elif candidate_better(metrics, best_metrics):
                best_metrics = dict(metrics)
                best_state = _module_state_cpu(module)

            print(
                f"  CHECK layer={layer_idx:02d} round={round_number:02d} "
                f"NMSE={metrics['nmse']:.4f} (≤{args.accept_nmse:.4f}) "
                f"cos={metrics['cosine']:.4f} (≥{args.accept_cosine:.4f}) "
                f"ΔNLL_inc={metrics['incremental_delta_nll']:+.5f} "
                f"(≤{args.accept_incremental_delta_nll:+.5f}) "
                f"ΔNLL_total={metrics['cumulative_delta_nll']:+.5f} "
                f"(≤{args.accept_cumulative_delta_nll:+.5f}) "
                f"local={metrics['local_gate_mean']:.3f} "
                f"=> {'PASS' if metrics['accepted'] else 'continue'}",
                flush=True,
            )
            if metrics["accepted"]:
                _load_module_state(module, best_state)
                return best_metrics, best_state
            student.train()

        del teacher_out, student_out, target, pred, loss

    if best_metrics is None:
        raise RuntimeError("acceptance was never evaluated")

    # Critical difference from the old trainer: resume from the best quality
    # checkpoint, not blindly from the final optimizer step of the round.
    _load_module_state(module, best_state)
    return best_metrics, best_state


def save_progress(
    output_dir: Path,
    student,
    config: PDelta3CLVRConfig,
    accepted_layers: list[int],
    layer_reports: list[dict],
) -> None:
    payload = {
        "format_version": 1,
        "architecture": "smollm2-pdelta3-gdn2-clvr-local-sequential",
        "accepted_layers": list(accepted_layers),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "attention_state": _selected_attention_state(student, accepted_layers),
    }
    _atomic_torch_save(payload, output_dir / "sequential_progress.pt")
    _atomic_json_save(
        {k: v for k, v in payload.items() if k != "attention_state"},
        output_dir / "sequential_progress.json",
    )


def load_progress(
    output_dir: Path,
    student,
    config: PDelta3CLVRConfig,
    *,
    resume: bool,
) -> tuple[list[int], list[dict]]:
    path = output_dir / "sequential_progress.pt"
    if not resume or not path.exists():
        return [], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config") != config.to_dict():
        raise RuntimeError(
            "resume architecture config differs from checkpoint; use a new output directory"
        )
    accepted = [int(x) for x in payload.get("accepted_layers", [])]
    if accepted != list(range(len(accepted))):
        raise RuntimeError(f"accepted layers are not a contiguous prefix: {accepted}")
    for idx in accepted:
        install_replacement(
            student,
            idx,
            config,
            warm_start_previous_core=False,
        )
    if accepted:
        incompatible = student.load_state_dict(payload["attention_state"], strict=False)
        expected = {
            k
            for k in student.state_dict()
            if any(k.startswith(_attention_prefix(i)) for i in accepted)
        }
        missing = [k for k in incompatible.missing_keys if k in expected]
        unexpected = [
            k for k in incompatible.unexpected_keys
            if any(k.startswith(_attention_prefix(i)) for i in accepted)
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"resume state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
    print(f"RESUME accepted layers={accepted}", flush=True)
    return accepted, list(payload.get("layer_reports", []))


def save_in_progress(
    output_dir: Path,
    student,
    config: PDelta3CLVRConfig,
    *,
    accepted_layers: list[int],
    current_layer: int,
    rounds_completed: int,
    pre_probe_nll: float,
    layer_reports: list[dict],
) -> None:
    layers = accepted_layers + [current_layer]
    payload = {
        "format_version": 1,
        "architecture": "smollm2-pdelta3-gdn2-clvr-local-sequential",
        "accepted_layers": list(accepted_layers),
        "current_layer": int(current_layer),
        "rounds_completed": int(rounds_completed),
        "pre_probe_nll": float(pre_probe_nll),
        "config": config.to_dict(),
        "layer_reports": list(layer_reports),
        "attention_state": _selected_attention_state(student, layers),
    }
    _atomic_torch_save(payload, output_dir / "sequential_in_progress.pt")
    _atomic_json_save(
        {k: v for k, v in payload.items() if k != "attention_state"},
        output_dir / "sequential_in_progress.json",
    )


def load_in_progress(
    output_dir: Path,
    student,
    config: PDelta3CLVRConfig,
    accepted_layers: list[int],
    *,
    resume: bool,
):
    path = output_dir / "sequential_in_progress.pt"
    if not resume or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config") != config.to_dict():
        print("Ignoring in-progress checkpoint with different architecture config.", flush=True)
        return None
    if [int(x) for x in payload.get("accepted_layers", [])] != accepted_layers:
        print("Ignoring stale in-progress checkpoint: accepted prefix changed.", flush=True)
        return None
    current = int(payload["current_layer"])
    if current != len(accepted_layers):
        print("Ignoring stale in-progress checkpoint: current layer is not next prefix layer.", flush=True)
        return None

    install_replacement(
        student,
        current,
        config,
        warm_start_previous_core=False,
    )
    incompatible = student.load_state_dict(payload["attention_state"], strict=False)
    expected_layers = accepted_layers + [current]
    expected = {
        k
        for k in student.state_dict()
        if any(k.startswith(_attention_prefix(i)) for i in expected_layers)
    }
    missing = [k for k in incompatible.missing_keys if k in expected]
    if missing:
        raise RuntimeError(f"in-progress checkpoint missing keys: {missing[:5]}")
    print(
        f"RESUME current layer={current} rounds_completed={payload.get('rounds_completed', 0)}",
        flush=True,
    )
    return payload


def remove_in_progress(output_dir: Path) -> None:
    for name in ("sequential_in_progress.pt", "sequential_in_progress.json"):
        p = output_dir / name
        if p.exists():
            p.unlink()


def write_status(output_dir: Path, payload: dict) -> None:
    _atomic_json_save(payload, output_dir / "sequential_run_status.json")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rounds_this_run = max(
        1, int(os.environ.get("SEQUENTIAL_MAX_ROUNDS_PER_RUN", "3"))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    amp = amp_factory(device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True

    config = PDelta3CLVRConfig(
        feature_dim=args.feature_dim,
        local_window=args.local_window,
        chunk_size=args.chunk_size,
        conv_kernel=args.conv_kernel,
        state_dtype=args.state_dtype,
        local_gate_init=args.local_gate_init,
        warm_start_previous_core=args.warm_start_previous_core,
    )

    print(
        f"device={device} dtype={dtype} feature_dim={config.feature_dim} "
        f"local_window={config.local_window} target_layers={args.target_layers}",
        flush=True,
    )
    print(f"output_dir={output_dir}", flush=True)
    print(f"max_rounds_this_run={max_rounds_this_run}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.config.use_cache = False

    student = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device)
    student.config.use_cache = False

    accepted_layers, layer_reports = load_progress(
        output_dir, student, config, resume=args.resume
    )
    in_progress = load_in_progress(
        output_dir, student, config, accepted_layers, resume=args.resume
    )

    raw = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.split,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    batch_iter = iter(
        seq.batches(
            seq.token_blocks(raw, tokenizer, args.text_field, args.context_length),
            args.batch_size,
        )
    )

    probe_blocks = seq.build_probe_blocks(
        tokenizer, context=args.probe_context, count=args.probe_blocks
    )
    teacher_probe_nll = seq.probe_nll(teacher, probe_blocks, device, amp)
    print(f"teacher probe NLL={teacher_probe_nll:.6f}", flush=True)

    start = time.perf_counter()
    num_layers = int(student.config.num_hidden_layers)
    target_layers = min(max(1, int(args.target_layers)), num_layers)

    for layer_idx in range(target_layers):
        if layer_idx in accepted_layers:
            continue
        if accepted_layers != list(range(layer_idx)):
            raise RuntimeError(
                f"accepted prefix must be 0..{layer_idx - 1}; got {accepted_layers}"
            )

        if in_progress is not None and int(in_progress["current_layer"]) == layer_idx:
            pre_probe_nll = float(in_progress["pre_probe_nll"])
            rounds_completed = int(in_progress.get("rounds_completed", 0))
            print(
                f"\nCONTINUE layer {layer_idx} from best saved quality checkpoint; "
                f"rounds={rounds_completed} pre-NLL={pre_probe_nll:.6f}",
                flush=True,
            )
        else:
            pre_probe_nll = seq.probe_nll(student, probe_blocks, device, amp)
            print("\n" + "=" * 112, flush=True)
            print(
                f"SEQUENTIAL PDELTA3-CLVR REPLACEMENT layer {layer_idx}/{target_layers - 1} "
                f"(model has {num_layers} layers)",
                flush=True,
            )
            print(
                f"before replacement NLL={pre_probe_nll:.6f} "
                f"Δteacher={pre_probe_nll - teacher_probe_nll:+.6f}",
                flush=True,
            )
            install_replacement(
                student,
                layer_idx,
                config,
                warm_start_previous_core=config.warm_start_previous_core,
            )
            rounds_completed = 0

        accepted = False
        last_report = None
        for local_round in range(1, max_rounds_this_run + 1):
            global_round = rounds_completed + local_round
            fw, kw, cw = round_weights(args, global_round)
            print(
                f"\n--- layer {layer_idx} round {global_round}: "
                f"weights functional={fw} KL={kw} CE={cw} ---",
                flush=True,
            )
            report, _ = train_one_round(
                teacher=teacher,
                student=student,
                layer_idx=layer_idx,
                batch_iter=batch_iter,
                probe_blocks=probe_blocks,
                teacher_probe_nll=teacher_probe_nll,
                pre_probe_nll=pre_probe_nll,
                args=args,
                round_number=global_round,
                device=device,
                amp=amp,
            )
            last_report = {
                "layer": layer_idx,
                **report,
            }
            layer_reports.append(last_report)

            if report["accepted"]:
                accepted = True
                break

            save_in_progress(
                output_dir,
                student,
                config,
                accepted_layers=accepted_layers,
                current_layer=layer_idx,
                rounds_completed=global_round,
                pre_probe_nll=pre_probe_nll,
                layer_reports=layer_reports,
            )
            print(
                f"Layer {layer_idx} not accepted; restored/saved the best quality "
                f"checkpoint from round {global_round}.",
                flush=True,
            )

            elapsed_min = (time.perf_counter() - start) / 60.0
            if elapsed_min >= args.max_runtime_minutes * 0.72:
                write_status(
                    output_dir,
                    {
                        "status": "paused_runtime_budget",
                        "accepted_layers": accepted_layers,
                        "current_layer": layer_idx,
                        "rounds_completed": global_round,
                        "last_report": last_report,
                        "message": "Rerun the Colab with RESUME=True.",
                    },
                )
                print("PAUSED: runtime budget reached; Drive checkpoint is safe.", flush=True)
                return 0

        if not accepted:
            total_rounds = rounds_completed + max_rounds_this_run
            save_in_progress(
                output_dir,
                student,
                config,
                accepted_layers=accepted_layers,
                current_layer=layer_idx,
                rounds_completed=total_rounds,
                pre_probe_nll=pre_probe_nll,
                layer_reports=layer_reports,
            )
            status = {
                "status": "current_layer_needs_more_training",
                "architecture": "PDelta3-GDN2-CLVR+LocalW",
                "accepted_layers": accepted_layers,
                "current_layer": layer_idx,
                "rounds_completed": total_rounds,
                "last_report": last_report,
                "message": (
                    "No next Transformer layer was replaced. Rerun to continue "
                    "from the best saved quality checkpoint."
                ),
            }
            write_status(output_dir, status)
            print("\nNOT A CRASH:", json.dumps(status, indent=2), flush=True)
            return 0

        accepted_layers.append(layer_idx)
        save_progress(output_dir, student, config, accepted_layers, layer_reports)
        remove_in_progress(output_dir)
        in_progress = None
        print(
            f"✅ ACCEPTED layer {layer_idx}: prefix now {accepted_layers}; "
            "moving to the next Transformer layer.",
            flush=True,
        )

    status = {
        "status": "target_layers_accepted",
        "architecture": "PDelta3-GDN2-CLVR+LocalW",
        "base_model": args.base_model,
        "teacher_probe_nll": teacher_probe_nll,
        "accepted_layers": accepted_layers,
        "target_layers": target_layers,
        "total_model_layers": num_layers,
        "config": config.to_dict(),
        "structure": structural_summary(student),
        "elapsed_minutes": (time.perf_counter() - start) / 60.0,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        ),
        "message": (
            "Pilot target reached. Increase --target-layers (up to the full model) "
            "and rerun with --resume to continue the same accepted prefix."
        ),
    }
    write_status(output_dir, status)
    _atomic_json_save(
        {
            **status,
            "layer_reports": layer_reports,
        },
        output_dir / "sequential_training_report.json",
    )
    tokenizer.save_pretrained(output_dir)
    print("\nPILOT TARGET COMPLETE", flush=True)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted by user. Previously saved accepted/current best checkpoints remain on Drive.",
            file=sys.stderr,
            flush=True,
        )
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise
