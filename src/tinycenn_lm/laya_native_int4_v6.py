"""Laya V6: hardware-probed native INT4 linear kernels with strict speed gates.

IMPORTANT: TorchAO's INT4 kernels are *not* the same bit format as Laya V5's
G128 signed-code + sparse-residual checkpoint. Quantizing a reconstructed V5
FP16 layer introduces an additional approximation and MUST be revalidated.

Designed for native INT4 GEMM on supported CUDA GPUs. No FP16 unpack cache in
the execution path. T4/SM75 support varies by kernel; always measure.
"""
from __future__ import annotations

import copy
import statistics
from typing import Any

import torch
from torch import nn


def backend_configs(group_size: int = 128,
                    include_gemlite: bool = True) -> dict[str, Any]:
    """Return only APIs actually exposed by the installed TorchAO build."""
    if group_size not in (32, 64, 128, 256):
        raise ValueError("TorchAO supported group sizes: 32,64,128,256")
    from torchao.quantization import Int4WeightOnlyConfig

    configs: dict[str, Any] = {}
    for name, packing, chooser in (
        ("torchao_tiled_hqq", "tile_packed_to_4d", "hqq"),
        ("torchao_tiled_tinygemm", "tile_packed_to_4d", "tinygemm"),
        ("torchao_plain_tinygemm", "plain", "tinygemm"),
    ):
        try:
            configs[name] = Int4WeightOnlyConfig(
                group_size=group_size,
                int4_packing_format=packing,
                int4_choose_qparams_algorithm=chooser,
                set_inductor_config=False,
            )
        except (TypeError, ValueError, AssertionError, AttributeError):
            pass

    # Historical TorchAO builds expose GemLite as GemliteUIntXWeightOnlyConfig;
    # some newer builds removed that spelling. Its absence is NOT an error.
    if include_gemlite:
        try:
            import gemlite  # noqa: F401
            from torchao.quantization import GemliteUIntXWeightOnlyConfig
            configs["gemlite_int4"] = GemliteUIntXWeightOnlyConfig(
                group_size=group_size, bit_width=4, mode="weight_only",
                set_inductor_config=False)
        except (ImportError, TypeError, ValueError, AttributeError):
            pass
    return configs


class PackedDtypeAdapter(nn.Module):
    """Keep native packed BF16 computation while accepting FP16 encoder input.

    This path includes both cast overheads in every measured benchmark. It
    must not be presented as a speedup unless the *whole* model measures faster.
    """
    def __init__(self, packed: nn.Linear):
        super().__init__()
        self.packed = packed
        self.in_features = packed.in_features
        self.out_features = packed.out_features

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            return self.packed(x.to(torch.bfloat16)).to(x.dtype)


@torch.inference_mode()
def quantize_linear(linear: nn.Linear, config: Any) -> nn.Module:
    """Install native packed INT4 and validate a live CUDA forward.

    Try the original FP16 activation dtype first; on Ampere+ try packed BF16
    with explicit input/output casts. Neither path decodes a full FP16 weight.
    Unsupported kernels raise and must never masquerade as accepted INT4.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError("Expected nn.Linear")
    if linear.weight.device.type != "cuda":
        raise RuntimeError("Native TorchAO INT4 benchmark requires CUDA")
    from torchao.quantization import quantize_

    failures = []
    dtypes = [linear.weight.dtype]
    if (torch.cuda.get_device_capability(linear.weight.device)[0]>=8
            and torch.bfloat16 not in dtypes):
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        try:
            m = nn.Sequential(copy.deepcopy(linear).to(
                dtype=dtype).eval().requires_grad_(False))
            quantize_(m, config)
            packed = m[0]
            if not isinstance(packed, nn.Linear) or type(packed.weight) is torch.Tensor:
                raise RuntimeError("TorchAO did not install packed weight")
            returned = (PackedDtypeAdapter(packed) if dtype==torch.bfloat16
                        and dtype!=linear.weight.dtype else packed)
            probe = torch.randn(
                16,linear.in_features,device=linear.weight.device,
                dtype=linear.weight.dtype)
            y = returned(probe)
            if tuple(y.shape)!=(16,linear.out_features) or not torch.isfinite(y).all():
                raise RuntimeError("Packed native INT4 forward failed validation")
            return returned
        except (RuntimeError,TypeError,ValueError,AssertionError,NotImplementedError) as exc:
            failures.append(f"{dtype}: {str(exc)[:210]}")
    raise RuntimeError("No executable native INT4 backend: "+"; ".join(failures))


@torch.inference_mode()
def cuda_events_ms(fn, *, warmup=8, repeats=16) -> dict[str, float]:
    """CUDA event timing including actual invocation but excluding sync overhead.

    Synchronizes each sample and reports robust central and tail latency.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[min(
            len(times)-1, int(0.95*(len(times)-1)))],
        "repeats": repeats,
    }


@torch.inference_mode()
def compare_linear_kernels(original: nn.Linear,
                           candidate: nn.Linear,
                           m_values=(128, 512, 2048),
                           repeats=12) -> list[dict[str, float]]:
    """Benchmark representative encoder M = batch * sequence length.

    For ModernBERT encoder: a B4,T512 batch has M=2048 projection rows.
    Unlike autoregressive LLM decode, this is NOT M=1.
    """
    if not original.weight.is_cuda:
        raise RuntimeError("Require CUDA baseline")
    rows = []
    for m in m_values:
        activations = torch.randn(
            int(m), original.in_features, device=original.weight.device,
            dtype=original.weight.dtype)
        # Prove candidate has a valid forward at every requested row count.
        result = candidate(activations)
        if tuple(result.shape) != (int(m), original.out_features):
            raise RuntimeError(f"Unexpected candidate shape: {result.shape}")
        if not torch.isfinite(result).all():
            raise RuntimeError("INT4 kernel produced nonfinite outputs")
        dense = cuda_events_ms(lambda: original(activations), repeats=repeats)
        packed = cuda_events_ms(lambda: candidate(activations), repeats=repeats)
        ratio = dense["median_ms"]/packed["median_ms"]
        rows.append({
            "m":int(m), "fp16_ms":dense["median_ms"],
            "int4_ms":packed["median_ms"], "speedup":ratio,
            "fp16_p95_ms":dense["p95_ms"],
            "int4_p95_ms":packed["p95_ms"],
        })
    return rows


def speed_gate(rows, *, min_gain=1.10, target_m=(512, 2048)) -> bool:
    """Require a speed advantage at EVERY requested encoder workload."""
    indexed = {int(row["m"]):row for row in rows}
    return bool(target_m) and all(
        m in indexed and indexed[m]["speedup"]>=min_gain
        and indexed[m]["int4_ms"]>0 for m in target_m)


def candidate_names(model: nn.Module) -> list[str]:
    """All four encoder projection sites per layer, largest first."""
    layers = getattr(model.encoder,"layers")
    found = []
    for index in range(len(layers)):
        for component in ("mlp.Wi","mlp.Wo","attn.Wqkv","attn.Wo"):
            name=f"encoder.layers.{index}.{component}"
            if isinstance(model.get_submodule(name),nn.Linear):
                found.append(name)
    return sorted(found,
                  key=lambda name:model.get_submodule(name).weight.numel(),
                  reverse=True)


def swap_module(model: nn.Module, name: str, replacement: nn.Module) -> nn.Module:
    parent,_,field=name.rpartition(".")
    owner=model.get_submodule(parent)
    old=getattr(owner,field)
    setattr(owner,field,replacement)
    return old
