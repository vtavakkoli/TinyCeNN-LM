#!/usr/bin/env python3
"""Colab entrypoint for the Tiny-LLM PDelta3 compatibility benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import scripts.benchmark_tiny_llm_pdelta3 as benchmark
from scripts.benchmark_cenn_research_layers import time_kernel
from tinycenn_lm.research_layers import softmax_reference


@torch.no_grad()
def fixed_diagnostics(core, spec, sample, device):
    q, k, v, _ = (x.to(device) for x in sample)
    route = benchmark.route_for(spec, v)
    eager_ms, eager_peak = time_kernel(
        lambda: core(q, k, v, routed_v=route), device, repeats=5
    )
    exact_ms, exact_peak = time_kernel(
        lambda: softmax_reference(q, k, v, core.groups), device, repeats=5
    )
    result = {
        "prefill_ms": eager_ms,
        "transformer_reference_ms": exact_ms,
        "prefill_speed_ratio": exact_ms / eager_ms,
        "peak_extra_bytes": eager_peak,
        "transformer_peak_extra_bytes": exact_peak,
        "state_bytes": core.recurrent_state_bytes(context=q.shape[2]),
    }
    result.update(core.decay_statistics())
    result.update(core.gate_statistics())
    return result


benchmark.diagnostics = fixed_diagnostics

if __name__ == "__main__":
    benchmark.main()
