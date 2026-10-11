"""Layerwise reconstruction QAT for Jev_CeNN ternary g128 experiments.

Optimize projected weight codes and group scales from *calibration-only* teacher I/O.
This is a Bonsai-inspired research experiment, not the proprietary Bonsai algorithm.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def collect_projection_io(model, names, batches, token_budget=2048, seed=2030):
    """Collect matched linear inputs and outputs from the unmodified target layer."""
    if token_budget < 64:
        raise ValueError("token_budget must be at least 64")
    # Spread the finite budget across ALL calibration batches instead of
    # taking the first few full-length examples (strong source of overfitting).
    samples_per_batch = max(16, math.ceil(token_budget / max(1, len(batches))))
    samples = {name: [[], [], 0] for name in names}
    hooks = []
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear before calibration: {name}")

        def observe(mod, args, output, key=name):
            xs, ys, collected = samples[key]
            if collected >= token_budget:
                return
            x = args[0].detach().reshape(-1, mod.in_features)
            y = output.detach().reshape(-1, mod.out_features)
            n = min(token_budget-collected, samples_per_batch, x.shape[0])
            if n <= 0:
                return
            ids = torch.linspace(0, x.shape[0]-1, steps=n, device=x.device).long()
            xs.append(x.index_select(0, ids).contiguous())
            ys.append(y.index_select(0, ids).contiguous())
            samples[key][2] += n

        hooks.append(module.register_forward_hook(observe))
    try:
        with torch.inference_mode():
            for batch in batches:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=next(model.parameters()).is_cuda):
                    model(**batch)
                if all(s[2] >= token_budget for s in samples.values()):
                    break
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for name, (xs, ys, n) in samples.items():
        if n < 64:
            raise RuntimeError(f"Insufficient calibration tokens for {name}: {n}")
        x, y = torch.cat(xs, dim=0), torch.cat(ys, dim=0)
        generator = torch.Generator(device=x.device).manual_seed(
            seed + sum(map(ord, name)))
        perm = torch.randperm(len(x), generator=generator, device=x.device)
        result[name] = (x.index_select(0, perm).contiguous(),
                        y.index_select(0, perm).contiguous())
    return result


def _dequant_fake(latent, scales, outputs, inputs, padded_inputs):
    w = F.pad(latent, (0, padded_inputs-inputs)).reshape(outputs, -1, 128)
    q = (w / scales.unsqueeze(-1)).round().clamp(-1, 1)
    # STE gradient through latent; exact group-scale gradient through q * scales.
    fake = q.detach()*scales.unsqueeze(-1) + w - w.detach()
    return fake.reshape(outputs, padded_inputs)[:, :inputs], q.detach().to(torch.int8)


def train_projection_g128(original, quantized, activation_xy, pack_fn, *,
                          steps=60, batch_tokens=128, lr=0.0003,
                          patience=4, validate_every=10, reg_strength=0.002,
                          verbose=False):
    """Return best (not necessarily last) packed ternary projection and diagnostics.

    Early stop uses disjoint calibration tokens, never the decision-selection or
    independent audit cases. Worst case restores initial PTQ parameters.
    """
    if not isinstance(original, nn.Linear) or steps < 0:
        raise ValueError("Expected nn.Linear and steps >= 0")
    x, y = activation_xy
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Unaligned calibration input/output pairs")
    n = len(x)
    if n < 64:
        raise ValueError("At least 64 calibration tokens are required")
    n_val = max(32, min(n//4, n-32))
    train_x, train_y = x[:-n_val], y[:-n_val]
    val_x, val_y = x[-n_val:], y[-n_val:]
    out_features, in_features = original.weight.shape
    padded = quantized.kpad
    orig_weight = original.weight.detach().float()
    latent = nn.Parameter(orig_weight.clone())
    init_scales = quantized.group_scales.float().clamp_min(1e-7).detach()
    log_scales = nn.Parameter(init_scales.log().clone())
    packed_before = quantized.packed_weight.detach().clone()
    scales_before = quantized.group_scales.detach().clone()
    bias = original.bias.detach() if original.bias is not None else None

    @torch.no_grad()
    def measure():
        sc = log_scales.exp().clamp(1e-7, 1.0)
        w, q = _dequant_fake(latent, sc, out_features, in_features, padded)
        with torch.autocast("cuda", dtype=torch.float16, enabled=val_x.is_cuda):
            pred = F.linear(val_x, w, bias)
        nmse = ((pred.float()-val_y.float()).square().mean()
                / val_y.float().square().mean().clamp_min(1e-7))
        return float(nmse.item()), q.clone(), sc.detach().half().clone()

    # Baseline reflects actual saved FP16 group scales, rather than a latent proxy.
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=val_x.is_cuda):
            pred0 = F.linear(val_x, quantized.unpack_weight(), bias)
        baseline = float(((pred0.float()-val_y.float()).square().mean()
                         / val_y.float().square().mean().clamp_min(1e-7)).item())
    best, best_q, best_sc = baseline, None, None
    curve = [{"step": 0, "val_nmse": baseline}]
    no_improve = 0
    if steps:
        optimizer = torch.optim.AdamW(
            [{"params": [latent], "lr": lr},
             {"params": [log_scales], "lr": lr*0.5}], weight_decay=0.0)
        for step in range(1, steps+1):
            index = torch.randint(0, len(train_x),
                                  (min(batch_tokens, len(train_x)),),
                                  device=train_x.device)
            xb, yb = train_x.index_select(0, index), train_y.index_select(0, index)
            sc = log_scales.exp().clamp(1e-7, 1.0)
            w, _ = _dequant_fake(latent, sc, out_features, in_features, padded)
            with torch.autocast("cuda", dtype=torch.float16, enabled=xb.is_cuda):
                pred = F.linear(xb, w, bias)
            loss = ((pred.float()-yb.float()).square().mean()
                    / yb.float().square().mean().clamp_min(1e-7))
            penalty = reg_strength*((latent-orig_weight).square().mean()
                                    / orig_weight.square().mean().clamp_min(1e-7))
            total = loss + penalty
            if not torch.isfinite(total):
                break
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_([latent, log_scales], max_norm=1.0)
            optimizer.step()
            if step % validate_every == 0 or step == steps:
                score, q, scales = measure()
                curve.append({"step": step, "val_nmse": score})
                if math.isfinite(score) and score < best-1e-6:
                    best, best_q, best_sc = score, q, scales
                    no_improve = 0
                else:
                    no_improve += 1
                if verbose:
                    print("  QAT", step, "val_NMSE", round(score, 6),
                          "best", round(best, 6))
                if no_improve >= patience:
                    break
    improved = best_q is not None
    with torch.no_grad():
        if improved:
            quantized.packed_weight.copy_(pack_fn(best_q.reshape(out_features,padded)))
            quantized.group_scales.copy_(best_sc)
        else:
            quantized.packed_weight.copy_(packed_before)
            quantized.group_scales.copy_(scales_before)
    diagnostics = {
        "baseline_nmse": baseline, "best_nmse": best,
        "relative_nmse_improvement": max(0.0,(baseline-best)/max(baseline,1e-9)),
        "improved": improved, "steps_used": curve[-1]["step"],
        "convergence_curve": curve, "tokens": n}
    return quantized, diagnostics
