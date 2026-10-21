"""Laya V5: activation-aware sparse PTQ and decision-margin fidelity evaluation.

Designed for native PyTorch SDPA + FP16 materialization, not low-bit GPU GEMM.
Requires the separately pinned laya_mixedbit_sparse_v4.py module (same process).
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
try:
    # Normal installed package import (CI/tests/library users).
    from .laya_mixedbit_sparse_v4 import PackedMixedLinear, PackedSparseLinear
except ImportError:
    # Standalone Colab downloads both pinned source files to /content.
    from laya_mixedbit_sparse_v4 import PackedMixedLinear, PackedSparseLinear

@torch.no_grad()
def capture_input_moments(model, names, batches, stride=8, max_batches=12):
    """Calibrate per-projection input RMS/mean without saving token activations.

    Accumulators live on GPU; each hook handles only sampled rows. The model is
    unmodified when context exits. This is calibration, not development/audit.
    """
    if stride<1:raise ValueError("stride must be positive")
    names=list(names)
    sums={}; sqs={}; counts={}; handles=[]
    for name in names:
        mod=model.get_submodule(name)
        if not isinstance(mod,nn.Linear):
            raise TypeError(f"Expected untouched nn.Linear: {name}")
        width=mod.in_features
        sums[name]=torch.zeros(width,device=mod.weight.device,dtype=torch.float32)
        sqs[name]=torch.zeros_like(sums[name])
        counts[name]=0
        def hook(_mod,args,key=name):
            src=args[0].detach()
            x=src.reshape(-1,src.shape[-1])[::stride].float()
            sums[key].add_(x.sum(dim=0))
            sqs[key].add_(x.square().sum(dim=0))
            counts[key]+=int(x.shape[0])
        handles.append(mod.register_forward_pre_hook(hook))
    try:
        model.eval()
        for batch in batches[:max_batches]:
            with torch.inference_mode(),torch.autocast(
                    "cuda",dtype=torch.float16,
                    enabled=next(model.parameters()).is_cuda):
                model(**batch)
    finally:
        for h in handles:h.remove()
    stats={}
    for name in names:
        n=max(1,counts[name])
        mean=sums[name]/n
        rms=(sqs[name]/n).clamp_min(1e-12).sqrt()
        # Cap huge activation outliers when choosing sparse weight errors.
        median=rms.median().clamp_min(1e-6)
        scaled=(rms/median).clamp(0.33,3.0)
        stats[name]={"rms":scaled.detach().cpu().contiguous(),
                     "mean":mean.detach().cpu().contiguous(),
                     "count":counts[name]}
        if counts[name]==0:raise RuntimeError(f"No input samples for {name}")
    return stats


@torch.no_grad()
def quantize_aware(linear,bits,sparse_k,stats=None,style="weighted",
                   correct_bias=False):
    """PackedSparseLinear using input-RMS-weighted top-k per output row.

    Only sparse indices and residual values change. The saved buffer structure
    remains compatible with the exact V4 packed delta loader.
    'plain' reproduces baseline top-k magnitude; 'weighted' uses activation RMS.
    """
    if style not in ("plain","weighted"):raise ValueError(style)
    packed=PackedSparseLinear.from_linear(linear,bits,sparse_k)
    if sparse_k==0 or style=="plain":
        if correct_bias and linear.bias is not None and stats is not None:
            residual=linear.weight.detach().float()-packed.weight_fp16().float()
            shift=residual@stats["mean"].to(residual.device)
            packed.bias.add_(shift.to(packed.bias.dtype))
            packed.release_cache()
        return packed
    if stats is None:
        raise ValueError("Activation stats required for weighted corrections")
    # Base quantization without any sparse residuals (exact same codes/scales).
    base=PackedMixedLinear.weight_fp16(packed).float()
    residual=linear.weight.detach().float()-base
    rms=stats["rms"].to(residual.device,dtype=torch.float32)
    if len(rms)!=linear.in_features:raise ValueError("Feature RMS length mismatch")
    weighted=residual.abs()*rms.reshape(1,-1)
    indices=weighted.topk(sparse_k,dim=1).indices
    values=residual.gather(1,indices)
    packed.outlier_indices.copy_(indices.to(torch.int16))
    packed.outlier_values.copy_(values.to(torch.float16))
    packed.release_cache()
    if correct_bias and linear.bias is not None:
        correction=(linear.weight.detach().float()-packed.weight_fp16().float())
        shift=correction@stats["mean"].to(correction.device)
        packed.bias.add_(shift.to(packed.bias.dtype))
        packed.release_cache()
    return packed


@torch.inference_mode()
def collect_decisions(model,batches,teacher_logits):
    """Agreement and robust confidence-margin diagnostics against FP16 teacher.

    Stores no per-example text. Relative-to-teacher new_error_count tracks
    mistakes not already present in a saved all-INT8 baseline mask.
    """
    agreement=0;count=0;total_js=0.0
    new_mask=[];max_drift=0.0
    erosion=0.0
    for batch,original in zip(batches,teacher_logits):
        pred=model(**batch)
        pred=pred[0] if isinstance(pred,(list,tuple)) else pred
        pred=pred.detach().float().cpu()
        original=original.float().cpu()
        marker=batch["marker_mask"].bool().cpu()
        if pred.shape!=original.shape or marker.shape!=pred.shape:
            raise ValueError("Teacher/student logits or marker shape mismatch")
        student=pred.masked_fill(~marker,-1e4)
        teacher=original.masked_fill(~marker,-1e4)
        probs=student.softmax(-1).clamp_min(1e-9)
        targets=teacher.softmax(-1).clamp_min(1e-9)
        average=(probs+targets)*0.5
        js=0.5*(probs*(probs/average).log()).sum(-1)
        js+=0.5*(targets*(targets/average).log()).sum(-1)
        tgt=targets.argmax(-1)
        arg=probs.argmax(-1)
        correct=(tgt==arg)
        new_mask.extend((~correct).tolist())
        agreement+=int(correct.sum())
        count+=len(correct)
        total_js+=float(js.sum())
        winner=teacher.gather(1,tgt[:,None]).squeeze(-1)
        winner_student=student.gather(1,tgt[:,None]).squeeze(-1)
        competitor=student.masked_fill(
            F.one_hot(tgt,student.shape[1]).bool(),-1e4).amax(dim=-1)
        teacher_runner=teacher.masked_fill(
            F.one_hot(tgt,teacher.shape[1]).bool(),-1e4).amax(dim=-1)
        original_margin=winner-teacher_runner
        student_margin=winner_student-competitor
        erosion+=float(((original_margin-student_margin).clamp_min(0)
                        / original_margin.clamp_min(0.1)).clamp_max(10).sum())
        max_drift=max(max_drift,float((student-teacher).abs().max()))
    if not count:raise ValueError("No decisions")
    return {
        "n":count,"agreement":agreement/count,
        "errors":count-agreement,
        "mean_js":total_js/count,
        "margin_erosion":erosion/count,
        "max_abs_logit_drift":max_drift,
        "error_mask":new_mask,
    }


def safe_accept(candidate,baseline,previous,limit_js,allowed_new=0,
                max_extra_erosion=0.030):
    """Pass both source-case-disjoint development splits using hard error mask.

    Never re-labels holdout data as development. Count *new* errors relative to
    the all-INT8 source baseline, not merely the total net accuracy change.
    """
    if candidate["n"]!=baseline["n"] or candidate["n"]!=previous["n"]:
        return False
    new=sum(bool(c) and not bool(b) for c,b in zip(
        candidate["error_mask"],baseline["error_mask"]))
    returned=sum(bool(c) and not bool(p) for c,p in zip(
        candidate["error_mask"],previous["error_mask"]))
    return (
        new<=allowed_new and returned<=allowed_new
        and candidate["errors"]<=previous["errors"]
        and candidate["mean_js"]<=float(limit_js)
        and candidate["margin_erosion"]<=baseline["margin_erosion"]+max_extra_erosion
    )


def public_metrics(x):
    return {k:v for k,v in x.items() if k!="error_mask"}
