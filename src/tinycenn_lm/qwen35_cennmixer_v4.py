from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen35_cennmixer_v4_core import CeNNMixerV4, CeNNMixerV4Config


def _primary(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def _replace_primary(result, primary):
    if isinstance(result, tuple):
        return (primary, *result[1:])
    if isinstance(result, list):
        return [primary, *result[1:]]
    return primary


def _joint_subspace(*weights: Tensor, rank: int) -> Tensor:
    """Return an orthonormal row projector spanning the dominant output subspace."""
    rows = weights[0].shape[0]
    rank = min(int(rank), rows)
    cov = torch.zeros(rows, rows, device=weights[0].device, dtype=torch.float32)
    for w in weights:
        x = w.detach().float()
        cov = cov + x @ x.t()
    _, vecs = torch.linalg.eigh(cov)
    return vecs[:, -rank:].t().contiguous()


@torch.no_grad()
def initialize_from_qwen35_v4(original: nn.Module, cenn: CeNNMixerV4) -> dict:
    """Spectrally compress one native Qwen3.5 Gated-Delta mixer into CeNNMixer-v4.

    For each native head we learn no arbitrary head permutation.  Instead we:
      * preserve the number of heads,
      * find a shared dominant q/k subspace so dot-product geometry is retained,
      * find a shared dominant v/z subspace so memory values and output gates align,
      * re-expand the compressed value coordinates through the native output map,
      * copy beta/decay projections and Qwen's learned continuous-time decay terms,
      * approximate each compressed depthwise-conv channel by energy-weighted
        aggregation of native channel kernels.

    This gives training a teacher-informed starting point rather than asking a
    random compact recurrent model to discover the native dynamics from scratch.
    """
    required = (
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
        "conv1d", "out_proj", "A_log", "dt_bias",
        "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim",
    )
    missing = [x for x in required if not hasattr(original, x)]
    if missing:
        return {"used": False, "reason": "missing native Gated-Delta attributes", "missing": missing}

    Hh = int(original.num_v_heads)
    Hk = int(original.num_k_heads)
    dkt = int(original.head_k_dim)
    dvt = int(original.head_v_dim)
    dks = int(cenn.cfg.key_dim)
    dvs = int(cenn.cfg.value_dim)

    if Hh != Hk:
        return {"used": False, "reason": "grouped value heads are not yet supported by spectral init"}
    if cenn.cfg.assoc_heads != Hh:
        return {
            "used": False,
            "reason": f"student heads={cenn.cfg.assoc_heads} but teacher heads={Hh}; preserve heads for aligned init",
        }
    if dks > dkt or dvs > dvt:
        return {"used": False, "reason": "student subspace cannot exceed teacher head dimension"}

    W = original.in_proj_qkv.weight.detach().float()
    Wz = original.in_proj_z.weight.detach().float()
    key_total = Hk * dkt
    val_total = Hh * dvt
    if W.shape[0] != 2 * key_total + val_total:
        return {"used": False, "reason": "unexpected QKV projection layout"}

    Wq = W[:key_total].view(Hh, dkt, -1)
    Wk = W[key_total:2 * key_total].view(Hh, dkt, -1)
    Wv = W[2 * key_total:].view(Hh, dvt, -1)
    Wz = Wz.view(Hh, dvt, -1)

    student_q = []
    student_k = []
    student_v = []
    student_z = []
    student_out = []
    qk_projectors = []
    v_projectors = []

    teacher_out = original.out_proj.weight.detach().float().view(cenn.cfg.hidden_size, Hh, dvt)

    for h in range(Hh):
        Pqk = _joint_subspace(Wq[h], Wk[h], rank=dks)
        Pv = _joint_subspace(Wv[h], Wz[h], rank=dvs)
        qk_projectors.append(Pqk)
        v_projectors.append(Pv)

        student_q.append(Pqk @ Wq[h])
        student_k.append(Pqk @ Wk[h])
        student_v.append(Pv @ Wv[h])
        student_z.append(Pv @ Wz[h])

        # If y_s = Pv y_t, then y_t ~= Pv^T y_s.  Compose native output
        # projection with Pv^T to initialize the compressed output map.
        student_out.append(teacher_out[:, h, :] @ Pv.t())

    cenn.in_proj_qkv.weight.copy_(
        torch.cat(
            (
                torch.cat(student_q, dim=0),
                torch.cat(student_k, dim=0),
                torch.cat(student_v, dim=0),
            ),
            dim=0,
        ).to(cenn.in_proj_qkv.weight)
    )
    cenn.z_proj.weight.copy_(torch.cat(student_z, dim=0).to(cenn.z_proj.weight))
    cenn.assoc_out.weight.copy_(torch.cat(student_out, dim=1).to(cenn.assoc_out.weight))

    cenn.beta_proj.weight.copy_(original.in_proj_b.weight.detach().to(cenn.beta_proj.weight))
    cenn.decay_proj.weight.copy_(original.in_proj_a.weight.detach().to(cenn.decay_proj.weight))
    cenn.assoc_log_rate.copy_(original.A_log.detach().to(cenn.assoc_log_rate))
    cenn.assoc_dt_bias.copy_(original.dt_bias.detach().to(cenn.assoc_dt_bias))

    # Qwen's RMS norm weight is zero-centered.  A scalar mean is a conservative
    # approximation after changing basis; training is free to specialize it.
    if hasattr(original, "norm") and hasattr(original.norm, "weight"):
        cenn.assoc_norm_weight.fill_(float(original.norm.weight.detach().float().mean()))

    tw = original.conv1d.weight.detach().float().squeeze(1)
    if tw.shape[-1] == cenn.cfg.conv_kernel and tw.shape[0] == 2 * key_total + val_total:
        qconv = tw[:key_total].view(Hh, dkt, -1)
        kconv = tw[key_total:2 * key_total].view(Hh, dkt, -1)
        vconv = tw[2 * key_total:].view(Hh, dvt, -1)
        cq, ck, cv = [], [], []
        for h in range(Hh):
            Pqk2 = qk_projectors[h].square()
            Pv2 = v_projectors[h].square()
            cq.append(Pqk2 @ qconv[h])
            ck.append(Pqk2 @ kconv[h])
            cv.append(Pv2 @ vconv[h])
        cenn.conv_weight.copy_(torch.cat((torch.cat(cq), torch.cat(ck), torch.cat(cv))).to(cenn.conv_weight))

    return {
        "used": True,
        "teacher_heads": Hh,
        "teacher_key_dim": dkt,
        "teacher_value_dim": dvt,
        "student_heads": cenn.cfg.assoc_heads,
        "student_key_dim": dks,
        "student_value_dim": dvs,
        "method": "per-head joint spectral q/k + joint spectral v/z + composed output projection",
    }


class DirectCeNNMixerV4(nn.Module):
    """Training wrapper: native mixer is target-only, CeNN output is always active."""

    def __init__(self, original: nn.Module, cenn: CeNNMixerV4, kind: str):
        super().__init__()
        self.original = original
        self.cenn = cenn
        self.kind = kind
        self.last_original: Optional[Tensor] = None
        self.last_cenn: Optional[Tensor] = None
        for p in self.original.parameters():
            p.requires_grad_(False)
        self.init_report = initialize_from_qwen35_v4(original, cenn) if kind == "linear_attention" else {
            "used": False,
            "reason": "spectral initialization currently targets Qwen3.5 linear-attention layers",
        }

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        # One replacement layer only: upstream is frozen, so the local teacher
        # path does not need an autograd graph.
        with torch.no_grad():
            original_result = self.original(hidden_states, *args, **kwargs)
        original_y = _primary(original_result)

        streaming = kwargs.get("cache_params") is not None or kwargs.get("past_key_values") is not None
        cenn_y = self.cenn(hidden_states, streaming=streaming)

        self.last_original = original_y.detach()
        self.last_cenn = cenn_y

        # No alpha interpolation.  The compact model is the model from step 1.
        return _replace_primary(original_result, cenn_y)


class CeNNOnlyMixerV4(nn.Module):
    def __init__(self, cenn: CeNNMixerV4, kind: str):
        super().__init__()
        self.cenn = cenn
        self.kind = kind

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        streaming = kwargs.get("cache_params") is not None or kwargs.get("past_key_values") is not None
        y = self.cenn(hidden_states, streaming=streaming)
        return (y, None) if self.kind == "full_attention" else y


def mixer_attr(layer: nn.Module):
    bt = getattr(layer, "block_type", None)
    if bt == "linear_attention" and hasattr(layer, "linear_attn"):
        return "linear_attn", "linear_attention"
    if bt == "full_attention" and hasattr(layer, "self_attn"):
        return "self_attn", "full_attention"
    raise RuntimeError(f"Unsupported Qwen3.5 mixer: {bt!r}")


def install_cenn_mixer_v4(model: nn.Module, layer_idx: int, cfg: CeNNMixerV4Config):
    layer = model.model.layers[layer_idx]
    attr, kind = mixer_attr(layer)
    original = getattr(layer, attr)
    ref = next(original.parameters())
    wrapper = DirectCeNNMixerV4(original, CeNNMixerV4(cfg, device=ref.device), kind)
    setattr(layer, attr, wrapper)
    return wrapper


def iter_wrappers_v4(model: nn.Module):
    for m in model.modules():
        if isinstance(m, DirectCeNNMixerV4):
            yield m


def reset_stream_state_v4(model: nn.Module):
    for m in model.modules():
        if isinstance(m, CeNNMixerV4):
            m.reset_stream_state()


def _rel_mse(s: Tensor, t: Tensor) -> Tensor:
    return (s - t).square().mean() / t.square().mean().clamp_min(1e-12)


def direct_mixer_losses_v4(model: nn.Module) -> dict[str, Tensor]:
    mses = []
    cosines = []
    deltas = []
    multiscales = []
    rmses = []
    tails = []

    for m in iter_wrappers_v4(model):
        if m.last_original is None or m.last_cenn is None:
            continue

        t = m.last_original.float()
        s = m.last_cenn.float()
        mses.append(_rel_mse(s, t))

        sf = s.reshape(-1, s.shape[-1])
        tf = t.reshape(-1, t.shape[-1])
        cosines.append((1.0 - F.cosine_similarity(sf, tf, dim=-1, eps=1e-6)).mean())

        lag_losses = []
        for lag in (1, 4, 16, 64):
            if s.shape[1] > lag:
                lag_losses.append(_rel_mse(s[:, lag:] - s[:, :-lag], t[:, lag:] - t[:, :-lag]))
        deltas.append(torch.stack(lag_losses).mean() if lag_losses else mses[-1].new_zeros(()))

        pooled = []
        for width in (8, 32, 128):
            if s.shape[1] >= width:
                ps = F.avg_pool1d(s.transpose(1, 2), kernel_size=width, stride=width).transpose(1, 2)
                pt = F.avg_pool1d(t.transpose(1, 2), kernel_size=width, stride=width).transpose(1, 2)
                pooled.append(_rel_mse(ps, pt))
        multiscales.append(torch.stack(pooled).mean() if pooled else mses[-1])

        sr = torch.sqrt(s.square().mean(-1).clamp_min(1e-12))
        tr = torch.sqrt(t.square().mean(-1).clamp_min(1e-12))
        rmses.append((torch.log(sr) - torch.log(tr)).square().mean())

        tail = min(128, s.shape[1])
        tails.append(_rel_mse(s[:, -tail:], t[:, -tail:]))

    if not mses:
        raise RuntimeError("No CeNN-v4 local outputs available")

    return {
        "mse": torch.stack(mses).mean(),
        "cosine": torch.stack(cosines).mean(),
        "delta": torch.stack(deltas).mean(),
        "multiscale": torch.stack(multiscales).mean(),
        "rms": torch.stack(rmses).mean(),
        "tail": torch.stack(tails).mean(),
    }


def freeze_all_except_cenn_v4(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    seen = set()
    for m in model.modules():
        if isinstance(m, CeNNMixerV4):
            for p in m.parameters():
                if id(p) not in seen:
                    p.requires_grad_(True)
                    params.append(p)
                    seen.add(id(p))
    return params


def clone_cenn_state_v4(model: nn.Module):
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, CeNNMixerV4):
            out[name] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return out


def load_cenn_state_v4(model: nn.Module, state: dict):
    mods = dict(model.named_modules())
    for name, sd in state.items():
        mods[name].load_state_dict(sd, strict=True)


def finalize_cenn_only_v4(model: nn.Module):
    for layer in model.model.layers:
        bt = getattr(layer, "block_type", None)
        if bt == "linear_attention" and isinstance(getattr(layer, "linear_attn", None), DirectCeNNMixerV4):
            w = layer.linear_attn
            layer.linear_attn = CeNNOnlyMixerV4(w.cenn, w.kind)
        elif bt == "full_attention" and isinstance(getattr(layer, "self_attn", None), DirectCeNNMixerV4):
            w = layer.self_attn
            layer.self_attn = CeNNOnlyMixerV4(w.cenn, w.kind)
    return model


__all__ = [
    "CeNNMixerV4Config", "CeNNMixerV4", "DirectCeNNMixerV4", "CeNNOnlyMixerV4",
    "initialize_from_qwen35_v4", "install_cenn_mixer_v4", "reset_stream_state_v4",
    "direct_mixer_losses_v4", "freeze_all_except_cenn_v4",
    "clone_cenn_state_v4", "load_cenn_state_v4", "finalize_cenn_only_v4",
]
