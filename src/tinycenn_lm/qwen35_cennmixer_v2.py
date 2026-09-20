from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CeNNMixerV2Config:
    hidden_size: int = 1024
    groups: int = 32
    cell_dim: int = 48
    graph_steps: int = 1
    neighbor_offsets: tuple[int, ...] = (1, 2, 4, 8)
    fast_decay_init: float = 0.55
    mid_decay_init: float = 0.88
    slow_decay_init: float = 0.985
    dropout: float = 0.0

    @property
    def state_dim(self) -> int:
        return self.groups * self.cell_dim

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.groups < 2:
            raise ValueError("groups must be >= 2")
        if self.cell_dim < 8:
            raise ValueError("cell_dim must be >= 8")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not self.neighbor_offsets:
            raise ValueError("neighbor_offsets must not be empty")
        for p in (self.fast_decay_init, self.mid_decay_init, self.slow_decay_init):
            if not 0.0 < p < 1.0:
                raise ValueError("decay initializers must be in (0,1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["neighbor_offsets"] = list(self.neighbor_offsets)
        return d


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(torch.log(torch.tensor(p / (1.0 - p))))


class CeNNMixerV2(nn.Module):
    """Multi-timescale sparse cellular recurrent sequence mixer.

    v2 is designed for *parallel teacher-mixer distillation*: the Qwen mixer
    remains frozen beside this module while alpha is increased from 0 to 1.
    Therefore this module does not use the zero-output initialization from v1;
    all of its parameters receive useful gradients immediately at alpha=0.
    """

    def __init__(self, cfg: CeNNMixerV2Config, *, device=None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        H, G, D, S = cfg.hidden_size, cfg.groups, cfg.cell_dim, cfg.state_dim

        self.in_proj = nn.Linear(H, S, bias=False, device=device, dtype=torch.float32)
        self.gate_proj = nn.Linear(H, G * 9, bias=True, device=device, dtype=torch.float32)

        self.state_fast = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.state_mid = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)
        self.state_slow = nn.Linear(D, D, bias=False, device=device, dtype=torch.float32)

        # Fuse all three timescales back to Qwen hidden width.
        self.out_proj = nn.Linear(S * 3, H, bias=False, device=device, dtype=torch.float32)

        nn.init.normal_(self.in_proj.weight, std=0.018)
        nn.init.normal_(self.gate_proj.weight, std=0.006)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.008)

        for m in (self.state_fast, self.state_mid, self.state_slow):
            nn.init.eye_(m.weight)
            m.weight.data.mul_(0.08)

        self.fast_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.fast_decay_init), device=device))
        self.mid_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.mid_decay_init), device=device))
        self.slow_decay_logit = nn.Parameter(torch.tensor(_logit(cfg.slow_decay_init), device=device))

        n = len(cfg.neighbor_offsets)
        self.neighbor_fast = nn.Parameter(torch.zeros(n, device=device))
        self.neighbor_mid = nn.Parameter(torch.zeros(n, device=device))
        self.neighbor_slow = nn.Parameter(torch.zeros(n, device=device))

        # Gain starts at 1.0, unlike v1's zero output scale.
        self.output_gain_raw = nn.Parameter(torch.tensor(0.0, device=device))

        self._stream_fast: Optional[Tensor] = None
        self._stream_mid: Optional[Tensor] = None
        self._stream_slow: Optional[Tensor] = None

    @property
    def output_gain(self) -> Tensor:
        # 2*sigmoid(0)=1. This keeps gain positive and bounded in (0,2).
        return 2.0 * torch.sigmoid(self.output_gain_raw)

    def reset_stream_state(self) -> None:
        self._stream_fast = self._stream_mid = self._stream_slow = None

    def _neighbor_mix(self, state: Tensor, weights: Tensor) -> Tensor:
        out = state
        for _ in range(self.cfg.graph_steps):
            mixed = out
            for w, off in zip(weights, self.cfg.neighbor_offsets):
                mixed = mixed + 0.5 * torch.tanh(w) * (
                    torch.roll(out, shifts=off, dims=1)
                    + torch.roll(out, shifts=-off, dims=1)
                )
            out = mixed
        return out

    def _step(self, x_t: Tensor, sf: Tensor, sm: Tensor, ss: Tensor):
        B = x_t.shape[0]
        G, D = self.cfg.groups, self.cfg.cell_dim

        u = self.in_proj(x_t.float()).view(B, G, D)
        gates = self.gate_proj(x_t.float()).view(B, G, 9)
        ef,wf,rf, em,wm,rm, es,ws,rs = [
            torch.sigmoid(gates[..., i:i+1]) for i in range(9)
        ]

        df = torch.sigmoid(self.fast_decay_logit)
        dm = torch.sigmoid(self.mid_decay_logit)
        ds = torch.sigmoid(self.slow_decay_logit)

        nf = self._neighbor_mix(sf, self.neighbor_fast)
        nm = self._neighbor_mix(sm, self.neighbor_mid)
        ns = self._neighbor_mix(ss, self.neighbor_slow)

        # Cross-timescale messages give slow cells a compressed view of faster state.
        cf = F.silu(u + self.state_fast(nf))
        cm = F.silu(u + self.state_mid(nm) + 0.12 * sf)
        cs = F.silu(u + self.state_slow(ns) + 0.08 * sm)

        sf = df * (1.0 - ef) * sf + wf * cf
        sm = dm * (1.0 - em) * sm + wm * cm
        ss = ds * (1.0 - es) * ss + ws * cs

        read = torch.cat((rf * sf, rm * sm, rs * ss), dim=-1).reshape(B, G * D * 3)
        y = self.out_proj(read)
        return y, sf, sm, ss

    def forward(self, hidden_states: Tensor, *, streaming: bool = False) -> Tensor:
        B, T, _ = hidden_states.shape
        dev = hidden_states.device

        if (
            streaming and T == 1
            and self._stream_fast is not None
            and self._stream_fast.shape[0] == B
        ):
            sf = self._stream_fast.to(dev)
            sm = self._stream_mid.to(dev)
            ss = self._stream_slow.to(dev)
        else:
            shape = (B, self.cfg.groups, self.cfg.cell_dim)
            sf = torch.zeros(shape, device=dev, dtype=torch.float32)
            sm = torch.zeros_like(sf)
            ss = torch.zeros_like(sf)

        outs = []
        for t in range(T):
            y, sf, sm, ss = self._step(hidden_states[:, t], sf, sm, ss)
            outs.append(y)

        out = torch.stack(outs, dim=1)
        out = self.output_gain * out
        if self.training and self.cfg.dropout:
            out = F.dropout(out, p=self.cfg.dropout)
        out = out.to(hidden_states.dtype)

        if streaming:
            self._stream_fast = sf.detach()
            self._stream_mid = sm.detach()
            self._stream_slow = ss.detach()
        return out


def _primary(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def _replace_primary(result, primary):
    if isinstance(result, tuple):
        return (primary, *result[1:])
    if isinstance(result, list):
        return [primary, *result[1:]]
    return primary


class ProgressiveCeNNMixerV2(nn.Module):
    """Wrap a frozen Qwen mixer and progressively hand control to CeNN.

    output = (1-alpha)*QwenMixer(x) + alpha*CeNN(x)

    The wrapper always computes the frozen teacher mixer during distillation and
    exposes both outputs for a direct mixer-output loss. At alpha=0 the model's
    externally visible output is exactly the original Qwen mixer output.
    """

    def __init__(self, original: nn.Module, cenn: CeNNMixerV2, kind: str):
        super().__init__()
        self.original = original
        self.cenn = cenn
        self.kind = kind
        self.register_buffer("alpha", torch.tensor(0.0, dtype=torch.float32), persistent=True)
        self.last_original: Optional[Tensor] = None
        self.last_cenn: Optional[Tensor] = None

        for p in self.original.parameters():
            p.requires_grad_(False)

    def set_alpha(self, alpha: float) -> None:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.alpha.fill_(float(alpha))

    def clear_last(self) -> None:
        self.last_original = self.last_cenn = None

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        # Original mixer remains the local teacher throughout progressive training.
        with torch.no_grad():
            original_result = self.original(hidden_states, *args, **kwargs)
        original_y = _primary(original_result)

        streaming = (
            kwargs.get("cache_params", None) is not None
            or kwargs.get("past_key_values", None) is not None
        )
        cenn_y = self.cenn(hidden_states, streaming=streaming)

        self.last_original = original_y.detach()
        self.last_cenn = cenn_y

        a = self.alpha.to(device=cenn_y.device, dtype=cenn_y.dtype)
        mixed = (1.0 - a) * original_y.to(cenn_y.dtype) + a * cenn_y
        return _replace_primary(original_result, mixed)


class CeNNOnlyMixerV2(nn.Module):
    """Final wrapper after alpha=1; contains no original Qwen mixer."""

    def __init__(self, cenn: CeNNMixerV2, kind: str):
        super().__init__()
        self.cenn = cenn
        self.kind = kind

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        streaming = (
            kwargs.get("cache_params", None) is not None
            or kwargs.get("past_key_values", None) is not None
        )
        y = self.cenn(hidden_states, streaming=streaming)
        if self.kind == "full_attention":
            return y, None
        return y


def mixer_attr(layer: nn.Module) -> tuple[str, str]:
    block_type = getattr(layer, "block_type", None)
    if block_type == "linear_attention" and hasattr(layer, "linear_attn"):
        return "linear_attn", "linear_attention"
    if block_type == "full_attention" and hasattr(layer, "self_attn"):
        return "self_attn", "full_attention"
    raise RuntimeError(f"Unsupported Qwen3.5 mixer: block_type={block_type!r}")


def install_cenn_mixer_v2(
    model: nn.Module, layer_idx: int, cfg: CeNNMixerV2Config
) -> ProgressiveCeNNMixerV2:
    layer = model.model.layers[layer_idx]
    attr, kind = mixer_attr(layer)
    original = getattr(layer, attr)
    ref = next(original.parameters())
    cenn = CeNNMixerV2(cfg, device=ref.device)
    wrapper = ProgressiveCeNNMixerV2(original, cenn, kind)
    setattr(layer, attr, wrapper)
    return wrapper


def iter_progressive_wrappers(model: nn.Module):
    for m in model.modules():
        if isinstance(m, ProgressiveCeNNMixerV2):
            yield m


def set_alpha_v2(model: nn.Module, alpha: float) -> None:
    for m in iter_progressive_wrappers(model):
        m.set_alpha(alpha)


def reset_stream_state_v2(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, CeNNMixerV2):
            m.reset_stream_state()


def direct_mixer_loss_v2(model: nn.Module) -> Tensor:
    losses = []
    for m in iter_progressive_wrappers(model):
        if m.last_original is None or m.last_cenn is None:
            continue
        t = m.last_original.float()
        s = m.last_cenn.float()
        den = t.square().mean().clamp_min(1e-12)
        losses.append((s - t).square().mean() / den)
    if not losses:
        raise RuntimeError("No progressive CeNN wrapper outputs available for mixer loss")
    return torch.stack(losses).mean()


def freeze_all_except_cenn_v2(model: nn.Module) -> list[Tensor]:
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    seen = set()
    for m in model.modules():
        if isinstance(m, CeNNMixerV2):
            for p in m.parameters():
                if id(p) not in seen:
                    p.requires_grad_(True)
                    params.append(p)
                    seen.add(id(p))
    return params


def clone_cenn_state_v2(model: nn.Module) -> dict:
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, CeNNMixerV2):
            state[name] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return state


def load_cenn_state_v2(model: nn.Module, state: dict) -> None:
    mods = dict(model.named_modules())
    for name, sd in state.items():
        mods[name].load_state_dict(sd, strict=True)


def finalize_cenn_only_v2(model: nn.Module) -> nn.Module:
    """Remove all frozen Qwen mixers from replaced layers after alpha=1."""
    for layer in model.model.layers:
        block_type = getattr(layer, "block_type", None)
        if block_type == "linear_attention" and isinstance(
            getattr(layer, "linear_attn", None), ProgressiveCeNNMixerV2
        ):
            w = layer.linear_attn
            layer.linear_attn = CeNNOnlyMixerV2(w.cenn, w.kind)
        elif block_type == "full_attention" and isinstance(
            getattr(layer, "self_attn", None), ProgressiveCeNNMixerV2
        ):
            w = layer.self_attn
            layer.self_attn = CeNNOnlyMixerV2(w.cenn, w.kind)
    return model


__all__ = [
    "CeNNMixerV2Config",
    "CeNNMixerV2",
    "ProgressiveCeNNMixerV2",
    "CeNNOnlyMixerV2",
    "mixer_attr",
    "install_cenn_mixer_v2",
    "iter_progressive_wrappers",
    "set_alpha_v2",
    "reset_stream_state_v2",
    "direct_mixer_loss_v2",
    "freeze_all_except_cenn_v2",
    "clone_cenn_state_v2",
    "load_cenn_state_v2",
    "finalize_cenn_only_v2",
]
