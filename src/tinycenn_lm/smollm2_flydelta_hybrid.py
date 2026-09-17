from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class FlyDeltaHybridConfig:
    """Hybrid local exact attention + FlyWire-controlled delta memory."""

    fly_nodes: int = 256
    local_window: int = 64
    anchor_window: int = 256
    anchor_every: int = 4
    graph_steps: int = 1
    delta_gate_init: float = 0.01
    graph_gain_init: float = 0.10
    eps: float = 1e-6

    def validate(self, model_config) -> None:
        if self.fly_nodes < 32:
            raise ValueError("fly_nodes must be >= 32")
        if self.local_window < 1 or self.anchor_window < self.local_window:
            raise ValueError("require 1 <= local_window <= anchor_window")
        if self.anchor_every < 2:
            raise ValueError("anchor_every must be >= 2")
        if self.graph_steps < 1:
            raise ValueError("graph_steps must be >= 1")
        if not 0.0 < self.delta_gate_init < 0.5:
            raise ValueError("delta_gate_init must be in (0, 0.5)")
        hidden = int(model_config.hidden_size)
        heads = int(model_config.num_attention_heads)
        kv_heads = int(model_config.num_key_value_heads)
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if heads % kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    def to_dict(self) -> dict:
        return asdict(self)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class FlyWireGateController(nn.Module):
    """Dense FlyWire controller producing decay/erase/write gates."""

    def __init__(
        self,
        hidden_size: int,
        kv_heads: int,
        adjacency: Tensor,
        graph_steps: int = 1,
        graph_gain_init: float = 0.10,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        self.fly_nodes = int(adjacency.shape[0])
        self.kv_heads = int(kv_heads)
        self.graph_steps = int(graph_steps)
        self.register_buffer("adjacency", adjacency, persistent=True)

        self.in_proj = nn.Linear(hidden_size, self.fly_nodes, bias=False)
        self.out_proj = nn.Linear(self.fly_nodes, self.kv_heads * 4, bias=True)
        self.graph_gain = nn.Parameter(torch.tensor(float(graph_gain_init), dtype=torch.float32))

        with torch.no_grad():
            self.out_proj.weight.zero_()
            bias = torch.zeros(self.kv_heads, 4)
            bias[:, 0] = 3.0
            bias[:, 1] = -2.0
            bias[:, 2] = -1.5
            bias[:, 3] = 0.0
            self.out_proj.bias.copy_(bias.reshape(-1))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.tanh(self.in_proj(hidden_states.float()))
        a = self.adjacency
        gain = torch.tanh(self.graph_gain.float())
        for _ in range(self.graph_steps):
            mixed = torch.matmul(x, a.t())
            x = torch.tanh(x + gain * mixed)

        gates = self.out_proj(x).view(*x.shape[:-1], self.kv_heads, 4)
        decay = 0.80 + 0.199 * torch.sigmoid(gates[..., 0])
        erase = 0.50 * torch.sigmoid(gates[..., 1])
        write = 0.75 * torch.sigmoid(gates[..., 2])
        content_mix = gates[..., 3]
        return decay, erase, write, content_mix


class FlyDeltaHybridAttention(nn.Module):
    """Exact local attention plus FlyWire-controlled delta-rule memory."""

    def __init__(
        self,
        original_attn: nn.Module,
        model_config,
        config: FlyDeltaHybridConfig,
        layer_idx: int,
        adjacency: Tensor,
    ) -> None:
        super().__init__()
        config.validate(model_config)
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = int(layer_idx)
        self.is_anchor = ((self.layer_idx + 1) % int(config.anchor_every) == 0)
        self.local_window = int(config.anchor_window if self.is_anchor else config.local_window)
        self.eps = float(config.eps)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(original_attn.q_proj.weight)
            self.k_proj.weight.copy_(original_attn.k_proj.weight)
            self.v_proj.weight.copy_(original_attn.v_proj.weight)
            self.o_proj.weight.copy_(original_attn.o_proj.weight)

        if self.is_anchor:
            self.register_parameter("delta_gate_logit", None)
            self.controller = None
        else:
            self.delta_gate_logit = nn.Parameter(
                torch.full((self.num_heads,), _logit(config.delta_gate_init), dtype=torch.float32)
            )
            self.controller = FlyWireGateController(
                hidden_size=self.hidden_size,
                kv_heads=self.num_key_value_heads,
                adjacency=adjacency,
                graph_steps=config.graph_steps,
                graph_gain_init=config.graph_gain_init,
            )

        self.streaming = False
        self._stream_s = None
        self._stream_k = None
        self._stream_v = None

    def reset_stream_state(self) -> None:
        self._stream_s = None
        self._stream_k = None
        self._stream_v = None

    def set_streaming(self, enabled: bool, reset: bool = False) -> None:
        self.streaming = bool(enabled)
        if reset:
            self.reset_stream_state()

    def _apply_rope(self, q: Tensor, k: Tensor, position_embeddings) -> tuple[Tensor, Tensor]:
        if position_embeddings is None:
            return q, k
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = position_embeddings
            return apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            return q, k

    def _local_exact_full(self, q: Tensor, k: Tensor, v: Tensor, attention_mask) -> Tensor:
        k_h = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_h = v.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.einsum("bhtd,bhsd->bhts", q.float(), k_h.float())
        scores = scores / math.sqrt(float(self.head_dim))

        t = q.shape[-2]
        pos = torch.arange(t, device=q.device)
        qpos = pos[:, None]
        kpos = pos[None, :]
        allowed = (kpos <= qpos) & (kpos >= qpos - self.local_window + 1)
        scores = scores.masked_fill(~allowed.view(1, 1, t, t), float("-inf"))

        if torch.is_tensor(attention_mask):
            try:
                if attention_mask.ndim == 2 and attention_mask.shape[-1] == t:
                    valid = attention_mask.to(q.device).bool().view(attention_mask.shape[0], 1, 1, t)
                    scores = scores.masked_fill(~valid, float("-inf"))
                elif attention_mask.ndim == 4 and attention_mask.shape[-2:] == (t, t):
                    scores = scores + attention_mask.to(scores)
            except Exception:
                pass

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.einsum("bhts,bhsd->bhtd", probs, v_h.float())

    def _local_exact_stream(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self._stream_k is None:
            out = self._local_exact_full(q, k, v, None)
        else:
            k_all = torch.cat([self._stream_k.to(k), k], dim=2)
            v_all = torch.cat([self._stream_v.to(v), v], dim=2)
            outs = []
            old = self._stream_k.shape[2]
            for j in range(q.shape[2]):
                upto = old + j + 1
                start = max(0, upto - self.local_window)
                kk = k_all[:, :, start:upto]
                vv = v_all[:, :, start:upto]
                kh = kk.repeat_interleave(self.num_key_value_groups, dim=1)
                vh = vv.repeat_interleave(self.num_key_value_groups, dim=1)
                score = torch.einsum("bhd,bhsd->bhs", q[:, :, j].float(), kh.float())
                score = score / math.sqrt(float(self.head_dim))
                prob = torch.softmax(score, dim=-1, dtype=torch.float32)
                outs.append(torch.einsum("bhs,bhsd->bhd", prob, vh.float()))
            out = torch.stack(outs, dim=2)

        keep = max(self.local_window - 1, 0)
        if keep:
            k_hist = k if self._stream_k is None else torch.cat([self._stream_k.to(k), k], dim=2)
            v_hist = v if self._stream_v is None else torch.cat([self._stream_v.to(v), v], dim=2)
            self._stream_k = k_hist[:, :, -keep:].detach()
            self._stream_v = v_hist[:, :, -keep:].detach()
        return out

    def _delta_memory(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.is_anchor:
            return torch.zeros_like(q, dtype=torch.float32), torch.zeros(
                q.shape[0], q.shape[2], self.num_heads, device=q.device, dtype=torch.float32
            )

        decay, erase, write, content_mix = self.controller(hidden_states)
        kf = F.normalize(k.float(), p=2, dim=-1)
        qf = F.normalize(q.float(), p=2, dim=-1)
        values = v.float()

        bsz, _, seq_len, d = kf.shape
        if self.streaming and self._stream_s is not None:
            state = self._stream_s.to(device=q.device, dtype=torch.float32)
            if state.shape[0] != bsz:
                state = torch.zeros(
                    bsz, self.num_key_value_heads, d, d, device=q.device, dtype=torch.float32
                )
        else:
            state = torch.zeros(
                bsz, self.num_key_value_heads, d, d, device=q.device, dtype=torch.float32
            )

        outputs = []
        mixes = []
        base_gate = self.delta_gate_logit.view(1, self.num_heads)
        for t in range(seq_len):
            kt = kf[:, :, t]
            vt = values[:, :, t]
            dec = decay[:, t].unsqueeze(-1).unsqueeze(-1)
            er = erase[:, t].unsqueeze(-1).unsqueeze(-1)
            wr = write[:, t].unsqueeze(-1).unsqueeze(-1)

            pred = torch.einsum("bkd,bkde->bke", kt, state)
            erase_outer = kt.unsqueeze(-1) * pred.unsqueeze(-2)
            write_outer = kt.unsqueeze(-1) * vt.unsqueeze(-2)
            state = dec * state - er * erase_outer + wr * write_outer

            state_h = state.repeat_interleave(self.num_key_value_groups, dim=1)
            out_t = torch.einsum("bhd,bhde->bhe", qf[:, :, t], state_h)
            outputs.append(out_t)

            cm = content_mix[:, t].repeat_interleave(self.num_key_value_groups, dim=1)
            mixes.append(torch.sigmoid(base_gate + 0.25 * torch.tanh(cm)))

        if self.streaming:
            self._stream_s = state.detach()

        return torch.stack(outputs, dim=2), torch.stack(mixes, dim=1)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        if self.streaming:
            local = self._local_exact_stream(q, k, v)
        else:
            local = self._local_exact_full(q, k, v, attention_mask)

        if self.is_anchor:
            mixed = local
        else:
            global_out, mix = self._delta_memory(q, k, v, hidden_states)
            mix_h = mix.transpose(1, 2).unsqueeze(-1)
            mixed = local.float() + mix_h * (global_out.float() - local.float())

        out = mixed.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out.to(dtype=hidden_states.dtype))
        return out, None


def replace_all_attention_with_flydelta(
    model: nn.Module,
    config: FlyDeltaHybridConfig,
    adjacency: Tensor,
) -> nn.Module:
    config.validate(model.config)
    for idx, layer in enumerate(model.model.layers):
        old = layer.self_attn
        if isinstance(old, FlyDeltaHybridAttention):
            continue
        new = FlyDeltaHybridAttention(old, model.config, config, idx, adjacency)
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        if new.controller is not None:
            new.controller.in_proj.weight.data = new.controller.in_proj.weight.data.to(old.q_proj.weight.dtype)
            new.controller.out_proj.weight.data = new.controller.out_proj.weight.data.to(old.q_proj.weight.dtype)
            new.controller.out_proj.bias.data = new.controller.out_proj.bias.data.to(old.q_proj.weight.dtype)
            new.controller.adjacency.data = new.controller.adjacency.data.float()
            new.controller.graph_gain.data = new.controller.graph_gain.data.float()
            new.delta_gate_logit.data = new.delta_gate_logit.data.float()
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model


def flydelta_modules(model: nn.Module) -> list[FlyDeltaHybridAttention]:
    return [m for m in model.modules() if isinstance(m, FlyDeltaHybridAttention)]


def set_flydelta_streaming(model: nn.Module, enabled: bool, reset: bool = False) -> None:
    for module in flydelta_modules(model):
        module.set_streaming(enabled, reset=reset)


def hybrid_layer_indices(model: nn.Module) -> list[int]:
    return [m.layer_idx for m in flydelta_modules(model) if not m.is_anchor]


def anchor_layer_indices(model: nn.Module) -> list[int]:
    return [m.layer_idx for m in flydelta_modules(model) if m.is_anchor]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def calibration_parameter_groups(
    model: nn.Module,
    layer_indices: list[int],
    main_lr: float,
    qkvo_lr: float,
    weight_decay: float = 0.01,
):
    selected = set(int(i) for i in layer_indices)
    freeze_all(model)
    main, qkvo = [], []
    for m in flydelta_modules(model):
        if m.is_anchor or m.layer_idx not in selected:
            continue
        for p in m.controller.parameters():
            p.requires_grad = True
            main.append(p)
        m.delta_gate_logit.requires_grad = True
        main.append(m.delta_gate_logit)
        for proj in (m.q_proj, m.k_proj, m.v_proj, m.o_proj):
            proj.weight.requires_grad = True
            qkvo.append(proj.weight)
    groups = [
        {"params": main, "lr": float(main_lr), "weight_decay": float(weight_decay)},
        {"params": qkvo, "lr": float(qkvo_lr), "weight_decay": float(weight_decay)},
    ]
    return groups, [*main, *qkvo]


def global_parameter_groups(
    model: nn.Module,
    main_lr: float,
    qkvo_lr: float,
    weight_decay: float = 0.01,
):
    return calibration_parameter_groups(
        model,
        hybrid_layer_indices(model),
        main_lr=main_lr,
        qkvo_lr=qkvo_lr,
        weight_decay=weight_decay,
    )


def flydelta_stats(model: nn.Module) -> dict[str, float | int]:
    modules = flydelta_modules(model)
    hybrids = [m for m in modules if not m.is_anchor]
    anchors = [m for m in modules if m.is_anchor]
    gates = []
    decays = []
    for m in hybrids:
        gates.append(float(torch.sigmoid(m.delta_gate_logit.detach().float()).mean().cpu()))
        with torch.no_grad():
            bias = m.controller.out_proj.bias.detach().float().view(m.num_key_value_heads, 4)
            decays.append(float((0.80 + 0.199 * torch.sigmoid(bias[:, 0])).mean().cpu()))
    return {
        "layers": len(modules),
        "hybrid_layers": len(hybrids),
        "anchor_layers": len(anchors),
        "mean_delta_gate": sum(gates) / max(len(gates), 1),
        "mean_initial_decay_proxy": sum(decays) / max(len(decays), 1),
    }


def assert_flydelta_replacement(model: nn.Module) -> None:
    modules = flydelta_modules(model)
    if len(modules) != int(model.config.num_hidden_layers):
        raise RuntimeError(
            f"expected {model.config.num_hidden_layers} FlyDelta attention modules, found {len(modules)}"
        )
    bad = [m.__class__.__name__ for m in model.modules() if m.__class__.__name__ == "LlamaAttention"]
    if bad:
        raise RuntimeError("standard LlamaAttention remains in model")
