from __future__ import annotations

import math
from dataclasses import dataclass, asdict
import torch
from torch import Tensor, nn
from .smollm2_amcenn_v2 import AdaptivePositiveSoftmaxFeatures

@dataclass(frozen=True)
class FlyGraphAttentionConfig:
    feature_dim: int = 256
    feature_seed: int = 7331
    graph_steps: int = 1
    graph_gate_init: float = 0.10
    edge_gain_scale: float = 0.50
    content_gate_init: float = 0.25
    eps: float = 1e-6
    antithetic_features: bool = True
    learnable_feature_correction: bool = True

    def validate(self, model_config) -> None:
        if self.feature_dim < 32:
            raise ValueError("feature_dim must be >= 32")
        if self.antithetic_features and self.feature_dim % 2:
            raise ValueError("antithetic feature_dim must be even")
        if self.graph_steps < 0:
            raise ValueError("graph_steps must be >= 0")
        if not 0.0 < self.graph_gate_init < 1.0:
            raise ValueError("graph_gate_init must be in (0, 1)")
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

class FlyGraphFeatureMixer(nn.Module):
    def __init__(self, src: Tensor, dst: Tensor, base_weight: Tensor, feature_dim: int,
                 graph_steps: int = 1, graph_gate_init: float = 0.10,
                 edge_gain_scale: float = 0.50, content_gate_init: float = 0.25,
                 eps: float = 1e-6) -> None:
        super().__init__()
        src = torch.as_tensor(src, dtype=torch.long)
        dst = torch.as_tensor(dst, dtype=torch.long)
        base_weight = torch.as_tensor(base_weight, dtype=torch.float32)
        if src.ndim != 1 or dst.ndim != 1 or base_weight.ndim != 1:
            raise ValueError("src, dst, base_weight must be 1D")
        if not (len(src) == len(dst) == len(base_weight)):
            raise ValueError("src, dst, base_weight lengths must match")
        if len(src) and (int(src.max()) >= feature_dim or int(dst.max()) >= feature_dim):
            raise ValueError("graph indices exceed feature_dim")
        self.feature_dim = int(feature_dim)
        self.graph_steps = int(graph_steps)
        self.edge_gain_scale = float(edge_gain_scale)
        self.eps = float(eps)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("base_weight", base_weight)
        self.edge_gain = nn.Parameter(torch.zeros(len(src), dtype=torch.float32))
        self.graph_gate_logit = nn.Parameter(torch.tensor(_logit(graph_gate_init), dtype=torch.float32))
        self.content_scale = nn.Parameter(torch.tensor(float(content_gate_init), dtype=torch.float32))
        self.self_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, phi: Tensor) -> Tensor:
        if self.graph_steps == 0 or self.src.numel() == 0:
            return phi
        x = phi.float()
        graph_gate = torch.sigmoid(self.graph_gate_logit)
        edge_gain = 1.0 + self.edge_gain_scale * torch.tanh(self.edge_gain)
        base = self.base_weight * edge_gain
        for _ in range(self.graph_steps):
            src_value = x.index_select(-1, self.src)
            dst_value = x.index_select(-1, self.dst)
            content_gate = torch.sigmoid(self.content_scale * (src_value - dst_value))
            messages = src_value * content_gate * base
            mixed = torch.zeros_like(x)
            mixed.index_add_(-1, self.dst, messages)
            self_residual = 0.10 * torch.tanh(self.self_gain) * x
            x = (x + graph_gate * mixed + self_residual).clamp_min(self.eps)
            x = x / x.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return x.to(dtype=phi.dtype)

class FlyGraphLinearAttention(nn.Module):
    def __init__(self, original_attn: nn.Module, model_config, config: FlyGraphAttentionConfig,
                 layer_idx: int, src: Tensor, dst: Tensor, base_weight: Tensor) -> None:
        super().__init__()
        config.validate(model_config)
        self.hidden_size = int(model_config.hidden_size)
        self.num_heads = int(model_config.num_attention_heads)
        self.num_key_value_heads = int(model_config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.feature_dim = int(config.feature_dim)
        self.eps = float(config.eps)
        self.layer_idx = int(layer_idx)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(original_attn.q_proj.weight)
            self.k_proj.weight.copy_(original_attn.k_proj.weight)
            self.v_proj.weight.copy_(original_attn.v_proj.weight)
            self.o_proj.weight.copy_(original_attn.o_proj.weight)

        self.features = AdaptivePositiveSoftmaxFeatures(
            self.head_dim, self.feature_dim,
            seed=int(config.feature_seed) + self.layer_idx,
            antithetic=bool(config.antithetic_features),
            learnable_correction=bool(config.learnable_feature_correction),
        )
        self.graph = FlyGraphFeatureMixer(
            src=src, dst=dst, base_weight=base_weight,
            feature_dim=self.feature_dim,
            graph_steps=int(config.graph_steps),
            graph_gate_init=float(config.graph_gate_init),
            edge_gain_scale=float(config.edge_gain_scale),
            content_gate_init=float(config.content_gate_init),
            eps=self.eps,
        )
        self.streaming = False
        self._stream_s = None
        self._stream_z = None

    def reset_stream_state(self) -> None:
        self._stream_s = None
        self._stream_z = None

    def set_streaming(self, enabled: bool, reset: bool = False) -> None:
        self.streaming = bool(enabled)
        if reset:
            self.reset_stream_state()

    def _apply_rope(self, q: Tensor, k: Tensor, position_embeddings):
        if position_embeddings is None:
            return q, k
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = position_embeddings
            return apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            return q, k

    def _padding_valid(self, attention_mask, bsz: int, seq_len: int, device):
        if not torch.is_tensor(attention_mask):
            return None
        if attention_mask.ndim == 2 and attention_mask.shape == (bsz, seq_len):
            return attention_mask.to(device=device).bool()
        return None

    def forward(self, hidden_states: Tensor, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache: bool = False, cache_position=None,
                position_embeddings=None, **kwargs):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, position_embeddings)

        phi_q = self.graph(self.features(q))
        phi_k = self.graph(self.features(k))
        phi_k_t = phi_k.transpose(1, 2).float()
        values = v.transpose(1, 2).float()
        valid = self._padding_valid(attention_mask, bsz, seq_len, hidden_states.device)
        if valid is not None:
            mask = valid[:, :, None, None].float()
            phi_k_t = phi_k_t * mask
            values = values * mask

        writes = phi_k_t.unsqueeze(-1) * values.unsqueeze(-2)
        prefix_s = writes.cumsum(dim=1)
        prefix_z = phi_k_t.cumsum(dim=1)

        if self.streaming and self._stream_s is not None:
            if self._stream_s.shape[0] != bsz:
                self.reset_stream_state()
            else:
                prefix_s = prefix_s + self._stream_s[:, None].to(prefix_s)
                prefix_z = prefix_z + self._stream_z[:, None].to(prefix_z)

        s_heads = prefix_s.repeat_interleave(self.num_key_value_groups, dim=2)
        z_heads = prefix_z.repeat_interleave(self.num_key_value_groups, dim=2)
        phi_q_t = phi_q.transpose(1, 2).float()
        numerator = torch.einsum("bthf,bthfd->bthd", phi_q_t, s_heads)
        denominator = torch.einsum("bthf,bthf->bth", phi_q_t, z_heads).unsqueeze(-1)
        context = numerator / denominator.clamp_min(self.eps)

        out = context.reshape(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out.to(dtype=hidden_states.dtype))

        if self.streaming:
            self._stream_s = prefix_s[:, -1].detach()
            self._stream_z = prefix_z[:, -1].detach()
        return out, None

def replace_all_attention_with_fly(model: nn.Module, config: FlyGraphAttentionConfig,
                                   src: Tensor, dst: Tensor, base_weight: Tensor) -> nn.Module:
    config.validate(model.config)
    for idx, layer in enumerate(model.model.layers):
        old = layer.self_attn
        new = FlyGraphLinearAttention(old, model.config, config, idx, src, dst, base_weight)
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        new.graph.edge_gain.data = new.graph.edge_gain.data.float()
        new.graph.graph_gate_logit.data = new.graph.graph_gate_logit.data.float()
        new.graph.content_scale.data = new.graph.content_scale.data.float()
        new.graph.self_gain.data = new.graph.self_gain.data.float()
        layer.self_attn = new
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    return model

def fly_attention_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, FlyGraphLinearAttention)]

def set_fly_streaming(model: nn.Module, enabled: bool, reset: bool = False) -> None:
    for module in fly_attention_modules(model):
        module.set_streaming(enabled, reset=reset)

def fly_parameter_groups(model: nn.Module, main_lr: float, qkvo_lr: float,
                         weight_decay: float = 0.01):
    for p in model.parameters():
        p.requires_grad = False
    main, qkvo = [], []
    for module in fly_attention_modules(model):
        for p in module.features.parameters():
            p.requires_grad = True
            main.append(p)
        for p in module.graph.parameters():
            p.requires_grad = True
            main.append(p)
        for proj in (module.q_proj, module.k_proj, module.v_proj, module.o_proj):
            proj.weight.requires_grad = True
            qkvo.append(proj.weight)
    groups = [
        {"params": main, "lr": float(main_lr), "weight_decay": float(weight_decay)},
        {"params": qkvo, "lr": float(qkvo_lr), "weight_decay": float(weight_decay)},
    ]
    return groups, [*main, *qkvo]

def fly_attention_stats(model: nn.Module):
    modules = fly_attention_modules(model)
    if not modules:
        raise RuntimeError("no FlyGraphLinearAttention modules found")
    gates = [float(torch.sigmoid(m.graph.graph_gate_logit.detach().float()).cpu()) for m in modules]
    scales = [float(m.graph.content_scale.detach().float().cpu()) for m in modules]
    return {
        "layers": len(modules),
        "mean_graph_gate": sum(gates) / len(gates),
        "min_graph_gate": min(gates),
        "max_graph_gate": max(gates),
        "mean_content_scale": sum(scales) / len(scales),
    }

def assert_pure_fly_attention(model: nn.Module) -> None:
    expected = int(model.config.num_hidden_layers)
    modules = fly_attention_modules(model)
    if len(modules) != expected:
        raise RuntimeError(f"expected {expected} fly attention layers, found {len(modules)}")
    leftovers = [m.__class__.__name__ for m in model.modules()
                 if "LlamaAttention" in m.__class__.__name__]
    if leftovers:
        raise RuntimeError(f"standard Llama attention remains: {leftovers[:4]}")
