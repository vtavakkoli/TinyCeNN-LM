"""PDelta3 temporal memory with shared nonlinear readout refinement.

Reference implementation: FP32 recurrent arithmetic, bounded local attention,
stateful decoding. No fused-kernel or speedup claim. Only unpadded causal inputs,
append-only caches, and greedy decoding are supported.
"""
from __future__ import annotations

import copy
import math
import weakref
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .pdelta3_frontier import FrontierPDelta3Layer, FrontierState

FORMAT = "tinycenn-nonlinear-readout-v1"


@dataclass(frozen=True)
class ReadoutConfig:
    feature_dim: int = 64
    window: int = 32
    rank: int = 16
    steps: int = 1
    chunk_size: int = 32
    conv_kernel: int = 4
    state_dtype: str = "fp32"
    local_gate_init: float = 0.75
    variant: str = "recurrent"

    def __post_init__(self):
        if min(self.feature_dim, self.window, self.rank, self.conv_kernel) < 1:
            raise ValueError("Feature, window, rank, and convolution sizes must be positive")
        if self.steps not in (0, 1, 2) or not 1 <= self.chunk_size <= 32:
            raise ValueError("Use 0/1/2 refinement steps and chunk_size in [1,32]")
        if self.state_dtype not in ("fp32", "fp16"):
            raise ValueError("state_dtype must be fp32 or fp16")
        if not 0 < self.local_gate_init < 1:
            raise ValueError("local_gate_init must be in (0,1)")
        if self.variant not in ("recurrent", "attention_control"):
            raise ValueError("Unknown variant")


class SharedNonlinearReadout(nn.Module):
    """u <- u + sigmoid(eta) A SiLU(B RMSNorm(u) + C RMSNorm(q)).

    q is the current token's projected query. A/B/C/eta are shared across depth;
    no temporal state is added. Zero A makes initialization an exact identity.
    """
    def __init__(self, heads, dim, rank, steps):
        super().__init__()
        self.steps = steps
        if steps:
            self.a = nn.Parameter(torch.zeros(heads, dim, rank))
            self.b = nn.Parameter(torch.randn(heads, rank, dim) / math.sqrt(dim))
            self.c = nn.Parameter(torch.randn(heads, rank, dim) / math.sqrt(dim))
            self.eta = nn.Parameter(torch.full((heads,), -2.0))

    @staticmethod
    def norm(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    def forward(self, read, query):
        if not self.steps:
            return read
        condition = torch.einsum("bhtd,hrd->bhtr", self.norm(query), self.c)
        scale = self.eta.sigmoid()[None, :, None, None]
        for _ in range(self.steps):
            hidden = F.silu(torch.einsum("bhtd,hrd->bhtr", self.norm(read), self.b) + condition)
            read = read + scale * torch.einsum("bhtr,hdr->bhtd", hidden, self.a)
        return read


def local_attention(q, k, v, window, groups):
    """Exact W-token attention; GQA views and O(B*Hq*T*W) scores, no T*T mask.

    k/v contain at most W-1 cached prefix tokens followed by the new tokens.
    """
    b, hq, t, d = q.shape
    hk, total = k.shape[1], k.shape[2]
    if hq != hk * groups or total < t:
        raise ValueError("Invalid GQA or prefix shapes")
    kw = F.pad(k, (0, 0, window - 1, 0)).unfold(2, window, 1)[:, :, -t:]
    vw = F.pad(v, (0, 0, window - 1, 0)).unfold(2, window, 1)[:, :, -t:]
    scores = torch.einsum("bhgtd,bhtdw->bhgtw", q.reshape(b, hk, groups, t, d), kw)
    positions = torch.arange(total - t, total, device=q.device)
    offsets = torch.arange(window, device=q.device) - window + 1
    valid = positions[:, None] + offsets[None, :] >= 0
    weights = (scores / math.sqrt(d)).masked_fill(~valid[None, None, None], -torch.inf).softmax(-1)
    return torch.einsum("bhgtw,bhtdw->bhgtd", weights, vw).reshape(b, hq, t, d)


@dataclass
class ReadoutState:
    recurrent: FrontierState
    keys: Tensor
    values: Tensor
    position: int

    @property
    def nbytes(self):
        tensors = [self.keys, self.values] + [getattr(self.recurrent, f.name) for f in fields(FrontierState)]
        return sum(x.numel() * x.element_size() for x in tensors if isinstance(x, Tensor))


class NonlinearMemory(nn.Module):
    def __init__(self, heads, kv_heads, dim, config: ReadoutConfig):
        super().__init__()
        self.config = config
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.groups = heads // kv_heads
        self.memory = FrontierPDelta3Layer(
            heads, kv_heads, dim, feature_dim=config.feature_dim,
            variant="conv4_gdn2_clvr_f96", chunk_size=config.chunk_size,
            conv_kernel=config.conv_kernel, state_dtype=config.state_dtype,
        )
        self.refine = SharedNonlinearReadout(heads, dim, config.rank, config.steps)
        self.gate_w = nn.Parameter(torch.zeros(heads, dim))
        self.gate_b = nn.Parameter(torch.full((heads,), math.log(config.local_gate_init / (1-config.local_gate_init))))

    def forward(self, q, k, v, routed_v=None, state=None, return_state=False):
        # Keep the chunk solve and recurrent update out of ambient autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            routed_v = v if routed_v is None else routed_v.float()
            read, recurrent = self.memory(
                q, k, v, routed_v=routed_v,
                state=None if state is None else state.recurrent, return_state=True,
            )
            read = self.refine(read, q)
            keys = k if state is None else torch.cat((state.keys.float(), k), dim=2)
            values = v if state is None else torch.cat((state.values.float(), v), dim=2)
            local = local_attention(q, keys, values, self.config.window, self.groups)
            gate = (torch.einsum("bhtd,hd->bht", q, self.gate_w) + self.gate_b[None, :, None]).sigmoid()
            output = gate[..., None] * local + (1 - gate[..., None]) * read
            if not return_state:
                return output
            keep = self.config.window - 1
            dtype = self.memory.storage_dtype
            # clone avoids retaining the whole prefill allocation through a view.
            kt = keys[:, :, -keep:] if keep else keys[:, :, :0]
            vt = values[:, :, -keep:] if keep else values[:, :, :0]
            new = ReadoutState(recurrent, kt.to(dtype).clone(), vt.to(dtype).clone(),
                               (0 if state is None else state.position) + q.shape[2])
            return output, new


class NonlinearAttention(nn.Module):
    def __init__(self, original, model_config, config, index, previous=None):
        super().__init__()
        self.original, self.recipe, self.layer_idx = original, config, index
        self.family = model_config.model_type
        self.h = model_config.num_attention_heads
        self.hk = model_config.num_key_value_heads
        self.d = getattr(original, "head_dim", model_config.hidden_size // self.h)
        self.config = model_config
        self.is_causal = True
        self.scaling = self.d ** -0.5
        self.attention_dropout = 0.0
        self.last_value = None
        object.__setattr__(self, "_previous", weakref.ref(previous) if previous is not None else None)
        if config.variant == "attention_control":
            self.readout = nn.Parameter(torch.eye(self.d).expand(self.h, -1, -1).clone())
            self.core = None
        else:
            self.core = NonlinearMemory(self.h, self.hk, self.d, config)
        self.to(original.q_proj.weight.device)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, **kwargs):
        if position_embeddings is None:
            raise ValueError("Position embeddings are required")
        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is not None and not hasattr(cache, "memory_states"):
            raise TypeError("Use nonlinear_readout.new_cache(model)")
        if cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Train with use_cache=False")
        if attention_mask is not None:
            if attention_mask.ndim != 4 or bool((attention_mask[..., -1, :] < 0).any()):
                raise ValueError("Only unpadded causal inputs are supported")
        b, t, _ = hidden_states.shape
        gate = None
        if self.family == "qwen3_5_text":
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            q, gate = self.original.q_proj(hidden_states).view(b, t, self.h, 2*self.d).chunk(2, -1)
            q = self.original.q_norm(q).transpose(1, 2)
            k = self.original.k_norm(self.original.k_proj(hidden_states).view(b, t, self.hk, self.d)).transpose(1, 2)
        else:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            q = self.original.q_proj(hidden_states).view(b, t, self.h, self.d).transpose(1, 2)
            k = self.original.k_proj(hidden_states).view(b, t, self.hk, self.d).transpose(1, 2)
        v = self.original.v_proj(hidden_states).view(b, t, self.hk, self.d).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        previous = self._previous() if self._previous is not None else None
        route = previous.last_value if previous is not None else v
        if route is None or route.shape != v.shape:
            route = v
        # Preserve gradients across jointly trained replacement layers.
        if self.core is None:
            if cache is not None:
                k, v = cache.update(k, v, self.layer_idx)
            output = F.scaled_dot_product_attention(
                q, k.repeat_interleave(self.h // self.hk, dim=1),
                v.repeat_interleave(self.h // self.hk, dim=1), attn_mask=attention_mask,
                is_causal=attention_mask is None and t > 1, scale=self.scaling,
            )
            with torch.autocast(device_type=q.device.type, enabled=False):
                output = torch.einsum("bhtd,hde->bhte", output.float(), self.readout)
        elif cache is not None:
            output, state = self.core(q, k, v, route, cache.memory_states.get(self.layer_idx), True)
            cache.memory_states[self.layer_idx] = state
        else:
            output = self.core(q, k, v, route)
        # This temporary is cleared at the end of each complete model forward.
        self.last_value = v
        flat = output.transpose(1, 2).reshape(b, t, self.h*self.d).to(hidden_states.dtype)
        if gate is not None:
            flat = flat * gate.reshape(b, t, -1).sigmoid()
        return self.original.o_proj(flat), None


def model_config(model):
    cfg = model.config
    return cfg.get_text_config(decoder=True) if hasattr(cfg, "get_text_config") else cfg


def eligible_layers(model):
    cfg = model_config(model)
    if cfg.model_type not in ("llama", "qwen3_5_text"):
        raise ValueError(f"Unsupported model family: {cfg.model_type}")
    return [i for i, layer in enumerate(model.model.layers)
            if cfg.model_type == "llama" or getattr(layer, "block_type", None) == "full_attention"]


def wrappers(model):
    return [layer.self_attn for layer in model.model.layers
            if isinstance(getattr(layer, "self_attn", None), NonlinearAttention)]


def clear_routes(model, args, output):
    for adapter in wrappers(model):
        adapter.last_value = None


def install(model, layers, config):
    allowed = eligible_layers(model)
    if not layers or len(set(layers)) != len(layers) or any(i not in allowed for i in layers):
        raise ValueError(f"Choose unique full-attention layers from {allowed}")
    model.requires_grad_(False)
    previous = None
    for index in sorted(layers):
        original = model.model.layers[index].self_attn
        if isinstance(original, NonlinearAttention):
            raise ValueError("Already replaced")
        # Identical memory initialization across R0/R1/R2, even at later layers.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() + 1009 * index)
            current = NonlinearAttention(original, model_config(model), config, index, previous)
        model.model.layers[index].self_attn = current
        previous = current
    model.register_forward_hook(clear_routes)
    return model.eval()


def build_student(teacher, layers, config):
    return install(copy.deepcopy(teacher), layers, config)


def new_cache(model):
    layers = [a.layer_idx for a in wrappers(model) if a.core is not None]
    if model_config(model).model_type == "qwen3_5_text":
        from .qwen35_integrated_memory import Qwen35IntegratedCache
        return Qwen35IntegratedCache(model_config(model), layers)
    from .integrated_memory import IntegratedCache
    return IntegratedCache(layers)


def adapter_payload(model, metadata=None):
    return {"format": FORMAT, "metadata": metadata or {}, "adapters": {
        str(a.layer_idx): {"config": asdict(a.recipe), "state_dict": {
            k: v.detach().cpu().clone() for k, v in a.state_dict().items() if not k.startswith("original.")
        }} for a in wrappers(model)}}


def load_adapter(model, payload):
    if payload.get("format") != FORMAT:
        raise ValueError("Wrong checkpoint format")
    if set(payload["adapters"]) != {str(a.layer_idx) for a in wrappers(model)}:
        raise ValueError("Checkpoint layer mismatch")
    for a in wrappers(model):
        saved = payload["adapters"][str(a.layer_idx)]
        if saved["config"] != asdict(a.recipe):
            raise ValueError("Checkpoint configuration mismatch")
        missing, unexpected = a.load_state_dict(saved["state_dict"], strict=False)
        if unexpected or any(not key.startswith("original.") for key in missing):
            raise ValueError("Incomplete checkpoint")


def restore_student(teacher, payload):
    if payload.get("format") != FORMAT or not payload.get("adapters"):
        raise ValueError("Wrong checkpoint format")
    configs = [ReadoutConfig(**x["config"]) for x in payload["adapters"].values()]
    if any(c != configs[0] for c in configs):
        raise ValueError("Mixed configurations unsupported")
    model = build_student(teacher, [int(i) for i in payload["adapters"]], configs[0])
    load_adapter(model, payload)
    return model
