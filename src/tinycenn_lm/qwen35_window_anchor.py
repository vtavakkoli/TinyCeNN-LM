"""Window-and-anchor token mixers for text-only Qwen3.5-0.8B.

Research-only: causal full-sequence forward, use_cache=False. Neither CeNN nor
DeltaNet recurrence is used by the student. Original FFNs are untouched.
DeltaNet pretrained Q/K/V, convolution, gating and output projections provide
initialization; the recurrence/delta rule itself is REMOVED.

The official Qwen/Qwen3.5-0.8B checkpoint is multimodal. load_official_text()
extracts only language_model.* and lm_head.* weights. This module is included
with any exported custom weights; standard AutoModel.from_pretrained alone
cannot reconstruct the altered architecture.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_model

from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM, apply_rotary_pos_emb,
)

SOURCE = "Qwen/Qwen3.5-0.8B"


def causal_window_anchor(q, k, v, attention_mask, window=128, anchors=16,
                         chunk_size=128):
    """Causal local-window + *past prefix sink* attention; never bidirectional.

    Chunks the QUERY dimension. Keys are limited to past anchors + local span.
    Memory per attention chunk O(B*H*chunk*(window+anchors)), though Python
    chunks can make this slower than optimized DeltaNet kernels.
    """
    batch, heads, tokens, _ = q.shape
    assert k.shape[-2] == tokens and v.shape[-2] == tokens
    pieces = []
    valid = None
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        valid = attention_mask.bool()
    for start in range(0, tokens, chunk_size):
        end = min(tokens, start + chunk_size)
        first = max(0, start - int(window) + 1)
        # A prefix of globally visible *past* keys plus local keys.
        prefix_end = min(int(anchors), first)
        local_k = torch.arange(first, end, device=q.device)
        if prefix_end:
            prefix_k = torch.arange(prefix_end, device=q.device)
            indices = torch.cat((prefix_k, local_k))
        else:
            indices = local_k
        queries = torch.arange(start, end, device=q.device)
        allowed = (indices[None, :] <= queries[:, None]) & (
            (queries[:, None] - indices[None, :] < int(window))
            | (indices[None, :] < int(anchors)))
        mask = allowed[None, None, :, :].expand(batch, 1, -1, -1)
        if valid is not None:
            mask = mask & valid[:, None, None, indices]
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            original = attention_mask[..., start:end, :][:, :, :, indices]
            if original.dtype == torch.bool:
                mask = mask & original
            else:
                mask = mask & (original > -1e4)
        result = F.scaled_dot_product_attention(
            q[:, :, start:end, :], k[:, :, indices, :], v[:, :, indices, :],
            attn_mask=mask, dropout_p=0.0, is_causal=False,
        )
        pieces.append(result)
    return torch.cat(pieces, dim=-2)


class WindowAnchorDelta(nn.Module):
    """DeltaNet replacement: causal window/anchors; NO delta recurrence."""
    def __init__(self, old, window=128, anchors=16, chunk_size=128):
        super().__init__()
        self.layer_idx = old.layer_idx
        self.num_k_heads = old.num_k_heads
        self.num_v_heads = old.num_v_heads
        self.head_k_dim = old.head_k_dim
        self.head_v_dim = old.head_v_dim
        self.key_dim = old.key_dim
        self.value_dim = old.value_dim
        assert self.head_k_dim == self.head_v_dim, (
            "SDPA requires equal key and value head dimensions")
        self.window, self.anchors, self.chunk_size = (
            int(window), int(anchors), int(chunk_size))
        self.in_proj_qkv = copy.deepcopy(old.in_proj_qkv)
        self.conv1d = copy.deepcopy(old.conv1d)
        self.in_proj_z = copy.deepcopy(old.in_proj_z)
        self.norm = copy.deepcopy(old.norm)
        self.out_proj = copy.deepcopy(old.out_proj)
        self.activation = old.activation

    def forward(self, hidden_states, cache_params=None, attention_mask=None,
                **kwargs):
        if cache_params is not None:
            raise RuntimeError("Experimental all-24 replacement requires use_cache=False")
        b, t, _ = hidden_states.shape
        packed = self.in_proj_qkv(hidden_states).transpose(1, 2)
        # Native Qwen DeltaNet uses a causal depthwise convolution before QKV.
        conv = F.conv1d(packed, self.conv1d.weight, self.conv1d.bias,
                        padding=self.conv1d.kernel_size[0] - 1,
                        groups=self.conv1d.groups)[..., :t]
        if self.activation in ("silu", "swish"):
            conv = F.silu(conv)
        elif self.activation == "relu":
            conv = F.relu(conv)
        else:
            raise ValueError("Unsupported delta-conv activation " + str(self.activation))
        q, k, v = torch.split(
            conv.transpose(1, 2),
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, t, self.num_k_heads, self.head_k_dim).transpose(1, 2)
        k = k.reshape(b, t, self.num_k_heads, self.head_k_dim).transpose(1, 2)
        v = v.reshape(b, t, self.num_v_heads, self.head_v_dim).transpose(1, 2)
        if self.num_v_heads != self.num_k_heads:
            factor = self.num_v_heads // self.num_k_heads
            assert factor * self.num_k_heads == self.num_v_heads
            q = q.repeat_interleave(factor, dim=1)
            k = k.repeat_interleave(factor, dim=1)
        # The DeltaNet kernel used L2-normalized keys and queries; retain
        # this preprocessing, but remove all recurrent delta-rule updates.
        q = F.normalize(q.float(), dim=-1).to(v.dtype)
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        output = causal_window_anchor(q, k, v, attention_mask,
                                      self.window, self.anchors, self.chunk_size)
        output = output.transpose(1, 2).reshape(-1, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
        output = self.norm(output, z).reshape(b, t, self.value_dim)
        return self.out_proj(output)


class WindowAnchorFull(nn.Module):
    """Full GQA attention replacement with pretrained Q/K/V/gate and MRoPE."""
    def __init__(self, old, window=128, anchors=16, chunk_size=128):
        super().__init__()
        self.layer_idx = old.layer_idx
        self.head_dim = old.head_dim
        self.num_key_value_groups = old.num_key_value_groups
        self.window, self.anchors, self.chunk_size = (
            int(window), int(anchors), int(chunk_size))
        self.q_proj = copy.deepcopy(old.q_proj)
        self.k_proj = copy.deepcopy(old.k_proj)
        self.v_proj = copy.deepcopy(old.v_proj)
        self.o_proj = copy.deepcopy(old.o_proj)
        self.q_norm = copy.deepcopy(old.q_norm)
        self.k_norm = copy.deepcopy(old.k_norm)

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise RuntimeError("Experimental all-24 replacement requires use_cache=False")
        if position_embeddings is None:
            raise ValueError("Qwen3.5 partial MRoPE embeddings required")
        b, t, _ = hidden_states.shape
        q_and_gate = self.q_proj(hidden_states).view(b, t, -1, self.head_dim*2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, t, -1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(b, t, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(b, t, -1, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if self.num_key_value_groups != 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        output = causal_window_anchor(query, key, value, attention_mask,
                                      self.window, self.anchors, self.chunk_size)
        output = output.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(output * torch.sigmoid(gate)), None


def install_all24(model, window=128, anchors=16, chunk_size=128):
    layers = model.model.layers
    if len(layers) != 24:
        raise ValueError(f"Expected 24 Qwen3.5-0.8B layers, got {len(layers)}")
    counts = {"delta_replaced":0, "full_replaced":0}
    for i, layer in enumerate(layers):
        if layer.block_type == "linear_attention":
            old = layer.linear_attn
            if not isinstance(old, WindowAnchorDelta):
                layer.linear_attn = WindowAnchorDelta(old, window, anchors, chunk_size)
            counts["delta_replaced"] += 1
        elif layer.block_type == "full_attention":
            old = layer.self_attn
            if not isinstance(old, WindowAnchorFull):
                layer.self_attn = WindowAnchorFull(old, window, anchors, chunk_size)
            counts["full_replaced"] += 1
        else:
            raise ValueError(f"Unexpected Qwen3.5 layer type: {layer.block_type}")
    if counts != {"delta_replaced":18,"full_replaced":6}:
        raise RuntimeError("Incorrect Qwen3.5 0.8B layer layout " + repr(counts))
    return counts


def freeze_all_but_attention_projections(model):
    model.requires_grad_(False)
    names = []
    for layer in model.model.layers:
        block = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
        assert isinstance(block, (WindowAnchorDelta, WindowAnchorFull))
        projections = (
            ("in_proj_qkv", "out_proj") if isinstance(block, WindowAnchorDelta)
            else ("q_proj", "k_proj", "v_proj", "o_proj"))
        for name in projections:
            getattr(block, name).requires_grad_(True)
            names.append(f"{layer.block_type}.{name}")
    return names


def load_official_text(repo=SOURCE, dtype=torch.float16):
    """Download official multimodal checkpoint; load TEXT backbone only.

    Requires transformers with Qwen3.5 support, CPU RAM adequate for model
    and state tensors. Full vision module is never instantiated.
    """
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo, allow_patterns=[
        "config.json", "*.safetensors", "*.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "chat_template.jinja", "special_tokens_map.json"
    ]))
    cfg = AutoConfig.from_pretrained(str(root))
    text_cfg = cfg.text_config
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = Qwen3_5ForCausalLM(text_cfg)
    finally:
        torch.set_default_dtype(old)
    state = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith("model.language_model."):
                    new_key = "model." + key[len("model.language_model."):]
                elif key.startswith("lm_head."):
                    new_key = key
                else:
                    continue
                state[new_key] = reader.get_tensor(key)
    if len(state) < 100:
        raise RuntimeError("No text-backbone tensors found; inspect upstream key layout")
    missing, unexpected = model.load_state_dict(state, strict=False)
    illegal_missing = [x for x in missing if x != "lm_head.weight"]
    if illegal_missing or unexpected:
        raise RuntimeError(f"Checkpoint incompatible: missing={illegal_missing[:8]}, unexpected={unexpected[:8]}")
    model.tie_weights()
    model.eval().requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(str(root)), root


def save_custom(model, tokenizer, root, manifest):
    from safetensors.torch import save_model
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    save_model(model, str(root/"model.safetensors"))
    model.config.save_pretrained(str(root))
    tokenizer.save_pretrained(str(root))
    (root/"window_anchor_manifest.json").write_text(json.dumps(manifest,indent=2))
    (root/"qwen35_window_anchor.py").write_text(Path(__file__).read_text())


def load_custom(root, device="cuda"):
    from transformers import Qwen3_5TextConfig
    root = Path(root)
    manifest = json.loads((root/"window_anchor_manifest.json").read_text())
    cfg = Qwen3_5TextConfig.from_pretrained(str(root))
    dtype = torch.float16 if manifest["dtype"] == "float16" else torch.bfloat16
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = Qwen3_5ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(old)
    counts = install_all24(model, **manifest["attention_settings"])
    load_model(model, str(root/"model.safetensors"), strict=True, device="cpu")
    model.eval().requires_grad_(False)
    model.to(device)
    return model, AutoTokenizer.from_pretrained(str(root)), counts
