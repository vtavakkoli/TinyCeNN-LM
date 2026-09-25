"""Standalone PDelta3-GDN2 typed-decision model and game-control runtime helpers.

This module intentionally does not import the ``laya`` package.  It reuses the
dependency-free checkpoint reconstruction and typed-decision runtime from
``tinycenn_lm.standalone_decision`` and replaces:

1. every ModernBERT encoder layer whose ``attention_type`` is ``full_attention``;
2. every full softmax self-attention block in the typed-decision transformer head.

ModernBERT sliding/local attention layers are intentionally kept unchanged.
The exported model is a complete checkpoint, not a Laya adapter.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from .laya_lab.pdelta import PDelta3GDN2CLVRAttention
from .standalone_decision import (
    StandaloneDecisionModel,
    StandaloneDecisionRuntime,
    build_checkpoint_model,
    _apply_rope_compat,
)


class _HeadAttentionAdapter(nn.Module):
    """Expose ``nn.MultiheadAttention`` weights with a ModernBERT-like interface.

    ``PDelta3GDN2CLVRAttention`` expects an object with ``Wqkv``, ``Wo``,
    ``out_drop`` and a small attention config.  The decision head uses PyTorch
    ``nn.MultiheadAttention`` instead, so this adapter converts its projection
    weights without ever running dense attention.
    """

    def __init__(self, mha: nn.MultiheadAttention, layer_idx: int):
        super().__init__()
        if not mha.batch_first:
            raise ValueError("standalone decision head requires batch_first=True")
        if mha.kdim not in (None, mha.embed_dim) or mha.vdim not in (None, mha.embed_dim):
            raise ValueError("cross-dimensional MultiheadAttention is not supported")
        if mha.in_proj_weight is None:
            raise ValueError("separate Q/K/V projection weights are not supported")

        d = int(mha.embed_dim)
        heads = int(mha.num_heads)
        if d % heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.config = SimpleNamespace(
            hidden_size=d,
            num_attention_heads=heads,
        )
        self.layer_idx = int(layer_idx)
        self.head_dim = d // heads

        self.Wqkv = nn.Linear(d, 3 * d, bias=mha.in_proj_bias is not None)
        with torch.no_grad():
            self.Wqkv.weight.copy_(mha.in_proj_weight)
            if mha.in_proj_bias is not None:
                self.Wqkv.bias.copy_(mha.in_proj_bias)

        self.Wo = copy.deepcopy(mha.out_proj)
        self.out_drop = nn.Dropout(float(mha.dropout))


class PDelta3DecisionHeadLayer(nn.Module):
    """Transformer-head layer with PDelta3 instead of full softmax attention.

    The residual/normalization/feed-forward structure is copied from the source
    ``nn.TransformerEncoderLayer``.  Only the self-attention sublayer is
    replaced, preserving the learned FFN and LayerNorm parameters.
    """

    def __init__(
        self,
        original: nn.TransformerEncoderLayer,
        layer_idx: int,
        *,
        feature_dim: int = 96,
        conv_kernel: int = 4,
        chunk_size: int = 64,
        local_kernel: int = 5,
        local_window: int = 32,
        local_gate_init: float = 0.72,
    ):
        super().__init__()
        if not bool(getattr(original, "norm_first", False)):
            raise ValueError("expected norm_first=True decision head")

        self.layer_idx = int(layer_idx)
        self.norm_first = True
        self.norm1 = copy.deepcopy(original.norm1)
        self.norm2 = copy.deepcopy(original.norm2)
        self.linear1 = copy.deepcopy(original.linear1)
        self.linear2 = copy.deepcopy(original.linear2)
        self.dropout = copy.deepcopy(original.dropout)
        self.dropout1 = copy.deepcopy(original.dropout1)
        self.dropout2 = copy.deepcopy(original.dropout2)
        self.activation = original.activation

        adapter = _HeadAttentionAdapter(original.self_attn, layer_idx)
        self.attn = PDelta3GDN2CLVRAttention(
            adapter,
            feature_dim=int(feature_dim),
            conv_kernel=int(conv_kernel),
            chunk_size=int(chunk_size),
            local_kernel=int(local_kernel),
            local_window=int(local_window),
            local_gate_init=float(local_gate_init),
        )

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        **_: Any,
    ) -> Tensor:
        if src_mask is not None:
            raise ValueError("PDelta3 decision head does not accept src_mask")
        if is_causal:
            raise ValueError("typed-decision head is bidirectional, not causal")

        if src_key_padding_mask is None:
            valid = torch.ones(src.shape[:2], dtype=torch.long, device=src.device)
        else:
            valid = (~src_key_padding_mask.bool()).long()

        n1 = self.norm1(src)
        mixed, _ = self.attn(n1, attention_mask=valid)
        x = src + self.dropout1(mixed)
        x = x + self._ff_block(self.norm2(x))
        return x


class PDelta3DecisionHead(nn.Module):
    """Sequential decision head with no ``nn.MultiheadAttention`` modules."""

    def __init__(self, layers: Iterable[PDelta3DecisionHeadLayer]):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
        **kwargs: Any,
    ) -> Tensor:
        x = src
        causal = bool(is_causal) if is_causal is not None else False
        for layer in self.layers:
            x = layer(
                x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=causal,
                **kwargs,
            )
        return x


def encoder_full_attention_indices(model: nn.Module) -> list[int]:
    """Return encoder layers marked ``full_attention`` by ModernBERT."""
    return [
        i
        for i, layer in enumerate(model.encoder.layers)
        if str(getattr(layer, "attention_type", "")) == "full_attention"
    ]


def install_pdelta3_encoder(
    model: nn.Module,
    layer_indices: Iterable[int] | None = None,
    *,
    feature_dim: int = 96,
    conv_kernel: int = 4,
    chunk_size: int = 64,
    local_kernel: int = 5,
    local_window: int = 32,
    local_gate_init: float = 0.72,
) -> list[int]:
    """Replace every selected ModernBERT full-attention module with PDelta3."""
    indices = (
        encoder_full_attention_indices(model)
        if layer_indices is None
        else [int(i) for i in layer_indices]
    )
    installed: list[int] = []
    for i in indices:
        layer = model.encoder.layers[i]
        if str(getattr(layer, "attention_type", "")) != "full_attention":
            raise ValueError(f"encoder layer {i} is not full_attention")
        if not isinstance(layer.attn, PDelta3GDN2CLVRAttention):
            layer.attn = PDelta3GDN2CLVRAttention(
                layer.attn,
                feature_dim=int(feature_dim),
                conv_kernel=int(conv_kernel),
                chunk_size=int(chunk_size),
                local_kernel=int(local_kernel),
                local_window=int(local_window),
                local_gate_init=float(local_gate_init),
            )
        installed.append(i)
    return installed


def converted_encoder_indices(model: nn.Module) -> list[int]:
    """Return all source full-attention encoder layers already converted."""
    return [
        i
        for i in encoder_full_attention_indices(model)
        if isinstance(model.encoder.layers[i].attn, PDelta3GDN2CLVRAttention)
    ]


def install_pdelta3_head(
    model: nn.Module,
    *,
    feature_dim: int = 96,
    conv_kernel: int = 4,
    chunk_size: int = 64,
    local_kernel: int = 5,
    local_window: int = 32,
    local_gate_init: float = 0.72,
) -> list[int]:
    """Replace every decision-head ``nn.MultiheadAttention`` layer with PDelta3."""
    if model.head is None:
        return []
    if isinstance(model.head, PDelta3DecisionHead):
        return list(range(len(model.head.layers)))

    source_layers = list(model.head.layers)
    converted: list[PDelta3DecisionHeadLayer] = []
    for i, layer in enumerate(source_layers):
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise TypeError(
                f"unsupported decision head layer {i}: {type(layer).__name__}"
            )
        converted.append(
            PDelta3DecisionHeadLayer(
                layer,
                i,
                feature_dim=feature_dim,
                conv_kernel=conv_kernel,
                chunk_size=chunk_size,
                local_kernel=local_kernel,
                local_window=local_window,
                local_gate_init=local_gate_init,
            )
        )
    model.head = PDelta3DecisionHead(converted)
    return list(range(len(converted)))


def remaining_full_attention(model: nn.Module) -> dict[str, int]:
    """Count dense/full attention that remains after conversion."""
    encoder = sum(
        not isinstance(model.encoder.layers[i].attn, PDelta3GDN2CLVRAttention)
        for i in encoder_full_attention_indices(model)
    )
    head = sum(isinstance(m, nn.MultiheadAttention) for m in model.head.modules()) if model.head is not None else 0
    return {"encoder": int(encoder), "decision_head": int(head)}


def pdelta_modules(model: nn.Module) -> list[PDelta3GDN2CLVRAttention]:
    return [m for m in model.modules() if isinstance(m, PDelta3GDN2CLVRAttention)]


def pdelta_core_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Unique PDelta3 core parameters, excluding copied QKV/Wo projections."""
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in pdelta_modules(model):
        for p in module.trainable_core_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def pdelta_projection_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Unique PDelta3 QKV/Wo parameters for very-low-LR calibration."""
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in pdelta_modules(model):
        for child in (module.Wqkv, module.Wo):
            for p in child.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
    return params


def decision_head_adaptation_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Non-PDelta parameters useful for the short global stabilization pass."""
    params: list[nn.Parameter] = []
    seen: set[int] = set()

    blocked = {id(p) for p in pdelta_core_parameters(model)}
    blocked.update(id(p) for p in pdelta_projection_parameters(model))

    modules = [model.type_emb, model.scorer, model.act_head]
    if model.head is not None:
        modules.append(model.head)

    for module in modules:
        for p in module.parameters():
            if id(p) in blocked or id(p) in seen:
                continue
            seen.add(id(p))
            params.append(p)
    return params


def build_source_model(root: str | Path) -> tuple[StandaloneDecisionModel, dict[str, Any]]:
    """Dependency-free source checkpoint reconstruction."""
    return build_checkpoint_model(root)


def load_standalone_pdelta3(
    repo_or_dir: str,
    *,
    device: str | torch.device | None = None,
    token: str | None = None,
) -> StandaloneDecisionRuntime:
    """Load a full standalone PDelta3 decision model from disk or Hugging Face."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    root = Path(repo_or_dir)
    if not root.exists():
        root = Path(
            snapshot_download(
                repo_or_dir,
                token=token,
                allow_patterns=[
                    "standalone_config.json",
                    "model.safetensors",
                    "encoder/*",
                    "tokenizer/*",
                ],
            )
        )

    metadata = json.loads(
        (root / "standalone_config.json").read_text(encoding="utf-8")
    )
    encoder_config = AutoConfig.from_pretrained(str(root / "encoder"))
    _apply_rope_compat(encoder_config)
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    model = StandaloneDecisionModel(
        encoder,
        head_layers=int(metadata.get("head_layers", 2)),
        n_act=int(metadata.get("n_act", 2)),
    )

    pdelta = metadata.get("pdelta3", {})
    install_pdelta3_encoder(
        model,
        metadata["converted_full_attention_layers"],
        feature_dim=int(pdelta.get("feature_dim", 96)),
        conv_kernel=int(pdelta.get("conv_kernel", 4)),
        chunk_size=int(pdelta.get("chunk_size", 64)),
        local_kernel=int(pdelta.get("local_kernel", 5)),
        local_window=int(pdelta.get("local_window", 32)),
        local_gate_init=float(pdelta.get("local_gate_init", 0.72)),
    )
    install_pdelta3_head(
        model,
        feature_dim=int(pdelta.get("feature_dim", 96)),
        conv_kernel=int(pdelta.get("conv_kernel", 4)),
        chunk_size=int(pdelta.get("chunk_size", 64)),
        local_kernel=int(pdelta.get("local_kernel", 5)),
        local_window=int(pdelta.get("local_window", 32)),
        local_gate_init=float(pdelta.get("local_gate_init", 0.72)),
    )

    state = load_file(str(root / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    left = remaining_full_attention(model)
    if any(left.values()):
        raise RuntimeError(f"export still contains full attention: {left}")

    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    return StandaloneDecisionRuntime(model, tokenizer, metadata, device=device)
