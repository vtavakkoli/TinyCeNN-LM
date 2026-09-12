from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .cenn import CeNNConfig, FastCeNNCore
from .modeling import DEFAULT_BASE_MODEL, _get_decoder_layers


class CeNNReplacementLayer(nn.Module):
    """Transformer-free decoder layer built only from a recurrent causal CeNN core.

    The surrounding pretrained language interface is intentionally kept: token
    embeddings, final RMSNorm, tokenizer and LM head. The original Transformer
    decoder layer itself is removed and no attention/MLP parameters remain in
    this layer.
    """

    def __init__(
        self,
        config: CeNNConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.cenn = FastCeNNCore(config)
        if device is not None or dtype is not None:
            move_kwargs: dict[str, object] = {}
            if device is not None:
                move_kwargs["device"] = device
            if dtype is not None:
                move_kwargs["dtype"] = dtype
            self.cenn.to(**move_kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        if kwargs.get("use_cache", False):
            raise RuntimeError(
                "CeNN student requires use_cache=False until a recurrent CeNN state cache "
                "is implemented."
            )
        if kwargs.get("output_attentions", False):
            raise RuntimeError("CeNN-only student has no attention matrices to return.")
        return hidden_states + self.cenn(hidden_states)


def _reference_parameter(module: nn.Module) -> nn.Parameter | None:
    return next((p for p in module.parameters() if p.is_floating_point()), None)


def replace_transformer_with_cenn(
    model: nn.Module,
    config: CeNNConfig,
    layer_indices: Sequence[int] = (0,),
) -> nn.Module:
    """Remove selected Transformer decoder layers and replace them with CeNN layers."""
    layers = _get_decoder_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    if config.hidden_size != hidden_size:
        raise ValueError(
            f"CeNN hidden_size={config.hidden_size} does not match model hidden_size={hidden_size}"
        )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    for index in layer_indices:
        if index < 0 or index >= len(layers):
            raise IndexError(f"layer index {index} out of range [0, {len(layers)})")
        if isinstance(layers[index], CeNNReplacementLayer):
            raise ValueError(f"layer {index} is already CeNN-only")
        reference = _reference_parameter(layers[index])
        device = reference.device if reference is not None else None
        dtype = reference.dtype if reference is not None else None
        layers[index] = CeNNReplacementLayer(config, device=device, dtype=dtype)
    return model


def freeze_student_interfaces(model: nn.Module, train_interfaces: str = "none") -> None:
    """Train the core and optionally adapt pretrained interfaces after distillation.

    ``norm`` trains the final normalization; ``all`` also trains the embeddings
    and output head. The default retains the original core-only experiment.
    """
    if train_interfaces not in {"none", "norm", "all"}:
        raise ValueError("train_interfaces must be none, norm, or all")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, CeNNReplacementLayer):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if train_interfaces != "none":
        norm = getattr(getattr(model, "model", None), "norm", None)
        if not isinstance(norm, nn.Module):
            raise ValueError("student has no supported final model.norm")
        norm.requires_grad_(True)
    if train_interfaces == "all":
        for interface in (model.get_input_embeddings(), model.get_output_embeddings()):
            if interface is None:
                raise ValueError("student must expose input and output embeddings")
            interface.requires_grad_(True)


def student_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def _student_state_dict(model: nn.Module) -> dict[str, Tensor]:
    # Keep interfaces loaded from a v2 checkpoint even when they are frozen in a
    # later stage. Otherwise re-saving silently reverts them to the base model.
    extra_keys = set(getattr(model, "_cenn_interface_keys", ()))
    extra_keys.update(name for name, p in model.named_parameters() if p.requires_grad)
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".cenn." in name or name in extra_keys:
            state[name] = tensor.detach().cpu()
    if not any(".cenn." in name for name in state):
        raise ValueError("no CeNN replacement weights found")
    return state


def save_cenn_student(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: CeNNConfig,
    base_model: str = DEFAULT_BASE_MODEL,
    layer_indices: Sequence[int] = (0,),
    extra_metadata: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _student_state_dict(model)
    torch.save(state, output_dir / "cenn_student.pt")
    metadata = {
        "format_version": 2,
        "architecture": "cenn-only-replacement",
        "base_model": base_model,
        "layer_indices": list(layer_indices),
        "cenn": config.to_dict(),
        "state_keys": sorted(state),
    }
    if extra_metadata:
        metadata["training"] = extra_metadata
    (output_dir / "student_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_dir


def load_cenn_student_weights(
    model: nn.Module,
    student_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    student_dir = Path(student_dir)
    state = torch.load(
        student_dir / "cenn_student.pt",
        map_location=map_location,
        weights_only=True,
    )
    metadata = json.loads((student_dir / "student_config.json").read_text())
    core_keys = {key for key in model.state_dict() if ".cenn." in key}
    expected = set(metadata.get("state_keys", core_keys)) | core_keys
    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected | state.keys() - model.state_dict().keys())
    if strict and missing:
        raise RuntimeError(f"missing CeNN student keys: {missing}")
    if strict and unexpected:
        raise RuntimeError(f"unexpected CeNN student keys: {unexpected}")
    model.load_state_dict(state, strict=False)
    model._cenn_interface_keys = tuple(key for key in state if ".cenn." not in key)
    return model


def build_cenn_student(
    student_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    attn_implementation: str = "sdpa",
):
    """Rebuild a Transformer-free CeNN student from the pretrained interfaces."""
    from transformers import AutoModelForCausalLM

    student_dir = Path(student_dir)
    metadata = json.loads((student_dir / "student_config.json").read_text())
    if metadata.get("architecture") != "cenn-only-replacement":
        raise ValueError("checkpoint is not a CeNN-only replacement student")

    kwargs: dict[str, object] = {"attn_implementation": attn_implementation}
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    config = CeNNConfig.from_dict(metadata["cenn"])
    replace_transformer_with_cenn(model, config, tuple(metadata["layer_indices"]))
    load_cenn_student_weights(model, student_dir)

    move_kwargs: dict[str, object] = {}
    if device is not None:
        move_kwargs["device"] = device
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if move_kwargs:
        model.to(**move_kwargs)
    model.config.use_cache = False
    return model
