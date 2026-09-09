from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .cenn import CeNNConfig, FastCeNNCore


DEFAULT_BASE_MODEL = "arnir0/Tiny-LLM"


def _floating_reference_parameter(module: nn.Module) -> nn.Parameter | None:
    """Return a representative floating-point parameter for device/dtype alignment."""
    return next((p for p in module.parameters() if p.is_floating_point()), None)


class HybridDecoderLayer(nn.Module):
    """Wrap a pretrained decoder layer with a zero-init recurrent CeNN residual.

    The base layer is kept intact, including its attention behavior. TinyCeNN-LM
    v0.1 intentionally disables Transformer-only KV caching because the CeNN branch
    also needs its own recurrent neighborhood state for exact incremental decoding.

    The newly created CeNN branch inherits the pretrained decoder layer's floating
    point dtype/device. This is essential for BF16/FP16 inference: a BF16 hidden
    state cannot be convolved with an FP32 CeNN kernel without an explicit cast.
    """

    def __init__(self, base_layer: nn.Module, config: CeNNConfig) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.cenn = FastCeNNCore(config)

        reference = _floating_reference_parameter(base_layer)
        if reference is None:
            self.residual_scale = nn.Parameter(torch.ones(()))
        else:
            self.cenn.to(device=reference.device, dtype=reference.dtype)
            self.residual_scale = nn.Parameter(
                torch.ones((), device=reference.device, dtype=reference.dtype)
            )

    def forward(self, *args, **kwargs):
        if kwargs.get("use_cache", False):
            raise RuntimeError(
                "TinyCeNN-LM v0.1 requires use_cache=False. The Transformer KV cache "
                "does not contain the per-step CeNN neighborhood state needed for exact "
                "incremental generation. Full-prefix generation is correct; a dedicated "
                "streaming CeNN cache is planned for a later version."
            )
        outputs = self.base_layer(*args, **kwargs)

        if torch.is_tensor(outputs):
            return outputs + self.residual_scale * self.cenn(outputs)

        if isinstance(outputs, tuple):
            hidden = outputs[0]
            hidden = hidden + self.residual_scale * self.cenn(hidden)
            return (hidden, *outputs[1:])

        if isinstance(outputs, list):
            hidden = outputs[0]
            hidden = hidden + self.residual_scale * self.cenn(hidden)
            return [hidden, *outputs[1:]]

        raise TypeError(
            "Unsupported decoder-layer output type: "
            f"{type(outputs)!r}. Expected Tensor, tuple, or list."
        )


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
    )
    for path in candidates:
        obj = model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(
        "Could not locate decoder layers. TinyCeNN-LM currently targets "
        "Llama-family causal language models such as arnir0/Tiny-LLM."
    )


def inject_cenn(
    model: nn.Module,
    config: CeNNConfig | None = None,
    layer_indices: Sequence[int] = (0,),
) -> nn.Module:
    layers = _get_decoder_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    if config is None:
        config = CeNNConfig(hidden_size=hidden_size)
    elif config.hidden_size != hidden_size:
        raise ValueError(
            f"CeNN hidden_size={config.hidden_size} does not match "
            f"model hidden_size={hidden_size}"
        )

    # A Transformer KV cache alone is insufficient for the recurrent CeNN state.
    # Disable it globally to prevent a silent train/inference mismatch.
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    for index in layer_indices:
        if index < 0 or index >= len(layers):
            raise IndexError(f"layer index {index} out of range [0, {len(layers)})")
        if isinstance(layers[index], HybridDecoderLayer):
            raise ValueError(f"layer {index} already has a CeNN adapter")
        layers[index] = HybridDecoderLayer(layers[index], config)
    return model


def freeze_for_adapter_training(
    model: nn.Module,
    train_lm_head: bool = False,
    train_embeddings: bool = False,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, HybridDecoderLayer):
            for parameter in module.cenn.parameters():
                parameter.requires_grad = True
            module.residual_scale.requires_grad = True

    if train_lm_head and hasattr(model, "lm_head"):
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
    if train_embeddings:
        embeddings = model.get_input_embeddings()
        for parameter in embeddings.parameters():
            parameter.requires_grad = True


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def _adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".cenn." in name or name.endswith(".residual_scale"):
            state[name] = tensor.detach().cpu()
    if not state:
        raise ValueError("no CeNN adapter found in model")
    return state


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    layer_indices: Sequence[int] = (0,),
    config: CeNNConfig,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "cenn_adapter.pt"
    metadata_path = output_dir / "cenn_config.json"

    torch.save(_adapter_state_dict(model), weights_path)
    metadata = {
        "format_version": 1,
        "base_model": base_model,
        "layer_indices": list(layer_indices),
        "cenn": config.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def load_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    adapter_dir = Path(adapter_dir)
    state = torch.load(
        adapter_dir / "cenn_adapter.pt",
        map_location=map_location,
        weights_only=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if ".cenn." not in k]
    if strict and unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected}")
    missing_adapter = [
        k
        for k in _adapter_state_dict(model)
        if k in incompatible.missing_keys
    ]
    if strict and missing_adapter:
        raise RuntimeError(f"missing adapter keys: {missing_adapter}")
    return model


def build_from_adapter(
    adapter_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    attn_implementation: str = "sdpa",
):
    from transformers import AutoModelForCausalLM

    adapter_dir = Path(adapter_dir)
    metadata = json.loads((adapter_dir / "cenn_config.json").read_text())
    base_model = metadata["base_model"]
    config = CeNNConfig.from_dict(metadata["cenn"])
    layer_indices = tuple(metadata["layer_indices"])

    kwargs = {"attn_implementation": attn_implementation}
    if dtype is not None:
        # Modern Transformers uses `dtype`; `torch_dtype` is deprecated.
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    inject_cenn(model, config=config, layer_indices=layer_indices)
    load_adapter(model, adapter_dir)

    move_kwargs: dict[str, object] = {}
    if device is not None:
        move_kwargs["device"] = device
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if move_kwargs:
        model.to(**move_kwargs)
    return model
