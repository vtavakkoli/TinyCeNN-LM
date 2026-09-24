"""Standalone typed-decision runtime with TinyCeNN Integrated Memory V2.3.

The module intentionally has no dependency on the ``laya`` Python package.  It
can reconstruct the public Laya-compatible typed-decision checkpoint layout
from ordinary Transformers config/tokenizer files plus a safetensors state
dict, then replace selected ModernBERT full-attention modules with a linear
Integrated Memory block.

The exported runtime is aimed at low-latency symbolic/game-control decisions:
encode a compact state, score a finite action set, and return the best action.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


def _serialize_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def normalize_question(question: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an external typed-decision question into the runtime format."""
    qtype = str(question.get("type", question.get("t", "choice"))).lower()
    if qtype not in QTYPES:
        raise ValueError(f"unsupported question type: {qtype!r}")

    criteria = question.get("criteria", question.get("crit"))
    if qtype == "choice":
        if isinstance(criteria, (list, tuple)):
            criteria = {str(x): None for x in criteria}
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise ValueError("choice questions require at least two criteria/options")
        criteria = dict(criteria)
    elif qtype == "score":
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
            raise ValueError("score questions require at least two ordered criteria")
        criteria = list(criteria)
    else:
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, Mapping):
            raise ValueError("noul criteria must be a mapping when provided")
        criteria = {str(k).lower(): v for k, v in criteria.items()}

    instructions = question.get("instructions", question.get("ins", ""))
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False, default=str)

    result = {"t": qtype, "ins": instructions, "crit": criteria}
    if "labels" in question:
        result["labels"] = dict(question["labels"])
    return result


def render_options(question: Mapping[str, Any]) -> list[str]:
    q = normalize_question(question) if "t" not in question else dict(question)
    qtype, criteria = q["t"], q.get("crit")
    if qtype == "choice":
        return [
            str(k) if v is None or v == "" else f"{k}: {_render_value(v)}"
            for k, v in criteria.items()
        ]
    if qtype == "score":
        return [f"level {i}: {_render_value(v)}" for i, v in enumerate(criteria)]

    labels = q.get("labels", _DEFAULT_NOUL_LABELS)
    false_label = str(labels.get("false", "false")).strip() or "false"
    true_label = str(labels.get("true", "true")).strip() or "true"
    false_desc = criteria.get("false")
    true_desc = criteria.get("true")
    return [
        f"{false_label}: " + (
            _render_value(false_desc) if false_desc not in (None, "") else "no, the statement does not hold"
        ),
        f"{true_label}: " + (
            _render_value(true_desc) if true_desc not in (None, "") else "yes, the statement holds"
        ),
    ]


def build_sequence(
    tokenizer,
    state: str | dict | list,
    question: Mapping[str, Any],
    *,
    max_len: int = 512,
    head_max_len: int = 192,
    truncate_left: bool = False,
    state_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Build a compact typed-decision sequence and option marker positions."""
    q = normalize_question(question) if "t" not in question else dict(question)
    options = render_options(q)
    mask_token = tokenizer.mask_token
    instruction = str(q["ins"]).replace(mask_token, " ")
    head_ids = tokenizer(
        f"{q['t']} question: {instruction}", add_special_tokens=False
    )["input_ids"]

    option_ids: list[list[int]] = []
    for option in options:
        ids = tokenizer(
            " " + option.replace(mask_token, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )["input_ids"]
        option_ids.append([tokenizer.mask_token_id] + ids)

    option_budget = head_max_len - sum(len(x) for x in option_ids)
    if option_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [x[:per] for x in option_ids]
        option_budget = head_max_len - sum(len(x) for x in option_ids)
    head_ids = head_ids[: max(8, option_budget)]

    ids = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers: list[int] = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)

    if state_ids is None:
        text = _serialize_state(state).replace(mask_token, " ")
        state_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    room = max(0, max_len - len(ids) - 1)
    if truncate_left:
        start = max(0, len(state_ids) - room)
        state_part = list(state_ids[start:])
    else:
        state_part = list(state_ids[:room])
    ids = (ids + state_part + [tokenizer.sep_token_id])[:max_len]
    markers = [m for m in markers if m < max_len]
    return ids, markers


def collate_items(items: Sequence[Mapping[str, Any]], pad_id: int) -> dict[str, Tensor]:
    if not items:
        raise ValueError("items must not be empty")
    n = len(items)
    length = max(len(x["ids"]) for x in items)
    kmax = max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), int(pad_id), dtype=torch.long)
    mask = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, kmax), dtype=torch.long)
    marker_mask = torch.zeros((n, kmax), dtype=torch.bool)
    qtype = torch.zeros(n, dtype=torch.long)

    for i, item in enumerate(items):
        row = torch.tensor(item["ids"], dtype=torch.long)
        ids[i, : row.numel()] = row
        mask[i, : row.numel()] = 1
        m = torch.tensor(item["markers"], dtype=torch.long)
        marker_pos[i, : m.numel()] = m
        marker_mask[i, : m.numel()] = True
        qtype[i] = int(item["qtype"])
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }


class StandaloneDecisionModel(nn.Module):
    """ModernBERT encoder plus the compact typed-decision head used by the checkpoint."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = int(encoder.config.hidden_size)
        nhead = max(1, d // 64)
        if head_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d,
                nhead,
                dim_feedforward=4 * d,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.head = nn.TransformerEncoder(layer, int(head_layers), enable_nested_tensor=False)
        else:
            self.head = None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, int(n_act))
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        marker_pos: Tensor,
        marker_mask: Tensor,
        qtype: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            h = self.head(h, src_key_padding_mask=pad)

        gather_idx = marker_pos.clamp_min(0)[:, :, None].expand(-1, -1, h.size(-1))
        marker_hidden = torch.gather(h, 1, gather_idx)
        logits = self.scorer(marker_hidden).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), dim=-1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.shape[-1] >= 2:
            top2 = p.topk(2, dim=-1).values
        else:
            top1 = p.topk(1, dim=-1).values
            top2 = torch.cat((top1, torch.zeros_like(top1)), dim=-1)
        features = torch.stack((top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0), dim=-1)
        act_logits = self.act_head(torch.cat((h[:, 0].float(), features), dim=-1))
        return logits, act_logits


def _apply_modernbert_rope(q: Tensor, k: Tensor, position_embeddings):
    if position_embeddings is None:
        return q, k
    try:
        from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    except Exception:
        return q, k
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)


def _valid_tokens(attention_mask: Tensor | None, hidden_states: Tensor) -> Tensor:
    b, t = hidden_states.shape[:2]
    if attention_mask is None:
        return torch.ones((b, t), dtype=torch.bool, device=hidden_states.device)
    if attention_mask.ndim == 2:
        return attention_mask.bool()
    if attention_mask.ndim != 4:
        return torch.ones((b, t), dtype=torch.bool, device=hidden_states.device)
    allowed = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e4
    valid_q = allowed.any(dim=-1).any(dim=1)
    valid_k = allowed.any(dim=-2).any(dim=1)
    valid = valid_q & valid_k
    if valid.shape[-1] != t:
        return torch.ones((b, t), dtype=torch.bool, device=hidden_states.device)
    return valid


def _depthwise_sequence_conv(x: Tensor, weight: Tensor) -> Tensor:
    b, h, t, d = x.shape
    y = x.permute(0, 1, 3, 2).reshape(b, h * d, t)
    kernel = int(weight.shape[-1])
    left = (kernel - 1) // 2
    right = kernel - 1 - left
    y = F.pad(y, (left, right))
    y = F.conv1d(y, weight, groups=h * d)
    return y.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


def _identity_conv_weight(heads: int, head_dim: int, kernel: int) -> Tensor:
    weight = torch.zeros(heads * head_dim, 1, kernel)
    weight[:, 0, (kernel - 1) // 2] = 1.0
    return weight


def _orthogonal_maps(heads: int, out_dim: int, in_dim: int) -> Tensor:
    maps = torch.empty(heads, out_dim, in_dim)
    for head in range(heads):
        nn.init.orthogonal_(maps[head])
    return maps


class IntegratedMemoryV23Attention(nn.Module):
    """Non-quadratic replacement for a bidirectional ModernBERT full-attention block.

    Four head-wise routes are mixed:
    1. symmetric positive softmax features,
    2. ELU+1 positive features,
    3. CeNN-style local depthwise sequence mixing,
    4. a direct value route.

    Both global branches use linear associative memory, so no T x T attention
    matrix is formed.
    """

    architecture = "integrated_memory_v23"

    def __init__(self, original: nn.Module, feature_dim: int = 96, local_kernel: int = 5):
        super().__init__()
        self.config = original.config
        self.layer_idx = int(getattr(original, "layer_idx", -1))
        self.hidden_size = int(self.config.hidden_size)
        self.num_heads = int(self.config.num_attention_heads)
        self.head_dim = int(getattr(original, "head_dim", self.hidden_size // self.num_heads))
        self.feature_dim = int(feature_dim)
        self.local_kernel = int(local_kernel)

        self.Wqkv = copy.deepcopy(original.Wqkv)
        self.Wo = copy.deepcopy(original.Wo)
        self.out_drop = copy.deepcopy(original.out_drop)
        for module in (self.Wqkv, self.Wo):
            for p in module.parameters():
                p.requires_grad = False

        warm = _orthogonal_maps(self.num_heads, self.feature_dim, self.head_dim)
        self.wq = nn.Parameter(warm.clone())
        self.wk = nn.Parameter(warm.clone())
        self.feature_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.feature_input_scale = float(self.head_dim ** -0.25)

        self.local_weight = nn.Parameter(
            _identity_conv_weight(self.num_heads, self.head_dim, self.local_kernel)
        )
        # soft-global, elu-global, local, direct
        self.mix_logits = nn.Parameter(
            torch.tensor([2.5, 0.8, -0.4, -1.4]).repeat(self.num_heads, 1)
        )
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads))
        self.last_core_output: Tensor | None = None

    def trainable_core_parameters(self) -> list[nn.Parameter]:
        frozen = {id(p) for p in self.Wqkv.parameters()} | {id(p) for p in self.Wo.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in frozen]

    def _qkv(self, hidden_states: Tensor, position_embeddings=None) -> tuple[Tensor, Tensor, Tensor]:
        b, t, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_modernbert_rope(q, k, position_embeddings)
        return q, k, v

    def _project(self, x: Tensor, weight: Tensor) -> Tensor:
        z = torch.einsum("bhtd,hfd->bhtf", x.float(), weight.float())
        z = z * self.feature_input_scale
        scale = self.feature_log_scale.float().clamp(-1.5, 1.5).exp()
        return z * scale[None, :, None, None]

    @staticmethod
    def _linear_memory(qf: Tensor, kf: Tensor, value: Tensor) -> Tensor:
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, value)
        normalizer = kf.sum(dim=2)
        numerator = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        denominator = torch.einsum("bhtf,bhf->bht", qf, normalizer).unsqueeze(-1)
        return numerator / denominator.clamp_min(1e-6)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings=None,
        attention_mask=None,
        **_: Any,
    ):
        q, k, v = self._qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].float()
        value = v.float() * mask

        zq = self._project(q, self.wq)
        zk = self._project(k, self.wk)

        soft_q = torch.cat((torch.softmax(zq, -1), torch.softmax(-zq, -1)), dim=-1).clamp_min(1e-6)
        soft_k = torch.cat((torch.softmax(zk, -1), torch.softmax(-zk, -1)), dim=-1).clamp_min(1e-6) * mask
        soft_global = self._linear_memory(soft_q, soft_k, value)

        elu_q = (F.elu(zq) + 1.0).clamp_min(1e-4)
        elu_k = (F.elu(zk) + 1.0).clamp_min(1e-4) * mask
        elu_global = self._linear_memory(elu_q, elu_k, value)

        local = _depthwise_sequence_conv(v.float(), self.local_weight) * mask
        direct = v.float() * mask
        mix = torch.softmax(self.mix_logits.float(), dim=-1)
        core = (
            mix[:, 0][None, :, None, None] * soft_global
            + mix[:, 1][None, :, None, None] * elu_global
            + mix[:, 2][None, :, None, None] * local
            + mix[:, 3][None, :, None, None] * direct
        )
        gain = self.output_gain.float().clamp(-2.0, 2.0).exp()
        core = core * gain[None, :, None, None] * mask
        self.last_core_output = core

        b, _, t, _ = core.shape
        flat = core.transpose(1, 2).reshape(b, t, self.hidden_size).to(hidden_states.dtype)
        return self.out_drop(self.Wo(flat)), None


def full_attention_indices(model: nn.Module) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.encoder.layers)
        if str(getattr(layer, "attention_type", "")) == "full_attention"
    ]


def install_integrated_memory(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    feature_dim: int = 96,
    local_kernel: int = 5,
) -> list[int]:
    installed: list[int] = []
    for idx in layer_indices:
        i = int(idx)
        current = model.encoder.layers[i].attn
        if isinstance(current, IntegratedMemoryV23Attention):
            installed.append(i)
            continue
        model.encoder.layers[i].attn = IntegratedMemoryV23Attention(
            current, feature_dim=feature_dim, local_kernel=local_kernel
        )
        installed.append(i)
    return installed


def converted_full_attention_indices(model: nn.Module) -> list[int]:
    return [
        i
        for i in full_attention_indices(model)
        if isinstance(model.encoder.layers[i].attn, IntegratedMemoryV23Attention)
    ]


def _apply_rope_compat(config) -> None:
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (
        ("full_attention", "global_rope_theta"),
        ("sliding_attention", "local_rope_theta"),
    ):
        values = rope.get(layer_type)
        theta = values.get("rope_theta") if isinstance(values, dict) else flat
        if theta is not None and hasattr(config, attr):
            setattr(config, attr, float(theta))


def build_checkpoint_model(root: str | os.PathLike[str]) -> tuple[StandaloneDecisionModel, dict[str, Any]]:
    """Reconstruct a source-compatible typed-decision model without importing Laya."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    root = Path(root)
    config = json.loads((root / "rl_agent_config.json").read_text(encoding="utf-8"))
    encoder_dir = root / "encoder"
    encoder_config = AutoConfig.from_pretrained(str(encoder_dir))
    _apply_rope_compat(encoder_config)
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    model = StandaloneDecisionModel(
        encoder,
        head_layers=int(config.get("head_layers", 2)),
        n_act=len(config.get("act_costs", {})) + 1,
    )
    weights = load_file(str(root / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    return model, config


@dataclass
class DecisionResult:
    choice: str
    probabilities: dict[str, float]
    confidence: float


class StandaloneDecisionRuntime:
    """Small inference wrapper used by game/control code."""

    def __init__(
        self,
        model: StandaloneDecisionModel,
        tokenizer,
        config: Mapping[str, Any],
        *,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = dict(config)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def decide(
        self,
        state: str | dict | list,
        actions: Mapping[str, Any] | Sequence[str],
        *,
        instruction: str = "Which action should the controller take next?",
        max_len: int | None = None,
    ) -> DecisionResult:
        if isinstance(actions, Mapping):
            criteria = dict(actions)
        else:
            criteria = {str(x): None for x in actions}
        q = normalize_question(
            {"type": "choice", "instructions": instruction, "criteria": criteria}
        )
        ids, markers = build_sequence(
            self.tokenizer,
            state,
            q,
            max_len=int(max_len or self.config.get("max_len", 512)),
            head_max_len=int(self.config.get("head_max_len", 192)),
            truncate_left=isinstance(state, list),
        )
        batch = collate_items(
            [{"ids": ids, "markers": markers, "qtype": QTYPES["choice"]}],
            int(self.tokenizer.pad_token_id),
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits, _ = self.model(**batch)
        probs = torch.softmax(logits[0, : len(criteria)], dim=-1).float().cpu()
        names = list(criteria)
        best = int(probs.argmax().item())
        probability_map = {name: float(probs[i]) for i, name in enumerate(names)}
        return DecisionResult(
            choice=names[best],
            probabilities=probability_map,
            confidence=float(probs[best]),
        )


def load_standalone(
    repo_or_dir: str,
    *,
    device: str | torch.device | None = None,
    token: str | None = None,
) -> StandaloneDecisionRuntime:
    """Load a fully exported V2.3 model from a local directory or Hugging Face repo."""
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

    metadata = json.loads((root / "standalone_config.json").read_text(encoding="utf-8"))
    encoder_config = AutoConfig.from_pretrained(str(root / "encoder"))
    _apply_rope_compat(encoder_config)
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    model = StandaloneDecisionModel(
        encoder,
        head_layers=int(metadata.get("head_layers", 2)),
        n_act=int(metadata.get("n_act", 2)),
    )
    install_integrated_memory(
        model,
        metadata["converted_full_attention_layers"],
        feature_dim=int(metadata.get("feature_dim", 96)),
        local_kernel=int(metadata.get("local_kernel", 5)),
    )
    state = load_file(str(root / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    return StandaloneDecisionRuntime(model, tokenizer, metadata, device=device)


def replacement_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        if isinstance(module, IntegratedMemoryV23Attention):
            for p in module.trainable_core_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
    return params


def count_replacement_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in replacement_trainable_parameters(model))
