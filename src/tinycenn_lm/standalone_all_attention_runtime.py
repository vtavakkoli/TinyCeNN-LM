"""Self-contained mixed global+sliding ModernBERT typed-decision runtime.

Copied as runtime.py into every published Laya V3.1 Hugging Face model.
Does not import TinyCeNN-LM, Laya, or other user project packages.
Depends only on torch, transformers==4.57.6, and safetensors via loader.

Global layers: V2.9 FastWindowMemory (W1024 default; quadratic at 512).
Sliding layers: V3.0 SlidingWindowFusion (W129 default; finite-window SDPA).
"""
from __future__ import annotations

"""Self-contained V2.9 FastWindowMemory typed-decision runtime.

This module is copied into each exported Hugging Face model repository.
It deliberately never imports TinyCeNN-LM, Laya, or local project modules.
Runtime dependencies: torch, transformers==4.57.6 and safetensors.
The Hugging Face Hub client is needed only to download a remote repository.
"""
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def _apply_rope_compat(config):
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


def _valid_tokens(attention_mask: Tensor | None, hidden_states: Tensor) -> Tensor:
    b, t = hidden_states.shape[:2]
    if attention_mask is None:
        return torch.ones((b, t), dtype=torch.bool, device=hidden_states.device)
    if attention_mask.ndim == 2:
        return attention_mask.bool()
    if attention_mask.ndim != 4:
        return torch.ones((b, t), dtype=torch.bool, device=hidden_states.device)
    allowed = (attention_mask if attention_mask.dtype == torch.bool
               else attention_mask > -1e4)
    valid_q = allowed.any(dim=-1).any(dim=1)
    valid_k = allowed.any(dim=-2).any(dim=1)
    valid = valid_q & valid_k
    return valid if valid.shape[-1] == t else torch.ones(
        (b, t), dtype=torch.bool, device=hidden_states.device
    )


def _depthwise_sequence_conv(x: Tensor, weight: Tensor) -> Tensor:
    b, h, t, d = x.shape
    kernel = int(weight.shape[-1])
    y = x.permute(0, 1, 3, 2).reshape(b, h * d, t)
    y = F.pad(y, ((kernel - 1) // 2, kernel - 1 - (kernel - 1) // 2))
    y = F.conv1d(y, weight, groups=h * d)
    return y.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


def _orthogonal_maps(heads: int, out_dim: int, in_dim: int) -> Tensor:
    maps = torch.empty(heads, out_dim, in_dim)
    for h in range(heads):
        nn.init.orthogonal_(maps[h])
    return maps


class FastWindowAttention(nn.Module):
    """State-dict-compatible V2.8 full/global attention replacement.

    The local path is quadratic when W >= 2*T-1. W1024 / MAX_LEN512 is
    a full-attention reconstruction control, not an attention-free speedup.
    """

    def __init__(self, original, *, local_window=1024, landmark_stride=16,
                 feature_dim=96, max_landmarks=32):
        super().__init__()
        self.config = original.config
        self.layer_idx = int(getattr(original, "layer_id", -1))
        self.hidden_size = int(self.config.hidden_size)
        self.num_heads = int(self.config.num_attention_heads)
        self.head_dim = int(getattr(original, "head_dim",
                                    self.hidden_size // self.num_heads))
        self.local_window = int(local_window)
        self.landmark_stride = int(landmark_stride)
        self.max_landmarks = int(max_landmarks)
        self.feature_dim = int(feature_dim)
        self.global_input_scale = float(self.head_dim ** -0.25)
        self.Wqkv = copy.deepcopy(original.Wqkv)
        self.Wo = copy.deepcopy(original.Wo)
        self.out_drop = copy.deepcopy(original.out_drop)
        self.rotary_emb = copy.deepcopy(original.rotary_emb)
        h, d = self.num_heads, self.head_dim
        init = _orthogonal_maps(h, self.feature_dim, d)
        self.global_wq = nn.Parameter(init.clone())
        self.global_wk = nn.Parameter(init.clone())
        self.global_log_scale = nn.Parameter(torch.zeros(h))
        # The original constructor initializes identity convolution weights.
        self.local_weight = nn.Parameter(torch.zeros(h * d, 1, 5))
        with torch.no_grad():
            self.local_weight[:, 0, 2] = 1.0
        self.route_logits = nn.Parameter(
            torch.tensor([1.0, 0.2, 0.2, -2.2]).repeat(h, 1)
        )
        self.route_query = nn.Parameter(torch.zeros(h, 4, d))
        self.local_log_temperature = nn.Parameter(torch.zeros(h))
        self.landmark_log_temperature = nn.Parameter(torch.zeros(h))
        self.conv_gain = nn.Parameter(torch.full((h,), -4.0))
        self.Wqkv.requires_grad_(False)
        self.rotary_emb.requires_grad_(False)
        self.last_core_output = None

    def _local_attention(self, q, k, v, valid):
        b, h, t, d = q.shape
        w = self.local_window
        if w < t:
            # Exact blocked centered-local attention inherited by the original
            # PDelta architecture. Never pad a short T up to huge W.
            left, right = (w - 1) // 2, w // 2
            blocks = (t + w - 1) // w
            tail = blocks * w - t
            qp = F.pad(q, (0, 0, 0, tail)).reshape(b, h, blocks, w, d)
            def windows(x):
                return F.pad(x, (0, 0, left, right + tail)).unfold(
                    2, 2 * w - 1, w
                ).transpose(-1, -2)
            kp, vp = windows(k), windows(v)
            key_valid = F.pad(valid, (left, right + tail)).unfold(
                1, 2 * w - 1, w
            )
            qi = torch.arange(w, device=q.device)[:, None]
            ki = torch.arange(2 * w - 1, device=q.device)[None, :]
            band = (ki >= qi) & (ki < qi + w)
            allowed = band[None, None, None] & key_valid[:, None, :, None, :]
            def batch_blocks(x):
                return x.transpose(1, 2).reshape(b * blocks, h, x.shape[-2], d)
            allowed = allowed.transpose(1, 2).reshape(
                b * blocks, 1, w, 2 * w - 1
            )
            z = F.scaled_dot_product_attention(
                batch_blocks(qp), batch_blocks(kp), batch_blocks(vp),
                attn_mask=allowed, dropout_p=0.0
            )
            z = z.reshape(b, blocks, h, w, d).transpose(1, 2)
            return z.reshape(b, h, blocks * w, d)[:, :, :t] * valid[:, None, :, None]
        allowed = valid[:, None, None, :]
        if w < 2 * t - 1:
            qi = torch.arange(t, device=q.device)[:, None]
            ki = torch.arange(t, device=q.device)[None, :]
            band = ((ki >= qi - (w - 1) // 2) &
                    (ki <= qi + w // 2))
            allowed = allowed & band[None, None]
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        return z * valid[:, None, :, None].to(z.dtype)

    def _landmarks(self, q, k, v, valid):
        b, h, t, d = k.shape
        stride = max(self.landmark_stride, math.ceil(t / self.max_landmarks))
        blocks = math.ceil(t / stride)
        pad = blocks * stride - t
        mask = valid[:, None, :, None].to(k.dtype)
        packed = F.pad(valid, (0, pad), value=False).reshape(b, blocks, stride)
        counts = packed.sum(-1)
        divisor = counts.clamp_min(1).to(k.dtype)[:, None, :, None]
        kp = F.pad(k * mask, (0, 0, 0, pad)).reshape(b, h, blocks, stride, d)
        vp = F.pad(v * mask, (0, 0, 0, pad)).reshape(b, h, blocks, stride, d)
        pooled_k = kp.sum(3) / divisor
        pooled_v = vp.sum(3) / divisor
        allowed = (counts > 0)[:, None, None, :]
        temperature = self.landmark_log_temperature.clamp(-0.4, 0.4).exp()
        qs = q * temperature[None, :, None, None]
        return F.scaled_dot_product_attention(
            qs, pooled_k, pooled_v, attn_mask=allowed, dropout_p=0.0
        ) * mask

    def _global_linear(self, q, k, v, mask):
        scale = self.global_log_scale.to(q.dtype).clamp(-1.5, 1.5).exp()
        qz = torch.einsum("bhtd,hfd->bhtf", q, self.global_wq)
        kz = torch.einsum("bhtd,hfd->bhtf", k, self.global_wk)
        qz = qz * self.global_input_scale * scale[None, :, None, None]
        kz = kz * self.global_input_scale * scale[None, :, None, None]
        qf = torch.cat((torch.softmax(qz, -1), torch.softmax(-qz, -1)), -1).clamp_min(1e-6)
        kf = torch.cat((torch.softmax(kz, -1), torch.softmax(-kz, -1)), -1).clamp_min(1e-6) * mask
        vf = v * mask
        memory = torch.einsum("bhtf,bhtd->bhfd", kf, vf)
        normalizer = kf.sum(2)
        numerator = torch.einsum("bhtf,bhfd->bhtd", qf, memory)
        denominator = torch.einsum("bhtf,bhf->bht", qf, normalizer).unsqueeze(-1)
        return numerator / denominator.clamp_min(1e-6)

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, position_ids=None, sliding_window_mask=None,
                **kwargs):
        b, t, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            b, t, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.transpose(3, 1).unbind(dim=2)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        valid = _valid_tokens(attention_mask, hidden_states)
        mask = valid[:, None, :, None].to(v.dtype)
        q, k, v = q * mask, k * mask, v * mask
        temperature = self.local_log_temperature.clamp(-0.4, 0.4).exp()
        local = self._local_attention(
            q * temperature[None, :, None, None], k, v, valid
        )
        landmarks = self._landmarks(q, k, v, valid)
        global_linear = self._global_linear(q, k, v, mask) * mask
        qnorm = F.normalize(q, dim=-1)
        dynamic = torch.einsum("bhtd,hkd->bhtk", qnorm, self.route_query)
        mixture = torch.softmax(
            self.route_logits.to(q.dtype)[None, :, None, :] + dynamic, dim=-1
        )
        fused = (mixture[..., 0:1] * local
                 + mixture[..., 1:2] * landmarks
                 + mixture[..., 2:3] * global_linear
                 + mixture[..., 3:4] * v)
        local_conv = _depthwise_sequence_conv(v, self.local_weight)
        conv_weight = torch.sigmoid(self.conv_gain).to(v.dtype)[None, :, None, None]
        fused = (fused + conv_weight * (local_conv - v)) * mask
        self.last_core_output = fused
        flat = fused.transpose(1, 2).reshape(b, t, self.hidden_size).to(hidden_states.dtype)
        return self.out_drop(self.Wo(flat)), None


class StandaloneDecisionModel(nn.Module):
    """Original Laya typed-decision head without a project-library dependency."""
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = int(encoder.config.hidden_size)
        nhead = max(1, d // 64)
        if head_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d, nhead, dim_feedforward=4*d, dropout=dropout,
                batch_first=True, norm_first=True
            )
            self.head = nn.TransformerEncoder(
                layer, int(head_layers), enable_nested_tensor=False
            )
        else:
            self.head = None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d+4, 256), nn.GELU(), nn.Linear(256, int(n_act))
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            h = self.head(h, src_key_padding_mask=~attention_mask.bool())
        index = marker_pos.clamp_min(0)[:, :, None].expand(-1, -1, h.size(-1))
        marker_hidden = torch.gather(h, 1, index)
        logits = self.scorer(marker_hidden).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p*torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.shape[-1] >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat((top1, torch.zeros_like(top1)), -1)
        features = torch.stack(
            (top2[:, 0], top2[:, 0]-top2[:, 1], ent, k/255.0), -1
        )
        act_logits = self.act_head(torch.cat((h[:, 0].float(), features), -1))
        return logits, act_logits


def _build_global_model(root: str | Path, manifest: dict):
    """Generic exporter protocol: build_model(bundle_directory, manifest)."""
    root = Path(root)
    if manifest["architecture"] != "laya-v29-fastwindow":
        raise ValueError("Expected laya-v29-fastwindow, got " + str(manifest["architecture"]))
    if manifest["runtime_format"] != 1:
        raise ValueError("Unsupported standalone bundle format")
    cfg = AutoConfig.from_pretrained(str(root / "encoder"))
    _apply_rope_compat(cfg)
    cfg.reference_compile = False
    encoder = AutoModel.from_config(cfg, attn_implementation="sdpa")
    model = StandaloneDecisionModel(
        encoder, head_layers=int(manifest["head_layers"]),
        n_act=int(manifest["n_act"]),
    )
    for index in manifest["global_layers"]:
        original = model.encoder.layers[index].attn
        if tuple(original.local_attention) != (-1, -1):
            raise ValueError(f"Layer {index} is not full/global attention")
        model.encoder.layers[index].attn = FastWindowAttention(
            original,
            local_window=int(manifest["window"]),
            landmark_stride=int(manifest["landmark_stride"]),
            feature_dim=int(manifest["feature_dim"]),
            max_landmarks=int(manifest["max_landmarks"]),
        )
    return model


QTYPES = {"choice": 0, "score": 1, "noul": 2}

def normalize_question(question: Mapping[str, Any]):
    qtype = str(question.get("type", question.get("t", "choice"))).lower()
    if qtype not in QTYPES:
        raise ValueError("Unsupported question type: " + qtype)
    criteria = question.get("criteria", question.get("crit"))
    if qtype == "choice":
        if isinstance(criteria, (list, tuple)):
            criteria = {str(x): None for x in criteria}
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise ValueError("Choice requires >=2 options")
        criteria = dict(criteria)
    elif qtype == "score":
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
            raise ValueError("Score requires >=2 levels")
        criteria = list(criteria)
    else:
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, Mapping):
            raise ValueError("Noul criteria must be a mapping")
        criteria = {str(k).lower(): v for k, v in criteria.items()}
    ins = question.get("instructions", question.get("ins", ""))
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False, default=str)
    result = {"t": qtype, "ins": ins, "crit": criteria}
    if "labels" in question:
        result["labels"] = dict(question["labels"])
    return result


def render_options(q):
    q = normalize_question(q) if "t" not in q else dict(q)
    criteria = q["crit"]
    def render(v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    if q["t"] == "choice":
        return [str(k) if v is None or v == "" else f"{k}: {render(v)}"
                for k,v in criteria.items()]
    if q["t"] == "score":
        return [f"level {i}: {render(v)}" for i,v in enumerate(criteria)]
    labels = q.get("labels", {"false": "false", "true": "true"})
    return [
        str(labels.get("false", "false"))+": "+(
            render(criteria.get("false")) if criteria.get("false") not in (None,"")
            else "no, the statement does not hold"),
        str(labels.get("true", "true"))+": "+(
            render(criteria.get("true")) if criteria.get("true") not in (None,"")
            else "yes, the statement holds"),
    ]


def build_sequence(tokenizer, state, question, *, max_len=512, head_max_len=192,
                   truncate_left=False, state_ids=None):
    q = normalize_question(question) if "t" not in question else dict(question)
    options = render_options(q)
    mask = tokenizer.mask_token
    ins = str(q["ins"]).replace(mask, " ")
    head_ids = tokenizer(
        f"{q['t']} question: {ins}", add_special_tokens=False
    )["input_ids"]
    option_ids = []
    for option in options:
        ids = tokenizer(
            " "+option.replace(mask, " "), add_special_tokens=False,
            truncation=True, max_length=48
        )["input_ids"]
        option_ids.append([tokenizer.mask_token_id] + ids)
    budget = head_max_len - sum(len(x) for x in option_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [x[:per] for x in option_ids]
        budget = head_max_len - sum(len(x) for x in option_ids)
    head_ids = head_ids[:max(8, budget)]
    ids = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers = []
    for opt in option_ids:
        markers.append(len(ids))
        ids.extend(opt)
    ids.append(tokenizer.sep_token_id)
    if state_ids is None:
        payload = state if isinstance(state, str) else json.dumps(
            state, ensure_ascii=False, separators=(",", ":"), default=str
        )
        state_ids = tokenizer(
            payload.replace(mask, " "), add_special_tokens=False
        )["input_ids"]
    room = max(0, max_len-len(ids)-1)
    state_part = list(state_ids[-room:] if truncate_left and room else
                      state_ids[:room])
    ids = (ids+state_part+[tokenizer.sep_token_id])[:max_len]
    markers = [i for i in markers if i < max_len]
    return ids, markers


def predict(model, tokenizer, state, question, *, max_len=512, device=None):
    """Simple task-level inference on a deployed standalone bundle."""
    q = normalize_question(question)
    ids, markers = build_sequence(tokenizer, state, q, max_len=max_len)
    if not markers:
        raise ValueError("No option markers survived truncation")
    parameter = next(model.parameters())
    device = device or parameter.device
    data = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        "marker_pos": torch.tensor([markers], dtype=torch.long, device=device),
        "marker_mask": torch.ones((1, len(markers)), dtype=torch.bool, device=device),
        "qtype": torch.tensor([QTYPES[q["t"]]], dtype=torch.long, device=device),
    }
    model.eval()
    with torch.inference_mode():
        scores, actions = model(**data)
    values = torch.softmax(scores.float(), -1)[0].cpu().tolist()
    choice = int(torch.tensor(values).argmax())
    return {
        "choice_index": choice,
        "choice": render_options(q)[choice],
        "probabilities": values,
        "action_logits": actions.float()[0].cpu().tolist(),
    }


# Bundled standalone sliding-fusion implementation
"""ModernBERT SlidingWindowFusion: exact finite-window SDPA plus gated CeNN-style local mixers.

An experiment in approximating the *sliding* ModernBERT layers (not the global
layers). It preserves pretrained QKV, RoPE and output projection. The 64-radius
configuration retains the teacher's 129-position local receptive field, but
remains windowed attention and is NOT an attention-free replacement.
"""

import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def valid_tokens(mask: Tensor | None, hidden: Tensor) -> Tensor:
    b, t = hidden.shape[:2]
    if mask is None:
        return torch.ones(b, t, dtype=torch.bool, device=hidden.device)
    if mask.ndim == 2:
        return mask.bool()
    if mask.ndim == 4:
        allowed = mask if mask.dtype == torch.bool else mask > -1e4
        return (allowed.any(-1).any(1) & allowed.any(-2).any(1)).bool()
    raise ValueError(f"Unrecognized attention mask shape {tuple(mask.shape)}")


def _identity_depthwise(channels: int, kernel: int) -> Tensor:
    weight = torch.zeros(channels, 1, kernel)
    weight[:, 0, kernel // 2] = 1.0
    return weight


def local_sliding_sdpa(
    q: Tensor, k: Tensor, v: Tensor, valid: Tensor, *,
    radius: int, block_size: int = 64
) -> Tensor:
    """O(T * (block_size + 2*radius)) memory local SDPA, exact ±radius semantics.

    Padding keys are excluded, and invalid query outputs are zero. Use this
    kernel as an alternative to the teacher's full T-by-T additive mask.
    """
    b, h, t, d = q.shape
    w = int(radius)
    bs = int(block_size)
    if w < 0 or bs < 1:
        raise ValueError("radius must be >=0, block_size >=1")
    blocks = math.ceil(t / bs)
    tail = blocks * bs - t
    # [B,H,blocks,BS,D] and [B,H,blocks,BS+2R,D]
    qp = F.pad(q, (0, 0, 0, tail)).reshape(b, h, blocks, bs, d)
    def keys_in_blocks(x):
        return F.pad(x, (0, 0, w, w + tail)).unfold(
            2, bs + 2*w, bs
        ).transpose(-1, -2)
    kp, vp = keys_in_blocks(k), keys_in_blocks(v)
    # SDPA bool mask: True means this key is permitted.
    kv = F.pad(valid, (w, w + tail)).unfold(1, bs + 2*w, bs)
    qi = torch.arange(bs, device=q.device)[:, None]
    ki = torch.arange(bs + 2*w, device=q.device)[None, :]
    band = (ki >= qi) & (ki <= qi + 2*w)
    allowed = (band[None, None, None, :, :]
               & kv[:, None, :, None, :])
    def flatten_blocks(x):
        return x.transpose(1, 2).reshape(
            b * blocks, h, x.shape[-2], d
        )
    allowed = allowed.transpose(1, 2).reshape(
        b * blocks, 1, bs, bs + 2*w
    )
    out = F.scaled_dot_product_attention(
        flatten_blocks(qp), flatten_blocks(kp), flatten_blocks(vp),
        attn_mask=allowed, dropout_p=0.0
    )
    out = out.reshape(b, blocks, h, bs, d).transpose(1, 2)
    return (out.reshape(b, h, blocks*bs, d)[:, :, :t]
            * valid[:, None, :, None].to(out.dtype))


def grouped_value_conv(v: Tensor, weight: Tensor, dilation: int) -> Tensor:
    b, h, t, d = v.shape
    k = weight.shape[-1]
    padding = dilation * (k - 1) // 2
    z = v.permute(0, 1, 3, 2).reshape(b, h*d, t)
    z = F.conv1d(z, weight.to(z.dtype), padding=padding,
                 dilation=dilation, groups=h*d)
    return z.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


class SlidingWindowFusion(nn.Module):
    """Learned per-head, per-token mixture of exact local window and local convs.

    A ±64 window matches ModernBERT's default native local attention radius.
    For radii <64, the context is genuinely truncated; no hidden teacher-mask
    attention path is available. Multiscale convolutions may recover some context.

    Fixed pretrained QKV/RoPE, tiny Wo calibration, FP32-master route weights.
    """

    architecture = "sliding_window_fusion_v30"

    def __init__(self, original: nn.Module, *,
                 radius: int = 64, block_size: int = 64):
        super().__init__()
        if not hasattr(original, "local_attention"):
            raise TypeError("Expected original ModernBERT attention")
        a, z = tuple(original.local_attention)
        if (a, z) == (-1, -1) or a != z:
            raise ValueError("SlidingWindowFusion accepts sliding attention only")
        if not (0 <= radius <= a):
            raise ValueError(f"Candidate radius {radius} exceeds native ±{a}")
        self.config = original.config
        self.layer_idx = int(getattr(original, "layer_id", -1))
        self.num_heads = int(original.num_heads)
        self.head_dim = int(original.head_dim)
        self.hidden_size = self.num_heads * self.head_dim
        self.radius = int(radius)
        self.native_radius = int(a)
        self.block_size = int(block_size)
        self.Wqkv = copy.deepcopy(original.Wqkv)
        self.Wo = copy.deepcopy(original.Wo)
        self.out_drop = copy.deepcopy(original.out_drop)
        self.rotary_emb = copy.deepcopy(original.rotary_emb)
        self.Wqkv.requires_grad_(False)
        self.rotary_emb.requires_grad_(False)
        channels = self.num_heads * self.head_dim
        self.short_weight = nn.Parameter(_identity_depthwise(channels, 7))
        self.dilated_weight = nn.Parameter(_identity_depthwise(channels, 9))
        self.wide_weight = nn.Parameter(_identity_depthwise(channels, 9))
        # 99.7% nominal route mass on the exact local attention path.
        # A small but nonzero residual path can be learned when W is narrowed.
        self.route_logits = nn.Parameter(
            torch.tensor([4.0, -3.0, -3.0, -3.0]).repeat(self.num_heads, 1)
        )
        self.route_query = nn.Parameter(
            torch.zeros(self.num_heads, 4, self.head_dim)
        )
        self.local_log_temperature = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, hidden_states: Tensor, position_embeddings=None,
                attention_mask=None, sliding_window_mask=None,
                position_ids=None, **kwargs):
        b, t, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            b, t, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.transpose(3, 1).unbind(2)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        valid = valid_tokens(attention_mask, hidden_states)
        key_mask = valid[:, None, :, None].to(v.dtype)
        # Distinct from native dense mask: no T×T tensor is constructed.
        q, k, v = q * key_mask, k * key_mask, v * key_mask
        temp = self.local_log_temperature.clamp(-0.25, 0.25).exp()
        local = local_sliding_sdpa(
            q * temp[None, :, None, None], k, v, valid,
            radius=self.radius, block_size=self.block_size
        )
        c7 = grouped_value_conv(v, self.short_weight, dilation=1)
        c9 = grouped_value_conv(v, self.dilated_weight, dilation=4)
        cwide = grouped_value_conv(v, self.wide_weight, dilation=8)
        # Dynamic, per-token routing; all branches share the current true
        # input features, so no teacher outputs leak into the student.
        qnorm = F.normalize(q, dim=-1)
        dynamic = torch.einsum("bhtd,hkd->bhtk", qnorm, self.route_query)
        gate = F.softmax(
            self.route_logits.to(q.dtype)[None, :, None, :] + dynamic, -1
        )
        fused = (gate[..., 0:1]*local
                 + gate[..., 1:2]*c7
                 + gate[..., 2:3]*c9
                 + gate[..., 3:4]*cwide) * key_mask
        out = self.Wo(fused.transpose(1, 2).reshape(
            b, t, self.hidden_size
        ).to(hidden_states.dtype))
        return (self.out_drop(out),)

    def config_dict(self):
        return {
            "architecture": self.architecture,
            "layer_idx": self.layer_idx,
            "radius": self.radius,
            "native_radius": self.native_radius,
            "block_size": self.block_size,
            "window_positions": 2*self.radius+1,
            "teacher_window_positions": 2*self.native_radius+1,
            "uses_local_sdpa": True,
            "attention_free": False,
            "full_context_dense_attention": False,
            "receptive_field_conv": [7, 33, 65],
        }



# ============================================================================
# Mixed-runtime reconstruction contract consumed by standalone_hf_loader.py
# ============================================================================
def build_model(root: str | Path, manifest: dict):
    """Build a COMPLETE fully converted model from ordinary config and manifest."""
    if manifest.get("architecture") != "laya-v31-all-attention":
        raise ValueError("Unsupported mixed-runtime architecture")
    if manifest.get("runtime_format") != 1:
        raise ValueError("Unsupported standalone bundle protocol")
    global_layers = [int(x) for x in manifest["global_layers"]]
    sliding_layers = [int(x) for x in manifest["sliding_layers"]]
    if set(global_layers) & set(sliding_layers):
        raise ValueError("Overlap between global and sliding layer indices")
    global_manifest = dict(manifest)
    global_manifest["architecture"] = "laya-v29-fastwindow"
    model = _build_global_model(root, global_manifest)
    if sorted(global_layers + sliding_layers) != list(range(len(model.encoder.layers))):
        raise ValueError("All encoder attention layers must be replaced exactly once")
    for i in sliding_layers:
        original = model.encoder.layers[i].attn
        if tuple(original.local_attention) == (-1, -1):
            raise ValueError(f"Sliding layer {i} is actually global")
        model.encoder.layers[i].attn = SlidingWindowFusion(
            original,
            radius=int(manifest["sliding_radius"]),
            block_size=int(manifest["sliding_block_size"]),
        )
    return model
