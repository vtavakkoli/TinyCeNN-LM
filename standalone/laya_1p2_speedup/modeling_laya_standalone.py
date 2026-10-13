"""Fully self-contained Laya-1.2-speedup inference code.

Requires only torch, transformers, safetensors. Does NOT download or import
convaiinnovations/laya-typed-decisions or TinyCeNN-LM at inference.
The model weights and tokenizer must be present in the same export directory.
Option-score model only; auxiliary action logits are intentionally unavailable.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoConfig, AutoTokenizer

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



def _apply_rope_compat(cfg):
    rope = getattr(cfg, "rope_parameters", None)
    if not isinstance(rope,dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention","global_rope_theta"),
                             ("sliding_attention","local_rope_theta")):
        v = rope.get(layer_type)
        theta = v.get("rope_theta") if isinstance(v,dict) else flat
        if theta is not None and hasattr(cfg,attr):
            setattr(cfg,attr,float(theta))


class LayaStandalone(nn.Module):
    """Native pretrained ModernBERT + exact 64-token option-scoring head.

    ALL weights are stored locally in model.safetensors. No auxiliary act_head.
    """
    def __init__(self, encoder_config=None, *, budget=64, head_layers=2,
                 pretrained_components=None):
        super().__init__()
        self.budget = int(budget)
        assert self.budget == 64, "This checkpoint implements the validated 64-token head"
        if pretrained_components is not None:
            # EXPORT ONLY: reuse the loaded upstream model without duplicating
            # ~400M weights; this branch is not needed during standalone load.
            self.encoder = pretrained_components.encoder
            self.head = pretrained_components.head
            self.type_emb = pretrained_components.type_emb
            self.scorer = pretrained_components.scorer
        else:
            assert encoder_config is not None
            _apply_rope_compat(encoder_config)
            self.encoder = AutoModel.from_config(
                encoder_config, attn_implementation="sdpa")
            d = int(encoder_config.hidden_size)
            nhead = max(1,d//64)
            layer = nn.TransformerEncoderLayer(
                d,nhead,dim_feedforward=4*d,dropout=0.1,
                batch_first=True,norm_first=True)
            self.head = nn.TransformerEncoder(
                layer,int(head_layers),enable_nested_tensor=False)
            self.type_emb = nn.Embedding(3,d)
            self.scorer = nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),
                                        nn.GELU(),nn.Linear(d,1))

    def _select_head_tokens(self,h,attention_mask,marker_pos,marker_mask):
        # Copied from V4.6, keeping exact order/masks for reproduction.
        b,t,d=h.shape
        cls=torch.zeros((b,1),dtype=torch.long,device=h.device)
        lm_count=min(self.budget,max(t-1,1))
        lm=torch.linspace(1,max(t-1,1),steps=lm_count,device=h.device).round().long()
        lm=lm.clamp(max=t-1).view(1,-1).expand(b,-1)
        marker_ok=marker_mask.bool()
        lm_ok=attention_mask.bool().gather(1,lm)
        lm_ok=lm_ok & (lm!=0)
        duplicate=(lm[:,:,None]==marker_pos[:,None,:]) & marker_ok[:,None,:]
        lm_ok=lm_ok & ~duplicate.any(-1)
        idx=torch.cat((cls,marker_pos.clamp(0,t-1),lm),dim=1)
        keep=torch.cat((torch.ones((b,1),dtype=torch.bool,device=h.device),
                        marker_ok,lm_ok),dim=1)
        gathered=h.gather(1,idx[:,:,None].expand(-1,-1,d))
        return gathered,keep

    def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype):
        h=self.encoder(input_ids=input_ids,
                       attention_mask=attention_mask).last_hidden_state
        h=h+self.type_emb(qtype)[:,None,:]
        compact,keep=self._select_head_tokens(
            h,attention_mask,marker_pos,marker_mask)
        transformed=self.head(compact,src_key_padding_mask=~keep)
        marker_hidden=transformed[:,1:1+marker_pos.shape[1],:]
        logits=self.scorer(marker_hidden).squeeze(-1).float()
        logits=logits.masked_fill(~marker_mask,-1e4)
        return logits,None

    @torch.inference_mode()
    def predict(self,tokenizer,state,questions,*,max_len=512,head_max_len=192):
        items=[];names=[];labels=[]
        for name,raw in questions.items():
            q=normalize_question(raw)
            if q["t"]!="choice":
                raise ValueError("Only choice questions supported by predict()")
            opts=list(q["crit"].keys())
            ids,markers=build_sequence(tokenizer,state,q,max_len=max_len,
                                       head_max_len=head_max_len)
            if len(markers)!=len(opts):
                raise ValueError("Truncated an option marker")
            names.append(name);labels.append(opts)
            items.append(dict(ids=ids,markers=markers,qtype=QTYPES[q["t"]]))
        batch=collate_items(items,int(tokenizer.pad_token_id))
        device=next(self.parameters()).device
        batch={k:v.to(device) for k,v in batch.items()}
        with (torch.autocast("cuda",dtype=torch.float16)
              if device.type=="cuda" else __import__("contextlib").nullcontext()):
            logits,_=self(**batch)
        probs=logits.float().masked_fill(~batch["marker_mask"],-1e9).softmax(-1).cpu()
        answers={}
        for i,name in enumerate(names):
            p=probs[i,:len(labels[i])]
            best=int(p.argmax())
            answers[name]=dict(choice=labels[i][best],
                               confidence=float(p[best]),
                               probabilities={label:float(pr) for label,pr in zip(labels[i],p)})
        return answers


def load_model(model_dir,*,device="cpu",local_files_only=True,compile_head=False):
    """Load entirely from the local standalone directory. No network needed."""
    root=Path(model_dir).resolve()
    metadata=json.loads((root/"config.json").read_text())
    assert metadata["format"]=="laya-1p2-full-standalone-v1"
    enc_cfg=AutoConfig.from_pretrained(
        str(root/"encoder"),local_files_only=local_files_only)
    model=LayaStandalone(enc_cfg,budget=int(metadata["head_context_budget"]),
                         head_layers=int(metadata["head_layers"]))
    state=load_file(str(root/"model.safetensors"),device="cpu")
    model.load_state_dict(state,strict=True)
    model=model.to(device).eval().requires_grad_(False)
    model.encoder.config.reference_compile=False
    if compile_head:
        model.head=torch.compile(model.head,mode="reduce-overhead",dynamic=False)
    tok=AutoTokenizer.from_pretrained(
        str(root/"tokenizer"),local_files_only=local_files_only)
    return model,tok
