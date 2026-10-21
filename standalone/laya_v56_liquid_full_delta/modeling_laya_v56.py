"""Laya V5.0 all-28-layer brute-force hybrid SVD compression, fully standalone.

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

def local_pair_indices(model):
    """Verify FULL / same-window LOCAL / LOCAL triples in ModernBERT.

    transformers==4.57.6 uses attn.local_attention: (-1,-1) means FULL,
    (64,64) is a 128-token sliding window. Newer builds can expose
    config.layer_types with attn.sliding_window instead. Support both.
    """
    layers=model.encoder.layers
    cfg=model.encoder.config
    n=len(layers)
    layer_types=getattr(cfg,"layer_types",None)
    if layer_types is not None and len(layer_types)!=n:
        raise ValueError("ModernBERT layer_types count does not match layers")

    def mode_at(i):
        a=layers[i].attn
        old_window=getattr(a,"local_attention",None)
        if old_window is not None:
            window=tuple(old_window)
            if window==(-1,-1):
                return ("full",None)
            if len(window)!=2 or min(window)<0:
                raise ValueError(f"Malformed local attention window at {i}: {window}")
            return ("local",window)
        typ=(layer_types[i] if layer_types is not None else
             getattr(layers[i],"attention_type",None))
        window=getattr(a,"sliding_window",None)
        if typ=="full_attention" and window is None:
            return ("full",None)
        if typ=="sliding_attention" and window is not None:
            return ("local",int(window))
        raise ValueError(f"Unsupported ModernBERT attention metadata at layer {i}: {typ}, {window}")

    pairs=[]
    for anchor in range(0,n,3):
        mode,_=mode_at(anchor)
        if mode!="full":
            raise ValueError(f"Layer {anchor} must be FULL attention")
        left_idx,right_idx=anchor+1,anchor+2
        if right_idx>=n:
            continue
        m1,w1=mode_at(left_idx)
        m2,w2=mode_at(right_idx)
        if m1!="local" or m2!="local" or w1!=w2:
            raise ValueError(
                f"Layers {(left_idx,right_idx)} must have identical local windows: {m1}/{w1}, {m2}/{w2}")
        pairs.append((left_idx,right_idx))
    return pairs


class SVDDeltaLinear(nn.Module):
    """Second-layer linear projection = shared first-layer projection + A @ B.

    The shared linear is a plain Python reference, NOT a registered submodule;
    its weights appear only once in state_dict, under the original first layer.
    """
    def __init__(self, first_linear, rank, *, A=None, B=None):
        super().__init__()
        if not isinstance(first_linear,nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(first_linear)}")
        out_dim,in_dim=first_linear.weight.shape
        rank=int(rank)
        if not (1<=rank<min(in_dim,out_dim)):
            raise ValueError(f"Unsupported rank {rank} for {out_dim}x{in_dim}")
        # Using object.__setattr__ prevents double registering the shared module.
        object.__setattr__(self,"_shared",first_linear)
        self.in_features=int(in_dim)
        self.out_features=int(out_dim)
        self.max_rank=rank
        self.active_rank=rank
        self.A=nn.Parameter(torch.zeros((out_dim,rank),device=first_linear.weight.device,
                                        dtype=first_linear.weight.dtype))
        self.B=nn.Parameter(torch.zeros((rank,in_dim),device=first_linear.weight.device,
                                        dtype=first_linear.weight.dtype))
        if A is not None and B is not None:
            with torch.no_grad():
                self.A.copy_(A.to(self.A.device,self.A.dtype))
                self.B.copy_(B.to(self.B.device,self.B.dtype))

    def set_rank(self, rank):
        rank=int(rank)
        if rank<1 or rank>self.max_rank:
            raise ValueError(f"rank {rank} outside 1..{self.max_rank}")
        self.active_rank=rank

    def shrink(self, rank):
        """Permanently discard extra SVD columns for a compact checkpoint."""
        rank=int(rank)
        if rank<1 or rank>self.max_rank:
            raise ValueError(rank)
        self.A=nn.Parameter(self.A[:,:rank].detach().clone())
        self.B=nn.Parameter(self.B[:rank,:].detach().clone())
        self.max_rank=rank
        self.active_rank=rank

    def forward(self,x):
        # fp16 autocast handles both paths identically, with FP32 stored weights.
        w=self._shared.weight
        bias=self._shared.bias
        base=F.linear(x,w,bias)
        r=self.active_rank
        return base+F.linear(F.linear(x,self.B[:r,:]),self.A[:,:r])



# V4.9 does NOT force compression of all nine local pairs.
# Each selected follower projection gets a separate low-rank residual;
# all unselected attention projection matrices retain pretrained weights.
def projection_slots(model):
    slots=[]
    for first,second in local_pair_indices(model):
        for component in ("Wqkv","Wo"):
            a=getattr(model.encoder.layers[first].attn,component)
            b=getattr(model.encoder.layers[second].attn,component)
            if not isinstance(a,nn.Linear) or not isinstance(b,nn.Linear):
                raise TypeError(f"Expected nn.Linear for layer {second}.{component}")
            if a.weight.shape != b.weight.shape:
                raise ValueError(f"Incompatible original projection: {second}.{component}")
            if a.bias is not None or b.bias is not None:
                raise ValueError(f"Biasful projection needs separate residual bias: {second}.{component}")
            slots.append((first,second,component))
    return slots


def _validate_slot(model, first, second, component):
    """Validate schedule independently of other already-compressed slots."""
    first,second=int(first),int(second)
    component=str(component)
    if (first,second) not in local_pair_indices(model):
        raise ValueError(f"Unknown local pair: {(first,second)}")
    if component not in ("Wqkv","Wo"):
        raise ValueError(f"Unknown projection: {component}")
    return first,second,component


def svd_factors(model, first, second, component, max_rank=256,
                oversampling=8, niter=2):
    """Compute one factorization per projection from ORIGINAL pretrained weights.

    A and B are float32 CPU tensors and can be sliced for different ranks.
    """
    first,second,component=_validate_slot(model,first,second,component)
    a=getattr(model.encoder.layers[first].attn,component)
    b=getattr(model.encoder.layers[second].attn,component)
    if not isinstance(a,nn.Linear) or not isinstance(b,nn.Linear):
        raise ValueError("SVD must be done BEFORE that follower is compressed")
    with torch.no_grad():
        delta=b.weight.detach().float()-a.weight.detach().float()
        r=min(int(max_rank),min(delta.shape)-1)
        q=min(r+int(oversampling),min(delta.shape))
        # Randomized SVD (approximation, not an exact mathematical SVD).
        U,S,V=torch.svd_lowrank(delta,q=q,niter=int(niter))
        root=S[:r].clamp_min(0).sqrt()
        A=(U[:,:r]*root.unsqueeze(0)).cpu().contiguous()
        B=(root.unsqueeze(1)*V[:,:r].T).cpu().contiguous()
        total=float(delta.square().sum().item())
        error=float((delta-(A.to(delta.device)@B.to(delta.device))).square().sum().item())
        del U,S,V,delta
    return A,B,{"rank":r,
        "relative_frobenius_error":(error/max(total,1e-30))**0.5,
        "saved_parameter_bytes_if_rank256":None}


def build_adapter(model, first, second, component, rank, factors=None,
                  *, initialized=True):
    """Replace ONE follower's projection; return original module for rollback.

    If initialized=False, A/B are zeros until strictly loaded from checkpoint.
    """
    first=int(first);second=int(second);component=str(component)
    if (first,second) not in local_pair_indices(model):
        raise ValueError(f"Invalid local pair: {(first,second)}")
    leader=getattr(model.encoder.layers[first].attn,component)
    follower_block=model.encoder.layers[second].attn
    old=getattr(follower_block,component)
    if not isinstance(leader,nn.Linear) or not isinstance(old,nn.Linear):
        raise TypeError("Projection already wrapped or unavailable")
    if leader.bias is not None or old.bias is not None:
        raise ValueError("Biasful attention projections are unsupported")
    if initialized:
        if factors is None:
            A,B,_=svd_factors(model,first,second,component,max_rank=rank)
        else:
            A,B=factors
        adapter=SVDDeltaLinear(leader,rank,A=A[:,:rank],B=B[:rank,:])
    else:
        adapter=SVDDeltaLinear(leader,rank)
    setattr(follower_block,component,adapter)
    return old


def restore_original_projection(model, second, component, original):
    if not isinstance(original,nn.Linear):
        raise TypeError("Rollback needs the original pretrained linear module")
    setattr(model.encoder.layers[int(second)].attn,str(component),original)


def selected_adapters(model):
    selected=[]
    for first,second in local_pair_indices(model):
        for component in ("Wqkv","Wo"):
            layer=getattr(model.encoder.layers[second].attn,component)
            if isinstance(layer,SVDDeltaLinear):
                selected.append((first,second,component,layer))
    return selected


def unique_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def estimated_saved_bytes(model, first, second, component, rank):
    leader=getattr(model.encoder.layers[first].attn,component)
    outdim,indim=leader.weight.shape
    # Full follower weight removed; A (out×rank), B (rank×in) remain.
    return 4*(outdim*indim-int(rank)*(outdim+indim))


def load_v49_model(model_dir, *, device="cpu", compile_head=False):
    """Strict, fully offline restoration of selectively compressed model."""
    root=Path(model_dir).resolve()
    meta=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if meta["format"]!="laya-v49-selective-svd-v1":
        raise ValueError("Wrong model format: expected V4.9 selective SVD")
    cfg=AutoConfig.from_pretrained(str(root/"encoder"),local_files_only=True)
    model=LayaStandalone(cfg,budget=int(meta["head_context_budget"]),
                         head_layers=int(meta["head_layers"]))
    for item in meta["compressed_projections"]:
        build_adapter(model,int(item["first"]),int(item["second"]),
                      str(item["component"]),int(item["rank"]),
                      initialized=False)
    weights=load_file(str(root/"model.safetensors"),device="cpu")
    model.load_state_dict(weights,strict=True)
    # No duplicated original follower weights or duplicated leader submodule.
    assert len(selected_adapters(model))==len(meta["compressed_projections"])
    for first,second,component,wrapper in selected_adapters(model):
        leader=getattr(model.encoder.layers[first].attn,component)
        assert wrapper._shared is leader
        assert "_shared" not in wrapper._modules
    model=model.to(device).eval().requires_grad_(False)
    model.encoder.config.reference_compile=False
    if compile_head:
        model.head=torch.compile(model.head,mode="reduce-overhead",dynamic=False)
    tokenizer=AutoTokenizer.from_pretrained(str(root/"tokenizer"),local_files_only=True)
    return model,tokenizer


# ---------------------------------------------------------------------------
# V5.0: brute-force all 28 ModernBERT blocks, attention + FFN, preserving the
# proven V4.9 pretrained-pair DELTA path as one selectable candidate family.
# ---------------------------------------------------------------------------

class IndependentSVDLinear(nn.Module):
    """Plain original projection compressed as A @ B; original dense W omitted.

    Compatible with ModernBERT's attention Wqkv, Wo and MLP Wi, Wo calls.
    A saved bias is kept if the original linear was biased.
    """
    def __init__(self, original: nn.Linear, rank: int, A=None, B=None):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("Expected original nn.Linear")
        out_dim,in_dim=original.weight.shape
        rank=int(rank)
        if rank<=0 or rank>=min(out_dim,in_dim):
            raise ValueError(f"Bad SVD rank {rank} for {out_dim}x{in_dim}")
        self.in_features=int(in_dim)
        self.out_features=int(out_dim)
        self.rank=rank
        self.A=nn.Parameter(torch.zeros((out_dim,rank),
            dtype=original.weight.dtype,device=original.weight.device))
        self.B=nn.Parameter(torch.zeros((rank,in_dim),
            dtype=original.weight.dtype,device=original.weight.device))
        if original.bias is None:
            self.register_parameter("bias",None)
        else:
            self.bias=nn.Parameter(original.bias.detach().clone())
        if A is not None and B is not None:
            with torch.no_grad():
                self.A.copy_(A[:,:rank].to(device=self.A.device,dtype=self.A.dtype))
                self.B.copy_(B[:rank,:].to(device=self.B.device,dtype=self.B.dtype))

    def forward(self,x):
        return F.linear(F.linear(x,self.B),self.A,self.bias)


def all_encoder_slots(model):
    """Enumerate available projections in EVERY pretrained encoder layer."""
    slots=[]
    for index,layer in enumerate(model.encoder.layers):
        for block,names in (("attn",("Wqkv","Wo")),("mlp",("Wi","Wo"))):
            component=getattr(layer,block)
            for name in names:
                module=getattr(component,name,None)
                if module is None:
                    raise ValueError(f"Required projection layer {index}.{block}.{name} missing")
                if not isinstance(module,nn.Linear):
                    raise TypeError(f"Not an uncompressed nn.Linear: {index}.{block}.{name}: {type(module)}")
                slots.append((index,block,name))
    if len(slots)!=4*len(model.encoder.layers):
        raise RuntimeError("Expected exactly two attention and two FFN projections per layer")
    return slots


def list_candidate_families(model):
    """Full brute-force search space: 112 own-SVD + 18 delta-shared slots."""
    options=[{"layer":int(layer),"block":block,"component":name,
              "mode":"own_svd"}
             for layer,block,name in all_encoder_slots(model)]
    for first,second in local_pair_indices(model):
        for component in ("Wqkv","Wo"):
            options.append({"layer":int(second),"leader":int(first),
                            "block":"attn","component":component,
                            "mode":"pair_delta"})
    return options


def own_svd_factors(original, *, max_rank=256, oversampling=8,niter=2):
    """Randomized, truncated SVD of ORIGINAL pretrained W (not layer delta)."""
    if not isinstance(original,nn.Linear):
        raise TypeError("Expected an uncompressed linear projection")
    with torch.no_grad():
        weight=original.weight.detach().float()
        rank=min(int(max_rank),min(weight.shape)-1)
        q=min(rank+int(oversampling),min(weight.shape))
        U,S,V=torch.svd_lowrank(weight,q=q,niter=int(niter))
        scale=S[:rank].clamp_min(0).sqrt()
        A=(U[:,:rank]*scale[None,:]).cpu().contiguous()
        B=(scale[:,None]*V[:,:rank].T).cpu().contiguous()
        approximated=A.to(weight.device)@B.to(weight.device)
        numerator=float((weight-approximated).square().sum().item())
        denominator=float(weight.square().sum().item())
    return A,B,{"rank":rank,
                "relative_frobenius_error":(numerator/max(denominator,1e-30))**.5}


def projection_at(model, spec):
    return getattr(getattr(model.encoder.layers[int(spec["layer"])],
                           str(spec["block"])),str(spec["component"]))


def replace_projection(model, spec, *, factors=None, initialized=True):
    """Apply one projection replacement; return original module for rollback."""
    idx=int(spec["layer"])
    block=str(spec["block"])
    comp=str(spec["component"])
    rank=int(spec["rank"])
    mode=str(spec["mode"])
    owner=getattr(model.encoder.layers[idx],block)
    old=getattr(owner,comp)
    if not isinstance(old,nn.Linear):
        raise TypeError(f"Already compressed or unsupported {idx}.{block}.{comp}")
    if mode=="own_svd":
        if initialized and factors is None:
            factors=own_svd_factors(old,max_rank=rank)[:2]
        if initialized:
            A,B=factors
            replacement=IndependentSVDLinear(old,rank,A,B)
        else:
            replacement=IndependentSVDLinear(old,rank)
    elif mode=="pair_delta":
        if block!="attn" or comp not in ("Wqkv","Wo"):
            raise ValueError("Pair delta only supported for local attention")
        leader=int(spec["leader"])
        if (leader,idx) not in local_pair_indices(model):
            raise ValueError(f"Not a valid local pair: {leader},{idx}")
        shared=getattr(model.encoder.layers[leader].attn,comp)
        if not isinstance(shared,nn.Linear):
            raise TypeError("Leader must retain original pretrained linear")
        if shared.bias is not None or old.bias is not None:
            raise ValueError("Pair delta only supported for bias-free attention")
        if initialized:
            if factors is None:
                A,B,_=svd_factors(model,leader,idx,comp,max_rank=rank)
            else:
                A,B=factors
            replacement=SVDDeltaLinear(shared,rank,A=A[:,:rank],B=B[:rank,:])
        else:
            replacement=SVDDeltaLinear(shared,rank)
    else:
        raise ValueError(f"Unsupported family: {mode}")
    setattr(owner,comp,replacement)
    return old


def revert_projection(model,spec,original):
    if not isinstance(original,nn.Linear):
        raise TypeError("Original projection must be nn.Linear")
    owner=getattr(model.encoder.layers[int(spec["layer"])],str(spec["block"]))
    setattr(owner,str(spec["component"]),original)


def applied_adapters(model,selected_specs):
    result=[]
    for spec in selected_specs:
        wrapper=projection_at(model,spec)
        if not isinstance(wrapper,(IndependentSVDLinear,SVDDeltaLinear)):
            raise AssertionError(f"Missing compressed adapter for {spec}")
        result.append(wrapper)
    return result


def expected_saved_bytes(model,spec):
    """FP32 parameter saving for one replacement, including any old bias."""
    idx=int(spec["layer"])
    owner=getattr(model.encoder.layers[idx],str(spec["block"]))
    original=getattr(owner,str(spec["component"]))
    if not isinstance(original,nn.Linear):
        raise TypeError("Call before installing candidate")
    out_dim,in_dim=original.weight.shape
    rank=int(spec["rank"])
    # Bias exists in both old and new for own_svd; cancels out.
    # Pair delta has no bias.
    return int(4*(out_dim*in_dim-rank*(out_dim+in_dim)))


def load_v50_model(model_dir, *, device="cpu",compile_head=False):
    """Strict reload of all hybrid low-rank projections from local disk only."""
    root=Path(model_dir).resolve()
    info=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if info["format"]!="laya-v50-all28-bruteforce-v1":
        raise ValueError("Not a V5.0 hybrid model")
    cfg=AutoConfig.from_pretrained(str(root/"encoder"),local_files_only=True)
    model=LayaStandalone(cfg,budget=int(info["head_context_budget"]),
                         head_layers=int(info["head_layers"]))
    # Rebuild in the same order: only uncompressed leaders are valid for delta.
    for spec in info["compressed_projections"]:
        replace_projection(model,spec,initialized=False)
    full=load_file(str(root/"model.safetensors"),device="cpu")
    model.load_state_dict(full,strict=True)
    for spec,adapter in zip(info["compressed_projections"],
                            applied_adapters(model,info["compressed_projections"])):
        if spec["mode"]=="pair_delta":
            leader=getattr(model.encoder.layers[int(spec["leader"])].attn,
                           str(spec["component"]))
            assert adapter._shared is leader
            assert "_shared" not in adapter._modules
    model.to(device).eval().requires_grad_(False)
    model.encoder.config.reference_compile=False
    if compile_head:
        model.head=torch.compile(model.head,mode="reduce-overhead",dynamic=False)
    tok=AutoTokenizer.from_pretrained(str(root/"tokenizer"),local_files_only=True)
    return model,tok


# ================================================================
# Laya V5.3: CeNN-inspired token-neighborhood liquid recurrence
# ================================================================

class LiquidCeNNProjection(nn.Module):
    """Replace/augment an FFN output projection with a 1D liquid CeNN cell.

    This is CeNN-*inspired*: local nearest-neighbor coupling via depthwise
    Conv1d over tokens plus gated state relaxation. It is not the original
    continuous-time Chua CeNN equation, nor an LTC ODE solver.

    Modes:
      * replace: remove the old projection entirely, f(x)=liquid(x).
      * residual: keep the original projection, f(x)=base(x)+scale*liquid(x).

    The residual scale is initially zero, so this mode is exactly the
    pretrained baseline at creation time, including identical predictions.
    """
    def __init__(self,original,width=64,steps=3,mode="residual",
                 dt_bound=0.50):
        super().__init__()
        if not hasattr(original,"in_features") or not hasattr(original,"out_features"):
            raise TypeError("LiquidCeNN requires a linear-like FFN projection")
        self.in_features=int(original.in_features)
        self.out_features=int(original.out_features)
        self.width=int(width)
        self.steps=int(steps)
        self.mode=str(mode)
        self.dt_bound=float(dt_bound)
        if self.width<4 or self.steps<1 or self.steps>10:
            raise ValueError("Bad cell width or recurrent step count")
        if self.mode not in ("replace","residual"):
            raise ValueError(self.mode)
        if not (0.0<self.dt_bound<=1.0):
            raise ValueError("dt_bound must be in (0,1]")
        # Only register the original dense/low-rank projection in residual
        # mode; replacement genuinely deletes its weights from state_dict.
        self.base=original if self.mode=="residual" else None
        self.drive=nn.Linear(self.in_features,self.width,bias=True)
        self.drive_norm=nn.LayerNorm(self.width)
        self.neighbor=nn.Conv1d(self.width,self.width,kernel_size=3,
                                padding=1,groups=self.width,bias=False)
        self.recurrent=nn.Linear(self.width,self.width,bias=False)
        self.liquid_gate=nn.Linear(2*self.width,self.width)
        self.readout=nn.Linear(self.width,self.out_features,bias=True)
        # Start with small dynamic corrections instead of random large logits.
        nn.init.normal_(self.readout.weight,mean=0.,std=0.008)
        nn.init.zeros_(self.readout.bias)
        self.scale=nn.Parameter(torch.tensor(0.,dtype=torch.float32))

        # Match original device and dtype without moving the original base.
        original_parameter=next(original.parameters(),None)
        if original_parameter is not None:
            for module in [self.drive,self.drive_norm,self.neighbor,
                           self.recurrent,self.liquid_gate,self.readout]:
                module.to(device=original_parameter.device,dtype=original_parameter.dtype)

    def liquid(self,x):
        # x must have [batch, tokens, d_ffn]. Spatial nearest neighbors
        # are token neighbors, not a two-dimensional image lattice.
        if x.ndim!=3:
            raise ValueError(f"Expected [B,T,D] FFN activation, got {tuple(x.shape)}")
        u=torch.tanh(self.drive_norm(self.drive(x)))
        h=torch.zeros_like(u)
        for _ in range(self.steps):
            neighbors=self.neighbor(h.transpose(1,2)).transpose(1,2)
            update_gate=torch.sigmoid(self.liquid_gate(torch.cat((u,h),dim=-1)))
            dt=self.dt_bound*update_gate
            proposal=torch.tanh(u+neighbors+self.recurrent(h))
            h=h+dt*(proposal-h)
        return self.readout(h)

    def forward(self,x):
        correction=self.liquid(x)
        if self.mode=="replace":
            return correction
        # Alpha=0 is an EXACT base-model control despite random CeNN weights.
        return self.base(x)+self.scale.to(dtype=correction.dtype)*correction


def install_liquid_ffn(model,layer_index,*,width=64,steps=3,mode="residual",
                       dt_bound=0.5):
    layer_index=int(layer_index)
    if not (0<=layer_index<len(model.encoder.layers)):
        raise IndexError(layer_index)
    owner=model.encoder.layers[layer_index].mlp
    previous=owner.Wo
    if isinstance(previous,LiquidCeNNProjection):
        raise ValueError(f"Liquid layer {layer_index} already installed")
    candidate=LiquidCeNNProjection(previous,width,steps,mode,dt_bound)
    owner.Wo=candidate
    return previous


def uninstall_liquid_ffn(model,layer_index,previous):
    layer_index=int(layer_index)
    if not isinstance(previous,nn.Module):
        raise TypeError("Replacement rollback must restore nn.Module")
    module=model.encoder.layers[layer_index].mlp
    if not isinstance(module.Wo,LiquidCeNNProjection):
        raise ValueError("Current layer is not LiquidCeNNProjection")
    module.Wo=previous


def find_liquid_cells(model):
    return [(i,block.mlp.Wo)
            for i,block in enumerate(model.encoder.layers)
            if isinstance(block.mlp.Wo,LiquidCeNNProjection)]


def load_v53_model(model_dir, *,device="cpu",compile_head=False):
    """Fully offline strict reconstruction of compressed V5.3 + liquid CeNN."""
    root=Path(model_dir).resolve()
    metadata=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if metadata.get("format")!="laya-v53-liquid-cenn-v1":
        raise ValueError("Not a V5.3 liquid CeNN export")
    cfg=AutoConfig.from_pretrained(str(root/"encoder"),local_files_only=True)
    model=LayaStandalone(cfg,budget=int(metadata["head_context_budget"]),
                         head_layers=int(metadata["head_layers"]))
    for spec in metadata["compressed_projections"]:
        replace_projection(model,spec,initialized=False)
    spec=metadata["liquid_cell"]
    install_liquid_ffn(model,int(spec["layer"]),width=int(spec["width"]),
       steps=int(spec["steps"]),mode=str(spec["mode"]),
       dt_bound=float(spec["dt_bound"]))
    state=load_file(str(root/"model.safetensors"),device="cpu")
    model.load_state_dict(state,strict=True)
    assert len(find_liquid_cells(model))==1
    model=model.to(device).eval().requires_grad_(False)
    model.encoder.config.reference_compile=False
    if compile_head:
        model.head=torch.compile(model.head,mode="reduce-overhead",dynamic=False)
    tokenizer=AutoTokenizer.from_pretrained(
       str(root/"tokenizer"),local_files_only=True)
    return model,tokenizer


# ================================================================
# V5.5 all-28 FFN-output CeNN substitution (no attention substitution)
# ================================================================
def load_v55_model(model_dir, *,device="cpu",compile_head=False):
    """Load fully standalone model with a subset or all 28 FFN liquid cells.

    Each selected encoder layer substitutes its mlp.Wo projection. Attention,
    residual connections, norms, and all other FFNs remain ModernBERT.
    """
    root=Path(model_dir).resolve()
    info=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if info.get("format")!="laya-v55-all28-liquid-cenn-v1":
        raise ValueError("This checkpoint is not an all-layer CeNN V5.5 export")
    cfg=AutoConfig.from_pretrained(str(root/"encoder"),local_files_only=True)
    model=LayaStandalone(cfg,budget=int(info["head_context_budget"]),
                         head_layers=int(info["head_layers"]))
    for spec in info["compressed_projections"]:
        replace_projection(model,spec,initialized=False)
    liquid_specs=info["liquid_cells"]
    assert len(liquid_specs)==len({int(s["layer"]) for s in liquid_specs})
    assert all(s.get("mode")=="replace" for s in liquid_specs)
    for cell in liquid_specs:
        install_liquid_ffn(model,int(cell["layer"]),
                          width=int(cell["width"]),
                          steps=int(cell["steps"]),
                          mode="replace",dt_bound=float(cell["dt_bound"]))
    full=load_file(str(root/"model.safetensors"),device="cpu")
    model.load_state_dict(full,strict=True)
    assert len(find_liquid_cells(model))==len(liquid_specs)
    model.to(device).eval().requires_grad_(False)
    model.encoder.config.reference_compile=False
    if compile_head:
        model.head=torch.compile(model.head,mode="reduce-overhead",dynamic=False)
    tokenizer=AutoTokenizer.from_pretrained(
        str(root/"tokenizer"),local_files_only=True)
    return model,tokenizer


# ================================================================
# V5.6 dual-model loader: FULL projection substitution vs DELTA addition
# ================================================================
def load_v56_model(model_dir, *,device="cpu",compile_head=False):
    """Strict, offline load of ONE of the five-layer liquid-CeNN variants.

    FULL: 5 FFN Wo modules are entirely replaced by CeNN recurrence.
    DELTA: 5 compressed original FFN Wo modules are retained and corrected
           with a residual CeNN (base(x) + scale*liquid(x)).
    The exporter must save each variant to a DIFFERENT directory.
    """
    root=Path(model_dir).resolve()
    info=json.loads((root/"config.json").read_text(encoding="utf-8"))
    if info.get("format")!="laya-v56-liquid-cenn-dual-v1":
        raise ValueError("Not a V5.6 dual-variant checkpoint")
    variant=str(info["variant"]).lower()
    if variant not in ("full","delta"):
        raise ValueError(f"Invalid V5.6 variant: {variant}")
    cfg=AutoConfig.from_pretrained(str(root/"encoder"),local_files_only=True)
    model=LayaStandalone(cfg,budget=int(info["head_context_budget"]),
                         head_layers=int(info["head_layers"]))
    for spec in info["compressed_projections"]:
        replace_projection(model,spec,initialized=False)
    cells=list(info["liquid_cells"])
    if len(cells)!=5 or len({int(s["layer"]) for s in cells})!=5:
        raise ValueError("V5.6 models require precisely five unique CeNN layers")
    required_mode="replace" if variant=="full" else "residual"
    for cell in cells:
        if cell["mode"]!=required_mode:
            raise ValueError("Mixed liquid modes are not a V5.6 variant")
        install_liquid_ffn(model,int(cell["layer"]),
            width=int(cell["width"]),steps=int(cell["steps"]),
            mode=required_mode,dt_bound=float(cell["dt_bound"]))
    full=load_file(str(root/"model.safetensors"),device="cpu")
    model.load_state_dict(full,strict=True)
    actual=find_liquid_cells(model)
    assert len(actual)==5
    assert all(m.mode==required_mode for _,m in actual)
    if variant=="full":
        assert all(m.base is None for _,m in actual)
    else:
        assert all(m.base is not None for _,m in actual)
    model.to(device).eval().requires_grad_(False)
    model.encoder.config.reference_compile=False
    if compile_head:
        model.head=torch.compile(model.head,mode="reduce-overhead",dynamic=False)
    tokenizer=AutoTokenizer.from_pretrained(
        str(root/"tokenizer"),local_files_only=True)
    return model,tokenizer
