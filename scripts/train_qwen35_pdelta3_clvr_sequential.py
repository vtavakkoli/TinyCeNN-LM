#!/usr/bin/env python3
"""Sequential PDelta3/GDN2-CLVR replacement of Qwen3.5 full-attention layers.

Qwen3.5 already uses a 3:1 hybrid stack: most layers are Gated DeltaNet
linear-attention layers and only every fourth layer is full attention. This
experiment leaves native linear-attention layers untouched and replaces only
full-attention layers, one at a time, with PDelta3/GDN2 + bounded Local-W +
cross-layer value routing.
"""
from __future__ import annotations

import argparse, copy, json, math, os, random, sys, time, weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, nn
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_smollm2_memory_fusion_sequential as seq
from tinycenn_lm.pdelta3_frontier import FrontierPDelta3Layer


@dataclass(frozen=True)
class QwenPDelta3CLVRConfig:
    feature_dim: int = 96
    local_window: int = 32
    chunk_size: int = 32
    conv_kernel: int = 4
    state_dtype: str = "fp16"
    variant: str = "conv4_gdn2_clvr_f96"
    local_gate_init: float = 0.72
    warm_start_previous_core: bool = True

    def validate(self, cfg):
        if self.feature_dim < 16 or self.local_window < 1 or self.conv_kernel < 1:
            raise ValueError("invalid PDelta3-CLVR dimensions")
        if not 1 <= self.chunk_size <= 32:
            raise ValueError("chunk_size must be in [1,32]")
        if self.state_dtype not in {"fp16", "fp32"}:
            raise ValueError("state_dtype must be fp16 or fp32")
        if int(cfg.num_attention_heads) % int(cfg.num_key_value_heads):
            raise ValueError("attention heads must be divisible by KV heads")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


def _text_config(model):
    return getattr(model.config, "text_config", model.config)


def full_attention_layers(model):
    kinds = list(getattr(_text_config(model), "layer_types", []))
    if not kinds:
        raise RuntimeError("Qwen3.5 config does not expose layer_types")
    return [i for i, kind in enumerate(kinds) if kind == "full_attention"]


def _repeat_kv(x, groups):
    return x.repeat_interleave(groups, dim=1)


class QwenPDelta3CLVRAttention(nn.Module):
    """Qwen3.5 Gated Attention replacement: Local-W + recurrent PDelta3/GDN2."""
    def __init__(self, original, cfg, config, layer_idx, previous_attention=None):
        super().__init__()
        config.validate(cfg)
        self.config = cfg
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(getattr(cfg, "head_dim", self.hidden_size // self.num_heads))
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = float(getattr(cfg, "attention_dropout", 0.0))
        self.is_causal = True
        self.local_window = int(config.local_window)
        object.__setattr__(self, "_previous_attention_ref", weakref.ref(previous_attention) if previous_attention is not None else None)

        # Qwen3.5 q_proj is doubled: query + post-attention output gate.
        self.q_proj = copy.deepcopy(original.q_proj)
        self.k_proj = copy.deepcopy(original.k_proj)
        self.v_proj = copy.deepcopy(original.v_proj)
        self.o_proj = copy.deepcopy(original.o_proj)
        self.q_norm = copy.deepcopy(original.q_norm)
        self.k_norm = copy.deepcopy(original.k_norm)

        self.core = FrontierPDelta3Layer(
            self.num_heads, self.num_kv_heads, self.head_dim,
            feature_dim=config.feature_dim, variant=config.variant,
            chunk_size=config.chunk_size, conv_kernel=config.conv_kernel,
            state_dtype=config.state_dtype,
        )
        init = min(max(float(config.local_gate_init), 1e-4), 1 - 1e-4)
        logit = math.log(init / (1 - init))
        self.local_gate_w = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.local_gate_b = nn.Parameter(torch.full((self.num_heads,), logit))
        self.last_value = None

    def _previous_attention(self):
        ref = object.__getattribute__(self, "_previous_attention_ref")
        return None if ref is None else ref()

    def _local_attention(self, q, k, v, attention_mask):
        kh, vh = _repeat_kv(k, self.groups), _repeat_kv(v, self.groups)
        scores = torch.matmul(q.float(), kh.float().transpose(-2, -1)) * self.scaling
        t = q.shape[-2]
        qi = torch.arange(t, device=q.device)[:, None]
        kj = torch.arange(t, device=q.device)[None, :]
        allowed = (kj <= qi) & (kj >= qi - self.local_window + 1)
        bias = torch.zeros((t, t), device=q.device, dtype=scores.dtype)
        bias.masked_fill_(~allowed, torch.finfo(scores.dtype).min)
        scores = scores + bias[None, None]
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                scores = scores + attention_mask[..., :t, :t].float()
            elif attention_mask.ndim == 2:
                key_mask = 1.0 - attention_mask[:, None, None, :t].float()
                scores = scores + key_mask * torch.finfo(scores.dtype).min
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout:
            probs = F.dropout(probs, p=self.attention_dropout)
        return torch.matmul(probs, vh.to(probs.dtype))

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("research replacement requires use_cache=False")
        shape = hidden_states.shape[:-1]
        qg = self.q_proj(hidden_states).view(*shape, self.num_heads, self.head_dim * 2)
        q, out_gate = torch.chunk(qg, 2, dim=-1)
        out_gate = out_gate.reshape(*shape, self.num_heads * self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(*shape, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*shape, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        self.last_value = v.detach()
        previous = self._previous_attention()
        routed_v = None
        if previous is not None and previous.last_value is not None and previous.last_value.shape == v.shape:
            routed_v = previous.last_value.to(device=v.device, dtype=v.dtype)
        if routed_v is None:
            routed_v = v

        global_out = self.core(q, k, v, routed_v=routed_v)
        local_out = self._local_attention(q, k, v, attention_mask)
        gate = torch.sigmoid(torch.einsum("bhtd,hd->bht", q.float(), self.local_gate_w.float()) + self.local_gate_b.float()[None, :, None]).to(q.dtype)
        mixed = gate[..., None] * local_out + (1 - gate[..., None]) * global_out
        out = mixed.transpose(1, 2).contiguous().reshape(*shape, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(out_gate)
        return self.o_proj(out.to(hidden_states.dtype)), None

    @torch.no_grad()
    def local_gate_mean(self):
        return float(torch.sigmoid(self.local_gate_b.float()).mean())


def replace_full_attention_layers(model, config, indices):
    cfg = _text_config(model)
    allowed = set(full_attention_layers(model))
    previous = None
    made = []
    for idx in sorted(indices):
        if idx not in allowed:
            raise ValueError(f"layer {idx} is not full attention")
        layer = model.model.layers[idx]
        if isinstance(layer.self_attn, QwenPDelta3CLVRAttention):
            wrapper = layer.self_attn
            object.__setattr__(wrapper, "_previous_attention_ref", weakref.ref(previous) if previous is not None else None)
        else:
            wrapper = QwenPDelta3CLVRAttention(layer.self_attn, cfg, config, idx, previous)
            layer.self_attn = wrapper
        previous = wrapper
        made.append(wrapper)
    return made


def _prefix(idx):
    return f"model.layers.{idx}.self_attn."


def selected_state(model, indices):
    prefixes = tuple(_prefix(i) for i in indices)
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if prefixes and k.startswith(prefixes)}


def atomic_torch(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp); tmp.replace(path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8"); tmp.replace(path)


def _load_selected(model, state, layers):
    inc = model.load_state_dict(state, strict=False)
    prefixes = tuple(_prefix(i) for i in layers)
    missing = [k for k in inc.missing_keys if prefixes and k.startswith(prefixes)]
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {missing[:8]}")


def save_progress(out, model, config, accepted, reports):
    payload = {
        "format_version": 1,
        "architecture": "qwen3.5-pdelta3-gdn2-clvr-localw",
        "accepted_full_attention_layers": list(accepted),
        "config": config.to_dict(), "reports": list(reports),
        "attention_state": selected_state(model, accepted),
    }
    atomic_torch(payload, out / "qwen35_progress.pt")
    atomic_json({k:v for k,v in payload.items() if k != "attention_state"}, out / "qwen35_progress.json")


def save_in_progress(out, model, config, accepted, current, rounds, pre_nll, reports, best):
    layers = accepted + [current]
    payload = {
        "format_version": 1, "status": "current_layer_needs_more_training",
        "accepted_full_attention_layers": list(accepted),
        "current_full_attention_layer": int(current), "rounds_completed": int(rounds),
        "pre_probe_nll": float(pre_nll), "config": config.to_dict(),
        "reports": list(reports), "best_report": dict(best),
        "attention_state": selected_state(model, layers),
    }
    atomic_torch(payload, out / "qwen35_in_progress.pt")
    atomic_json({k:v for k,v in payload.items() if k != "attention_state"}, out / "qwen35_in_progress.json")


def load_progress(out, model, config, resume):
    path = out / "qwen35_progress.pt"
    if not resume or not path.exists(): return [], []
    p = torch.load(path, map_location="cpu", weights_only=False)
    accepted = [int(x) for x in p.get("accepted_full_attention_layers", [])]
    if QwenPDelta3CLVRConfig.from_dict(p["config"]) != config:
        raise RuntimeError("resume architecture differs from saved config")
    if accepted:
        replace_full_attention_layers(model, config, accepted)
        _load_selected(model, p["attention_state"], accepted)
    print(f"RESUME accepted full-attention layers={accepted}", flush=True)
    return accepted, list(p.get("reports", []))


def load_in_progress(out, model, config, accepted, targets, resume):
    path = out / "qwen35_in_progress.pt"
    if not resume or not path.exists(): return None
    p = torch.load(path, map_location="cpu", weights_only=False)
    if [int(x) for x in p.get("accepted_full_attention_layers", [])] != accepted: return None
    expected = targets[len(accepted)]
    if int(p.get("current_full_attention_layer", -1)) != expected: return None
    layers = accepted + [expected]
    replace_full_attention_layers(model, config, layers)
    _load_selected(model, p["attention_state"], layers)
    print(f"RESUME current layer={expected}, rounds={p.get('rounds_completed',0)}", flush=True)
    return p


def remove_in_progress(out):
    for name in ("qwen35_in_progress.pt", "qwen35_in_progress.json"):
        p = out / name
        if p.exists(): p.unlink()


def warm_start(model, current, accepted):
    if not accepted: return
    prev = model.model.layers[accepted[-1]].self_attn
    cur = model.model.layers[current].self_attn
    if isinstance(prev, QwenPDelta3CLVRAttention) and isinstance(cur, QwenPDelta3CLVRAttention):
        cur.core.load_state_dict(prev.core.state_dict(), strict=True)
        cur.local_gate_w.data.copy_(prev.local_gate_w.data)
        cur.local_gate_b.data.copy_(prev.local_gate_b.data)
        print(f"  warm-started layer {current} from full-attention layer {accepted[-1]}", flush=True)


def trainable_groups(model, idx, lr, qkv_scale, train_qkv):
    for p in model.parameters(): p.requires_grad = False
    m = model.model.layers[idx].self_attn
    main = []
    for p in m.core.parameters(): p.requires_grad = True; main.append(p)
    for p in (m.local_gate_w, m.local_gate_b): p.requires_grad = True; main.append(p)
    slow = []
    if train_qkv:
        for child in (m.q_proj, m.k_proj, m.v_proj, m.q_norm, m.k_norm):
            for p in child.parameters(): p.requires_grad = True; slow.append(p)
    groups = [{"params": main, "lr": lr}]
    if slow: groups.append({"params": slow, "lr": lr * qkv_scale})
    return groups, main + slow


def make_optimizer(groups, device):
    try: return torch.optim.AdamW(groups, weight_decay=0.01, fused=device.type == "cuda")
    except Exception: return torch.optim.AdamW(groups, weight_decay=0.01)


def capture_attention_input(model, ids, idx, amp, with_output):
    cap: dict[str, Any] = {}
    module = model.model.layers[idx].self_attn
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None: raise RuntimeError("hidden_states not found")
        cap["hidden"] = h
        for key in ("position_embeddings","position_ids","attention_mask","cache_position"):
            if key in kwargs and kwargs[key] is not None:
                v = kwargs[key]
                if torch.is_tensor(v): v = v.detach()
                elif isinstance(v, tuple): v = tuple(x.detach() if torch.is_tensor(x) else x for x in v)
                cap[key] = v
    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        if with_output:
            with amp(): out = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
        else:
            with torch.no_grad(), amp(): out = model(input_ids=ids, use_cache=False, return_dict=True)
    finally: handle.remove()
    if "hidden" not in cap: raise RuntimeError(f"failed to capture layer {idx}")
    return cap, out


def attention_kwargs(cap):
    return {k:cap[k] for k in ("position_embeddings","position_ids","attention_mask","cache_position") if k in cap}


def call_attention(module, hidden, kwargs):
    out = module(hidden, past_key_values=None, **kwargs)
    return out[0] if isinstance(out, (tuple,list)) else out


@torch.no_grad()
def function_metrics(teacher, student, ids, idx, amp):
    tc, _ = capture_attention_input(teacher, ids, idx, amp, False)
    sc, _ = capture_attention_input(student, ids, idx, amp, False)
    hidden = sc["hidden"].detach(); kwargs = attention_kwargs(tc)
    target = call_attention(teacher.model.layers[idx].self_attn, hidden, kwargs)
    pred = call_attention(student.model.layers[idx].self_attn, hidden, kwargs)
    nmse, cosine = seq.alignment_metrics(pred, target)
    return float(nmse), float(cosine)


def distill_kl(student_logits, teacher_logits, temperature):
    s, t = student_logits.float()/temperature, teacher_logits.float()/temperature
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean") * temperature**2 / max(1, student_logits.shape[1])


def passes(nmse, cosine, inc, total, args):
    nll_ok = inc <= args.accept_incremental_delta_nll and total <= args.accept_cumulative_delta_nll
    return nll_ok if not args.strict_acceptance else (nmse <= args.accept_nmse and cosine >= args.accept_cosine and nll_ok)


def score(r):
    return (float(r["cumulative_delta_nll"]), float(r["incremental_delta_nll"]), float(r["nmse"]))


def train_round(teacher, student, idx, batch_iter, probe_blocks, teacher_nll, pre_nll, args, device, amp, round_idx):
    rescue = round_idx > 1
    lr = args.layer_lr * (args.rescue_lr_scale if rescue else 1.0)
    fw = args.rescue_functional_weight if rescue else args.functional_weight
    kw = args.rescue_kl_weight if rescue else args.kl_weight
    cw = args.rescue_ce_weight if rescue else args.ce_weight
    groups, trainable = trainable_groups(student, idx, lr, args.qkv_lr_scale, args.train_qkv)
    opt = make_optimizer(groups, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and seq.choose_dtype(device) == torch.float16))
    student.train(); module = student.model.layers[idx].self_attn
    best = best_state = None
    for step in range(1, args.max_layer_steps + 1):
        ids = next(batch_iter).to(device, non_blocking=True)
        tc, to = capture_attention_input(teacher, ids, idx, amp, True)
        sc, so = capture_attention_input(student, ids, idx, amp, True)
        hidden = sc["hidden"].detach(); kwargs = attention_kwargs(tc)
        with torch.no_grad(), amp(): target = call_attention(teacher.model.layers[idx].self_attn, hidden, kwargs)
        with amp():
            pred = call_attention(module, hidden, kwargs)
            functional = seq.alignment_loss(pred, target, args.cosine_weight)
            kl = distill_kl(so.logits, to.logits, args.temperature)
            local_penalty = torch.sigmoid(module.local_gate_b.float()).mean()
            loss = fw*functional + kw*kl + cw*so.loss + args.local_gate_penalty*local_penalty
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss layer {idx} step {step}")
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        grad = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0)); scaler.step(opt); scaler.update()
        if step == 1 or step % args.log_every == 0:
            bn, bc = seq.alignment_metrics(pred.detach(), target.detach())
            print(f"layer={idx:02d} round={round_idx:02d} step={step:03d}/{args.max_layer_steps} loss={float(loss.detach()):.4f} func={float(functional.detach()):.4f} nmse={float(bn):.4f} cos={float(bc):.4f} kl={float(kl.detach()):.4f} ce={float(so.loss.detach()):.4f} local={module.local_gate_mean():.3f} grad={grad:.3f}", flush=True)
        if step >= args.min_layer_steps and (step % args.check_every == 0 or step == args.max_layer_steps):
            nmse, cosine = function_metrics(teacher, student, ids, idx, amp)
            nll = seq.probe_nll(student, probe_blocks, device, amp); student.train()
            inc, cum = nll - pre_nll, nll - teacher_nll
            ok = passes(nmse, cosine, inc, cum, args)
            cand = {"layer":idx,"nmse":nmse,"cosine":cosine,"probe_nll":nll,"incremental_delta_nll":inc,"cumulative_delta_nll":cum,"local_gate_mean":module.local_gate_mean(),"step":step,"round":round_idx,"accepted":ok}
            if best is None or score(cand) < score(best): best, best_state = cand, selected_state(student, [idx])
            print(f"  CHECK layer={idx:02d} round={round_idx:02d} NMSE={nmse:.4f} (≤{args.accept_nmse:.4f}) cos={cosine:.4f} (≥{args.accept_cosine:.4f}) ΔNLL_inc={inc:+.5f} (≤{args.accept_incremental_delta_nll:+.5f}) ΔNLL_total={cum:+.5f} (≤{args.accept_cumulative_delta_nll:+.5f}) local={module.local_gate_mean():.3f} => {'PASS' if ok else 'continue'}", flush=True)
            if ok: best, best_state = cand, selected_state(student, [idx]); break
        del to, so, pred, target, loss
    if best is None or best_state is None: raise RuntimeError("acceptance was never evaluated")
    _load_selected(student, best_state, [idx])
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B"); p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu"); p.add_argument("--dataset-config", default="sample-10BT"); p.add_argument("--split", default="train"); p.add_argument("--text-field", default="text"); p.add_argument("--shuffle-buffer", type=int, default=2048); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=96); p.add_argument("--local-window", type=int, default=32); p.add_argument("--chunk-size", type=int, default=32); p.add_argument("--conv-kernel", type=int, default=4); p.add_argument("--state-dtype", choices=("fp16","fp32"), default="fp16"); p.add_argument("--local-gate-init", type=float, default=0.72); p.add_argument("--warm-start-previous-core", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--target-full-layers", type=int, default=3)
    p.add_argument("--context-length", type=int, default=128); p.add_argument("--probe-context", type=int, default=128); p.add_argument("--probe-blocks", type=int, default=6); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min-layer-steps", type=int, default=60); p.add_argument("--max-layer-steps", type=int, default=250); p.add_argument("--check-every", type=int, default=25); p.add_argument("--layer-lr", type=float, default=2e-4); p.add_argument("--qkv-lr-scale", type=float, default=0.10); p.add_argument("--train-qkv", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--functional-weight", type=float, default=0.30); p.add_argument("--kl-weight", type=float, default=1.0); p.add_argument("--ce-weight", type=float, default=0.08); p.add_argument("--cosine-weight", type=float, default=0.20); p.add_argument("--local-gate-penalty", type=float, default=0.001)
    p.add_argument("--rescue-lr-scale", type=float, default=0.50); p.add_argument("--rescue-functional-weight", type=float, default=0.15); p.add_argument("--rescue-kl-weight", type=float, default=1.50); p.add_argument("--rescue-ce-weight", type=float, default=0.12)
    p.add_argument("--accept-nmse", type=float, default=0.15); p.add_argument("--accept-cosine", type=float, default=0.94); p.add_argument("--accept-incremental-delta-nll", type=float, default=0.015); p.add_argument("--accept-cumulative-delta-nll", type=float, default=0.05); p.add_argument("--strict-acceptance", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--max-runtime-minutes", type=float, default=240.0); p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    max_rounds = max(1, int(os.environ.get("SEQUENTIAL_MAX_ROUNDS_PER_RUN", "2")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype = seq.choose_dtype(device); amp = seq.amp_factory(device, dtype)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(); torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} dtype={dtype} base={args.base_model} feature_dim={args.feature_dim} local_window={args.local_window}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    load_kwargs = dict(dtype=dtype, token=False, attn_implementation="eager")
    teacher = Qwen3_5ForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device); teacher.eval(); teacher.config.use_cache=False; teacher.requires_grad_(False)
    student = Qwen3_5ForCausalLM.from_pretrained(args.base_model, **load_kwargs).to(device); student.config.use_cache=False
    all_full = full_attention_layers(student)
    if not 1 <= args.target_full_layers <= len(all_full): raise ValueError(f"target-full-layers must be in [1,{len(all_full)}]")
    targets = all_full[:args.target_full_layers]
    print(f"Qwen3.5 full-attention layers: {all_full}", flush=True); print(f"target prefix: {targets}", flush=True)
    config = QwenPDelta3CLVRConfig(args.feature_dim,args.local_window,args.chunk_size,args.conv_kernel,args.state_dtype,"conv4_gdn2_clvr_f96",args.local_gate_init,args.warm_start_previous_core)
    accepted, reports = load_progress(out, student, config, args.resume)
    if accepted != targets[:len(accepted)]: raise RuntimeError(f"accepted={accepted} is not target prefix={targets}")
    inprog = load_in_progress(out, student, config, accepted, targets, args.resume) if len(accepted)<len(targets) else None
    raw = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=True).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    batch_iter = iter(seq.batches(seq.token_blocks(raw, tok, args.text_field, args.context_length), args.batch_size))
    probes = seq.build_probe_blocks(tok, context=args.probe_context, count=args.probe_blocks)
    teacher_nll = seq.probe_nll(teacher, probes, device, amp); print(f"teacher probe NLL={teacher_nll:.6f}", flush=True)
    start=time.perf_counter()
    while len(accepted)<len(targets):
        slot=len(accepted); idx=targets[slot]
        print("\n"+"="*112, flush=True); print(f"QWEN3.5 REPLACEMENT slot {slot+1}/{len(targets)} — actual layer {idx}", flush=True); print("="*112, flush=True)
        if inprog is not None and int(inprog["current_full_attention_layer"])==idx:
            pre_nll=float(inprog["pre_probe_nll"]); rounds=int(inprog.get("rounds_completed",0)); best=dict(inprog.get("best_report",{})); print(f"continuing layer {idx} from saved best checkpoint", flush=True)
        else:
            pre_nll=seq.probe_nll(student, probes, device, amp); print(f"before replacement NLL={pre_nll:.6f} Δteacher={pre_nll-teacher_nll:+.6f}", flush=True)
            replace_full_attention_layers(student, config, accepted+[idx]);
            if config.warm_start_previous_core: warm_start(student, idx, accepted)
            rounds=0; best={}
        accepted_now=False
        for local_round in range(1,max_rounds+1):
            round_idx=rounds+local_round; print(f"\n--- full-attention layer {idx} round {round_idx} ---", flush=True)
            report=train_round(teacher,student,idx,batch_iter,probes,teacher_nll,pre_nll,args,device,amp,round_idx); reports.append(report)
            if not best or score(report)<score(best): best=report
            if report["accepted"]:
                accepted.append(idx); save_progress(out,student,config,accepted,reports); remove_in_progress(out); inprog=None; accepted_now=True
                print(f"✅ ACCEPTED Qwen3.5 full-attention layer {idx}; prefix={accepted}", flush=True); break
            save_in_progress(out,student,config,accepted,idx,round_idx,pre_nll,reports,best); print(f"Layer {idx} not accepted; best checkpoint saved.", flush=True)
            if (time.perf_counter()-start)/60 >= args.max_runtime_minutes*0.75:
                atomic_json({"status":"paused_runtime_budget","accepted_full_attention_layers":accepted,"current_full_attention_layer":idx,"best_report":best}, out/"qwen35_run_status.json"); return 0
        if not accepted_now:
            status={"status":"current_layer_needs_more_training","architecture":"Qwen3.5-PDelta3-GDN2-CLVR+LocalW","base_model":args.base_model,"native_full_attention_layers":all_full,"target_full_attention_layers":targets,"accepted_full_attention_layers":accepted,"current_full_attention_layer":idx,"rounds_completed":rounds+max_rounds,"best_report":best,"message":"Rerun with RESUME=True to continue from the best saved checkpoint."}
            atomic_json(status,out/"qwen35_run_status.json"); print("\nNOT A CRASH:",json.dumps(status,indent=2),flush=True); return 0
    final_nll=seq.probe_nll(student,probes,device,amp)
    status={"status":"target_full_attention_prefix_accepted","architecture":"Qwen3.5-PDelta3-GDN2-CLVR+LocalW","base_model":args.base_model,"native_full_attention_layers":all_full,"target_full_attention_layers":targets,"accepted_full_attention_layers":accepted,"teacher_probe_nll":teacher_nll,"final_probe_nll":final_nll,"final_delta_nll":final_nll-teacher_nll,"config":config.to_dict(),"elapsed_minutes":(time.perf_counter()-start)/60,"peak_vram_gib":torch.cuda.max_memory_allocated()/(1024**3) if device.type=="cuda" else 0.0}
    atomic_json(status,out/"qwen35_run_status.json"); tok.save_pretrained(out/"tokenizer"); print("\nFINAL STATUS",json.dumps(status,indent=2),flush=True); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Persistent best checkpoints remain resumable.", file=sys.stderr, flush=True); raise
