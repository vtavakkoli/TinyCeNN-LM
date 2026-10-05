"""V2.5 typed decisions: windowed Delta + multiscale recurrent CeNN.

The reference Delta backend is for correctness tests/CPU reloads only. CUDA runs
use FLA and never silently fall back to the slow Python scan. Chunk overlap is an
approximation to sliding attention, not an exact moving-window recurrence.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .standalone_decision import _valid_tokens, _apply_rope_compat


def layer_kinds(encoder):
    """Support both ModernBERT 4.x and newer layer-type metadata."""
    kinds = []
    for i, layer in enumerate(encoder.layers):
        kind = getattr(layer, "attention_type", None)
        if kind is None:
            types = getattr(encoder.config, "layer_types", None)
            if types:
                kind = types[i]
            else:
                every = encoder.config.global_attn_every_n_layers
                kind = "full_attention" if i % every == 0 else "sliding_attention"
        if kind not in ("full_attention", "sliding_attention"):
            raise ValueError(f"Unsupported encoder layer {i}: {kind}")
        kinds.append(kind)
    return kinds


def reference_delta(q, k, v, g, beta):
    """FP32 gated delta recurrence, [B,T,H,D], matching FLA's default scale."""
    q, k, v = q.float(), F.normalize(k.float(), dim=-1), v.float()
    q = F.normalize(q, dim=-1) * q.shape[-1] ** -0.5
    state = v.new_zeros(q.shape[0], q.shape[2], q.shape[-1], v.shape[-1])
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].float().exp()[..., None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        delta = beta[:, t].float()[..., None] * (v[:, t] - prediction)
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(out, dim=1).to(v.dtype)


class WindowedBiDelta(nn.Module):
    """Independent overlapping windows; zero initial state in each direction."""
    def __init__(self, original, window=128, backend="fla"):
        super().__init__()
        if window < 2 or window % 2:
            raise ValueError("window must be an even integer >= 2")
        if backend not in ("fla", "reference"):
            raise ValueError("backend must be fla or reference")
        self.config = original.config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.window, self.stride, self.backend = int(window), int(window // 2), backend
        self.Wqkv = copy.deepcopy(original.Wqkv).requires_grad_(False)
        self.Wo = copy.deepcopy(original.Wo).requires_grad_(False)
        self.out_drop = copy.deepcopy(original.out_drop)
        self.rotary_emb = copy.deepcopy(getattr(original, "rotary_emb", None))
        self.gate_proj = nn.Linear(self.hidden_size, 2 * self.num_heads)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        with torch.no_grad():
            self.gate_proj.bias[self.num_heads:].fill_(-1.5)
        self.q_log_scale = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.k_log_scale = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.full((self.num_heads,), -4.0))
        self.mix_logits = nn.Parameter(torch.zeros(self.num_heads, 2))
        self.output_gain = nn.Parameter(torch.zeros(self.num_heads))

    def trainable_core_parameters(self):
        frozen = {id(p) for m in (self.Wqkv, self.Wo) for p in m.parameters()}
        return [p for p in self.parameters() if id(p) not in frozen]

    def _scan(self, q, k, v, g, beta):
        if self.backend == "reference":
            return reference_delta(q, k, v, g, beta)
        if not q.is_cuda:
            raise RuntimeError("FLA requires CUDA; use backend='reference' only for correctness tests.")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        out, _ = chunk_gated_delta_rule(
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
            g=g.contiguous(), beta=beta.contiguous(),
            output_final_state=False, use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
            state_v_first=True,
        )
        return out

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                position_ids=None, **kwargs):
        b, t, d = hidden_states.shape
        valid = _valid_tokens(attention_mask, hidden_states)
        qkv = self.Wqkv(hidden_states).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if position_embeddings is None and self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(t, device=hidden_states.device)[None, :]
            position_embeddings = self.rotary_emb(qkv, position_ids=position_ids)
        if position_embeddings is not None:
            from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2),
                                       *position_embeddings)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        q = (q * self.q_log_scale.clamp(-1, 1).exp()).to(qkv.dtype)
        k = (k * self.k_log_scale.clamp(-1, 1).exp()).to(qkv.dtype)
        raw_g, raw_beta = self.gate_proj(hidden_states).chunk(2, -1)
        g = -(self.A_log.float().clamp(-4, 4).exp() *
              F.softplus(raw_g.float() + self.dt_bias)).clamp_max(1.0)
        beta = raw_beta.sigmoid()
        m = valid[:, :, None, None]
        q, k, v = (x * m for x in (q, k, v))
        g, beta = g * valid[:, :, None], beta * valid[:, :, None]

        # Window starts are fixed in absolute token coordinates. Appending padding
        # cannot alter the context of an existing valid query.
        w, s = self.window, self.stride
        left = s
        n = math.ceil(t / s) + 1
        padded = (n - 1) * s + w
        right = padded - left - t
        def windows(x):
            tail = x.shape[2:]
            z = x.reshape(b, t, -1).transpose(1, 2)
            z = F.pad(z, (left, right)).unfold(-1, w, s)
            return z.permute(0, 2, 3, 1).reshape(b*n, w, *tail).contiguous()
        qw, kw, vw = (windows(x) for x in (q, k, v))
        gw, bw = (windows(x).to(qw.dtype) for x in (g, beta))
        forward = self._scan(qw, kw, vw, gw, bw)
        backward = self._scan(*(x.flip(1).contiguous() for x in (qw, kw, vw, gw, bw))).flip(1)
        mix = self.mix_logits.softmax(-1)
        y = forward.float()*mix[:, 0][None, None, :, None]
        y = y + backward.float()*mix[:, 1][None, None, :, None]
        y = y * self.output_gain.clamp(-2, 2).exp()[None, None, :, None]
        cols = y.reshape(b, n, w, d).permute(0, 3, 2, 1).reshape(b, d*w, n)
        merged = F.fold(cols, (1, padded), (1, w), stride=(1, s))
        # With 50% overlap and both boundary windows, every real token appears twice.
        merged = merged[:, :, 0, left:left+t].transpose(1, 2) * 0.5
        y = self.out_drop(self.Wo(merged.to(hidden_states.dtype)))
        return y * valid[:, :, None], None


class MultiscaleCeNN(nn.Module):
    """Leaky recurrent cellular templates at fixed scales plus global input.

    Global masked pooling broadcasts whole-sequence context; fixed coarse bins
    preserve padding invariance. No QK attention or token-by-token matrix.
    """
    def __init__(self, original, cell_dim=256, steps=2, scales=(1, 4, 16)):
        super().__init__()
        if steps < 1 or cell_dim < 1 or not scales or any(s < 1 for s in scales):
            raise ValueError("Invalid CeNN dimensions, recurrence count, or scales")
        self.config = original.config
        self.cell_dim, self.steps = int(cell_dim), int(steps)
        self.scales = tuple(int(s) for s in scales)
        d = self.config.hidden_size
        self.down = nn.Linear(d, cell_dim, bias=False)
        self.up = nn.Linear(cell_dim, d, bias=False)
        # Transfer a subspace of pretrained V/Wo; both projections remain trainable.
        with torch.no_grad():
            take = min(cell_dim, d)
            self.down.weight[:take].copy_(original.Wqkv.weight[2*d:2*d+take])
            self.up.weight[:, :take].copy_(original.Wo.weight[:, :take])
        self.norm = nn.LayerNorm(cell_dim)
        self.template = nn.Parameter(torch.zeros(cell_dim, 1, 3))
        nn.init.normal_(self.template, std=0.05)
        self.input_gain = nn.Parameter(torch.ones(cell_dim))
        self.leak_logit = nn.Parameter(torch.zeros(cell_dim))
        self.bias = nn.Parameter(torch.zeros(cell_dim))
        self.global_proj = nn.Linear(cell_dim, cell_dim, bias=False)
        nn.init.eye_(self.global_proj.weight)
        self.global_gain = nn.Parameter(torch.full((cell_dim,), 0.1))
        self.mix_logits = nn.Parameter(torch.zeros(len(scales)))
        self.output_gain = nn.Parameter(torch.zeros(()))
        self.out_drop = copy.deepcopy(original.out_drop)

    def trainable_core_parameters(self):
        return list(self.parameters())

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        valid = _valid_tokens(attention_mask, hidden_states)
        m = valid[:, :, None].to(hidden_states.dtype)
        u = self.norm(self.down(hidden_states)) * m
        global_input = self.global_proj(u.sum(1) / m.sum(1).clamp_min(1))
        global_input = global_input[:, :, None] * self.global_gain[None, :, None]
        template = self.template.tanh()
        template = 0.9 * template / template.abs().sum(-1, keepdim=True).clamp_min(1.0)
        eta = self.leak_logit.sigmoid()[None, :, None]
        branches = []
        b, t, c = u.shape
        for scale in self.scales:
            extra = (-t) % scale
            upad = F.pad(u.transpose(1, 2), (0, extra))
            mpad = F.pad(m.transpose(1, 2), (0, extra))
            count = mpad.reshape(b, 1, -1, scale).sum(-1)
            pooled = upad.reshape(b, c, -1, scale).sum(-1) / count.clamp_min(1)
            cell_mask = (count > 0).to(u.dtype)
            state = pooled.tanh() * cell_mask
            drive = pooled * self.input_gain[None, :, None] + global_input + self.bias[None, :, None]
            for _ in range(self.steps):
                feedback = F.conv1d(state, template.to(state.dtype), padding=1, groups=c)
                state = ((1-eta)*state + eta*torch.tanh(feedback + drive)) * cell_mask
            branches.append(state.repeat_interleave(scale, -1)[:, :, :t].transpose(1, 2))
        mix = self.mix_logits.softmax(0)
        z = sum(weight * branch for weight, branch in zip(mix, branches)) * m
        out = self.up(z.to(hidden_states.dtype)) * self.output_gain.clamp(-2, 2).exp()
        return self.out_drop(out) * m, None


def replacement_for(original, kind, config):
    if kind == "full_attention":
        return MultiscaleCeNN(original, config["cell_dim"], config["cell_steps"], config["scales"])
    if kind == "sliding_attention":
        return WindowedBiDelta(original, config["window"], config["backend"])
    raise ValueError(kind)


def core_parameters(model):
    seen, out = set(), []
    for module in model.modules():
        if isinstance(module, (WindowedBiDelta, MultiscaleCeNN)):
            for p in module.trainable_core_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
    return out


class LinearMaskEncoder(nn.Module):
    """Preserve all pretrained FFNs/norms/residuals without making T x T masks."""
    def __init__(self, encoder):
        super().__init__()
        if not all(isinstance(l.attn, (WindowedBiDelta, MultiscaleCeNN)) for l in encoder.layers):
            raise ValueError("Convert every encoder attention before wrapping")
        self.config = encoder.config
        self.embeddings, self.layers, self.final_norm = encoder.embeddings, encoder.layers, encoder.final_norm

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        h = self.embeddings(input_ids=input_ids)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        for layer in self.layers:
            mixed = layer.attn(layer.attn_norm(h), attention_mask=attention_mask, position_ids=pos)[0]
            h = h + mixed
            h = h + layer.mlp(layer.mlp_norm(h))
        return SimpleNamespace(last_hidden_state=self.final_norm(h) * attention_mask[:, :, None])


class PooledMarkerDecisionModel(nn.Module):
    """Shared option MLP conditioned on CLS, whole-sequence and option means."""
    def __init__(self, source_model, compact_dim=384, layers=2, copy_encoder=True):
        super().__init__()
        self.encoder = copy.deepcopy(source_model.encoder) if copy_encoder else source_model.encoder
        self.type_emb = copy.deepcopy(source_model.type_emb)
        d = self.encoder.config.hidden_size
        self.compact_dim, self.head_layers = int(compact_dim), int(layers)
        n_act = source_model.act_head[-1].out_features
        self.in_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, compact_dim, bias=False))
        self.context_proj = nn.Linear(3*compact_dim, compact_dim)
        self.marker_mixer = nn.Sequential(*[
            nn.Sequential(nn.LayerNorm(compact_dim), nn.Linear(compact_dim, compact_dim), nn.GELU())
            for _ in range(layers)
        ])
        self.scorer = nn.Sequential(nn.LayerNorm(compact_dim), nn.Linear(compact_dim, compact_dim),
                                    nn.GELU(), nn.Linear(compact_dim, 1))
        self.act_head = nn.Sequential(nn.Linear(compact_dim+4, 128), nn.GELU(), nn.Linear(128, n_act))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        idx = marker_pos[:, :, None].expand(-1, -1, h.shape[-1])
        options = self.in_proj(h.gather(1, idx))
        mask = marker_mask[:, :, None]
        pooled = (h * attention_mask[:, :, None]).sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
        context = self.context_proj(torch.cat([
            self.in_proj(h[:, 0]), self.in_proj(pooled),
            (options * mask).sum(1) / mask.sum(1).clamp_min(1),
        ], -1))
        z = options + context[:, None, :]
        for block in self.marker_mixer:
            z = z + block(z)
        logits = self.scorer(z).squeeze(-1).float().masked_fill(~marker_mask.bool(), -1e4)
        p = logits.detach().softmax(-1)
        k = marker_mask.sum(-1).clamp_min(2).float()
        top2 = p.topk(2, dim=-1).values
        entropy = -(p*p.clamp_min(1e-9).log()).sum(-1) / k.log()
        features = torch.stack([top2[:, 0], top2[:, 0]-top2[:, 1], entropy, k/255], -1)
        return logits, self.act_head(torch.cat([context.float(), features], -1))


def attention_audit(model):
    encoder = getattr(model, "encoder", None)
    bad_layers = [] if encoder is None else [i for i, l in enumerate(encoder.layers)
        if not isinstance(l.attn, (WindowedBiDelta, MultiscaleCeNN))]
    head_attention = [n for n, m in model.named_modules() if isinstance(
        m, (nn.MultiheadAttention, nn.TransformerEncoder, nn.TransformerEncoderLayer))]
    return {"passed": isinstance(model, PooledMarkerDecisionModel)
            and isinstance(encoder, LinearMaskEncoder) and not bad_layers and not head_attention,
            "unconverted_encoder_layers": bad_layers, "transformer_modules": head_attention}


def export_gate(teacher, student, audit, max_accuracy_drop=0.02, min_speedup=1.10):
    """Fail closed on missing/NaN metrics. Require paired CI as well as point accuracy."""
    if not 0 <= max_accuracy_drop <= 1 or min_speedup <= 0:
        raise ValueError("Invalid export gate thresholds")
    reasons = []
    try:
        tg, sg = teacher["gold_test"], student["gold_test"]
        accuracy_gap = float(sg["accuracy"]) - float(tg["accuracy"])
        ci_low = float(student["quality_vs_teacher"]["accuracy_gap_95pct_case_bootstrap"][0])
        speedup = float(student["speedup_vs_teacher"])
        values = [accuracy_gap, ci_low, speedup, float(sg["kl_from_gold"]), float(sg["brier"])]
        if not all(math.isfinite(v) for v in values): reasons.append("non-finite metrics")
        if sg["cases"] != tg["cases"] or sg["decisions"] != tg["decisions"] or sg["cases"] <= 0:
            reasons.append("invalid paired evaluation counts")
        if accuracy_gap < -max_accuracy_drop: reasons.append("accuracy loss exceeds tolerance")
        if ci_low < -max_accuracy_drop: reasons.append("paired confidence interval exceeds tolerance")
        if speedup < min_speedup: reasons.append("speedup below threshold")
        if not audit.get("passed", False): reasons.append("attention-free architecture audit failed")
    except (KeyError, TypeError, ValueError, IndexError):
        reasons.append("missing or invalid metrics")
    return {"passed": not reasons, "reasons": reasons,
            "max_accuracy_drop": max_accuracy_drop, "min_speedup": min_speedup}


def save_v25(model, directory, tokenizer, architecture_config, source_config, report):
    """Save a reloadable local checkpoint; Hub upload is a separate gated action."""
    from safetensors.torch import save_file
    import shutil
    import inspect
    from . import standalone_decision
    if not attention_audit(model)["passed"]:
        raise ValueError("Only the final attention-free pooled-head model can be exported")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"format_version": 1, "architecture": architecture_config,
                "encoder_config": model.encoder.config.to_dict(), "source_config": source_config,
                "compact_dim": model.compact_dim, "head_layers": model.head_layers,
                "n_act": model.act_head[-1].out_features}
    (root / "v25_config.json").write_text(json.dumps(metadata, indent=2))
    (root / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    save_file({n: p.detach().cpu().contiguous().clone() for n, p in model.state_dict().items()},
              str(root / "model.safetensors"))
    tokenizer.save_pretrained(root / "tokenizer")
    package = root / "tinycenn_lm"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    shutil.copy2(__file__, package / "decision_v25.py")
    shutil.copy2(inspect.getfile(standalone_decision), package / "standalone_decision.py")
    (root / "requirements.txt").write_text(
        "torch>=2.6\ntransformers==4.57.6\nsafetensors>=0.4\nhuggingface_hub>=0.25\nflash-linear-attention[cuda]==0.5.2\n")
    return root


def load_v25(directory, device="cpu", backend=None):
    """Load locally, without downloading the teacher or executing Hub remote code."""
    from transformers import AutoModel, ModernBertConfig
    from safetensors.torch import load_file
    from .standalone_decision import StandaloneDecisionModel
    root = Path(directory)
    meta = json.loads((root / "v25_config.json").read_text())
    if meta["format_version"] != 1:
        raise ValueError("Unsupported V2.5 checkpoint format")
    cfg = ModernBertConfig.from_dict(meta["encoder_config"])
    _apply_rope_compat(cfg)
    cfg.reference_compile = False
    enc = AutoModel.from_config(cfg, attn_implementation="sdpa")
    arch = dict(meta["architecture"])
    arch["backend"] = backend or ("fla" if str(device).startswith("cuda") else "reference")
    for layer, kind in zip(enc.layers, layer_kinds(enc)):
        layer.attn = replacement_for(layer.attn, kind, arch)
    source = StandaloneDecisionModel(LinearMaskEncoder(enc), head_layers=0, n_act=meta["n_act"])
    model = PooledMarkerDecisionModel(source, meta["compact_dim"], meta["head_layers"], copy_encoder=False)
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    return model.to(device).eval()
