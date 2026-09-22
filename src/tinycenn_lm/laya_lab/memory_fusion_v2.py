from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinycenn_lm.memory_attention import MemoryAugmentedCellularLayer
from .core import BaseLayaReplacementAttention, _valid_tokens
from .data import _batch_from_items, _load_typed_split, build_training_items, dataset_cases
from .evaluate import _accept, adapter_payload, benchmark_latency, evaluate_agent


class MemoryFusionV2Attention(BaseLayaReplacementAttention):
    """Bidirectional Laya adapter using the repo's successful MemoryFusion core."""

    architecture = "memory_fusion_v2"

    def __init__(
        self,
        original: nn.Module,
        feature_dim: int = 32,
        memory_rank: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    ):
        super().__init__(original)
        if getattr(original, "sliding_window", None) is not None:
            raise ValueError("MemoryFusionV2 currently supports full_attention only")
        self.feature_dim = int(feature_dim)
        self.memory_rank = int(memory_rank)
        self.dilations = tuple(int(x) for x in dilations)
        self.core = MemoryAugmentedCellularLayer(
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            feature_dim=self.feature_dim,
            variant="cellular_memory_fusion",
            dilations=self.dilations,
            shifted_window=8,
            memory_rank=self.memory_rank,
        )
        self.direction_logits = nn.Parameter(torch.zeros(self.num_heads, 2))
        for p in self.Wo.parameters():
            p.requires_grad = True

    def _core_masked(self, q: Tensor, k: Tensor, v: Tensor, valid: Tensor, reverse: bool):
        out = torch.zeros_like(v, dtype=torch.float32)
        for b in range(q.shape[0]):
            n = int(valid[b].sum().item())
            if n <= 0:
                continue
            qs, ks, vs = q[b:b+1, :, :n], k[b:b+1, :, :n], v[b:b+1, :, :n]
            if reverse:
                qs, ks, vs = qs.flip(2), ks.flip(2), vs.flip(2)
            y = self.core(qs.float(), ks.float(), vs.float())
            if reverse:
                y = y.flip(2)
            out[b:b+1, :, :n] = y
        return out

    def forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None, **kwargs):
        q, k, v = self.qkv(hidden_states, position_embeddings)
        valid = _valid_tokens(attention_mask, hidden_states)
        fwd = self._core_masked(q, k, v, valid, False)
        rev = self._core_masked(q, k, v, valid, True)
        w = self.direction_logits.float().softmax(dim=-1)
        out = (
            w[:, 0][None, :, None, None] * fwd
            + w[:, 1][None, :, None, None] * rev
        )
        out = out * valid[:, None, :, None].float()
        return self.finish(out, hidden_states)

    def core_parameters(self):
        return [*self.core.parameters(), self.direction_logits]

    def output_parameters(self):
        return list(self.Wo.parameters())

    def config_dict(self):
        d = super().config_dict()
        d.update(
            architecture=self.architecture,
            feature_dim=self.feature_dim,
            memory_rank=self.memory_rank,
            dilations=list(self.dilations),
            bidirectional=True,
            fusion_core="cellular_memory_fusion",
            fusion_prior=[2.0, -1.0, -1.0],
            train_output_projection=True,
            supported_attention_type="full_attention",
        )
        return d


@dataclass
class LayaMemoryFusionV2Config:
    model_id: str = "convaiinnovations/laya"
    seed: int = 2026
    output_dir: str = "/content/laya_tinycenn"
    candidate_layers: tuple[int, ...] = (12, 15)
    feature_dim: int = 32
    memory_rank: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    train_cases: int = 400
    train_max_len: int = 512
    batch_size: int = 2
    steps_per_round: int = 300
    max_rounds: int = 4
    check_every: int = 25
    min_steps_before_check: int = 50
    core_learning_rate: float = 2e-4
    output_learning_rate: float = 2e-5
    weight_decay: float = 1e-3
    cosine_weight: float = 0.25
    decision_kl_weight: float = 0.10
    action_kl_weight: float = 0.02
    distill_temperature: float = 1.0
    teacher_alpha_start: float = 0.90
    resume_alpha_start: float = 0.25
    teacher_alpha_end: float = 0.0
    max_local_nmse: float = 0.20
    min_local_cosine: float = 0.90
    min_teacher_agreement: float = 0.95
    max_mean_kl: float = 0.05
    max_accuracy_drop: float = 0.02
    gate_cases: int = 80
    final_cases: int = 160


def _tree_detach(x):
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_tree_detach(v) for v in x)
    if isinstance(x, list):
        return [_tree_detach(v) for v in x]
    return x


def _args(batch, device):
    return [batch[k].to(device) for k in (
        "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"
    )]


def _forward_capture(agent, layer_idx: int, args, grad: bool):
    capture: dict[str, Any] = {}
    module = agent.model.encoder.layers[layer_idx].attn

    def pre_hook(mod, hook_args, kwargs):
        capture["x"] = (hook_args[0] if hook_args else kwargs["hidden_states"]).detach()
        capture["pos"] = _tree_detach(kwargs.get("position_embeddings"))
        mask = kwargs.get("attention_mask")
        capture["mask"] = None if mask is None else mask.detach()

    h = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast(
            device_type=agent.device.type,
            dtype=agent.dtype,
            enabled=agent.device.type == "cuda",
        ):
            out = agent.model(*args)
    finally:
        h.remove()
    if "x" not in capture:
        raise RuntimeError(f"failed to capture layer {layer_idx} input")
    return capture, out


def _local_metrics(pred, target, valid2d):
    mask = valid2d[:, :, None].float()
    mse = (((pred.float() - target.float()) ** 2) * mask).sum()
    mse = mse / (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    denom = ((target.float() ** 2) * mask).sum()
    denom = denom / (mask.sum() * target.shape[-1]).clamp_min(1.0)
    nmse = mse / denom.clamp_min(1e-8)
    p, t = pred.float()[valid2d], target.float()[valid2d]
    cos = F.cosine_similarity(p, t, dim=-1).mean() if p.numel() else pred.new_tensor(0.0)
    return nmse, cos


def _kl(student, teacher, temperature: float, mask=None):
    t = max(float(temperature), 1e-3)
    s, q = student.float() / t, teacher.detach().float() / t
    if mask is not None:
        mask = mask.bool()
        s, q = s.masked_fill(~mask, -1e4), q.masked_fill(~mask, -1e4)
    return F.kl_div(
        F.log_softmax(s, -1), F.softmax(q, -1), reduction="none"
    ).sum(-1).mean() * (t * t)


def _alpha(step, total, start, end):
    return end if total <= 1 else start + (step - 1) / (total - 1) * (end - start)


def _stratified_cases(ds, gate_count, final_count, seed):
    grouped: dict[str, list] = {}
    for case in dataset_cases(ds, None):
        grouped.setdefault(str(case[3]), []).append(case)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)

    def take(n):
        keys = [k for k in sorted(grouped) if grouped[k]]
        out, cursor = [], 0
        while len(out) < n and keys:
            key = keys[cursor % len(keys)]
            out.append(grouped[key].pop())
            if not grouped[key]:
                keys = [k for k in keys if grouped[k]]
                cursor = 0
            else:
                cursor += 1
        return out

    return take(gate_count), take(final_count)


@torch.no_grad()
def _real_hidden_metrics(teacher, student, replacement, layer_idx, probe_batch):
    args = _args(probe_batch, teacher.device)
    tc, _ = _forward_capture(teacher, layer_idx, args, False)
    sc, _ = _forward_capture(student, layer_idx, args, False)
    x = sc["x"]
    with torch.autocast(
        device_type=teacher.device.type,
        dtype=teacher.dtype,
        enabled=teacher.device.type == "cuda",
    ):
        target, _ = teacher.model.encoder.layers[layer_idx].attn(
            x, position_embeddings=tc["pos"], attention_mask=tc["mask"]
        )
        pred, _ = replacement(
            x, position_embeddings=tc["pos"], attention_mask=tc["mask"]
        )
    nmse, cos = _local_metrics(
        pred, target, probe_batch["attention_mask"].to(teacher.device).bool()
    )
    return {"nmse": nmse.item(), "cosine": cos.item()}


def _train_round(
    teacher, student, replacement, layer_idx, cfg, train_items,
    probe_batch, gate_cases, teacher_gate, round_idx, offset_steps,
):
    device = teacher.device
    teacher.model.eval().requires_grad_(False)
    student.model.eval().requires_grad_(False)
    replacement.train()
    replacement.out_drop.eval()
    core, outp = replacement.core_parameters(), replacement.output_parameters()
    for p in core:
        p.requires_grad = True
        if p.is_floating_point():
            p.data = p.data.float()
    for p in outp:
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": core, "lr": cfg.core_learning_rate},
        {"params": outp, "lr": cfg.output_learning_rate},
    ], weight_decay=cfg.weight_decay)

    alpha_start = cfg.teacher_alpha_start if round_idx == 1 else cfg.resume_alpha_start
    best = {"nmse": float("inf"), "cosine": -1.0, "step": 0, "round": round_idx}
    last_gate = last_checks = last_drop = None

    for step in range(1, cfg.steps_per_round + 1):
        batch = _batch_from_items(
            train_items, (offset_steps + step - 1) * cfg.batch_size,
            cfg.batch_size, teacher.tok.pad_token_id
        )
        args = _args(batch, device)
        tc, tout = _forward_capture(teacher, layer_idx, args, False)
        sc, sout = _forward_capture(student, layer_idx, args, True)
        a = _alpha(step, cfg.steps_per_round, alpha_start, cfg.teacher_alpha_end)
        mixed = (a * tc["x"] + (1.0 - a) * sc["x"]).detach()

        use_amp = device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=teacher.dtype, enabled=use_amp
        ):
            target, _ = teacher.model.encoder.layers[layer_idx].attn(
                mixed, position_embeddings=tc["pos"], attention_mask=tc["mask"]
            )

        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=teacher.dtype, enabled=use_amp
        ):
            pred, _ = replacement(
                mixed, position_embeddings=tc["pos"], attention_mask=tc["mask"]
            )
            nmse, cos = _local_metrics(
                pred, target, batch["attention_mask"].to(device).bool()
            )
            functional = nmse + cfg.cosine_weight * (1.0 - cos)
            tlog, tact = tout
            slog, sact = sout
            dkl = _kl(
                slog, tlog, cfg.distill_temperature,
                batch["marker_mask"].to(device)
            )
            akl = _kl(sact, tact, cfg.distill_temperature)
            loss = functional + cfg.decision_kl_weight * dkl + cfg.action_kl_weight * akl

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss layer={layer_idx} round={round_idx} step={step}"
            )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(core + outp, 1.0)
        opt.step()

        if step == 1 or step % 10 == 0:
            print(
                f"layer={layer_idx:02d} round={round_idx} "
                f"step={step:03d}/{cfg.steps_per_round} alpha={a:.3f} "
                f"nmse={nmse.detach().item():.4f} cos={cos.detach().item():.4f} "
                f"dkl={dkl.detach().item():.4f} akl={akl.detach().item():.4f} "
                f"grad={float(grad):.3f}"
            )

        if step < cfg.min_steps_before_check or (
            step % cfg.check_every and step != cfg.steps_per_round
        ):
            continue

        replacement.eval()
        local = _real_hidden_metrics(
            teacher, student, replacement, layer_idx, probe_batch
        )
        local.update(step=step, round=round_idx)
        if local["nmse"] < best["nmse"]:
            best = dict(local)
        local_ok = (
            local["nmse"] <= cfg.max_local_nmse
            and local["cosine"] >= cfg.min_local_cosine
        )
        print(
            f"  LOCAL CHECK: NMSE={local['nmse']:.4f} "
            f"cos={local['cosine']:.4f} => "
            f"{'decision gate' if local_ok else 'continue'}"
        )
        if local_ok:
            last_gate = evaluate_agent(
                student, gate_cases, teacher_agent=teacher, label="student"
            )
            accepted, last_checks, last_drop = _accept(
                local, teacher_gate, last_gate, cfg
            )
            print(json.dumps({
                "accepted": accepted,
                "teacher_agreement": last_gate.get("teacher_agreement"),
                "mean_teacher_kl": last_gate.get("mean_teacher_kl"),
                "accuracy": last_gate.get("accuracy"),
                "accuracy_drop": last_drop,
                "checks": last_checks,
            }, indent=2))
            if accepted:
                replacement.eval()
                return {
                    "accepted": True, "round": round_idx, "steps": step,
                    "local": local, "gate": last_gate,
                    "checks": last_checks, "accuracy_drop": last_drop,
                }
        replacement.train()
        replacement.out_drop.eval()

    replacement.eval()
    return {
        "accepted": False, "round": round_idx, "steps": cfg.steps_per_round,
        "local": best, "gate": last_gate,
        "checks": last_checks, "accuracy_drop": last_drop,
    }


def _demo_and_latency(teacher, student):
    state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or leave?",
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    }
    return (
        {"teacher": teacher.predict(state, questions), "student": student.predict(state, questions)},
        {
            "teacher": benchmark_latency(teacher, state, questions),
            "student": benchmark_latency(student, state, questions),
        },
    )


def run_memory_fusion_v2(cfg: LayaMemoryFusionV2Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    out_dir = Path(cfg.output_dir) / "memory_fusion_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    import laya
    print("Loading Laya teacher:", cfg.model_id)
    teacher = laya.load(
        cfg.model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    teacher.model.eval().requires_grad_(False)
    try:
        teacher.model.encoder.config.reference_compile = False
    except Exception:
        pass
    student = copy.copy(teacher)
    student.model = copy.deepcopy(teacher.model).eval().requires_grad_(False)

    train_ds, test_ds = _load_typed_split("train"), _load_typed_split("test")
    items = build_training_items(
        teacher, train_ds, cfg.train_cases, cfg.train_max_len, cfg.seed
    )
    if len(items) < max(16, cfg.batch_size * 6):
        raise RuntimeError("too few Laya training items")
    probe_n = min(max(12, cfg.batch_size * 6), max(1, len(items) // 10))
    train_items, probe_items = items[:-probe_n], items[-probe_n:]
    probe_batch = _batch_from_items(
        probe_items, 0, min(len(probe_items), max(4, cfg.batch_size * 2)),
        teacher.tok.pad_token_id
    )

    gate_cases, final_cases = _stratified_cases(
        test_ds, cfg.gate_cases, cfg.final_cases, cfg.seed
    )
    teacher_gate = evaluate_agent(teacher, gate_cases, label="teacher")
    print("Teacher gate accuracy:", round(teacher_gate["accuracy"], 4))
    print("Gate workflows:", {k: v["n"] for k, v in teacher_gate["by_workflow"].items()})

    candidates = []
    for idx in map(int, cfg.candidate_layers):
        if not 0 <= idx < len(student.model.encoder.layers):
            continue
        typ = str(student.model.encoder.layers[idx].attention_type)
        if typ != "full_attention":
            print(f"Skipping layer {idx} ({typ}); V2 is full_attention-only.")
            continue
        candidates.append(idx)
    print("Candidates:", [(i, student.model.encoder.layers[i].attention_type) for i in candidates])

    history, offset_steps = [], 0
    for idx in candidates:
        print(f"
{'='*88}
MemoryFusionV2 layer {idx}
{'='*88}")
        layer = student.model.encoder.layers[idx]
        original = layer.attn
        replacement = MemoryFusionV2Attention(
            teacher.model.encoder.layers[idx].attn,
            cfg.feature_dim, cfg.memory_rank, cfg.dilations
        ).to(student.device)
        dtype = teacher.model.encoder.layers[idx].attn.Wqkv.weight.dtype
        replacement.Wqkv.to(device=student.device, dtype=dtype)
        replacement.Wo.to(device=student.device, dtype=dtype)
        layer.attn = replacement
        accepted = False

        for round_idx in range(1, cfg.max_rounds + 1):
            print(f"
--- layer {idx} round {round_idx}/{cfg.max_rounds} ---")
            result = _train_round(
                teacher, student, replacement, idx, cfg, train_items,
                probe_batch, gate_cases, teacher_gate, round_idx, offset_steps
            )
            offset_steps += cfg.steps_per_round
            rec = {"layer": idx, "attention_type": "full_attention", **result}
            history.append(rec)
            torch.save({
                "layer": idx,
                "round": round_idx,
                "accepted": result["accepted"],
                "config": replacement.config_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in replacement.state_dict().items()},
                "metrics": result,
            }, out_dir / f"layer_{idx}_round_{round_idx}.pt")
            if result["accepted"]:
                print(f"✅ accepted layer {idx}")
                accepted = True
                break
            print("Not accepted yet; retaining weights for the next round.")

        if not accepted:
            layer.attn = original
            print(f"❌ layer {idx} exhausted all rounds; original Laya attention restored.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    teacher_final = evaluate_agent(teacher, final_cases, label="teacher")
    student_final = evaluate_agent(
        student, final_cases, teacher_agent=teacher, label="memory_fusion_v2"
    )
    demo, latency = _demo_and_latency(teacher, student)
    accepted_layers = [
        i for i, layer in enumerate(student.model.encoder.layers)
        if isinstance(layer.attn, MemoryFusionV2Attention)
    ]
    trainable_params = sum(
        p.numel() for i in accepted_layers
        for p in student.model.encoder.layers[i].attn.parameters()
        if p.requires_grad
    )
    report = {
        "architecture": "memory_fusion_v2",
        "status": "ok" if accepted_layers else "failed_no_accepted_layers",
        "model_id": cfg.model_id,
        "config": asdict(cfg),
        "candidate_layers": candidates,
        "accepted_layers": accepted_layers,
        "replacement_trainable_parameters": trainable_params,
        "history": history,
        "teacher_gate": teacher_gate,
        "teacher_final": teacher_final,
        "student_final": student_final,
        "latency": latency,
        "demo": demo,
        "gate_final_disjoint": True,
        "notes": [
            "convaiinnovations/laya remains the untouched teacher.",
            "Uses the repository's successful Cellular+Hedgehog+GDN2 MemoryFusion core.",
            "The causal core is run forward and reverse with shared weights for bidirectional Laya.",
            "Wo is calibrated with a small LR while Wqkv remains frozen.",
            "Failed rounds retain weights; exhausted candidates are restored before the next layer.",
            "Local gates are measured on real current-student hidden states.",
        ],
    }
    torch.save(adapter_payload(student.model, cfg, report), out_dir / "adapter.pt")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("
Accepted layers:", accepted_layers)
    print("Final student metrics:")
    print(json.dumps(student_final, indent=2))
    print("Latency:", json.dumps(latency, indent=2))
    print("Saved:", out_dir / "adapter.pt")
    return teacher, student, report
