#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from tinycenn_lm.smollm2_amcenn import DEFAULT_SMOLLM2


def int_list(value: str) -> list[int]:
    vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


def str_list(value: str) -> list[str]:
    vals = [x.strip() for x in value.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated strings")
    return vals


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate learned constant-state CeNN attention kernels")
    p.add_argument("--base-model", default=DEFAULT_SMOLLM2)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--train-sequences", type=int, default=6)
    p.add_argument("--eval-sequences", type=int, default=2)
    p.add_argument("--layers", type=int_list, default=[18])
    p.add_argument(
        "--variants",
        type=str_list,
        default=["learned_softplus", "learned_softmax", "norm_softplus", "norm_taylor2", "taylor2_learned"],
    )
    p.add_argument("--feature-dims", type=int_list, default=[256, 512])
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--output-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.5)
    p.add_argument("--partition-weight", type=float, default=0.25)
    p.add_argument("--kernel-weight", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dataset-seed", type=int, default=9107)
    p.add_argument("--gradient-check", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--output-dir", default="result/cenn-learned-kernels")
    return p.parse_args()


def choose_dtype(device: torch.device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def token_blocks(dataset, tokenizer, context_length):
    buf, eos = [], tokenizer.eos_token_id
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= context_length:
            yield torch.tensor(buf[:context_length], dtype=torch.long)
            del buf[:context_length]


def exact_attention(q: Tensor, k: Tensor, v: Tensor, groups: int):
    kh = k.repeat_interleave(groups, dim=1)
    vh = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhtd,bhsd->bhts", q, kh) / math.sqrt(q.shape[-1])
    t = q.shape[-2]
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril().view(1, 1, t, t)
    masked = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(masked, dim=-1)
    out = torch.einsum("bhts,bhsd->bhtd", weights, vh)
    logz = torch.logsumexp(masked, dim=-1)
    return out, weights, logz, scores, causal


@dataclass
class AttentionSample:
    q: Tensor
    k: Tensor
    v: Tensor
    exact_out: Tensor
    exact_weights: Tensor
    exact_logz: Tensor
    exact_scores: Tensor

    def to(self, device: torch.device) -> "AttentionSample":
        return AttentionSample(*[x.to(device) for x in self.__dict__.values()])


class LearnedPositiveKernel(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, feature_dim: int, mode: str):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.mode = mode

        self.wq = nn.Parameter(torch.empty(num_heads, feature_dim, head_dim))
        self.wk = nn.Parameter(torch.empty(num_kv_heads, feature_dim, head_dim))
        self.bq = nn.Parameter(torch.zeros(num_heads, feature_dim))
        self.bk = nn.Parameter(torch.zeros(num_kv_heads, feature_dim))
        nn.init.normal_(self.wq, std=0.02)
        nn.init.normal_(self.wk, std=0.02)

        self.log_q_scale = nn.Parameter(torch.full((num_heads,), -0.5 * math.log(float(feature_dim))))
        self.log_k_scale = nn.Parameter(torch.full((num_kv_heads,), -0.5 * math.log(float(feature_dim))))
        if mode == "softmax":
            with torch.no_grad():
                self.log_q_scale.fill_(0.5 * math.log(float(feature_dim)))
                self.log_k_scale.fill_(0.5 * math.log(float(feature_dim)))
        if mode == "norm_softplus":
            self.log_tau_q = nn.Parameter(torch.full((num_heads,), math.log(4.0)))
            self.log_tau_k = nn.Parameter(torch.full((num_kv_heads,), math.log(4.0)))
        else:
            self.register_parameter("log_tau_q", None)
            self.register_parameter("log_tau_k", None)

    def _project(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum("bhtd,hfd->bhtf", x, w) + b.view(1, b.shape[0], 1, b.shape[1])

    def features(self, q: Tensor, k: Tensor):
        if self.mode == "norm_softplus":
            q = F.normalize(q, dim=-1) * self.log_tau_q.exp().clamp(0.2, 20.0).view(1, -1, 1, 1)
            k = F.normalize(k, dim=-1) * self.log_tau_k.exp().clamp(0.2, 20.0).view(1, -1, 1, 1)
        ql = torch.einsum("bhtd,hfd->bhtf", q, self.wq) + self.bq.view(1, self.num_heads, 1, self.feature_dim)
        kl = torch.einsum("bhtd,hfd->bhtf", k, self.wk) + self.bk.view(1, self.num_kv_heads, 1, self.feature_dim)
        if self.mode == "softmax":
            qphi = torch.softmax(ql, dim=-1)
            kphi = torch.softmax(kl, dim=-1)
        else:
            qphi = F.softplus(ql) + 1e-6
            kphi = F.softplus(kl) + 1e-6
        qphi = qphi * self.log_q_scale.clamp(-10, 10).exp().view(1, -1, 1, 1)
        kphi = kphi * self.log_k_scale.clamp(-10, 10).exp().view(1, -1, 1, 1)
        return qphi, kphi

    def kernel(self, q: Tensor, k: Tensor, groups: int) -> Tensor:
        qphi, kphi = self.features(q, k)
        kh = kphi.repeat_interleave(groups, dim=1)
        return torch.einsum("bhtf,bhsf->bhts", qphi, kh).clamp_min(1e-12)

    @property
    def effective_feature_dim(self):
        return self.feature_dim


class NormalizedTaylor2Kernel(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.log_tau_q = nn.Parameter(torch.full((num_heads,), math.log(4.0)))
        self.log_tau_k = nn.Parameter(torch.full((num_kv_heads,), math.log(4.0)))
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def kernel(self, q: Tensor, k: Tensor, groups: int) -> Tensor:
        qn = F.normalize(q, dim=-1) * self.log_tau_q.exp().clamp(0.2, 30.0).view(1, -1, 1, 1)
        kn = F.normalize(k, dim=-1) * self.log_tau_k.exp().clamp(0.2, 30.0).view(1, -1, 1, 1)
        kh = kn.repeat_interleave(groups, dim=1)
        s = torch.einsum("bhtd,bhsd->bhts", qn, kh) / math.sqrt(self.head_dim)
        return (1.0 + s + 0.5 * s.square()).clamp_min(1e-8)

    @property
    def effective_feature_dim(self):
        d = self.head_dim
        return 1 + d + d * (d + 1) // 2


class Taylor2PlusLearnedKernel(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, feature_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.learned = LearnedPositiveKernel(num_heads, num_kv_heads, head_dim, feature_dim, "softplus")
        self.log_taylor_scale = nn.Parameter(torch.zeros(()))
        self.raw_alpha = nn.Parameter(torch.tensor(-3.0))

    def kernel(self, q: Tensor, k: Tensor, groups: int) -> Tensor:
        kh = k.repeat_interleave(groups, dim=1)
        s = torch.einsum("bhtd,bhsd->bhts", q, kh) / math.sqrt(self.head_dim)
        taylor = 1.0 + s + 0.5 * s.square()
        learned = self.learned.kernel(q, k, groups)
        alpha = F.softplus(self.raw_alpha)
        return (self.log_taylor_scale.clamp(-5, 5).exp() * taylor + alpha * learned).clamp_min(1e-8)

    @property
    def effective_feature_dim(self):
        d = self.head_dim
        return 1 + d + d * (d + 1) // 2 + self.learned.feature_dim


def build_variant(name: str, num_heads: int, num_kv_heads: int, head_dim: int, feature_dim: int) -> nn.Module:
    if name == "learned_softplus":
        return LearnedPositiveKernel(num_heads, num_kv_heads, head_dim, feature_dim, "softplus")
    if name == "learned_softmax":
        return LearnedPositiveKernel(num_heads, num_kv_heads, head_dim, feature_dim, "softmax")
    if name == "norm_softplus":
        return LearnedPositiveKernel(num_heads, num_kv_heads, head_dim, feature_dim, "norm_softplus")
    if name == "norm_taylor2":
        return NormalizedTaylor2Kernel(num_heads, num_kv_heads, head_dim)
    if name == "taylor2_learned":
        return Taylor2PlusLearnedKernel(num_heads, num_kv_heads, head_dim, feature_dim)
    raise ValueError(f"unknown variant: {name}")


def candidate_from_kernel(kernel: Tensor, v: Tensor, groups: int):
    vh = v.repeat_interleave(groups, dim=1)
    t = kernel.shape[-1]
    causal = torch.ones(t, t, dtype=torch.bool, device=kernel.device).tril().view(1, 1, t, t)
    masked = kernel.masked_fill(~causal, 0.0)
    den = masked.sum(dim=-1).clamp_min(1e-12)
    weights = masked / den.unsqueeze(-1)
    out = torch.einsum("bhts,bhsd->bhtd", weights, vh)
    return out, weights, den.log(), causal


def nmse(ref: Tensor, app: Tensor):
    return (app.sub(ref).square().mean() / ref.square().mean().clamp_min(1e-12))


def cosine(ref: Tensor, app: Tensor):
    return F.cosine_similarity(ref.reshape(-1, ref.shape[-1]), app.reshape(-1, app.shape[-1]), dim=-1).mean()


def attention_kl(ref: Tensor, app: Tensor):
    eps = 1e-9
    p, q = ref.clamp_min(eps), app.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


def topk_overlap(ref: Tensor, app: Tensor, top_k: int):
    b, h, t, _ = ref.shape
    vals = []
    for pos in torch.linspace(max(1, t // 4), t - 1, steps=min(10, t), device=ref.device).round().long().unique().tolist():
        kk = min(top_k, pos + 1)
        r = torch.topk(ref[:, :, pos, : pos + 1], kk, dim=-1).indices
        a = torch.topk(app[:, :, pos, : pos + 1], kk, dim=-1).indices
        for bi in range(b):
            for hi in range(h):
                vals.append(len(set(r[bi, hi].tolist()) & set(a[bi, hi].tolist())) / kk)
    return statistics.fmean(vals) if vals else 0.0


def training_loss(model: nn.Module, sample: AttentionSample, groups: int, args):
    ker = model.kernel(sample.q, sample.k, groups)
    out, weights, logz, causal = candidate_from_kernel(ker, sample.v, groups)
    l_out = nmse(sample.exact_out, out)
    l_kl = attention_kl(sample.exact_weights, weights)
    l_z = F.smooth_l1_loss(logz, sample.exact_logz)
    mask = causal.expand_as(ker)
    target_scores = sample.exact_scores.masked_select(mask)
    log_kernel = ker.clamp_min(1e-12).log().masked_select(mask)
    l_kernel = F.smooth_l1_loss(log_kernel, target_scores)
    total = (
        args.output_weight * l_out
        + args.kl_weight * l_kl
        + args.partition_weight * l_z
        + args.kernel_weight * l_kernel
    )
    return total, {"out": l_out, "kl": l_kl, "logz": l_z, "kernel": l_kernel}


def tensor_grad_metrics(ref: Tensor, app: Tensor):
    return {
        "cosine": float(F.cosine_similarity(ref.reshape(1, -1), app.reshape(1, -1), dim=-1).item()),
        "nmse": float(nmse(ref, app).item()),
    }


def gradient_fidelity(model: nn.Module, sample: AttentionSample, groups: int):
    probe_gen = torch.Generator(device=sample.q.device).manual_seed(424242)
    probe = torch.randn(sample.exact_out.shape, generator=probe_gen, device=sample.q.device)

    q1 = sample.q.detach().clone().requires_grad_(True)
    k1 = sample.k.detach().clone().requires_grad_(True)
    v1 = sample.v.detach().clone().requires_grad_(True)
    exact, _, _, _, _ = exact_attention(q1, k1, v1, groups)
    loss1 = (exact * probe).mean()
    gq1, gk1, gv1 = torch.autograd.grad(loss1, (q1, k1, v1))

    q2 = sample.q.detach().clone().requires_grad_(True)
    k2 = sample.k.detach().clone().requires_grad_(True)
    v2 = sample.v.detach().clone().requires_grad_(True)
    ker = model.kernel(q2, k2, groups)
    app, _, _, _ = candidate_from_kernel(ker, v2, groups)
    loss2 = (app * probe).mean()
    gq2, gk2, gv2 = torch.autograd.grad(loss2, (q2, k2, v2))

    qmet, kmet, vmet = tensor_grad_metrics(gq1, gq2), tensor_grad_metrics(gk1, gk2), tensor_grad_metrics(gv1, gv2)
    return {
        "grad_q_cosine": qmet["cosine"],
        "grad_q_nmse": qmet["nmse"],
        "grad_k_cosine": kmet["cosine"],
        "grad_k_nmse": kmet["nmse"],
        "grad_v_cosine": vmet["cosine"],
        "grad_v_nmse": vmet["nmse"],
        "grad_mean_cosine": statistics.fmean([qmet["cosine"], kmet["cosine"], vmet["cosine"]]),
        "grad_mean_nmse": statistics.fmean([qmet["nmse"], kmet["nmse"], vmet["nmse"]]),
    }


def memory_stats(effective_f: int, context: int, kvh: int, d: int):
    kv = 2 * context * kvh * d
    cenn = kvh * effective_f * (d + 1)
    return {
        "state_vs_kv_ratio": cenn / max(kv, 1),
        "break_even_tokens": math.ceil(effective_f * (d + 1) / (2 * d)),
        "cenn_state_mib_fp32": cenn * 4 / (1024 ** 2),
    }


def collect_samples(model, tokenizer, blocks, layers: list[int], count: int, device: torch.device):
    result: dict[int, list[AttentionSample]] = {i: [] for i in layers}
    for seq in range(count):
        ids = next(blocks).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
        t = ids.shape[1]
        pos = torch.arange(t, device=device).unsqueeze(0)
        for idx in layers:
            layer = model.model.layers[idx]
            h0 = out.hidden_states[idx]
            x = layer.input_layernorm(h0)
            cos, sin = model.model.rotary_emb(x, pos)
            attn = layer.self_attn
            b = x.shape[0]
            q = attn.q_proj(x).view(b, t, model.config.num_attention_heads, -1).transpose(1, 2).float()
            k = attn.k_proj(x).view(b, t, model.config.num_key_value_heads, -1).transpose(1, 2).float()
            v = attn.v_proj(x).view(b, t, model.config.num_key_value_heads, -1).transpose(1, 2).float()
            q, k = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
            groups = model.config.num_attention_heads // model.config.num_key_value_heads
            exact_out, exact_w, exact_logz, scores, _ = exact_attention(q, k, v, groups)
            sample = AttentionSample(q, k, v, exact_out, exact_w, exact_logz, scores)
            result[idx].append(AttentionSample(*[z.detach().cpu() for z in sample.__dict__.values()]))
        print(f"captured sequence {seq + 1}/{count}")
    return result


def evaluate(model: nn.Module, samples: list[AttentionSample], groups: int, device: torch.device, top_k: int):
    vals = []
    runtime = []
    for cpu_sample in samples:
        s = cpu_sample.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            ker = model.kernel(s.q, s.k, groups)
            out, weights, logz, _ = candidate_from_kernel(ker, s.v, groups)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime.append((time.perf_counter() - started) * 1000)
        vals.append({
            "output_cosine": float(cosine(s.exact_out, out).item()),
            "output_nmse": float(nmse(s.exact_out, out).item()),
            "partition_log_mae": float((logz - s.exact_logz).abs().mean().item()),
            "attention_kl": float(attention_kl(s.exact_weights, weights).item()),
            "topk_overlap": topk_overlap(s.exact_weights, weights, top_k),
        })
    keys = vals[0].keys()
    out = {k: statistics.fmean(v[k] for v in vals) for k in keys}
    out["runtime_ms"] = statistics.fmean(runtime)
    return out


def train_one(model: nn.Module, train_samples: list[AttentionSample], groups: int, device: torch.device, args, seed: int):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(seed)
    last = {}
    for step in range(1, args.train_steps + 1):
        sample = train_samples[rng.randrange(len(train_samples))].to(device)
        opt.zero_grad(set_to_none=True)
        total, parts = training_loss(model, sample, groups, args)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last = {k: float(v.detach().cpu()) for k, v in parts.items()}
        if step == 1 or step % 20 == 0 or step == args.train_steps:
            print(
                f"step={step:4d}/{args.train_steps} total={float(total.detach().cpu()):.5f} "
                f"out={last['out']:.4f} kl={last['kl']:.4f} logz={last['logz']:.4f} "
                f"kernel={last['kernel']:.4f} grad={float(grad_norm):.3f}"
            )
    return last


def write_csv(path: Path, rows: list[dict]):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def selection_score(row: dict):
    c = max(0.0, min(1.0, row["output_cosine"]))
    n = math.exp(-max(0.0, row["output_nmse"]))
    k = math.exp(-max(0.0, row["attention_kl"]))
    z = math.exp(-0.25 * max(0.0, row["partition_log_mae"]))
    g = max(0.0, min(1.0, row.get("grad_mean_cosine", 0.0)))
    return 0.40 * c + 0.20 * n + 0.15 * k + 0.10 * z + 0.15 * g


def main():
    args = parse_args()
    allowed = {"learned_softplus", "learned_softmax", "norm_softplus", "norm_taylor2", "taylor2_learned"}
    unknown = set(args.variants) - allowed
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    print(f"device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(device).eval()
    model.config.use_cache = False
    n_layers = int(model.config.num_hidden_layers)
    if any(i < 0 or i >= n_layers for i in args.layers):
        raise ValueError(f"invalid layer list for {n_layers} layers: {args.layers}")

    raw = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=True).shuffle(
        seed=args.dataset_seed, buffer_size=2048
    )
    blocks = token_blocks(raw, tokenizer, args.context_length)
    total_sequences = args.train_sequences + args.eval_sequences
    samples = collect_samples(model, tokenizer, blocks, args.layers, total_sequences, device)

    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size) // num_heads
    groups = num_heads // num_kv_heads
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for layer in args.layers:
        train_samples = samples[layer][: args.train_sequences]
        eval_samples = samples[layer][args.train_sequences :]
        for variant in args.variants:
            dims = [0] if variant == "norm_taylor2" else args.feature_dims
            for fd in dims:
                print("=" * 100)
                print(f"layer={layer} variant={variant} requested_F={fd}")
                candidate = build_variant(variant, num_heads, num_kv_heads, head_dim, fd).to(device)
                started = time.time()
                train_last = train_one(candidate, train_samples, groups, device, args, args.seed + layer * 1009 + fd)
                metrics = evaluate(candidate, eval_samples, groups, device, args.top_k)
                row = {
                    "layer": layer,
                    "variant": variant,
                    "requested_feature_dim": fd,
                    "effective_feature_dim": int(candidate.effective_feature_dim),
                    "train_steps": args.train_steps,
                    "elapsed_seconds": time.time() - started,
                    **metrics,
                    **memory_stats(int(candidate.effective_feature_dim), args.context_length, num_kv_heads, head_dim),
                    **{f"last_train_{k}": v for k, v in train_last.items()},
                }
                if args.gradient_check:
                    row.update(gradient_fidelity(candidate, eval_samples[0].to(device), groups))
                row["selection_score"] = selection_score(row)
                rows.append(row)
                print(json.dumps(row, indent=2))

                if args.save_checkpoints:
                    ckpt = outdir / "checkpoints"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "variant": variant,
                            "layer": layer,
                            "requested_feature_dim": fd,
                            "effective_feature_dim": int(candidate.effective_feature_dim),
                            "state_dict": {k: v.detach().cpu() for k, v in candidate.state_dict().items()},
                        },
                        ckpt / f"layer{layer:02d}_{variant}_f{fd}.pt",
                    )
                del candidate
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    ranked = sorted(rows, key=lambda r: r["selection_score"], reverse=True)
    report = {
        "status": "PASS",
        "experiment": "cenn-learned-kernel-concepts",
        "base_model": args.base_model,
        "context_length": args.context_length,
        "layers": args.layers,
        "variants": args.variants,
        "feature_dims": args.feature_dims,
        "train_sequences": args.train_sequences,
        "eval_sequences": args.eval_sequences,
        "train_steps": args.train_steps,
        "gradient_checked": bool(args.gradient_check),
        "rows": rows,
        "ranking": ranked,
        "best": ranked[0] if ranked else None,
        "success_targets": {
            "promising": {"output_cosine": 0.80, "output_nmse": 0.30, "attention_kl": 0.80, "grad_mean_cosine": 0.70},
            "strong": {"output_cosine": 0.90, "output_nmse": 0.15, "attention_kl": 0.40, "grad_mean_cosine": 0.80},
            "near_equivalence": {"output_cosine": 0.97, "output_nmse": 0.05},
        },
        "note": "Selection score is heuristic. Inspect forward, partition, attention-distribution, gradient, state-size and runtime metrics separately.",
    }
    (outdir / "learned_kernel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(outdir / "learned_kernel_summary.csv", rows)
    print("=" * 100)
    print("Best candidate:")
    print(json.dumps(report["best"], indent=2))
    print(f"Saved report to {outdir}")


if __name__ == "__main__":
    main()
