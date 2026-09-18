#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
from huggingface_hub import list_repo_files, snapshot_download

from tinycenn_lm.qwen35_flycore_v1 import (
    FlyFFNV3Config,
    FlyVocabConfig,
    assert_qwen35_flycore,
    factorize_embedding_weight,
    install_fly_vocab,
)
from tinycenn_lm.qwen35_flycore_standalone import (
    export_flycore_standalone,
    upload_flycore_standalone,
)
from tinycenn_lm.qwen35_flyffn_v3 import routing_schedule
from tinycenn_lm.qwen35_standalone import load_standalone

_HELPER_PATH = Path(__file__).with_name("run_qwen35_flycore_v1.py")
_spec = importlib.util.spec_from_file_location("flycore_base_runner", _HELPER_PATH)
helper = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(helper)
base = helper.base


DEFAULT_SOURCE_REPO = "vtava/Qwen35-0.8B-FlyFFN-v3-AllFFN"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Qwen3.5 FlyCore-v1 initialized from a trained FlyFFN-v3 standalone model"
        )
    )
    p.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO)
    p.add_argument("--source-dir", default="/content/qwen35_v3_source")
    p.add_argument("--run-mode", choices=["quick", "strong"], default="quick")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--vocab-latent-dim", type=int, default=384)
    p.add_argument("--vocab-graph-mix-init", type=float, default=0.10)
    p.add_argument("--factor-chunk-rows", type=int, default=4096)
    p.add_argument("--output-dir", default="results/flycore_from_v3_qwen35_08b")
    p.add_argument("--standalone-dir", default=None)
    p.add_argument("--upload-hf", action="store_true")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--seed", type=int, default=7521)
    return p.parse_args()


def download_source(repo_id: str, local_dir: str | Path) -> Path:
    local_dir = Path(local_dir)
    token = os.environ.get("HF_TOKEN") or None

    print(f"STAGE source: inspecting {repo_id}", flush=True)
    files = list_repo_files(repo_id, repo_type="model", token=token)
    print(f"Source repo files ({len(files)}):", flush=True)
    for name in files:
        print("  ", name, flush=True)

    required = {
        "standalone_state.pt",
        "flyffn_config.json",
        "config.json",
    }
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(
            "The source repository is not the verified FlyFFN-v3 standalone format. "
            f"Missing: {missing}. Found files: {files}"
        )

    patterns = [
        "standalone_state.pt",
        "standalone_manifest.json",
        "flyffn_config.json",
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "special_tokens_map.json",
        "chat_template*",
        "*.jinja",
    ]

    print("STAGE source: downloading verified v3 standalone from Hugging Face", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token,
    )
    path = Path(path)

    state_path = path / "standalone_state.pt"
    manifest_path = path / "standalone_manifest.json"
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("verified", False):
            raise RuntimeError("Source standalone_manifest.json is not verified")
        print(
            f"Source manifest verified=True | state_keys={manifest.get('state_keys')} | "
            f"flyffn_keys={manifest.get('flyffn_state_keys')}",
            flush=True,
        )
    else:
        print(
            "WARNING source has no standalone_manifest.json; checkpoint keys will be "
            "validated by the loader.",
            flush=True,
        )

    return path


def load_source_twice(source_dir: Path, device, dtype):
    print("STAGE source: reconstructing trained FlyFFN-v3 teacher", flush=True)
    teacher, tokenizer = load_standalone(
        source_dir,
        device=device,
        dtype=dtype,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("STAGE source: reconstructing second v3 copy for FlyCore conversion", flush=True)
    student, _ = load_standalone(
        source_dir,
        device=device,
        dtype=dtype,
    )
    student.eval()
    return teacher, student, tokenizer


def source_ffn_config(source_dir: Path) -> FlyFFNV3Config:
    data = json.loads((source_dir / "flyffn_config.json").read_text(encoding="utf-8"))
    return FlyFFNV3Config(
        fly_nodes=int(data["fly_nodes"]),
        router_rank=int(data["router_rank"]),
        num_shards=int(data["num_shards"]),
        graph_steps=int(data["graph_steps"]),
        graph_mix_init=float(data["graph_mix_init"]),
        router_noise_std=float(data.get("router_noise_std", 5e-3)),
    )


def main():
    args = parse_args()
    base.base.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.base.choose_dtype(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"DEVICE {device} | dtype={dtype} | "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )

    source_dir = download_source(args.source_repo_id, args.source_dir)
    teacher, student, tokenizer = load_source_twice(source_dir, device, dtype)

    assert len(teacher.model.layers) == 24
    source_schedule = routing_schedule(teacher)
    print("SOURCE V3 ROUTING", json.dumps(source_schedule, indent=2), flush=True)

    adjacency = teacher.flyffn_shared_graph.adjacency.detach().float()
    print(
        f"Using FlyWire adjacency directly from source checkpoint: "
        f"{tuple(adjacency.shape)}",
        flush=True,
    )

    ffn_cfg = source_ffn_config(source_dir)
    vocab_cfg = FlyVocabConfig(
        latent_dim=args.vocab_latent_dim,
        fly_nodes=int(adjacency.shape[0]),
        graph_steps=ffn_cfg.graph_steps,
        graph_mix_init=args.vocab_graph_mix_init,
    )

    print(
        f"STAGE factorizing SOURCE v3 embedding: vocab={teacher.config.vocab_size} "
        f"hidden={teacher.config.hidden_size} rank={args.vocab_latent_dim}",
        flush=True,
    )
    factorization = factorize_embedding_weight(
        teacher.model.embed_tokens.weight,
        args.vocab_latent_dim,
        args.factor_chunk_rows,
    )
    factor_stats = {
        k: v for k, v in factorization.items() if not isinstance(v, torch.Tensor)
    }
    print("VOCAB FACTORIZATION", json.dumps(factor_stats, indent=2), flush=True)

    print("STAGE replacing only embedding + LM head with shared Fly vocabulary core", flush=True)
    install_fly_vocab(
        student,
        vocab_cfg,
        adjacency,
        factorization=factorization,
    )
    assert_qwen35_flycore(student)

    # Verify that the pre-trained v3 routing survived the vocabulary conversion.
    student_schedule_before = routing_schedule(student)
    if student_schedule_before != source_schedule:
        raise RuntimeError("FlyFFN routing changed while installing FlyEmbedding/FlyLMHead")
    print("✓ 24/24 trained FlyFFN-v3 layers preserved exactly", flush=True)

    if args.run_mode == "quick":
        vocab_updates = 80
        joint_updates = 400
        grad_accum = 1
        eval_count = 10
        probe_count = 3
    else:
        vocab_updates = 400
        joint_updates = 3000
        grad_accum = 2
        eval_count = 20
        probe_count = 6

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STAGE preparing FineWeb-Edu calibration/train/eval blocks", flush=True)
    train_needed = max(
        vocab_updates * args.batch_size,
        joint_updates * grad_accum * args.batch_size,
    )
    train_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 10,
                train_needed,
                args.seq_len,
            )
        ),
        args.batch_size,
    )
    probe_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 777,
                probe_count * args.batch_size,
                args.seq_len,
            )
        ),
        args.batch_size,
    )
    eval_batches = base.base.make_batches(
        list(
            base.base.token_blocks(
                tokenizer,
                args.seed + 999,
                eval_count * args.batch_size,
                args.seq_len,
            )
        ),
        args.batch_size,
    )

    initial_probe = helper.probe_model(
        student,
        teacher,
        probe_batches,
        device,
        dtype,
    )
    print("INITIAL FLY VOCAB PROBE", json.dumps(initial_probe, indent=2), flush=True)

    print("STAGE FlyEmbedding + FlyLMHead calibration", flush=True)
    vocab_metrics, vocab_hist = helper.calibrate_vocab(
        student,
        teacher,
        train_batches,
        probe_batches,
        vocab_updates,
        device,
        dtype,
    )
    pd.DataFrame(vocab_hist).to_csv(
        out_dir / "vocab_training_history.csv",
        index=False,
    )
    print("VOCAB CALIBRATION RESULT", json.dumps(vocab_metrics, indent=2), flush=True)

    print("STAGE joint low-LR FlyCore distillation", flush=True)
    metrics, joint_hist = helper.global_train_flycore(
        "flycore-from-v3",
        student,
        teacher,
        train_batches,
        eval_batches,
        args,
        device,
        dtype,
        joint_updates,
        grad_accum,
    )
    pd.DataFrame(joint_hist).to_csv(
        out_dir / "joint_training_history.csv",
        index=False,
    )

    assert_qwen35_flycore(student)
    final_schedule = routing_schedule(student)
    if set(final_schedule) != set(source_schedule):
        raise RuntimeError("FlyFFN layer set changed unexpectedly")

    test_ids = eval_batches[0][:, :-1].to(device)
    print("STAGE benchmarking source FlyFFN-v3 vs FlyCore", flush=True)
    source_bench = base.base.benchmark(teacher, test_ids, device)
    flycore_bench = base.base.benchmark(student, test_ids, device)

    print("STAGE generating FlyCore chat samples", flush=True)
    chat_samples = helper.generate_chat_samples(student, tokenizer, device)
    (out_dir / "chat_samples.json").write_text(
        json.dumps(chat_samples, indent=2),
        encoding="utf-8",
    )

    vocab_stats = student.fly_vocab_core.parameter_stats()
    report = {
        "architecture": (
            "FlyCore-v1 initialized from trained Qwen3.5 FlyFFN-v3 AllFFN standalone"
        ),
        "source_repo_id": args.source_repo_id,
        "source_is_trained_flyffn_v3": True,
        "source_graph_download_required": False,
        "source_flywire_adjacency_from_checkpoint": True,
        "source_routing_schedule": source_schedule,
        "final_routing_schedule": final_schedule,
        "token_mixers_unchanged": True,
        "fly_embedding": True,
        "fly_lm_head": True,
        "shared_fly_vocab_core": True,
        "flyffn_layers": 24,
        "dense_anchor_layers": [],
        "factorization": factor_stats,
        "initial_fly_vocab_probe": initial_probe,
        "vocab_calibration": vocab_metrics,
        "vocab_parameter_stats": vocab_stats,
        "source_v3": {
            "ce": metrics["teacher_ce"],
            "perplexity": metrics["teacher_perplexity"],
            **source_bench,
        },
        "flycore": {
            "ce": metrics["ce"],
            "perplexity": metrics["perplexity"],
            "teacher_kl": metrics["teacher_kl"],
            **flycore_bench,
        },
        "ce_gap_vs_source_v3": metrics["ce"] - metrics["teacher_ce"],
        "ppl_ratio_vs_source_v3": (
            metrics["perplexity"] / metrics["teacher_perplexity"]
        ),
        "parameter_ratio_flycore_over_source_v3": (
            sum(p.numel() for p in student.parameters())
            / sum(p.numel() for p in teacher.parameters())
        ),
        "decode_speed_ratio_flycore_over_source_v3": (
            flycore_bench["decode_tokens_s"] / source_bench["decode_tokens_s"]
        ),
        "config": {
            "run_mode": args.run_mode,
            "seq_len": args.seq_len,
            "vocab_latent_dim": args.vocab_latent_dim,
            "vocab_graph_mix_init": args.vocab_graph_mix_init,
            "vocab_calibration_updates": vocab_updates,
            "joint_train_updates": joint_updates,
            "grad_accum": grad_accum,
            "fly_nodes": ffn_cfg.fly_nodes,
            "router_rank": ffn_cfg.router_rank,
            "num_shards": ffn_cfg.num_shards,
            "graph_steps": ffn_cfg.graph_steps,
            "graph_mix_init": ffn_cfg.graph_mix_init,
        },
    }

    rows = {
        "Source FlyFFN-v3 AllFFN": report["source_v3"],
        "FlyCore-v1 from v3": report["flycore"],
    }
    pd.DataFrame(rows).T.to_csv(out_dir / "summary.csv")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    state = helper.state_for_flycore(student)
    torch.save(state, out_dir / "biological_qwen35_flycore_from_v3.pt")

    standalone_dir = (
        Path(args.standalone_dir)
        if args.standalone_dir
        else out_dir / "standalone"
    )
    print("STAGE exporting verified FlyCore standalone", flush=True)
    manifest = export_flycore_standalone(
        student,
        tokenizer,
        ffn_cfg,
        vocab_cfg,
        standalone_dir,
        metadata=report,
        artifacts_dir=out_dir,
    )
    print(
        f"STANDALONE verified=True state_keys={manifest['state_keys']} "
        f"flycore_keys={manifest['flycore_state_keys']} "
        f"size_gb={manifest['state_bytes']/1024**3:.3f}",
        flush=True,
    )

    hf_url = None
    if args.upload_hf:
        if not args.hf_repo_id:
            raise RuntimeError("--upload-hf requires --hf-repo-id")
        print(f"STAGE uploading FlyCore to {args.hf_repo_id}", flush=True)
        hf_url = upload_flycore_standalone(
            standalone_dir,
            args.hf_repo_id,
            private=args.hf_private,
        )
        print("HUGGINGFACE", hf_url, flush=True)

    print("\nFLYCORE FROM V3 SUMMARY", flush=True)
    print(pd.DataFrame(rows).T, flush=True)
    print("\nVOCAB PARAMETER STATS", flush=True)
    print(json.dumps(vocab_stats, indent=2), flush=True)
    print("\nKEY RESULTS", flush=True)
    print(
        json.dumps(
            {
                "ce_gap_vs_source_v3": report["ce_gap_vs_source_v3"],
                "ppl_ratio_vs_source_v3": report["ppl_ratio_vs_source_v3"],
                "parameter_ratio_flycore_over_source_v3": report[
                    "parameter_ratio_flycore_over_source_v3"
                ],
                "decode_speed_ratio_flycore_over_source_v3": report[
                    "decode_speed_ratio_flycore_over_source_v3"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    print("\nSaved:", out_dir, flush=True)
    print("Source v3:", source_dir, flush=True)
    print("Standalone:", standalone_dir, flush=True)
    if hf_url:
        print("Hugging Face:", hf_url, flush=True)


if __name__ == "__main__":
    main()
