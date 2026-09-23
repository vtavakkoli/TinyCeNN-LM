from __future__ import annotations
import copy, json, random
from pathlib import Path
import numpy as np
import torch
from .core import ARCHITECTURES, LayaLabConfig
from .factory import choose_candidate_layers, replaced_layers, replacement_parameters
from .data import _load_typed_split, build_training_items, dataset_cases
from .train import train_one_replacement
from .evaluate import evaluate_agent, _accept, benchmark_latency, adapter_payload

def run_experiment(cfg: LayaLabConfig):
    if cfg.architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}")
    if cfg.architecture == "pdelta3_gdn2_clvr":
        # PDelta3 has a dedicated end-to-end distillation runner.  Keep the
        # generic runner for the other Laya attention experiments.
        from .pdelta_optimized import run_pdelta3_optimized
        return run_pdelta3_optimized(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    settings = cfg.mode_settings()
    out_dir = Path(cfg.output_dir) / cfg.architecture
    out_dir.mkdir(parents=True, exist_ok=True)

    import laya
    print("Loading Laya teacher:", cfg.model_id)
    teacher = laya.load(cfg.model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    teacher.model.eval().requires_grad_(False)
    try:
        teacher.model.encoder.config.reference_compile = False
    except Exception:
        pass

    student = copy.copy(teacher)
    student.model = copy.deepcopy(teacher.model).eval().requires_grad_(False)

    train_ds = _load_typed_split("train")
    test_ds = _load_typed_split("test")
    model_max_len = int(teacher.cfg.get("max_len", settings["train_max_len"]))
    train_max_len = min(int(settings["train_max_len"]), model_max_len)
    functional = cfg.architecture == "integrated_memory_v22" and cfg.integrated_functional_training
    if cfg.target_all_attention and cfg.architecture != "integrated_memory_v22":
        raise ValueError("target_all_attention is supported here only for integrated_memory_v22")
    if sum((bool(cfg.target_layers), cfg.target_all_full_attention, cfg.target_all_attention)) > 1:
        raise ValueError("choose one target policy")
    probe_items = None
    if functional:
        from .pdelta_optimized import _split_transfer_rows
        fit_rows, probe_rows = _split_transfer_rows(list(train_ds), cfg.seed)
        train_ds = fit_rows
        probe_items = build_training_items(teacher, probe_rows, len(probe_rows), train_max_len, cfg.seed)
    train_items = build_training_items(
        teacher, train_ds, settings["train_cases"], train_max_len, cfg.seed
    )
    print(
        "Attention-transfer sequences:",
        len(train_items),
        "| train max length:",
        train_max_len,
    )
    from .pdelta_optimized import _stratified_disjoint_cases
    gate_cases, final_cases = _stratified_disjoint_cases(
        test_ds, settings["gate_cases"], settings["final_cases"], cfg.seed)

    teacher_gate = evaluate_agent(teacher, gate_cases, label="teacher")
    print("Teacher gate accuracy:", round(teacher_gate["accuracy"], 4))

    if cfg.target_layers and cfg.target_all_full_attention:
        raise ValueError(
            "Use either target_layers or target_all_full_attention, not both"
        )

    if cfg.target_all_attention:
        candidates = list(range(len(student.model.encoder.layers)))
        # Convert in execution order so later modules can learn compensation
        # for the accepted upstream replacements during decision distillation.
    elif cfg.target_all_full_attention:
        if cfg.architecture != "integrated_memory_v22":
            raise ValueError(
                "target_all_full_attention is currently supported only for "
                "integrated_memory_v22"
            )
        full_layers = [
            i for i, layer in enumerate(student.model.encoder.layers)
            if str(layer.attention_type) == "full_attention"
        ]
        # Start with the layers that already showed the strongest transfer
        # behavior, then cover every remaining full-attention block.
        empirical_priority = [18, 21, 6, 12, 15, 9]
        candidates = [i for i in empirical_priority if i in full_layers]
        center = (len(student.model.encoder.layers) - 1) / 2.0
        candidates.extend(
            sorted(
                (i for i in full_layers if i not in candidates),
                key=lambda i: abs(i - center),
            )
        )
    elif cfg.target_layers:
        n_layers = len(student.model.encoder.layers)
        candidates = list(dict.fromkeys(int(i) for i in cfg.target_layers))
        invalid = [i for i in candidates if i < 0 or i >= n_layers]
        if invalid:
            raise ValueError(
                f"target_layers contains invalid ModernBERT layer indices: {invalid}"
            )
    elif cfg.architecture == "integrated_memory_v22":
        candidates = choose_candidate_layers(
            student.model,
            settings["max_candidates"],
            preferred_attention_type="full_attention",
        )
    else:
        candidates = choose_candidate_layers(
            student.model,
            settings["max_candidates"],
        )
    if cfg.architecture == "pdelta3_gdn2_clvr":
        # PDelta3 keeps its existing mixed shortlist but tries full-attention
        # candidates first.
        candidates.sort(key=lambda i: (
            0 if str(student.model.encoder.layers[i].attention_type) == "full_attention" else 1,
            abs(i - (len(student.model.encoder.layers) - 1) / 2.0),
        ))
    print("Candidate ModernBERT layers:", [
        (i, student.model.encoder.layers[i].attention_type) for i in candidates
    ])
    train_steps = cfg.training_steps if cfg.training_steps is not None else settings["steps"]
    history = []
    for idx in candidates:
        print(
            f"\n=== {cfg.architecture}: layer {idx} "
            f"({student.model.encoder.layers[idx].attention_type}) ==="
        )
        if functional:
            from .integrated_train import train_integrated_replacement
            replacement, local = train_integrated_replacement(
                teacher, student, idx, cfg, train_items, probe_items,
                train_steps, settings["batch_size"])
        else:
            replacement, local = train_one_replacement(
                teacher, idx, cfg, train_items, train_steps, settings["batch_size"])
        if cfg.architecture == "integrated_memory_v22":
            replacement.requires_grad_(False)
        old = student.model.encoder.layers[idx].attn
        student.model.encoder.layers[idx].attn = replacement.to(student.device)
        student.model.eval()
        gate = evaluate_agent(student, gate_cases, teacher_agent=teacher, label="student")
        accepted, checks, accuracy_drop = _accept(local, teacher_gate, gate, cfg)
        rec = {
            "layer": idx,
            "attention_type": str(student.model.encoder.layers[idx].attention_type),
            "local": local,
            "gate": gate,
            "checks": checks,
            "accuracy_drop": accuracy_drop,
            "accepted": accepted,
        }
        history.append(rec)
        print(json.dumps({
            "layer": idx,
            "accepted": accepted,
            "local": local,
            "agreement": gate.get("teacher_agreement"),
            "mean_kl": gate.get("mean_teacher_kl"),
            "accuracy": gate.get("accuracy"),
            "checks": checks,
        }, indent=2))
        if not accepted:
            student.model.encoder.layers[idx].attn = old
            del replacement
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    teacher_final = evaluate_agent(teacher, final_cases, label="teacher")
    student_final = evaluate_agent(
        student, final_cases, teacher_agent=teacher, label=cfg.architecture
    )

    demo_state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    }
    demo_questions = {
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
    demo = {
        "teacher": teacher.predict(demo_state, demo_questions),
        "student": student.predict(demo_state, demo_questions),
    }
    latency = {
        "teacher": benchmark_latency(teacher, demo_state, demo_questions),
        "student": benchmark_latency(student, demo_state, demo_questions),
    }
    report = {
        "architecture": cfg.architecture,
        "model_id": cfg.model_id,
        "mode": cfg.mode,
        "candidate_layers": candidates,
        "target_policy": (
            "all_attention"
            if cfg.target_all_attention
            else "all_full_attention"
            if cfg.target_all_full_attention
            else "explicit_layers"
            if cfg.target_layers
            else "auto_shortlist"
        ),
        "accepted_layers": replaced_layers(student.model),
        "replacement_trainable_parameters": (
            sum(p.numel() for i in replaced_layers(student.model)
                for name, p in student.model.encoder.layers[i].attn.named_parameters()
                if not name.startswith(("Wqkv.", "Wo.", "out_drop.")))
            if cfg.architecture == "integrated_memory_v22"
            else replacement_parameters(student.model)
        ),
        "training_steps_per_candidate": train_steps,
        "history": history,
        "teacher_final": teacher_final,
        "student_final": student_final,
        "latency": latency,
        "demo": demo,
        "note": (
            (
                "Laya uses a bidirectional ModernBERT encoder. Integrated Memory V2.2 "
                "supports full and sliding kernels with optional cumulative decision transfer; "
                "these results are not interchangeable with causal-LM conversions."
            )
            if cfg.architecture == "integrated_memory_v22"
            else (
                "PDelta3 is a bidirectional encoder adaptation; these Laya results are "
                "not interchangeable with causal-LM results."
            )
            if cfg.architecture == "pdelta3_gdn2_clvr"
            else (
                "MemoryFusion is evaluated as a bidirectional Laya encoder adaptation; "
                "these results are not interchangeable with causal-LM results."
            )
        ),
    }
    accepted = set(report["accepted_layers"])
    report["target_acceptance_rate"] = len(accepted.intersection(candidates)) / max(1, len(candidates))
    report["all_targets_replaced"] = bool(candidates) and set(candidates).issubset(accepted)
    report["remaining_attention_layers"] = [i for i in range(len(student.model.encoder.layers)) if i not in accepted]
    report["all_attention_replaced"] = not report["remaining_attention_layers"]
    report["final_quality_checks"] = {
        "teacher_agreement": student_final.get("teacher_agreement", 0) >= cfg.min_teacher_agreement,
        "teacher_kl": student_final.get("mean_teacher_kl", float("inf")) <= cfg.max_mean_kl,
        "accuracy_drop": teacher_final["accuracy"] - student_final["accuracy"] <= cfg.max_accuracy_drop,
    }
    report["final_quality_passed"] = all(report["final_quality_checks"].values())
    report["evaluation_split"] = "disjoint_stratified_gate_and_final"
    report["conversion_succeeded"] = bool(report["accepted_layers"])
    report["student_is_unmodified_teacher"] = not report["conversion_succeeded"]
    torch.save(adapter_payload(student.model, cfg, report), out_dir / "adapter.pt")
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nAccepted layers:", report["accepted_layers"])
    print("Final student metrics:")
    print(json.dumps(student_final, indent=2))
    print("Latency:", json.dumps(latency, indent=2))
    print("Saved:", out_dir / "adapter.pt")
    return teacher, student, report
