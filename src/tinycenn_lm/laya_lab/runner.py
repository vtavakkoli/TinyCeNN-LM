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
    train_items = build_training_items(
        teacher, train_ds, settings["train_cases"], settings["train_max_len"], cfg.seed
    )
    gate_cases = dataset_cases(test_ds, settings["gate_cases"])
    final_cases = dataset_cases(test_ds, settings["final_cases"])
    teacher_gate = evaluate_agent(teacher, gate_cases, label="teacher")
    print("Teacher gate accuracy:", round(teacher_gate["accuracy"], 4))

    candidates = choose_candidate_layers(student.model, settings["max_candidates"])
    if cfg.architecture == "pdelta3_gdn2_clvr":
        # PDelta3 is a global bidirectional recurrence. Full-attention layers
        # are the closer structural match, so try them before sliding attention.
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
        replacement, local = train_one_replacement(
            teacher, idx, cfg, train_items, train_steps, settings["batch_size"]
        )
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
        "accepted_layers": replaced_layers(student.model),
        "replacement_trainable_parameters": replacement_parameters(student.model),
        "training_steps_per_candidate": train_steps,
        "history": history,
        "teacher_final": teacher_final,
        "student_final": student_final,
        "latency": latency,
        "demo": demo,
        "note": (
            "PDelta3 is a bidirectional encoder adaptation; these Laya results are "
            "not interchangeable with causal-LM results."
        ),
    }
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
