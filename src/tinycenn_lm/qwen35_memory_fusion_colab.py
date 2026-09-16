from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def progress_info(output_dir: Path, target_layers: list[int]) -> dict:
    status = _json(output_dir / "sequential_run_status.json")
    progress = _json(output_dir / "sequential_progress.json")
    current = _json(output_dir / "sequential_in_progress.json")
    final = _json(output_dir / "sequential_training_report.json")
    accepted = [int(x) for x in progress.get("accepted_layers", status.get("accepted_layers", []))]
    targets = [int(x) for x in progress.get("target_layers", status.get("target_layers", target_layers))]
    reports = []
    for source in (progress, current, final):
        if source.get("layer_reports"):
            reports = source["layer_reports"]
    return {
        "accepted": accepted,
        "target": targets,
        "current": current.get("current_layer", status.get("current_layer")),
        "rounds": int(current.get("rounds_completed", status.get("rounds_completed", 0)) or 0),
        "status": status.get("status", final.get("status", "not_started")),
        "reports": reports,
    }


def dashboard_html(info: dict, live: dict | None = None) -> str:
    target = info["target"]
    accepted = info["accepted"]
    current = info.get("current")
    pct = 100.0 * len(accepted) / max(1, len(target))
    pills = []
    for layer in target:
        if layer in accepted:
            bg, fg, label = "#dcfce7", "#166534", f"✓ L{layer}"
        elif current is not None and int(current) == layer:
            bg, fg, label = "#fef3c7", "#92400e", f"▶ L{layer}"
        else:
            bg, fg, label = "#e5e7eb", "#374151", f"L{layer}"
        pills.append(
            f"<span style='padding:6px 10px;margin:3px;border-radius:9px;background:{bg};color:{fg};display:inline-block;font-weight:600'>{label}</span>"
        )
    live_text = ""
    if live:
        pieces = []
        if live.get("round") is not None:
            pieces.append(f"round <b>{live['round']}</b>")
        if live.get("step") is not None:
            pieces.append(f"step <b>{live['step']}/{live.get('max_step', '?')}</b>")
        if live.get("nmse") is not None:
            pieces.append(f"NMSE <b>{live['nmse']:.4f}</b>")
        if live.get("cosine") is not None:
            pieces.append(f"cos <b>{live['cosine']:.4f}</b>")
        if live.get("inc") is not None:
            pieces.append(f"ΔNLL inc <b>{live['inc']:+.5f}</b>")
        if live.get("cum") is not None:
            pieces.append(f"ΔNLL total <b>{live['cum']:+.5f}</b>")
        if pieces:
            live_text = "<p><b>Live:</b> " + " | ".join(pieces) + "</p>"
    return f"""
    <div style='border:1px solid #cbd5e1;border-radius:12px;padding:14px;margin:6px 0'>
      <h3 style='margin-top:0'>Qwen3.5 Memory Fusion progress</h3>
      <div>{''.join(pills)}</div>
      <p><b>{len(accepted)}/{len(target)}</b> full-attention anchors accepted ({pct:.1f}%)</p>
      <div style='height:15px;background:#e5e7eb;border-radius:8px;overflow:hidden'>
        <div style='height:15px;width:{pct:.1f}%;background:#22c55e'></div>
      </div>
      <p>Current anchor: <b>{current}</b> | saved rounds: <b>{info.get('rounds', 0)}</b> | status: <b>{info.get('status')}</b></p>
      {live_text}
    </div>
    """


def show_progress(output_dir: Path, target_layers: list[int]):
    import pandas as pd
    from IPython.display import HTML, display

    info = progress_info(output_dir, target_layers)
    display(HTML(dashboard_html(info)))
    if info["reports"]:
        df = pd.DataFrame(info["reports"])
        cols = [c for c in [
            "layer", "round", "accepted", "steps", "nmse", "cosine", "probe_nll",
            "incremental_delta_nll", "cumulative_delta_nll",
        ] if c in df.columns]
        display(df[cols].tail(24))
        last = info["reports"][-1]
        checks = pd.DataFrame([
            ["NMSE", last.get("nmse"), "≤ 0.20", last.get("nmse", 9) <= 0.20],
            ["Cosine", last.get("cosine"), "≥ 0.90", last.get("cosine", -9) >= 0.90],
            ["Incremental ΔNLL", last.get("incremental_delta_nll"), "≤ +0.015", last.get("incremental_delta_nll", 9) <= 0.015],
            ["Cumulative ΔNLL", last.get("cumulative_delta_nll"), "≤ +0.050", last.get("cumulative_delta_nll", 9) <= 0.05],
        ], columns=["metric", "latest", "required", "pass"])
        display(checks)
    return info


def run_live_training(*, cmd: list[str], repo: Path, output_dir: Path, target_layers: list[int], max_step: int, env: dict | None = None) -> int:
    from IPython.display import HTML, display

    log_path = output_dir / "last_colab_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    info = progress_info(output_dir, target_layers)
    live = {"layer": info.get("current"), "round": None, "step": None, "max_step": max_step, "nmse": None, "cosine": None, "inc": None, "cum": None}
    handle = display(HTML(dashboard_html(info, live)), display_id=True)

    rx_layer = re.compile(r"QWEN3\.5 FULL-ATTENTION REPLACEMENT: layer (\d+)")
    rx_round = re.compile(r"--- layer (\d+) round (\d+) ---")
    rx_step = re.compile(r"layer=(\d+)\s+step=(\d+)/(\d+)")
    rx_check = re.compile(r"ACCEPTANCE CHECK layer=(\d+):\s+NMSE=([0-9.]+).*?cos=([0-9.]+).*?ΔNLL_inc=([+-]?[0-9.]+).*?ΔNLL_total=([+-]?[0-9.]+)")

    last_ui = 0.0
    print("▶ TRAIN / RESUME")
    print("Targets:", target_layers)
    print("Command:", " ".join(cmd))
    print("Live log:", log_path)

    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(cmd, cwd=repo, env=env or os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                m = rx_layer.search(line)
                if m:
                    live.update(layer=int(m.group(1)), round=None, step=None)
                m = rx_round.search(line)
                if m:
                    live.update(layer=int(m.group(1)), round=int(m.group(2)), step=0)
                m = rx_step.search(line)
                if m:
                    live.update(layer=int(m.group(1)), step=int(m.group(2)), max_step=int(m.group(3)))
                m = rx_check.search(line)
                if m:
                    live.update(layer=int(m.group(1)), nmse=float(m.group(2)), cosine=float(m.group(3)), inc=float(m.group(4)), cum=float(m.group(5)))
                now = time.time()
                if now - last_ui > 1.0 or "ACCEPTANCE CHECK" in line or "accepted Qwen3.5" in line:
                    current = progress_info(output_dir, target_layers)
                    if live.get("layer") is not None:
                        current["current"] = live["layer"]
                    handle.update(HTML(dashboard_html(current, live)))
                    last_ui = now
            return_code = proc.wait()
    return return_code


DEFAULT_USER_CHATS = [
    {"name": "factual", "messages": [{"role": "system", "content": "You are helpful and concise."}, {"role": "user", "content": "What is the capital of Austria? Answer in one sentence."}]},
    {"name": "explanation", "messages": [{"role": "system", "content": "Explain clearly for non-experts."}, {"role": "user", "content": "Why does the sky look blue? Explain it to a 10-year-old in two sentences."}]},
    {"name": "coding", "messages": [{"role": "system", "content": "You are a careful Python assistant."}, {"role": "user", "content": "Write a small Python function is_prime(n) and explain the key idea briefly."}]},
    {"name": "reasoning", "messages": [{"role": "system", "content": "Show the calculation briefly, then answer."}, {"role": "user", "content": "A train travels 120 km in 1.5 hours. What is its average speed in km/h?"}]},
    {"name": "instruction_following", "messages": [{"role": "system", "content": "Follow the requested format exactly."}, {"role": "user", "content": "Give exactly three benefits of unit tests. Use three short bullet points and nothing else."}]},
    {"name": "multi_turn", "messages": [{"role": "system", "content": "You are a practical software architecture assistant."}, {"role": "user", "content": "I am building a small Python REST API."}, {"role": "assistant", "content": "What would you like to improve?"}, {"role": "user", "content": "Give me three concrete ways to make it more reliable in production."}]},
]
