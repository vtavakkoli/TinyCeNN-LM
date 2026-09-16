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
    replaced = [int(x) for x in progress.get("replaced_layers", status.get("replaced_layers", []))]
    targets = [int(x) for x in progress.get("target_layers", status.get("target_layers", target_layers))]
    reports = []
    for source in (progress, current, final):
        if source.get("layer_reports"):
            reports = source["layer_reports"]
    strict = sorted({int(r["layer"]) for r in reports if r.get("selected") and r.get("selection") == "strict"})
    fallback = sorted({int(r["layer"]) for r in reports if r.get("selected") and r.get("selection") == "closest_fallback"})
    return {
        "replaced": replaced,
        "target": targets,
        "strict": strict,
        "fallback": fallback,
        "current": current.get("current_layer", status.get("current_layer")),
        "rounds": int(current.get("rounds_completed", status.get("rounds_completed", 0)) or 0),
        "status": status.get("status", final.get("status", "not_started")),
        "reports": reports,
    }


def dashboard_html(info: dict, live: dict | None = None) -> str:
    target = info["target"]
    replaced = info["replaced"]
    strict = set(info.get("strict", []))
    fallback = set(info.get("fallback", []))
    current = info.get("current")
    pct = 100.0 * len(replaced) / max(1, len(target))
    pills = []
    for layer in target:
        if layer in strict:
            bg, fg, label = "#dcfce7", "#166534", f"✓{layer}"
        elif layer in fallback or (layer in replaced and layer not in strict):
            bg, fg, label = "#dbeafe", "#1d4ed8", f"≈{layer}"
        elif current is not None and int(current) == layer:
            bg, fg, label = "#fef3c7", "#92400e", f"▶{layer}"
        else:
            bg, fg, label = "#e5e7eb", "#374151", str(layer)
        pills.append(
            f"<span style='padding:4px 7px;margin:2px;border-radius:7px;background:{bg};color:{fg};display:inline-block;font-size:12px;font-weight:600'>{label}</span>"
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
        if live.get("distance") is not None:
            pieces.append(f"distance <b>{live['distance']:.4f}</b>")
        if pieces:
            live_text = "<p><b>Live:</b> " + " | ".join(pieces) + "</p>"
    return f"""
    <div style='border:1px solid #cbd5e1;border-radius:12px;padding:14px;margin:6px 0'>
      <h3 style='margin-top:0'>Gemma 4 E2B — all-attention Memory Fusion progress</h3>
      <div>{''.join(pills)}</div>
      <p><b>{len(replaced)}/{len(target)}</b> attention layers replaced ({pct:.1f}%) —
         strict: <b>{len(strict)}</b>, closest fallback: <b>{len(fallback)}</b></p>
      <div style='height:15px;background:#e5e7eb;border-radius:8px;overflow:hidden'>
        <div style='height:15px;width:{pct:.1f}%;background:#22c55e'></div>
      </div>
      <p>Current layer: <b>{current}</b> | saved rounds: <b>{info.get('rounds', 0)}</b> | status: <b>{info.get('status')}</b></p>
      {live_text}
      <p style='font-size:12px'>✓ strict gate pass &nbsp; ≈ best/closest fallback after configured rounds</p>
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
            "layer", "layer_type", "round", "step", "strict_accepted", "selected", "selection",
            "nmse", "cosine", "probe_nll", "incremental_delta_nll", "cumulative_delta_nll", "closeness_score",
        ] if c in df.columns]
        display(df[cols].tail(40))
    return info


def run_live_training(*, cmd: list[str], repo: Path, output_dir: Path, target_layers: list[int], max_step: int, env: dict | None = None) -> int:
    from IPython.display import HTML, display

    log_path = output_dir / "last_colab_run.log"
    runtime_status = output_dir / "colab_runtime_status.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    info = progress_info(output_dir, target_layers)
    live = {"layer": info.get("current"), "round": None, "step": None, "max_step": max_step,
            "nmse": None, "cosine": None, "inc": None, "cum": None, "distance": None}
    handle = display(HTML(dashboard_html(info, live)), display_id=True)

    rx_layer = re.compile(r"GEMMA4 ALL-ATTENTION REPLACEMENT: layer (\d+)")
    rx_round = re.compile(r"--- layer (\d+) round (\d+)/(\d+) ---")
    rx_step = re.compile(r"layer=(\d+)\s+step=(\d+)/(\d+)")
    rx_check = re.compile(
        r"ACCEPTANCE CHECK layer=(\d+):\s+NMSE=([0-9.]+).*?cos=([0-9.]+).*?"
        r"ΔNLL_inc=([+-]?[0-9.]+).*?ΔNLL_total=([+-]?[0-9.]+).*?distance=([0-9.]+)"
    )

    print("▶ TRAIN / RESUME — Gemma 4 E2B all 35 text-attention layers")
    print("Command:", " ".join(cmd))
    print("Live log:", log_path)
    last_ui = 0.0
    rc = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(cmd, cwd=repo, env=env or os.environ.copy(), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line); log.flush()
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
                        live.update(layer=int(m.group(1)), nmse=float(m.group(2)), cosine=float(m.group(3)),
                                    inc=float(m.group(4)), cum=float(m.group(5)), distance=float(m.group(6)))
                    now = time.time()
                    if now - last_ui > 1.0 or "ACCEPTANCE CHECK" in line or "selected Gemma 4 layer" in line:
                        current = progress_info(output_dir, target_layers)
                        if live.get("layer") is not None:
                            current["current"] = live["layer"]
                        handle.update(HTML(dashboard_html(current, live)))
                        last_ui = now
                rc = proc.wait()
    except Exception as exc:
        runtime_status.write_text(json.dumps({"status": "trainer_failed", "error": str(exc)}, indent=2), encoding="utf-8")
        raise
    if rc:
        runtime_status.write_text(json.dumps({"status": "trainer_failed", "returncode": rc}, indent=2), encoding="utf-8")
    else:
        runtime_status.write_text(json.dumps({"status": "trainer_returned", "returncode": 0}, indent=2), encoding="utf-8")
    return int(rc or 0)


DEFAULT_COMPLETION_PROMPTS = [
    "Austria is a country in Central Europe. The capital of Austria is",
    "The largest planet in the Solar System is",
    "A reliable REST API should handle failures by",
    "In Python, a function that checks whether an integer is prime can be written as",
    "Artificial intelligence can help scientists by",
    "The main advantage of recurrent bounded-state memory over a full KV cache is",
]
