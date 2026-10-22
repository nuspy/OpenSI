"""Probe t4: Tool observation in-context effectiveness measurement.

Compares answer accuracy on configs/prototype.yaml tasks under two conditions:
1. Without tool summary (baseline one-shot in-context)
2. With tool summary in-context (recurrent observation folded in)

Operates against the live OpenAI-compatible endpoint if reachable,
or falls back to the deterministic mock backend when the GPU/server is offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_runtime.controller import SCHEMA_PROMPT, parse_decision
from harness_adapter import ToolGateway
from model_adapter import get_backend


def load_config() -> dict[str, Any]:
    with open(ROOT / "configs" / "prototype.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def is_server_reachable(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def run_evaluation(backend, tasks: list[dict], gateway: ToolGateway) -> dict[str, Any]:
    results = []
    baseline_correct = 0
    recurrent_correct = 0

    for task in tasks:
        goal = task["goal"]
        expected = str(task["expected"]).strip()
        tool_name = task.get("tool")
        tool_args = task.get("tool_args", {})

        # Execute tool to get real observation summary
        tool_res = gateway.execute(tool_name, tool_args)
        tool_summary = tool_res.summary

        # Condition 1: Baseline (no tool summary in payload)
        msg_baseline = [
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": json.dumps({"goal": goal})},
        ]
        out_base_raw = backend.generate(msg_baseline)
        try:
            dec_base = parse_decision(out_base_raw)
            ans_base = str(dec_base.get("answer", dec_base.get("rationale", ""))).strip()
        except Exception:
            ans_base = str(out_base_raw).strip()

        ok_base = (expected.lower() in ans_base.lower()) or (ans_base == expected)
        if ok_base:
            baseline_correct += 1

        # Condition 2: Recurrent with tool summary in payload
        msg_recurrent = [
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": json.dumps({
                "goal": goal,
                "tool_history": [{"tool": tool_name, "summary": tool_summary}],
            })},
        ]
        out_rec_raw = backend.generate(msg_recurrent)
        try:
            dec_rec = parse_decision(out_rec_raw)
            ans_rec = str(dec_rec.get("answer", dec_rec.get("rationale", ""))).strip()
        except Exception:
            ans_rec = str(out_rec_raw).strip()

        ok_rec = (expected.lower() in ans_rec.lower()) or (ans_rec == expected)
        if ok_rec:
            recurrent_correct += 1

        results.append({
            "goal": goal,
            "expected": expected,
            "tool_used": tool_name,
            "tool_summary": tool_summary,
            "baseline": {"answer": ans_base[:100], "correct": ok_base},
            "recurrent": {"answer": ans_rec[:100], "correct": ok_rec},
        })

    n = len(tasks)
    base_acc = round(baseline_correct / n, 3) if n > 0 else 0.0
    rec_acc = round(recurrent_correct / n, 3) if n > 0 else 0.0

    return {
        "tasks_evaluated": n,
        "baseline_accuracy": base_acc,
        "recurrent_accuracy": rec_acc,
        "accuracy_delta": round(rec_acc - base_acc, 3),
        "task_breakdown": results,
    }


def main() -> None:
    cfg = load_config()
    base = cfg.get("openai_base_url", "http://127.0.0.1:1234/v1")
    live_up = is_server_reachable(base)

    backend_name = "openai_compatible" if live_up else "mock"
    backend = get_backend(backend_name, cfg)
    gateway = ToolGateway(cfg)

    eval_tasks = cfg.get("eval_tasks", [])
    eval_report = run_evaluation(backend, eval_tasks, gateway)

    report = {
        "backend": backend_name,
        "server_reachable": live_up,
        "note": (
            "Evaluated with live server." if live_up
            else "Server offline/GPU busy. Evaluated with deterministic mock backend on CPU."
        ),
        "results": eval_report,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
