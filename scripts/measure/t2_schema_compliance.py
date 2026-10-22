"""Probe t2: Schema compliance measurement.

Measures the fraction of outputs that conform to the controller's JSON
decision schema under SCHEMA_PROMPT across 20 synthetic states.
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

from cognitive_runtime.controller import SCHEMA_PROMPT, parse_decision, ParseError
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


SYNTHETIC_GOALS = [
    "Compute 15 + 27.",
    "Calculate the square of 19.",
    "Read config from data/eval_corpus/notes.txt.",
    "Check the current server time using clock.",
    "Evaluate expression 100 / 4.",
    "What is the capital of Italy?",
    "Summarize document data/eval_corpus/notes.txt.",
    "Find files matching *.txt in sandbox.",
    "Multiply 13 by 7.",
    "Wait for external event before continuing.",
    "Perform sanity check on system state.",
    "Calculate 2^8.",
    "Report the code word in notes.txt.",
    "Compute remainder of 155 % 7.",
    "Ask user for clarification on ambiguous parameter.",
    "Verify integrity of local dataset.",
    "Compute 45 - 19.",
    "Calculate (12 + 8) * 5.",
    "Retrieve system clock timestamp.",
    "Emit final conclusion based on observations.",
]


def test_compliance(backend, goals: list[str]) -> dict[str, Any]:
    results = []
    valid_count = 0
    action_counts: dict[str, int] = {}

    for i, goal in enumerate(goals):
        payload = {"goal": goal}
        messages = [
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            raw_text = backend.generate(messages)
            decision = parse_decision(raw_text)
            action = decision.get("action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1
            valid_count += 1
            results.append({
                "index": i,
                "goal": goal,
                "valid": True,
                "action": action,
                "output_excerpt": str(raw_text)[:120],
            })
        except (ParseError, Exception) as exc:
            results.append({
                "index": i,
                "goal": goal,
                "valid": False,
                "error": str(exc),
            })

    total = len(goals)
    compliance_rate = round(valid_count / total, 3) if total > 0 else 0.0

    return {
        "total_probed": total,
        "valid_decisions": valid_count,
        "parse_failures": total - valid_count,
        "compliance_rate": compliance_rate,
        "action_distribution": action_counts,
        "samples": results[:5],  # preview first 5
    }


def main() -> None:
    cfg = load_config()
    base = cfg.get("openai_base_url", "http://127.0.0.1:1234/v1")
    live_up = is_server_reachable(base)

    if live_up:
        backend_name = "openai_compatible"
        backend = get_backend(backend_name, cfg)
        note = "Evaluated against live OpenAI-compatible endpoint."
    else:
        backend_name = "mock"
        backend = get_backend(backend_name, cfg)
        note = "Server offline/GPU busy. Evaluated against deterministic mock backend."

    metrics = test_compliance(backend, SYNTHETIC_GOALS)

    report = {
        "backend": backend_name,
        "server_reachable": live_up,
        "note": note,
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
