"""Run the Milestone 4 comparison harness.

Runs every eval task from configs/prototype.yaml twice:

- baseline arm: one-shot, no tool feedback (Milestone 4 arm A)
- recurrent arm: multi-step loop with tool observations (arm C)

and prints per-task results. With the mock backend the results are
deterministic and CPU-free; point ``runtime.backend`` at a real serving
endpoint to measure real behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_runtime.controller import CognitiveController  # noqa: E402
from harness_adapter import ToolGateway  # noqa: E402
from model_adapter import get_backend  # noqa: E402


def load_config() -> dict:
    with open(ROOT / "configs" / "prototype.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate() -> list[dict]:
    cfg = load_config()
    results = []
    for task in cfg["eval_tasks"]:
        row: dict[str, dict] = {}
        for arm in ("baseline", "recurrent"):
            arm_cfg = {**cfg, "mode": arm}
            gateway = ToolGateway(arm_cfg)
            backend = get_backend(cfg["backend"], arm_cfg)
            controller = CognitiveController(
                backend, gateway, arm_cfg,
                trace_dir=ROOT / "data" / "traces")
            outcome = controller.run(task["goal"])
            row[arm] = {
                "final_text": outcome.final_text.strip(),
                "stop_reason": outcome.stop_reason,
                "steps": outcome.steps,
                "tool_calls": outcome.tool_calls,
                "correct": outcome.final_text.strip() == task["expected"],
            }
        row["goal"] = task["goal"]
        row["expected"] = task["expected"]
        results.append(row)
    return results


def main() -> None:
    results = evaluate()
    for r in results:
        print(f"goal: {r['goal']}")
        for arm in ("baseline", "recurrent"):
            a = r[arm]
            mark = "OK " if a["correct"] else "FAIL"
            print(f"  {mark} {arm:9s} steps={a['steps']} tool_calls={a['tool_calls']} "
                  f"answer={a['final_text']!r} reason={a['stop_reason']}")
    n = len(results)
    for arm in ("baseline", "recurrent"):
        score = sum(1 for r in results if r[arm]["correct"])
        print(f"{arm}: {score}/{n} correct")


if __name__ == "__main__":
    main()
