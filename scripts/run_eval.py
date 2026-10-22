"""Run the Milestone 4 comparison harness across multiple evaluation arms.

Compares:
- Arm A (baseline): one-shot, no tool feedback
- Arm B (reactive): reactive single-turn tool caller without cognitive state
- Arm C (recurrent): OpenSI multi-step cognitive loop with bounded state,
  hypothesis tracking, and error recovery

Evaluates tasks defined in configs/prototype.yaml, computes success rate,
step counts, tool usage efficiency, and recovery performance.
Outputs formatted console tables and an optional JSON benchmark report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_runtime.controller import CognitiveController  # noqa: E402
from harness_adapter import ToolGateway  # noqa: E402
from model_adapter import get_backend  # noqa: E402


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or (ROOT / "configs" / "prototype.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate(config: dict | None = None, arms: tuple[str, ...] = ("baseline", "reactive", "recurrent")) -> dict:
    cfg = config or load_config()
    tasks = cfg.get("eval_tasks", [])
    results: list[dict] = []
    arm_metrics: dict[str, dict] = {
        arm: {"correct": 0, "total_steps": 0, "total_tool_calls": 0, "failures": 0}
        for arm in arms
    }

    t0 = time.time()

    for idx, task in enumerate(tasks, start=1):
        row: dict[str, Any] = {
            "task_id": idx,
            "goal": task["goal"],
            "expected": task["expected"],
            "arms": {},
        }

        for arm in arms:
            arm_cfg = {**cfg, "mode": arm}
            gateway = ToolGateway(arm_cfg)
            backend = get_backend(cfg["backend"], arm_cfg)
            controller = CognitiveController(
                backend,
                gateway,
                arm_cfg,
                trace_dir=ROOT / "data" / "traces",
            )
            outcome = controller.run(task["goal"])
            ans = outcome.final_text.strip()
            is_correct = (ans.lower() == str(task["expected"]).strip().lower())

            arm_res = {
                "final_text": ans,
                "stop_reason": outcome.stop_reason,
                "steps": outcome.steps,
                "tool_calls": outcome.tool_calls,
                "correct": is_correct,
            }
            row["arms"][arm] = arm_res

            # Accumulate metrics
            if is_correct:
                arm_metrics[arm]["correct"] += 1
            else:
                arm_metrics[arm]["failures"] += 1
            arm_metrics[arm]["total_steps"] += outcome.steps
            arm_metrics[arm]["total_tool_calls"] += outcome.tool_calls

        results.append(row)

    total_time = time.time() - t0
    n_tasks = len(tasks)

    # Compute averages
    summary = {}
    for arm in arms:
        corr = arm_metrics[arm]["correct"]
        pct = (corr / n_tasks * 100) if n_tasks > 0 else 0.0
        avg_steps = (arm_metrics[arm]["total_steps"] / n_tasks) if n_tasks > 0 else 0.0
        avg_tools = (arm_metrics[arm]["total_tool_calls"] / n_tasks) if n_tasks > 0 else 0.0
        summary[arm] = {
            "correct": corr,
            "total": n_tasks,
            "accuracy_pct": round(pct, 1),
            "avg_steps": round(avg_steps, 2),
            "avg_tool_calls": round(avg_tools, 2),
            "total_tool_calls": arm_metrics[arm]["total_tool_calls"],
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": cfg.get("backend", "mock"),
        "total_tasks": n_tasks,
        "elapsed_seconds": round(total_time, 3),
        "summary": summary,
        "tasks": results,
    }


def print_report(eval_output: dict) -> None:
    print("=" * 80)
    print(f"OpenSI Milestone 4 Evaluation Benchmark (Backend: {eval_output['backend']})")
    print(f"Tasks: {eval_output['total_tasks']} | Runtime: {eval_output['elapsed_seconds']}s")
    print("=" * 80)

    for item in eval_output["tasks"]:
        print(f"\n[Task {item['task_id']}] {item['goal']}")
        print(f"  Expected: {item['expected']}")
        for arm, res in item["arms"].items():
            mark = "PASS" if res["correct"] else "FAIL"
            print(f"    [{mark:4s}] {arm:9s} | steps={res['steps']} | tools={res['tool_calls']} | ans={res['final_text']!r}")

    print("\n" + "-" * 80)
    print("SUMMARY COMPARISON MATRIX:")
    print(f"{'Arm':12s} | {'Accuracy':10s} | {'Score':8s} | {'Avg Steps':10s} | {'Avg Tools':10s}")
    print("-" * 80)
    for arm, stats in eval_output["summary"].items():
        score_str = f"{stats['correct']}/{stats['total']}"
        acc_str = f"{stats['accuracy_pct']}%"
        print(f"{arm:12s} | {acc_str:10s} | {score_str:8s} | {stats['avg_steps']:<10.2f} | {stats['avg_tool_calls']:<10.2f}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSI Milestone 4 Evaluation Runner")
    parser.add_argument("--json", type=str, default=None, help="Path to write JSON results report")
    parser.add_argument("--config", type=str, default=None, help="Custom config yaml path")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    output = evaluate(cfg)
    print_report(output)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nWrote benchmark results JSON to {out_path}")


if __name__ == "__main__":
    main()
