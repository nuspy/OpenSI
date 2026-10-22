"""OpenSI measurement driver (Milestone 1/4 calibration suite).

Pattern agreed with the user on 2026-10-15: every measurement task is
executed as a four-part cycle inside ``main``:

  1. write a HANDOFF block into results.txt
       A. context   - why this measurement matters (ADR link)
       B. purpose   - what quantity is being measured
       C. script    - the script that is launched
       D. data      - what data the script provides
       E. action    - what to do with the results
  2. launch the measurement script exactly once
  3. append its output to results.txt
  4. move to the next test

New measurement tasks are appended to ``TESTS`` following the same
five keys. When the target model is swapped in the serving process,
re-running ``python test.py`` re-measures everything and appends a
new session block.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results.txt"

TESTS: list[dict] = [
    {
        "name": "t1_serving_probe",
        "script": "scripts/measure/t1_serving_probe.py",
        "context": (
            "Milestone 1: confirm the OpenAI-compatible endpoint "
            "serves the intended model and measure raw throughput. "
            "Reference: docs/research/runtime-decision.md."
        ),
        "purpose": "Model id served, request latency, tokens/second.",
        "data": "3 repeats of the eval-task goals from "
               "configs/prototype.yaml against the live endpoint.",
        "action": (
            "If served model id and tok/s match the target model, "
            "calibrate budgets in configs/prototype.yaml from the "
            "measured seconds-per-step; otherwise flag the mismatch "
            "in the runtime-decision log."
        ),
    },
    {
        "name": "t2_schema_compliance",
        "script": "scripts/measure/t2_schema_compliance.py",
        "context": (
            "The controller requires strict JSON decisions "
            "(Section 6). Non-compliant output must be counted, "
            "not silently accepted."
        ),
        "purpose": "Fraction of generations that parse as valid "
                   "decision JSON; parse-failure rate under the "
                   "SCHEMA_PROMPT instruction.",
        "data": "20 synthetic states (varied goals) prompted "
               "through the live endpoint.",
        "action": (
            "If compliance is below 0.8, revise the schema prompt "
            "and re-run before adopting structured-output mode."
        ),
    },
    {
        "name": "t3_state_resume",
        "script": "scripts/measure/t3_state_resume.py",
        "context": (
            "Acceptance criterion 5: state survives process "
            "restart. This probe validates the JSON serialization "
            "round-trip without a GPU."
        ),
        "purpose": "Bit-exact round-trip fidelity of CognitiveState "
                   "save/load, including bounded-list behavior.",
        "data": "Synthetic states with full observation and "
               "history lists.",
        "action": (
            "Any fidelity loss invalidates the baseline arm; "
            "record it in the runtime-decision log before Milestone 4."
        ),
    },
    {
        "name": "t4_tool_effectiveness",
        "script": "scripts/measure/t4_tool_effectiveness.py",
        "context": (
            "Milestone 4 compares baseline vs recurrent arms. The "
            "recurrent arm's advantage comes from tool "
            "observations; this probe measures whether the live "
            "server benefits from tool results in-context."
        ),
        "purpose": "Answer accuracy with and without the tool "
                   "summary prepended to the prompt.",
        "data": "The eval-task goals from configs/prototype.yaml, "
               "run with and without tool summaries in context.",
        "action": (
            "A non-positive delta argues for the latent-recurrence "
            "branch (Milestone 5) rather than more context "
            "recurrence."
        ),
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config() -> dict:
    with open(ROOT / "configs" / "prototype.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_once(script: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(ROOT / script)],
                          capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr)
                                if proc.returncode else "")
    return proc.returncode, out


def main() -> None:
    cfg = load_config()
    with open(RESULTS_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== session {now_iso()} "
                 f"(model={cfg.get('model')}, "
                 f"endpoint={cfg.get('openai_base_url')}) =====\n")
        for test in TESTS:
            # 1. handoff block
            fh.write(f"\n--- HANDOFF: {test['name']} ---\n")
            fh.write(f"A. context: {test['context']}\n")
            fh.write(f"B. purpose: {test['purpose']}\n")
            fh.write(f"C. script:  {test['script']}\n")
            fh.write(f"D. data:    {test['data']}\n")
            fh.write(f"E. action:  {test['action']}\n")
            # 2. launch once
            rc, out = run_once(test["script"])
            # 3. append results
            fh.write(f"--- RESULTS {now_iso()} (exit {rc}) ---\n{out}")
            # 4. next test
            fh.flush()
        fh.write(f"\n===== end session {now_iso()} =====\n")
    print(f"results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
