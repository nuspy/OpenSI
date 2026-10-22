"""Probe t3: Cognitive state round-trip resume and boundedness fidelity.

Verifies acceptance criterion 5: CognitiveState survives persistence to disk
and reload, checking bit-exact fidelity and enforcing bounded list constraints
(MAX_RECENT_OBSERVATIONS=8, MAX_TOOL_HISTORY=32).
Runs purely on CPU.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_runtime.state import (
    CognitiveState,
    StateStore,
    MAX_RECENT_OBSERVATIONS,
    MAX_TOOL_HISTORY,
)


def run_state_probe() -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix="opensi_probe_t3_"))
    try:
        store = StateStore(temp_dir)
        session_id = "probe_session_test_01"

        # Generate large observation and tool history to test truncation
        observations = [
            {"kind": "tool", "content": f"obs_{i}", "index": i}
            for i in range(25)
        ]
        tools = [
            {
                "request_id": f"req_{j}",
                "tool": "calculator",
                "arguments": {"expression": f"{j}*2"},
                "status": "success",
                "result_ref": f"tool:calculator:{j}",
                "summary": str(j * 2),
                "started_unix": time.time(),
                "duration_s": 0.01,
            }
            for j in range(50)
        ]

        original_state = CognitiveState(
            session_id=session_id,
            goal="Test persistent memory round-trip fidelity.",
            state_version=5,
            success_criteria="State must match identically on reload.",
            open_questions=["Is observation ordering preserved?"],
            plan_summary="Step 1 -> Step 2 -> Resume.",
            recent_observations=list(observations),
            tool_history=list(tools),
            budgets={"max_steps": 10, "max_seconds": 30.0},
            stop_conditions=["manual_test"],
        )

        saved_path = store.save(original_state)
        exists = store.exists(session_id)

        # Reload state from disk
        reloaded_state = store.load(session_id)

        # Verification checks
        checks = {
            "file_exists_on_disk": saved_path.exists(),
            "store_exists_method": exists,
            "session_id_match": reloaded_state.session_id == original_state.session_id,
            "goal_match": reloaded_state.goal == original_state.goal,
            "state_version_match": reloaded_state.state_version == original_state.state_version,
            "success_criteria_match": reloaded_state.success_criteria == original_state.success_criteria,
            "open_questions_match": reloaded_state.open_questions == original_state.open_questions,
            "plan_summary_match": reloaded_state.plan_summary == original_state.plan_summary,
            "budgets_match": reloaded_state.budgets == original_state.budgets,
            "stop_conditions_match": reloaded_state.stop_conditions == original_state.stop_conditions,
            "observations_bounded_to_max": (
                len(reloaded_state.recent_observations) == MAX_RECENT_OBSERVATIONS
            ),
            "observations_most_recent_preserved": (
                reloaded_state.recent_observations[-1]["content"] == "obs_24"
            ),
            "tool_history_bounded_to_max": (
                len(reloaded_state.tool_history) == MAX_TOOL_HISTORY
            ),
            "tool_history_most_recent_preserved": (
                reloaded_state.tool_history[-1]["request_id"] == "req_49"
            ),
        }

        all_passed = all(checks.values())

        return {
            "session_id": session_id,
            "all_checks_passed": all_passed,
            "checks": checks,
            "limits": {
                "max_recent_observations": MAX_RECENT_OBSERVATIONS,
                "max_tool_history": MAX_TOOL_HISTORY,
            },
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    result = run_state_probe()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
