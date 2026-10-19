"""Integration tests: full recurrent loop against the mock backend.

These run on CPU with zero GPU access (acceptance criterion 7) and
assert that the harness returns structured tool observations to the
model rather than deciding the next step itself.
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    with open(ROOT / "configs" / "prototype.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_recurrent_loop_returns_structured_observation():
    from cognitive_runtime.controller import CognitiveController
    from harness_adapter import ToolGateway
    from model_adapter import get_backend

    cfg = {**_cfg(), "mode": "recurrent"}
    gw = ToolGateway(cfg)
    backend = get_backend(cfg["backend"], cfg)
    controller = CognitiveController(
        backend, gw, cfg, trace_dir=ROOT / "data" / "traces")

    task = cfg["eval_tasks"][0]
    outcome = controller.run(task["goal"])
    assert outcome.stop_reason == "final"
    assert outcome.tool_calls >= 1
    assert outcome.final_text.strip() == task["expected"]
    # Trace must record at least one decision event.
    events = controller.tracer.read(outcome.session_id)
    assert any(e["kind"] == "decision" for e in events)


def test_state_survives_process_restart(tmp_path):
    """State persistence: save, reload in a 'fresh process', compare."""
    from cognitive_runtime.state import new_state, StateStore

    state = new_state("resume goal", "criteria")
    state.recent_observations.append({"kind": "tool", "content": "obs-1"})
    state.tool_history.append({
        "request_id": "x", "tool": "calculator", "arguments": {},
        "status": "success", "result_ref": "tool:calculator",
        "summary": "42", "started_unix": 0.0, "duration_s": 0.0,
    })
    store = StateStore(tmp_path)
    store.save(state)
    reloaded = store.load(state.session_id)
    assert reloaded.goal == state.goal
    assert reloaded.tool_history[0]["summary"] == "42"


def test_mock_backend_is_deterministic():
    from model_adapter import get_backend

    cfg = {**_cfg(), "mode": "recurrent"}
    a = get_backend("mock", cfg)
    b = get_backend("mock", cfg)
    goal = _cfg()["eval_tasks"][0]["goal"]
    payload = json.dumps({"goal": goal})
    msg = [{"role": "system", "content": "sys"}, {"role": "user", "content": payload}]
    assert a.generate(msg) == b.generate(msg)
