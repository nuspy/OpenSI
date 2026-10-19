"""Unit tests for the controller's decision parsing and budgets."""

import json

import pytest

from cognitive_runtime.controller import parse_decision, ParseError, ACTIONS


def test_valid_actions_all_parse():
    for action in ACTIONS:
        obj = parse_decision(json.dumps({"action": action}))
        assert obj["action"] == action


def test_fenced_json_is_accepted():
    text = "```json\n{\"action\": \"EMIT_FINAL\", \"answer\": \"42\"}\n```"
    obj = parse_decision(text)
    assert obj["action"] == "EMIT_FINAL"
    assert obj["answer"] == "42"


def test_prose_around_json_is_tolerated():
    text = 'Sure! {"action": "CALL_TOOL", "tool": "calculator"} done.'
    obj = parse_decision(text)
    assert obj["action"] == "CALL_TOOL"


def test_unknown_action_rejected():
    with pytest.raises(ParseError):
        parse_decision('{"action": "NOT_A_REAL_ACTION"}')


def test_non_json_rejected():
    with pytest.raises(ParseError):
        parse_decision("the answer is forty two")


def test_budget_limits_enforced():
    """A controller with max_steps=0 must stop immediately."""
    from cognitive_runtime.controller import CognitiveController
    from harness_adapter import ToolGateway

    cfg = {
        "mode": "recurrent",
        "policy": {"sandbox_roots": ["data/eval_corpus"]},
        "budgets": {"max_steps": 0, "max_seconds": 60},
    }
    gw = ToolGateway(cfg)

    class DeadBackend:
        def generate(self, messages):
            raise AssertionError("backend must not be called when steps exhausted")

    controller = CognitiveController(DeadBackend(), gw, cfg)
    outcome = controller.run("anything")
    assert outcome.stop_reason in ("budget", "budget_steps")
    assert outcome.steps == 0


def test_policy_denies_unknown_tool():
    from cognitive_runtime.policies import PolicyEngine, PolicyConfig

    engine = PolicyEngine(PolicyConfig())
    decision = engine.check_tool("rm_rf_root", 1)
    assert decision.requires_approval is True
