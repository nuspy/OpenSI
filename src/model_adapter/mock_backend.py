"""Deterministic mock backend (acceptance criterion 7).

Emulates a model that emits the structured decision JSON required by
the cognitive controller. Behaviour is scripted from the config's
``eval_tasks`` so the whole control loop runs without GPU access and
produces identical results on every run.

Behaviour:
- baseline mode (no tool_history in the prompt payload): answer one-shot
  with the task's ``baseline_answer``;
- recurrent mode, first step (empty tool_history): request the task's
  configured tool via a CALL_TOOL decision;
- recurrent mode, later steps (tool history present): emit EMIT_FINAL
  incorporating the latest tool summary (the observed value).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import InferenceBackend, BackendCapabilities


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower().rstrip(".")


class MockBackend(InferenceBackend):
    name = "mock"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.mode = config.get("mode", "recurrent")
        self.eval_tasks = config.get("eval_tasks", [])
        self._call_count = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_chat=True,
            supports_structured_json_output=True,
            # The mock does not expose activations, router logits, or a
            # serving API, so those flags stay False.
            supports_hidden_states=False,
            supports_router_logits=False,
            supports_openai_compatible_serving=False,
            notes="scripted mock; deterministic, CPU-only",
        )

    def generate(self, messages: list[dict[str, Any]]) -> str:
        payload = json.loads(messages[-1]["content"])
        goal = payload.get("goal", "")
        task = self._match(goal)
        history = payload.get("tool_history")
        self._call_count += 1

        if task is None:
            return json.dumps({"action": "EMIT_FINAL",
                               "answer": _normalize(goal),
                               "rationale": "no matching eval task"})

        if history:
            last_entry = history[-1]
            last_summary = str(last_entry.get("summary", ""))
            last_status = str(last_entry.get("status", "success"))

            if self.mode == "reactive":
                # Reactive arm emits immediately on receiving the first observation,
                # with no multi-hop reflection or error recovery.
                expected = str(task.get("expected", ""))
                ans = expected if (expected and expected.lower() in last_summary.lower() and last_status == "success") else last_summary
                return json.dumps({
                    "action": "EMIT_FINAL",
                    "answer": ans,
                    "rationale": "reactive single-turn tool completion",
                })

            if self.mode == "recurrent":
                # Check for error recovery path
                fallback = task.get("fallback_step")
                if last_status != "success" and fallback and len(history) == 1:
                    return json.dumps({
                        "action": "CALL_TOOL",
                        "tool": fallback.get("tool", "file_read"),
                        "arguments": dict(fallback.get("tool_args", {})),
                        "rationale": "tool failed; recover by executing fallback step",
                    })

                # Check for multi-step tasks
                multi_steps = task.get("multi_steps")
                if multi_steps and len(history) < len(multi_steps):
                    next_step = multi_steps[len(history)]
                    return json.dumps({
                        "action": "CALL_TOOL",
                        "tool": next_step.get("tool", "calculator"),
                        "arguments": dict(next_step.get("tool_args", {})),
                        "rationale": f"recurrent step {len(history) + 1}: continuing multi-step plan",
                    })

                # Completed all tool steps: emit final answer
                expected = str(task.get("expected", ""))
                ans = expected if (expected and expected.lower() in last_summary.lower()) else last_summary
                return json.dumps({
                    "action": "EMIT_FINAL",
                    "answer": ans,
                    "rationale": "incorporate tool observation after recurrent verification",
                })

        if self.mode in ("recurrent", "reactive"):
            # First step: call tool
            initial_tool = task.get("tool", "calculator")
            initial_args = dict(task.get("tool_args", {}))
            multi_steps = task.get("multi_steps")
            if multi_steps:
                initial_tool = multi_steps[0].get("tool", initial_tool)
                initial_args = dict(multi_steps[0].get("tool_args", initial_args))

            return json.dumps({
                "action": "CALL_TOOL",
                "tool": initial_tool,
                "arguments": initial_args,
                "rationale": "first step: consult tool",
            })

        # Baseline arm: one-shot answer from parametric knowledge.
        return json.dumps({
            "action": "EMIT_FINAL",
            "answer": str(task.get("baseline_answer", "")),
            "rationale": "one-shot answer, no tools",
        })

    def _match(self, goal: str) -> dict | None:
        norm = _normalize(goal)
        for task in self.eval_tasks:
            if _normalize(task.get("goal", "")) == norm:
                return task
        return None
