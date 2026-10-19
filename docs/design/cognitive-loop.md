# Cognitive Loop Design

The prototype runs a recurrent control loop (Section 6 of the
handoff). Each iteration:

```text
1. flush events        pending events -> recent_observations
2. serialize state    CognitiveState -> JSON user message
3. backend.generate   model emits a decision JSON object
4. parse + validate   invalid JSON -> bounded repair (never blind execution)
5. policy check       allowed? approval required?
6. execute            gateway runs the tool, returns ToolResult
7. fold observation   result summary -> state
```

## Loop termination conditions

| Condition | Config key | Effect |
|---|---|---|
| step budget | `budgets.max_steps` | hard cap on iterations |
| tool-call budget | `budgets.max_tool_calls` | caps side effects |
| wall-clock budget | `budgets.max_seconds` | prevents stalls |
| repair budget | `budgets.max_repair_attempts` | bounds malformed-output retries |
| terminal action | `EMIT_FINAL`, `ABORT` | normal exits |

## Decision schema

The model must answer with one JSON object:

```json
{
  "action": "CALL_TOOL | CONTINUE_INTERNAL | WAIT_FOR_EVENT | ASK_USER | EMIT_FINAL | ABORT",
  "rationale": "short audit note",
  "tool": "name (only for CALL_TOOL)",
  "arguments": {}
}
```

`EMIT_FINAL` carries `answer`. The controller never executes an action
that failed schema validation.

## Baseline vs recurrent arms

- `mode: baseline` — single generation, no tool feedback. Tests
  whether the model already "knows" the answer.
- `mode: recurrent` — multi-step loop; tool results are fed back as
  observations before the final answer.

The Milestone 4 evaluation compares both arms on identical tasks. The
third task in `configs/prototype.yaml` (a file whose content the model
cannot know parametrically) is the discriminating case: the baseline
arm fails it, the recurrent arm succeeds.
