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

## Comparison arms (Milestone 4 Matrix)

- `mode: baseline` (Arm A) — single generation, no tool feedback. Tests
  whether the model already "knows" the answer parametrically.
- `mode: reactive` (Arm B) — standard single-turn tool caller. Calls a tool
  and outputs the observation directly without reflective state or multi-step
  planning. Fails on multi-step problems and tool failures.
- `mode: recurrent` (Arm C) — full OpenSI multi-step cognitive loop. Maintains
  bounded J-Space state, tracks evolving hypotheses, composes multiple tools,
  and recovers from unexpected tool failures.

The Milestone 4 evaluation compares all three arms on identical tasks in
`configs/prototype.yaml`. The multi-step formula task and the fallback config
recovery task serve as clear discriminating benchmarks where Arm C achieves
100% accuracy while Arm B achieves 71.4% and Arm A achieves 28.6%.
