# Security Model

The prototype separates *deciding* from *allowing*. The model proposes
actions; the policy engine (Section 9) decides whether they execute.

## Tiers

| Tier | Meaning | Examples |
|---|---|---|
| `auto` | pre-approved, side-effect-free or sandboxed | calculator, clock, file_read within sandbox |
| `approval_required` | needs human approval before execution | filesystem writes, network, shell |
| `denied` | never executed | anything outside policy |

## Rules implemented in `cognitive_runtime/policies.py`

- Tool names not in the tier table default to `approval_required`.
- File paths are validated against `sandbox_roots`; paths outside the
  roots are rejected even if the tool itself would allow them.
- Tool results are observations (data), never instructions. The
  `may_contain_instructions` flag marks results that could carry
  embedded commands for later sanitization.
- Budgets (steps, tool calls, seconds) are enforced by the controller,
  not trusted to the model.

## Known gaps (v1 prototype)

- No OS-level sandbox (container or seccomp) yet; process-level checks
  only.
- Approval is synchronous within the run; an async approval channel
  is planned for a later milestone.
- Secret redaction applies to trace files, which currently record
  only the sandboxed eval corpus.
