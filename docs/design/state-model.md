# State Model (J-Space)

Four distinct stores, deliberately kept separate (Section 7):

1. **Working state** (`cognitive_runtime.state.CognitiveState`) — the
   active goal, bounded observations, and tool history for the current
   task. Serialized to `data/sessions/<session_id>.json` after each
   step. Bounded lists keep checkpoints small.
2. **Persistent memory** — cross-session facts; out of scope for
   Milestones 1-2 (the `eval_corpus` stands in for it).
3. **KV cache** — an inference optimization owned by the serving
   layer, not persisted by the runtime.
4. **Latent activations** — out of scope until Milestone 5.

## Boundedness

- `recent_observations`: max 8 entries.
- `tool_history`: max 32 entries.
- Everything is plain JSON: auditable, diffable, no pickle.

## What is deliberately not stored

Full transcripts of tool outputs (only bounded summaries), and any
tensor data. A natural-language summary is *not* a reproduction of the
model's internal activation state; the prototype treats that gap as
an open research question, not as a solved problem.
