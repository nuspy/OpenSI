# State Model (J-Space)

Four distinct stores, deliberately kept separate (Section 7):

1. **Working state** (`cognitive_runtime.state.CognitiveState`) — the
   active goal, bounded observations, and tool history for the current
   task. Serialized to `data/sessions/<session_id>.json` after each
   step. Bounded lists and `compress_state()` keep checkpoints small
   by summarizing older observations into `working_hypotheses`.
2. **Persistent memory** (`cognitive_runtime.memory.PersistentMemoryStore`) —
   durable cross-session facts, heuristics, and skill references stored in
   `data/memory/knowledge.json`. Referenced in working state via compact
   `persistent_memory_refs` IDs.
3. **KV cache** — an inference optimization owned by the serving
   layer, not persisted by the runtime.
4. **Latent activations** (`src/model_adapter/transformers_recurrent_backend.py`) —
   ephemeral hidden state representations for Milestone 5 latent recurrence
   experiments with residual adapter blending.

## Boundedness

- `recent_observations`: max 8 entries (older entries condensed into `working_hypotheses`).
- `tool_history`: max 32 entries.
- `decisions`: max 16 entries.
- Everything is plain JSON: auditable, diffable, no pickle.

## What is deliberately not stored

Full transcripts of tool outputs (only bounded summaries), and any
tensor data. A natural-language summary is *not* a reproduction of the
model's internal activation state; the prototype treats that gap as
an open research question, not as a solved problem.
