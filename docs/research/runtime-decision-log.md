# Runtime Decision Log

Append-only history of runtime reviews (Section 15.5). Each entry
records: the milestone or issue prompting the review, new evidence,
the previous decision, re-evaluated alternatives, the outcome, and the
next trigger for reconsideration.

## Entry 1 — 2026-10-15 — Project start (Milestones 1-2)

- **Trigger:** initial implementation.
- **Previous decision:** none (initial).
- **Alternatives evaluated:** Transformers in-process primary; llama.cpp
  native fork; hybrid with serving gateway.
- **Evidence gathered:** HF API metadata for Qwen/Qwen3-30B-A3B and
  ggml-org/Qwen3-30B-A3B-GGUF fetched on 2026-10-15; host is a 32 GB
  RTX 5090 (bf16 weights ≈ 61 GB, do not fit; GGUF Q4_K_M fits).
- **Outcome:** Split roles. Mock backend for CI and unit tests,
  OpenAI-compatible serving (llama-server) planned for Milestone 1
  end-to-end runs, Transformers reserved for the Milestone 5 latent
  experiments. Baseline vs recurrent arms live in the controller
  configuration, not the backend.
- **Impact on tests:** all integration tests must pass CPU-only.
- **Next trigger:** first real-model measurement run, or Milestone 5
  start.
