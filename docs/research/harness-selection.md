# Harness Selection Report

**Date:** 15 October 2026
**Scope:** Milestone 0 of the Cognitive Recurrence Handoff (Section 4)
**Status:** Provisional. Live API evidence collected 2026-10-15; no deep fork performed.

## What was evaluated

| Candidate | Repo (observed) | Language | License | Activity on 2026-10-15 |
|---|---|---|---|---|
| Hermes Agent | `NousResearch/hermes-agent` | Python | MIT | pushed same day |
| OpenClaw | `openclaw/openclaw` | TypeScript | MIT | pushed same day |
| Goose | was `block/goose`, now redirects to `aaif-goose/goose` | Rust | Apache-2.0 | pushed same day |
| OpenHands | was `All-Hands-AI/OpenHands`, now `OpenHands/OpenHands` | TypeScript | MIT | pushed same day |

Note: the Goose redirect is itself evidence — the project's governance
changed since the handoff document was drafted (it previously lived
under `block`). This is exactly why the handoff insists on live
inspection rather than trusting cached knowledge.

## Rubric (1-5, weights per Section 4)

Weights: endpoint abstraction 15%, MCP lifecycle 15%, skills/tools extensibility 15%,
memory/resumability 10%, recurring goals and event injection 10%,
delegating next-step decision to the model 15%, modularity of a small fork 10%,
security boundaries 5%, tests/docs/maintenance 5%.

| Criterion | Hermes Agent | OpenClaw | Goose | OpenHands |
|---|---|---|---|---|
| Model/endpoint abstraction | 4 | 3 | 4 | 3 |
| MCP discovery & lifecycle | 4 | 4 | 4 | 3 |
| Skills/tools extensibility | 4 | 4 | 3 | 3 |
| Memory/resumability | 4 | 3 | 3 | 3 |
| Recurring goals, events | 4 | 4 | 3 | 3 |
| Model-controlled next step | 3 | 2 | 3 | 2 |
| Modularity for small fork | 3 | 2 | 3 | 2 |
| Security/sandboxing | 3 | 3 | 3 | 4 |
| Tests/docs/activity | 4 | 4 | 3 | 4 |

Language alignment matters: the cognitive runtime prototype is
Python; Hermes Agent is Python, OpenClaw and OpenHands are
TypeScript, Goose is Rust. A Python runtime calling a TypeScript
harness adds an inter-process boundary for no functional gain.

## Recommendation (provisional)

**Primary: Hermes Agent (Python, MIT)** for the eventual harness
adapter, because it is language-aligned with the Python cognitive
runtime, MIT-licensed, and actively maintained (pushed on the review
date). **For Milestones 1-2 specifically, no deep harness integration is
needed at all:** the local `harness_adapter` package already provides
the tool gateway and event surface, and the recommended serving path
for real models is an OpenAI-compatible endpoint (which any of the
candidates can front). **Defer the final harness commitment until
Milestone 4 results are in.**

Rejected for now: OpenClaw and OpenHands (cross-language boundary),
Goose (recent organizational churn suggests less stable foundations).

## Unresolved questions

- Which exact model revision will the evaluation set use once real
  weights are attached (Qwen3-30B-A3B or the Thinking-2507 variant)?
- Will the harness run on the same host as the model (32 GB VRAM) or
  on a separate machine? This changes the gateway interface
  (in-process import vs. network RPC).
- Approval policy for filesystem writes: per-call or per-session?
