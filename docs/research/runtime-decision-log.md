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

## Entry 2 — 2026-10-20 — Alignment on Model Quantization, Harness, and Latent Recurrence

- **Trigger:** review of handoff, bug resolution, and architectural clarification with user.
- **Decisions & Evidence:**
  1. **Quantization Level:** Confirmed **Q4** (GGUF Q4_K_M for llama.cpp / quantized serving) as the experimental target to guarantee fit and context headroom on the 32 GB RTX 5090 host.
  2. **Harness Integration:** Cloned `NousResearch/hermes-agent` to `external/hermes-agent` to serve as the reference upstream harness for adapter integration.
  3. **Latent Recurrence Fallback (Milestone 5):** Confirmed that if zero-shot unadapted forward-pass recurrence fails (due to numerical drift or MoE router disruption), a lightweight adapter (e.g., LoRA or a compact recurrent projection layer) is approved as an architectural contingency rather than treating zero-shot as a strict blocker.
  4. **GPU Status & Calibration Suite:** GPU currently occupied with external tasks. Calibration and measurement suites (`scripts/measure/` and `test.py`) consolidated to run dual-mode: 100% deterministic on CPU (mock), automatically switching to live inference when the server endpoint is active.
- **Impact on code & tests:**
  - `src/cognitive_runtime/state.py` bug in `StateStore.exists` resolved.
  - `src/model_prep/inspect.py` hub lookup bug resolved.
  - `test.py` syntax error fixed and 4 calibration probes (`t1` to `t4`) implemented in `scripts/measure/`.
  - `pytest.ini` configured to isolate OpenSI test suites from external submodules.
- **Next trigger:** GPU availability / launching the local Q4 server for full live benchmark run.

## Entry 3 — 2026-10-22 — Hermes Skills Bridging, MCP Catalog Integration, and Multi-Arm Benchmark Matrix

- **Trigger:** Implementation of passive harness extension capabilities (Section 4, 6C, 8) and expansion of Milestone 4 comparative evaluation.
- **Decisions & Implementations:**
  1. **Skill Loading & Persistent Memory:** Implemented `SkillLoader` to dynamically parse `SKILL.md` frontmatter and instructions from `external/hermes-agent/skills/`. Integrated with `PersistentMemoryStore` so discovered skills are cataloged in long-term durable memory (`persistent_memory_refs`) without prompt bloat.
  2. **MCP Catalog & Client Protocol:** Implemented `MCPManifestLoader` to discover manifests in `external/hermes-agent/optional-mcps/` (Wolfram, Semgrep, etc.). Implemented `MCPClient` abstraction (`MockMCPClient`, `StdioMCPClient`, `HttpMCPClient`) and `bridge_mcp_tools()` enforcing Section 8.1 tool proposal and approval lifecycles.
  3. **Multi-Arm Evaluation Matrix (Milestone 4):** Expanded `scripts/run_eval.py` to compare 3 distinct arms:
     - **Arm A (Baseline):** One-shot parametric answer without tools (28.6% accuracy on test corpus).
     - **Arm B (Reactive):** Single-turn tool caller without cognitive reflection or hypothesis tracking (71.4% accuracy).
     - **Arm C (Recurrent):** OpenSI multi-step cognitive loop with bounded state, hypothesis tracking, and error recovery (100.0% accuracy).
     Added multi-step calculation (`formula.txt`) and error recovery (`config.txt` -> `backup_config.txt`) tasks to `data/eval_corpus/`.
  4. **Integrated Test Suite:** Added `t5_multi_arm_eval` into `test.py` calibration battery. All 43 test items (36 unit/integration + 7 new MCP/skill tests) pass with 0 failures on CPU.
- **Next trigger:** Launch of local Q4 model server to record live token throughput, latency, and real model metrics.
