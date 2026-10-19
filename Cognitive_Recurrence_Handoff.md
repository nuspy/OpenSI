# Cognitive Recurrence Runtime --- Codex Engineering Handoff

**Version:** 0.1 --- research and prototype brief\
**Date:** 15 October 2026\
**Audience:** OpenAI Codex / coding agent\
**Project status:** Discovery + prototype specification; no repository
has yet been selected or cloned.

------------------------------------------------------------------------

## 1. Mission

Build a research prototype for an LLM inference system that supports
**persistent, self-directed cognitive recurrence**. The system should
reuse an existing open-weight model and its original weights wherever
possible. It must not require training a foundation model from scratch.

The intended system has three separable deliverables:

1.  **Cognitive inference runtime:** a runtime/control path that lets
    the model continue internal reasoning, inspect intermediate results,
    revise a candidate, request external information or tools,
    incorporate tool results, and decide whether to continue or emit a
    final response.
2.  **Model preparation/conversion utility:** a tool that inspects a
    model, validates whether the proposed runtime can support it,
    produces required metadata/configuration/adapters, and packages
    runtime-specific artifacts. It must not falsely claim that ordinary
    file conversion creates new cognitive abilities.
3.  **Passive environment / harness adapter:** adapt a mature
    open-source agent harness so it supplies tools, MCP servers, skills,
    external events, task execution, and user input, while the
    **LLM-controlled cognitive loop** determines the next
    task/tool/action. The harness performs mechanics and policy
    enforcement; it should not impose a fixed reasoning plan.

Codex must begin with repository discovery and a written recommendation
before making major implementation choices.

## 2. Core hypothesis

A capable pretrained model may already contain useful abilities for
planning, evaluation, revision, and tool selection. We want to test
whether changing the inference/control topology can expose those
abilities in a recurrent process without updating the base model
weights.

The system should support this cycle:

1.  Maintain a goal and a compact persistent cognitive state.
2.  Perform inference.
3.  Evaluate the current candidate and uncertainty / missing
    information.
4.  Choose one of: continue internal reasoning, request a tool, await an
    external event, ask the user, revise the state, or emit a final
    answer.
5.  If a tool is requested, the environment executes it and returns
    structured results.
6.  Incorporate results into the next cognitive iteration.
7.  Stop on a model-selected completion condition, a hard resource
    limit, an interruption, or a safety/policy barrier.

"J-Space" is a project-specific term here: a persistent memory/state
space that preserves the active goal and relevant cognition across
inference iterations. Do not assume it is the same as any research
paper's use of that term.

### Important distinction

The first prototype can implement recurrence at the **agent-turn /
token-context level**. A later experimental branch may investigate
**latent-state recurrence** by reusing or transforming hidden
activations. These are not equivalent.

-   **Context recurrence:** run the model again with a compact state,
    prior outputs, and tool feedback.
-   **Latent recurrence:** intervene in the neural computation graph and
    feed an internal representation into further computation.
-   **Weight sharing:** reuse the same parameter tensors across repeated
    computation. This does not guarantee stable behavior if the model
    was trained for a different forward-pass topology.
-   **Weight preservation:** keep original parameters unchanged. This is
    possible in both prototypes, but the latent-recurrence experiment
    may need architecture-specific code and may perform worse without
    adaptation.

Do not conflate chain-of-thought text with hidden activations. Visible
reasoning text is a token sequence, not a complete dump of the model's
internal state.

## 3. Preliminary model target

Start with **Qwen3-30B-A3B** as a candidate, subject to current
compatibility verification: - approximately 30.5B total parameters; -
approximately 3.3B active parameters per token; - Mixture-of-Experts
architecture, commonly described as 128 routed experts with 8 selected
per token, plus shared components depending on the exact model
configuration.

Official model card: https://huggingface.co/Qwen/Qwen3-30B-A3B\
A later or specialized checkpoint may be useful, but Codex must verify
the exact repository, license, tokenizer, chat template, context limits,
and runtime support before locking it in.

The model is an experimental target, not a hard dependency. Design
interfaces so another causal decoder / MoE model can be substituted.

## 4. Repository discovery: harness decision is a required first milestone

Before coding, compare current open-source candidates on GitHub. **Do
not assume the first candidate is the winner.** Inspect the live
repository, default branch, recent commits/releases, license,
architecture, extension APIs, tests, documentation, and issues. Record
the date and links for every claim.

### Candidate A --- Hermes Agent (primary candidate)

Repository: https://github.com/NousResearch/hermes-agent\
Docs: https://hermes-agent.nousresearch.com/docs

Why it appears promising: - model/provider endpoint can be configured,
including local endpoints; - persistent memory; - agent-managed skills
and skill improvement; - MCP client and tool discovery; - recurring
loops / goal-oriented work patterns; - an established agent loop with
separable orchestration modules.

Relevant docs to inspect: - MCP:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/mcp.md -
Skills:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/skills.md -
Memory:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/memory.md -
Recurring loops:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/loops.md -
Agent loop internals:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/agent-loop.md

Potential mismatch: its existing orchestration may own too much of the
control loop for the "passive harness" ideal. Do not fork deeply before
identifying clean extension points. Prefer an adapter/plugin or thin
fork with minimal divergence.

### Candidate B --- OpenClaw

Repository: https://github.com/openclaw/openclaw

Relevant docs: - Skills:
https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md -
Agent runtime:
https://github.com/openclaw/openclaw/blob/main/docs/concepts/agent.md -
Skill creation:
https://github.com/openclaw/openclaw/blob/main/docs/tools/creating-skills.md

Why compare it: - integrated agent runtime and session handling; -
extensible skills, tools, and multiple communication surfaces; - useful
if event delivery, gateway/channel integration, or skill lifecycle is
more important than a minimal inference integration.

Potential mismatch: its runtime is integrated and may be harder to
reduce to a passive execution substrate. Evaluate architecture and
license before selection.

### Candidate C --- Goose

Repository: https://github.com/block/goose\
Docs: https://block.github.io/goose/

Why compare it: - open-source extensible agent; - tool/extension
architecture and MCP ecosystem; - designed for connecting models to
executable capabilities.

Potential mismatch: originally strongly oriented toward interactive task
execution and coding workflows. Verify whether recurring autonomous
state and lifecycle hooks are sufficient or would require a large
redesign.

### Candidate D --- OpenHands

Repository: https://github.com/All-Hands-AI/OpenHands

Why compare it: - mature agent/tool execution and software-task
infrastructure; - MCP integration documented at
https://github.com/OpenHands/docs/blob/main/openhands/usage/cli/mcp-servers.mdx.

Potential mismatch: potentially heavier and coding-centric. Use if
sandboxed execution and task infrastructure outweigh the cost of
adapting its orchestration.

### Harness selection rubric

Score each candidate 1--5 and justify every score with source links:

  ---------------------------------------------------------------------
  Criterion                                                      Weight
  ----------------------------- ---------------------------------------
  Clear model-provider / local                                      15%
  endpoint abstraction          

  MCP discovery, invocation,                                        15%
  lifecycle, and auth           

  Extensible tools and skills;                                      15%
  model can propose new skills  

  Memory/session state and                                          10%
  resumability                  

  Recurring goals, asynchronous                                     10%
  tasks, event injection        

  Ability to delegate next-step                                     15%
  decision to an                
  external/model-controlled     
  loop                          

  Modularity and ease of                                            10%
  maintaining a small fork      

  Security boundaries,                                               5%
  approvals, sandboxing         

  Tests, maintenance, docs,                                          5%
  active development            
  ---------------------------------------------------------------------

The winner is not necessarily the one with the most features. It is the
one that can be made a **passive execution and integration layer** with
the least invasive changes.

Deliver `docs/research/harness-selection.md` containing: - date of
review; - candidate URLs and exact repository names; - license and any
compatibility caveats; - key modules / extension points; - recent
maintenance evidence; - rubric table; - recommendation and reasons; -
unresolved questions. If the evidence is insufficient, report that and
keep the choice provisional.

## 5. Runtime research and implementation strategy

### 5.1 Do not start by rewriting all of llama.cpp

`llama.cpp` is a credible C/C++ inference backend with CUDA support,
local execution, and an OpenAI-compatible server API: -
https://github.com/ggml-org/llama.cpp -
https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md -
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

It is a useful later target for production-ish local inference. But
direct experimentation with PyTorch hidden states and model submodules
is likely easier initially with Hugging Face Transformers. Keep a
backend interface so the cognitive controller is not coupled to one
inference engine.

**Phase 1:** Python prototype with Transformers and a local model;
preserve original weights.\
**Phase 2:** llama.cpp/OpenAI-compatible endpoint adapter for
context-level recurrence.\
**Phase 3:** investigate latent recurrence, potentially in a dedicated
Transformers branch first.\
**Phase 4:** only if justified, port selected changes to a native
C++/GGML backend.

### 5.2 Latent recurrence is a research branch, not an assumed feature

A standard Transformer is trained around a particular forward-pass
structure. Repeating layers or feeding an arbitrary hidden state back
into the network can break distributional assumptions and degrade
output. A Universal Transformer is a useful conceptual reference for
shared-weight recurrence, but it is not evidence that arbitrary
repetition of a pretrained Qwen architecture will work.

Potential experimental designs, each isolated behind a flag: - repeat a
selected block or subset of blocks using shared weights; - retain a
compact recurrent latent state and combine it with the current input
representation; - create a "review pass" that consumes intermediate
activations, if the architecture supports it; - compare against ordinary
context recurrence at matched approximate compute.

Do not silently change residual-stream shape, positional encoding,
normalization, attention cache semantics, MoE routing, or expert tensor
layout. Document exactly what is recurrent and what remains unchanged.

### 5.3 MoE constraints

For Qwen3-30B-A3B, expert routing happens at token/layer level. "Active
parameters" does not mean only active experts need to be stored in
memory; most/all expert weights may still need to be resident or
accessible. Preserve the original router behavior initially. Do not
prune experts or force a reduced expert set as part of the first
prototype.

## 6. System architecture

Create three main packages with narrow contracts.

### A. `cognitive_runtime`

Responsibilities: - maintain the current cognitive state and goal; -
call the configured inference backend; - interpret a structured control
decision; - choose whether to continue, call a tool, wait, ask the user,
or finish; - enforce iteration, token, time, and tool budgets; - persist
and resume state; - emit trace events for evaluation.

Suggested interface concepts (names are illustrative; adapt after repo
inspection): -
`InferenceBackend.generate(request) -> GenerationResult` -
`InferenceBackend.inspect_capabilities() -> BackendCapabilities` -
`CognitiveController.step(state, observation) -> ControlDecision` -
`StateStore.load(session_id) / save(session_id, state)` -
`EventBus.publish(event)` and
`EventBus.wait_for(event_filter, timeout)` -
`ToolGateway.execute(tool_request, policy_context) -> ToolResult`

The control decision should use a validated schema with enumerated
actions: - `CONTINUE_INTERNAL` - `CALL_TOOL` - `WAIT_FOR_EVENT` -
`ASK_USER` - `EMIT_FINAL` - `ABORT`

Require structured outputs / constrained JSON where supported. If
parsing fails, use a bounded recovery path; never execute malformed tool
requests.

### B. `model_adapter` / `model_prep`

Responsibilities: - inspect model config and architecture; - verify
tokenizer and chat template; - identify architecture features:
dense/MoE, layer count, hidden size, attention variant, RoPE, expert
count/top-k where applicable; - determine backend support and fail
clearly if unsupported; - generate a manifest of base model files and
hashes; - produce a runtime config and optional sidecar metadata; -
validate quantized or converted artifacts without implying that
conversion itself changes model cognition.

**No destructive rewrite of original model files.** All generated
artifacts go into a separate output directory. Never overwrite source
weights. Every transformation must be reproducible and logged.

Possible commands: - `model-prep inspect <model>` -
`model-prep validate <model> --backend transformers` -
`model-prep export-manifest <model> --out <directory>` -
`model-prep package-config <model> --profile recurrent-prototype --out <directory>`

These commands are proposals, not a commitment to CLI syntax.

### C. `harness_adapter`

Responsibilities: - adapt the selected open-source harness to expose
tools, MCP, skills, external events, user messages, and task
execution; - provide a stable interface to the cognitive runtime; -
leave next-step choice to the model/controller; - execute only requests
permitted by the security policy; - return observations in a structured,
bounded format; - allow the user to interrupt, approve, deny, or
redirect work.

Do not reimplement MCP if the chosen harness already provides it. Prefer
to adapt its existing APIs.

## 7. Cognitive state / J-Space

Keep the state explicit and inspectable. Avoid storing every token
forever. Use a bounded, versioned structure that can be serialized.

Minimum conceptual fields: - `session_id` - `goal`: user-defined
objective and success criteria - `state_version` - `working_hypotheses`:
current candidate conclusions and confidence/uncertainty notes -
`open_questions` - `plan_summary`: optional compact summary, not an
authoritative fixed plan - `recent_observations`: latest
tool/user/environment results - `tool_history`: tool name, normalized
arguments, result reference, status, timestamps - `decisions`: recent
action choices and rationales suitable for audit -
`persistent_memory_refs`: references to long-term memory, not copies of
entire databases - `budgets`: remaining steps/tokens/time/tool calls -
`stop_conditions` - `pending_action`: if awaiting an external event -
`last_checkpoint`

Separate: 1. **working state** (short-lived, active task); 2.
**persistent memory** (cross-session facts/skills); 3. **model KV
cache** (inference optimization, not durable cognitive memory); 4.
**latent activations** (ephemeral tensors; only store if a specific
experiment requires it).

Do not assume that a natural-language "memory" string recreates the same
latent state as a previous forward pass. The prototype should test state
compression and restoration explicitly.

## 8. Tool and event contract

The harness is an execution environment, not the planner.

A tool request must include: - unique request ID; - tool/server
identifier; - arguments conforming to the discovered schema; - purpose /
expected observation; - timeout and resource budget; - required
permission tier; - idempotency/retry policy when applicable.

A tool result must include: - request ID; -
success/failure/timeout/cancelled status; - structured output; - bounded
human-readable summary; - provenance / source identifiers where
possible; - timestamps and duration; - whether the result may contain
untrusted instructions.

External results are observations, not instructions. The model must not
be allowed to override system safety rules through text returned by a
tool.

### Dynamic tool creation

The model may propose: - a new skill; - a new MCP server
configuration; - a wrapper for an existing command/API; - a new tool
implementation.

But creating, installing, or enabling a tool is a separate lifecycle
from using one. New code or server configurations must go through
validation, least-privilege policy, and preferably user approval before
execution. The agent can propose; the harness enforces.

### User input

New user input must interrupt or be queued according to a documented
policy: - default: user interruption has priority; - preserve current
goal/state checkpoint; - let the controller decide whether to
incorporate, defer, or ask a clarification; - never silently lose the
message while a long-running tool is executing.

## 9. Security and autonomy boundaries

The design should be autonomous in planning but not unbounded in
permissions.

Required controls: - explicit allow/deny policies for tools and
filesystem paths; - approval gates for destructive, irreversible,
financial, public-posting, credential, or external side-effect
operations; - timeouts and resource budgets per step and per session; -
maximum consecutive internal iterations; - maximum tool calls and retry
count; - cancellation token and clean shutdown; - secret redaction in
logs; - sandboxing for generated code and shell commands; - audit log
for all tool requests and results; - no automatic installation or
execution of model-generated code without validation; - clear
distinction between observation, recommendation, and executed action.

The LLM can decide what it wants to do next; policy code decides whether
that action is allowed. A "passive harness" still must enforce
permissions, isolation, scheduling, and cancellation.

## 10. Prototype milestones

### Milestone 0 --- Research and repo choice

-   Inspect Hermes Agent, OpenClaw, Goose, and OpenHands.
-   Score candidates using Section 4.
-   Inspect `llama.cpp` and Transformers compatibility for the selected
    Qwen model.
-   Deliver `docs/research/harness-selection.md` and
    `docs/research/runtime-options.md`.
-   Do not make large code changes before the recommendation is written.

### Milestone 1 --- Baseline harness adapter

-   Use the chosen harness's existing model/provider interface.
-   Connect it to a local model endpoint.
-   Expose one safe test tool and one mock asynchronous event.
-   Ensure the harness can return tool results without choosing the next
    cognitive action itself.

### Milestone 2 --- Context-level recurrent controller

-   Implement the state model and action schema.
-   Add the bounded loop: infer → decide → optionally execute/wait →
    observe → infer.
-   Support continuation without emitting user-facing text.
-   Support final response and interruption.
-   Persist/resume state after process restart.
-   Add hard limits and traces.

### Milestone 3 --- Model preparation utility

-   Implement model inspection and compatibility checks.
-   Produce model manifest, config, and reproducibility metadata.
-   No weight mutation in this milestone.
-   Add a dry-run mode and negative tests for unsupported architectures.

### Milestone 4 --- Evaluation harness

Compare at least: A. one-shot baseline; B. normal tool-using agent loop;
C. context-level recurrent controller; D. optionally, latent-recurrence
experimental branch.

Use matched tasks and approximate compute/token budgets. Measure: - task
success / correctness; - tool-call precision and unnecessary calls; -
recovery after wrong tool results; - hallucination rate for missing
information; - number of iterations to completion; - token usage,
latency, peak VRAM; - ability to stop correctly; - state-resume
fidelity; - reproducibility across repeated runs.

Do not declare success merely because the model produces longer
reasoning or more tool calls.

### Milestone 5 --- Latent recurrence experiment

Only after Milestones 1--4 work: - create a separate experimental
branch; - capture hidden states at well-defined points; - specify the
exact tensor and layer where recurrence occurs; - compare a small
supported model first, then the Qwen MoE target; - preserve the baseline
path and make the experiment feature-flagged; - run numerical stability,
shape, cache, and regression tests; - compare against the matched
context-level baseline.

If latent recurrence does not improve measurable performance, document
the negative result rather than hiding it.

## 11. Acceptance criteria for the first usable prototype

The prototype is acceptable when: 1. A user can define an objective. 2.
The model can choose internal continuation, tool call, wait, ask user,
final answer, or abort. 3. The environment can execute a tool and return
a structured observation. 4. The next inference can incorporate the
observation and revise its decision. 5. State survives a restart and can
be resumed. 6. Tool calls obey permissions and resource budgets. 7. A
deterministic mock backend makes integration tests run without GPU
access. 8. A real local model can be configured through a backend
adapter. 9. Baseline and recurrent modes can be compared on the same
test set. 10. All behavior is logged well enough to diagnose why a cycle
continued or stopped.

## 12. Suggested repository layout

Adapt to the chosen harness rather than imposing this layout if it
creates unnecessary duplication.

``` text
project/
  docs/
    research/
      harness-selection.md
      runtime-options.md
      architecture-decisions.md
    design/
      cognitive-loop.md
      state-model.md
      security-model.md
  src/
    cognitive_runtime/
      controller.py
      state.py
      events.py
      policies.py
      tracing.py
    model_adapter/
      base.py
      transformers_backend.py
      openai_compatible_backend.py
    model_prep/
      inspect.py
      validate.py
      manifest.py
      package.py
    harness_adapter/
      interface.py
      selected_harness.py
  tests/
    unit/
    integration/
    evals/
  configs/
    prototype.yaml
  scripts/
```

## 13. Instructions to Codex

Work in small, reviewable increments. Begin with repository research,
not implementation.

1.  Read this handoff fully.
2.  Browse current GitHub repositories and documentation using public
    web/repository access; do not require the user's personal GitHub
    account or credentials.
3.  Inspect the actual source code of the top two harness candidates,
    especially the agent loop, model adapter, tool registry, MCP
    integration, skill lifecycle, state persistence, and approval logic.
4.  Produce the two research documents from Milestone 0.
5.  Explain the recommended choice and its tradeoffs before substantial
    code changes.
6.  Then implement Milestone 1 and Milestone 2 only, with tests. Do not
    attempt latent recurrence in the first implementation.
7.  Preserve the original model files and the upstream harness wherever
    possible. Keep modifications behind a small adapter or a clearly
    isolated patch.
8.  Never claim that a model converter adds cognitive capabilities
    unless the conversion actually changes a documented
    runtime/architecture behavior.
9.  Ask the user only when a genuinely blocking decision cannot be
    resolved by repository evidence. Otherwise make a documented,
    reversible choice.
10. At the end of each milestone, report files changed, tests run,
    results, limitations, and the next smallest step.

## 14. Research sources

Primary repositories and documentation: - Hermes Agent:
https://github.com/NousResearch/hermes-agent - Hermes MCP docs:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/mcp.md -
Hermes skills docs:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/skills.md -
Hermes memory docs:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/memory.md -
Hermes loops docs:
https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/loops.md -
OpenClaw: https://github.com/openclaw/openclaw - Goose:
https://github.com/block/goose - OpenHands:
https://github.com/All-Hands-AI/OpenHands - llama.cpp:
https://github.com/ggml-org/llama.cpp - llama.cpp server API:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md -
llama.cpp build/CUDA docs:
https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md -
Qwen3-30B-A3B model card: https://huggingface.co/Qwen/Qwen3-30B-A3B

These links are starting points, not substitutes for checking the live
source code and license at implementation time.

------------------------------------------------------------------------

# 15. Mandatory runtime-engine comparison and recursive architecture review

**This section extends and overrides any earlier wording that could be
read as a fixed commitment to a particular inference engine.** The
runtime choice is a living engineering hypothesis. Codex must
investigate it at the beginning, document the tradeoffs, and revisit it
whenever evidence or implementation problems materially change the
assumptions.

## 15.1 Compare three distinct layers, not three interchangeable products

Codex must compare:

1.  **PyTorch** --- tensor computation framework and low-level substrate
    for custom neural execution.
2.  **Hugging Face Transformers** --- model
    implementation/loading/generation ecosystem, commonly running on
    PyTorch, with model classes, tokenizer/chat-template support,
    caches, hidden-state outputs, and architecture-specific
    implementations.
3.  **llama.cpp** --- a separate native inference runtime with its own
    model formats, kernels, memory management, sampling, and server
    interface.

PyTorch and Transformers are complementary rather than direct
substitutes: Transformers commonly uses PyTorch as its execution
backend. The comparison must say this explicitly. The real decision is
often between: - Transformers + PyTorch for inspectable, modifiable
model internals; - llama.cpp for optimized local inference and a native
serving/runtime path; - a hybrid architecture where a research runtime
and production-serving runtime coexist behind one backend interface.

Do not call all three "inference engines" without explaining the layer
each occupies.

## 15.2 Evaluate against the intended final architecture

Score and discuss every option against the following needs:

  -----------------------------------------------------------------------
  Requirement                         Why it matters
  ----------------------------------- -----------------------------------
  Load and run the exact selected     Basic feasibility
  Qwen MoE checkpoint                 

  Preserve original weights           First prototype must not require
                                      foundation-model training

  Access hidden states at specific    Needed for latent-state experiments
  layers/tokens                       

  Modify or wrap the forward pass     Needed for recurrent computation
                                      experiments

  Reuse selected layers/blocks with   Candidate latent-recurrence design
  shared weights                      

  Preserve KV-cache correctness       Needed for reliable autoregressive
  across iterations                   decoding

  Inspect router logits / MoE expert  Helps diagnose MoE behavior and
  selection                           recurrent-pass effects

  Keep original MoE routing by        Avoid changing model behavior
  default                             before baseline is established

  Add custom sampling and             Needed for model-controlled
  stop/continue decisions             cognitive loop

  Support long-running state and      Runtime/controller integration
  external tool feedback              

  Expose an API to an external        Keeps harness separate from neural
  harness                             execution

  Fit GPU VRAM and provide measurable Practical local feasibility
  performance                         

  Quantization / format support       Useful for deployment, but must not
                                      block research access

  Debuggability and testability       Essential for comparing hypotheses

  Future native/optimized deployment  Avoid locking the project to a
                                      research-only implementation
  -----------------------------------------------------------------------

For each requirement, label the evidence as **verified**, **partially
verified**, **unknown**, or **not supported**. Provide links to current
official docs/source and a minimal test where possible.

## 15.3 Preliminary hypotheses, not final verdicts

Use these as initial expectations to verify, not as assumptions to
preserve regardless of evidence.

### Hugging Face Transformers + PyTorch

Likely strongest starting point for latent-recurrence research because
it exposes architecture-specific model classes, forward methods, hidden
states, cache objects, and, for supported MoE models, router outputs.
The official Qwen3-MoE docs describe `hidden_states`, `past_key_values`,
`inputs_embeds`, and router logits. See: -
https://huggingface.co/docs/transformers/main/model_doc/qwen3_moe -
https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_moe.md -
https://pytorch.org/docs/stable/

Potential disadvantages to test: - peak memory and speed on the target
GPU; - support and kernel performance for the exact model and
dtype/quantization; - version-sensitive model implementations; -
complexity of preserving cache and attention semantics during custom
recurrence.

### llama.cpp

Likely strong for efficient local inference, quantized GGUF execution,
CUDA support, and serving behind an API: -
https://github.com/ggml-org/llama.cpp -
https://github.com/ggml-org/llama.cpp/tree/master/tools/server -
https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

Potential disadvantages to test: - whether the exact selected Qwen
checkpoint and quantization are supported; - how much internal
activation state can be inspected or intercepted through the public
API; - whether the required recurrence can be implemented cleanly
without a large invasive native-code fork; - whether its serving
abstractions permit the cognitive controller to retain control without
the server/harness deciding the next reasoning step.

Do not assume that API access to token generation gives access to latent
activations. A normal text-generation endpoint is not an
activation-inspection interface.

### PyTorch without relying entirely on Transformers generation

PyTorch can provide the low-level flexibility to prototype custom tensor
graphs and repeated computation. However, implementing an entire modern
decoder, MoE routing, attention/cache handling, tokenization, and
optimized kernels from scratch would be wasteful and error-prone. Prefer
reusing a validated architecture implementation and modifying the
smallest possible section.

## 15.4 Required experiments before selecting the primary research runtime

Build a small compatibility spike, not a full application, to answer:

1.  Can the exact model load and generate correctly?
2.  Can the backend expose a chosen layer's hidden state for a chosen
    token?
3.  Can it report or inspect router information if the architecture
    supports it?
4.  Can the forward path be instrumented without changing baseline
    logits beyond expected numerical tolerance?
5.  Can a one-step recurrent experiment be implemented behind a feature
    flag?
6.  Can the ordinary baseline path still run unchanged?
7.  Can the same user/tool protocol be driven through an
    OpenAI-compatible local endpoint, independently of whether that
    endpoint is backed by llama.cpp or another server?
8.  What are measured startup time, peak VRAM, token throughput,
    latency, and numerical differences?
9.  Which requirements are impossible or too invasive in the candidate
    runtime?
10. Can tests reproduce the result on a small model or CPU/mock path
    even when the large model is unavailable?

Record exact package versions, commit hashes, model revision,
quantization, CUDA/PyTorch versions, GPU, commands, logs, and failures.
Do not use a single successful text-generation sample as proof of
architectural compatibility.

## 15.5 Recursive decision process: runtime Architecture Decision Record (ADR)

Create `docs/research/runtime-decision.md` and keep it current. It must
include: - current selected runtime and role (research, serving, or
both); - date and revision; - the intended final architecture as
currently understood; - evidence and links; - test results and resource
measurements; - known blockers and workarounds; - alternatives rejected
or deferred; - confidence level and unresolved unknowns; - explicit
conditions that would trigger reconsideration.

At the end of every milestone, Codex must run a **runtime decision
review**. Reconsider the decision if any of the following occurs: - a
required hidden state or router output is inaccessible; - a recurrent
forward-pass change requires an invasive fork; - the target model is
unsupported or unreliable; - memory/performance is incompatible with the
available GPU; - cache correctness or reproducibility cannot be
maintained; - a version or kernel regression invalidates a previous
assumption; - the harness requires capabilities that the runtime's
serving API cannot provide; - measured results show the alternative
backend is materially better for the next phase; - the final
architecture changes from context-level recurrence to latent recurrence,
or vice versa.

The review must choose one of four outcomes: 1. **Keep** the current
decision with evidence. 2. **Adjust** the implementation while keeping
the same runtime. 3. **Split roles** (for example, Transformers/PyTorch
for research and llama.cpp for serving). 4. **Switch** the primary
runtime, with a migration plan and regression tests.

Do not switch merely because a new framework is fashionable. Do not
remain with a runtime merely because code has already been written. Use
the requirements and evidence.

Each decision review must append an entry to
`docs/research/runtime-decision-log.md` with: - milestone / issue
prompting review; - new evidence; - previous decision; - alternatives
re-evaluated; - outcome and rationale; - impact on architecture and
tests; - next trigger for reconsideration.

## 15.6 Runtime / harness boundary must remain stable

The harness choice and runtime choice must be evaluated jointly but not
fused into a single decision. The runtime owns neural execution and
model-specific capabilities. The cognitive controller owns the next-step
decision policy and recurrent state. The harness owns external tools,
MCP connections, skill execution, event delivery, user input,
permissions, and side effects.

Required abstraction: - `InferenceBackend`: load/generate/inspect model
capabilities; optionally expose experimental activation APIs. -
`CognitiveController`: choose continue/tool/wait/ask/final/abort. -
`ToolGateway`: execute permitted tool requests and return
observations. - `StateStore`: persist goal and cognitive state. -
`HarnessAdapter`: connect the selected harness to the
gateway/events/user interface.

The common interface must not pretend all backends expose the same
internals. Use explicit capabilities such as: -
`supports_hidden_states` - `supports_router_logits` -
`supports_custom_forward` - `supports_latent_recurrence_experiment` -
`supports_openai_compatible_serving` - `supports_model_quantization` -
`supports_resumeable_kv_cache`

Unsupported capabilities must be reported as unavailable, not simulated
or silently ignored.

## 15.7 Decision gates for the project

-   **Gate A --- Before the first implementation:** complete a
    source-backed comparison and a minimal load/generate test.
-   **Gate B --- Before committing to latent recurrence:** demonstrate
    access to the necessary internal tensors and define the precise
    recurrence point.
-   **Gate C --- Before adopting a native runtime fork:** estimate
    maintenance burden and show why a backend adapter or upstream
    extension is insufficient.
-   **Gate D --- Before production-style serving:** benchmark
    quantization, throughput, memory, robustness, and harness
    integration independently of research instrumentation.

Codex must not begin by writing the converter or modifying llama.cpp.
First establish what the architecture requires, then choose the smallest
viable substrate for each phase.

## 15.8 Required deliverables added by this section

-   `docs/research/runtime-options.md` --- evidence-backed comparison of
    PyTorch, Transformers, llama.cpp, and a hybrid option.
-   `docs/research/runtime-decision.md` --- current ADR.
-   `docs/research/runtime-decision-log.md` --- append-only review
    history.
-   `tests/runtime_compat/` --- reproducible compatibility probes and
    capability checks.
-   `tests/evals/` --- matched
    baseline/context-recurrence/latent-recurrence evaluation cases when
    available.

The recommendation must remain provisional until these checks are run
against the exact model revision and actual local environment.
