# OpenSI: cognitive recurrence runtime

A research prototype of an inference runtime in which a language model keeps a persistent, self-directed loop: it can continue
reasoning, inspect intermediate results, revise a candidate answer, ask for tools or information, take the results in, and
decide for itself whether to continue or to answer. It reuses an existing open-weight model; nothing is trained from scratch.

The design notes are in [`docs/design`](docs/design) (cognitive loop, state model, security model) and the choices made in
[`docs/research`](docs/research). The original brief is in `Cognitive_Recurrence_Handoff.md`.

## Layout

| Folder | What it holds |
| --- | --- |
| `src/cognitive_runtime` | the controller of the loop, events, memory, policies, state and tracing |
| `src/model_adapter` | model backends: a mock, an OpenAI-compatible server, and Transformers (plain and recurrent) |
| `src/harness_adapter` | the adapter that lets an agent harness supply tools, MCP servers and skills, and run what the loop decides |
| `src/model_prep` | a utility that inspects a model and prepares the runtime's metadata (`python -m model_prep`) |
| `scripts` | evaluation, serving benchmark, interactive CLI and measurement probes |
| `tests` | unit, integration and runtime-compatibility tests (`pytest`, configured in `pytest.ini`) |
| `data` | evaluation corpus and results; traces and sessions are not kept in the repository |

## Running the tests

```
python -m venv .venv
.venv\Scripts\pip install pytest
.venv\Scripts\python -m pytest
```

The tests use the mock backend; the Transformers and OpenAI-compatible backends need their own dependencies and a model.

## Status

Research prototype. The selected agent harness (`external/hermes-agent`) is a separate checkout and is not part of this
repository.
