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
| `src/harness_adapter` | the adapter that lets an agent harness supply tools, MCP servers and skills, and run what the loop decides; the harness is TaDaMoo |
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

The runtime tests use the mock backend. Running the Transformers or OpenAI-compatible backends for real needs their own dependencies and a model.

## Status

Research prototype.

## The harness: TaDaMoo

OpenSI uses [TaDaMoo](https://github.com/nuspy/tadamoo) as its agent harness. It reads what TaDaMoo has installed from
its user data folder (`$TADAMOO_DATA`, else `%APPDATA%\TaDaMoo` on Windows) and imports none of its code:

- skills: the `SKILL.md` folders in `capabilities/skills`;
- MCP servers: the entries of `capabilities/installed.json`.

The tools themselves (calculator, clock, file read and write inside the sandbox roots) are OpenSI's own, and a skill or
an MCP tool becomes available only after its proposal is approved. Without a TaDaMoo data folder everything else works and
no skills or servers are listed. The folder can be set in the configuration under `harness: {tadamoo_data: ...}`, and a
directory of `manifest.yaml` files under `harness: {mcp_catalog: ...}`. A model served by TaDaMoo's OpenAI-compatible
API can be used through the OpenAI-compatible backend.
