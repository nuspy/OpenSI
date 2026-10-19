# Runtime Options

**Date:** 15 October 2026
**Purpose:** compare the three layers named in Section 15 of the
handoff — PyTorch, Hugging Face Transformers, and llama.cpp — plus the
hybrid option, for the OpenSI cognitive recurrence prototype.

## The three layers are not interchangeable products

- **PyTorch** is the tensor-computation substrate. It provides no
  tokenizer, chat template, or model zoo on its own; it provides the
  tensors and autograd primitives.
- **Hugging Face Transformers** is the model implementation and
  loading ecosystem that *runs on* PyTorch. It supplies the model
  classes, chat templates, cache handling, and (for supported MoE
  models) router-logit access. It is the practical path to inspect
  hidden states and instrument the forward pass.
- **llama.cpp** is a separate native inference runtime with its own
  GGUF format, kernels, memory management, and OpenAI-compatible
  server. It exposes text in/out; the ordinary server API does not
  expose internal activations.

## Scoring against Section 15.2 requirements

| Requirement | Transformers | llama.cpp (server API) | Hybrid |
|---|---|---|---|
| Run Qwen3 MoE checkpoint | verified support (transformers `qwen3_moe`) | GGUF conversions exist (ggml-org repo) | both |
| Preserve original weights | yes (safetensors, unmodified) | yes (GGUF is a repackaging, weights preserved) | n/a |
| Access hidden states at chosen layer | **yes** (`output_hidden_states=True`) | not via standard API | research arm exposes them |
| Modify/wrap forward pass | **yes** (hooks, custom classes) | requires native fork | research arm |
| Reuse layers with shared weights | possible but invasive | hard | research arm |
| KV-cache correctness across iterations | library-managed | server-managed | controller-side cache reuse is the open question |
| Router-logit inspection | partial (`output_router_logits`) | no | research arm |
| Custom sampling/stop control | yes | partial (server sampling params) | — |
| Long-running state + tool feedback | controller-side (fine) | stateless HTTP calls are fine for context recurrence | — |
| VRAM fit (32 GB) | needs quantization or offload; bf16 ≈ 61 GB does not fit | Q4/Q8 GGUF fits (Q8 ≈ 32 GB is marginal; Q4_K_M comfortably fits) | serving side handles it |
| Debuggability | best | moderate | depends |
| Quantization support | optional (bnb/GPTQ) | best-in-class GGUF quantization | — |

## Evidence

- Qwen3-30B-A3B hub metadata (fetched 2026-10-15): architecture
  `Qwen3MoeForCausalLM`, 128 experts, 8 experts/token, Apache-2.0.
  The HF API `config` block did not expose `num_hidden_layers` in this
  response; the runtime probe records the exact value at install time.
- ggml-org publishes `Qwen3-30B-A3B-GGUF` (Q4_K_M, Q8_0, f16 shards),
  confirming llama.cpp support for this architecture.
- The 5090 host has 32 GB VRAM; bf16 weights (≈61 GB) will not fit, so
  the first real-weight experiments should use Q4/Q8 GGUF through a
  llama-server endpoint, while Transformers remains the place for
  activation-interception experiments once weights fit or a small
  proxy model is used.

## Consequence for this repo

`src/model_adapter` exposes one interface with three adapters:
`mock` (CI, deterministic), `openai_compatible` (serving via llama.cpp
or any OpenAI-compatible server), and `transformers` (in-process, for
research instrumentation). Capability flags make unsupported
capabilities explicit rather than implicit.
