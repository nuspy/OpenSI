# Runtime Decision Record (ADR)

**Revision:** 2
**Date:** 2026-10-20
**Status:** Accepted (Milestones 1-4 calibration consolidated; Q4 quantized serving confirmed; LoRA fallback approved for M5)

## Decision

For the first implementation (Milestones 1-2):

1. **Primary evaluation runtime:** an OpenAI-compatible serving
   endpoint (llama.cpp `llama-server` on the local RTX 5090 host).
   In Milestones 1-2 the endpoint is exercised by the mock backend;
   the OpenAI-compatible adapter in `src/model_adapter` is the bridge.
2. **CI runtime:** the deterministic mock backend. All tests must
   pass on CPU with no GPU and no model download.
3. **Research runtime (later):** Hugging Face Transformers in-process,
   chosen for hidden-state and router-logit access when the latent
   recurrence experiment (Milestone 5) begins.
4. **Not chosen yet:** a native llama.cpp fork or a full Transformers
   deployment as the primary runtime. Both remain open options after
   the Milestone 4 measurements.

## Rationale

- The handoff's Section 15.5 requires the decision to remain a living
  hypothesis; this record captures what is known today.
- The local environment is a 32 GB RTX 5090. The Qwen3-30B-A3B target
  in bf16 (~61 GB of weights) exceeds it, so early loops must either
  use quantized GGUF weights served locally or a small proxy model.
- The control loop needs only text-in/text-out from the serving
  layer; structured-decision parsing lives in the controller. That
  keeps the runtime swappable (Section 15.6).
- Transformers is the right home for activation experiments, but
  committing to it now would force large-weight residency before the
  measurement data justifies it.

## Evidence (as of 2026-10-15)

- HF model metadata for `Qwen/Qwen3-30B-A3B`: architecture
  `Qwen3MoeForCausalLM`, Apache-2.0 license, 16 safetensors shards
  (~61 GB total), chat template present.
- `ggml-org/Qwen3-30B-A3B-GGUF` publishes Q4_K_M and Q8_0 quants; Q8_0 is
  ~31 GB, marginal on 32 GB VRAM; Q4_K_M fits with context headroom.
- `torch` is not installed on this host yet; `transformers` is
  installed; the mock backend requires neither for CI.

## Consequences

- All runtime coupling in the prototype flows through
  `model_adapter.base.InferenceBackend` and its capability flags.
- Milestone 4 must record measured tokens/sec, peak VRAM, and latency
  before any switch is considered (Section 15.5 review triggers).
- If any trigger fires (inaccessible hidden states, invasive fork,
  VRAM incompatibility, reproducibility failures), append an entry
  to `runtime-decision-log.md` and re-decide.

## Next trigger for reconsideration

First successful multi-step recurrent run against a real served
model, or the start of Milestone 5 (latent recurrence), whichever
comes first.
