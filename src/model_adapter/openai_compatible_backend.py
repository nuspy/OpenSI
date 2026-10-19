"""OpenAI-compatible serving backend adapter.

Works with any OpenAI-compatible ``/v1/chat/completions`` endpoint
(llama.cpp server, vLLM, LMStudio, ...). Requires no GPU locally if the
server runs elsewhere on the LAN.
"""

from __future__ import annotations

from typing import Any

from .base import InferenceBackend, BackendCapabilities

try:  # requests is optional until an install makes it available
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None


class OpenAICompatibleBackend(InferenceBackend):
    name = "openai_compatible"

    def __init__(self, config: dict[str, Any]):
        self.base_url = str(config.get("openai_base_url", "")).rstrip("/")
        self.model = config.get("model", "")
        self.max_tokens = int(config.get("max_tokens", 512))
        self.temperature = float(config.get("temperature", 0.1))
        if _requests is None:
            raise RuntimeError("the 'requests' package is required for this backend")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_chat=True,
            supports_structured_json_output=False,
            supports_openai_compatible_serving=True,
            # llama-server exposes some token counts; hidden-state access
            # through the serving API is not available.
            supports_hidden_states=False,
            supports_router_logits=False,
        )

    def generate(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = _requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
