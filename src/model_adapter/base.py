"""Base contract for inference backends.

A backend wraps one inference engine behind ``generate()`` and reports
its capabilities explicitly. The capability flags are the contract
(Section 15.6): unsupported capabilities must report false, never be
silently simulated, because the controller must know whether tool
observations can carry real model output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendCapabilities:
    supports_chat: bool = False
    supports_structured_json_output: bool = False
    supports_hidden_states: bool = False
    supports_router_logits: bool = False
    supports_custom_forward: bool = False
    supports_latent_recurrence_experiment: bool = False
    supports_openai_compatible_serving: bool = False
    supports_model_quantization: bool = False
    supports_resumeable_kv_cache: bool = False
    max_context_tokens_hint: int = 0
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}


class InferenceBackend(ABC):
    """Interface between the cognitive controller and an engine.

    ``generate`` receives chat-style messages (the serialized
    cognitive state plus the decision instruction) and returns the
    model's text. The backend executes; it never decides policy.
    """

    name: str = "abstract"

    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]]) -> str: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self.capabilities().as_dict()}
