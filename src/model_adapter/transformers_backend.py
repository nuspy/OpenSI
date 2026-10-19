"""Transformers backend adapter (optional, requires torch + transformers).

Used when running the model natively in-process. Import is lazy so
that systems without torch installed can still import the package
and fall back to the mock backend.
"""

from __future__ import annotations

from typing import Any

from .base import InferenceBackend, BackendCapabilities

_HAS_TORCH = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    pass


class TransformersBackend(InferenceBackend):
    name = "transformers"

    def __init__(self, config: dict[str, Any]):
        if not _HAS_TORCH:
            raise RuntimeError(
                "transformers/torch not installed; install them or use the mock backend"
            )
        self.model_id = config.get("model", "")
        self.device = config.get("device", "cuda" if _HAS_TORCH else "cpu")
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto"
            )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_chat=True,
            supports_structured_json_output=False,
            supports_hidden_states=_HAS_TORCH,
            supports_router_logits=_HAS_TORCH,
            supports_custom_forward=_HAS_TORCH,
            supports_latent_recurrence_experiment=False,
        )

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.load()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=256)
        raw = self._tokenizer.decode(out[0][inputs.input_ids.shape[1] :],
                                    skip_special_tokens=False)
        return raw
