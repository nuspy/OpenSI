"""Model adapter layer.

Wraps inference engines behind one ``InferenceBackend`` interface so
the cognitive controller never depends on a particular runtime
(Section 15.6). Backends declare explicit capability flags;
unsupported capabilities must report ``False``, never silently fake
support.
"""

from .base import InferenceBackend, BackendCapabilities
from .mock_backend import MockBackend

__all__ = ["InferenceBackend", "BackendCapabilities", "MockBackend", "get_backend"]

AVAILABLE = {}


def _lazy_register() -> None:
    """Register backends whose heavy imports succeeded."""
    global AVAILABLE
    if "mock" not in AVAILABLE:
        AVAILABLE["mock"] = MockBackend
    try:
        from .openai_compatible_backend import OpenAICompatibleBackend

        AVAILABLE["openai_compatible"] = OpenAICompatibleBackend
    except Exception:  # pragma: no cover - requests missing
        pass
    try:
        from .transformers_backend import TransformersBackend

        AVAILABLE.setdefault("transformers", TransformersBackend)
    except Exception:  # pragma: no cover
        pass


def get_backend(name: str, config: dict) -> InferenceBackend:
    _lazy_register()
    try:
        cls = AVAILABLE[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; options: {sorted(AVAILABLE)}")
    return cls(config)
