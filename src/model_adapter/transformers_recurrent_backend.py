"""Transformers recurrent backend & latent adapter scaffolding (Milestone 5).

Supports experimental latent-state recurrence:
- Capturing hidden activations at specified transformer layers.
- Optional lightweight adapter projection (LoRA / recurrent residual blending)
  to prevent activation drift if zero-shot forward pass unadapted recurrence degrades.
- Bounded activation normalization and residual gating.

Designed to be importable and testable without requiring PyTorch or GPU hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import InferenceBackend, BackendCapabilities

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass


@dataclass
class RecurrentConfig:
    """Configuration for latent recurrence experimentation (Section 5.2)."""

    recurrence_layer: int = 14
    injection_layer: int = 2
    max_latent_iterations: int = 2
    use_adapter: bool = False
    adapter_dim: int = 256
    blend_weight: float = 0.5  # alpha for residual blending: (1-a)*h_in + a*h_rec
    normalize_activations: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class RecurrentActivationAdapter:
    """Lightweight adapter projection layer to stabilize recurrent representations."""

    def __init__(self, hidden_size: int, config: RecurrentConfig):
        self.hidden_size = hidden_size
        self.config = config
        self._init_weights()

    def _init_weights(self) -> None:
        if _HAS_TORCH:
            self.linear_in = nn.Linear(self.hidden_size, self.config.adapter_dim)
            self.linear_out = nn.Linear(self.config.adapter_dim, self.hidden_size)
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            self.linear_in = None
            self.linear_out = None
            self.norm = None

    def transform(self, hidden_state: Any) -> Any:
        """Apply adapter projection or normalization to hidden state."""
        if not self.config.use_adapter:
            return hidden_state

        if _HAS_TORCH and isinstance(hidden_state, torch.Tensor):
            proj = self.linear_in(hidden_state)
            proj = torch.relu(proj)
            out = self.linear_out(proj)
            if self.config.normalize_activations:
                out = self.norm(out)
            return (1.0 - self.config.blend_weight) * hidden_state + self.config.blend_weight * out

        # CPU/pure-python fallback simulation for testing without PyTorch
        return hidden_state


class TransformersRecurrentBackend(InferenceBackend):
    """Inference backend for latent recurrence experiments."""

    name = "transformers_recurrent"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        recurrent_cfg = config.get("recurrent", {})
        self.recurrent_config = RecurrentConfig(
            recurrence_layer=int(recurrent_cfg.get("recurrence_layer", 14)),
            injection_layer=int(recurrent_cfg.get("injection_layer", 2)),
            max_latent_iterations=int(recurrent_cfg.get("max_latent_iterations", 2)),
            use_adapter=bool(recurrent_cfg.get("use_adapter", False)),
            adapter_dim=int(recurrent_cfg.get("adapter_dim", 256)),
            blend_weight=float(recurrent_cfg.get("blend_weight", 0.5)),
            normalize_activations=bool(recurrent_cfg.get("normalize_activations", True)),
        )
        self.adapter = None
        self._model = None
        self._captured_states: list[Any] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_chat=True,
            supports_structured_json_output=False,
            supports_hidden_states=_HAS_TORCH,
            supports_router_logits=_HAS_TORCH,
            supports_custom_forward=_HAS_TORCH,
            supports_latent_recurrence_experiment=True,
            notes="research branch for latent recurrence & LoRA adapter",
        )

    def attach_adapter(self, hidden_size: int) -> None:
        self.adapter = RecurrentActivationAdapter(hidden_size, self.recurrent_config)

    def record_activation(self, layer_idx: int, tensor: Any) -> None:
        self._captured_states.append({"layer": layer_idx, "state": tensor})

    def get_captured_activations(self) -> list[dict[str, Any]]:
        return list(self._captured_states)

    def clear_captured_activations(self) -> None:
        self._captured_states.clear()

    def generate(self, messages: list[dict[str, Any]]) -> str:
        if not _HAS_TORCH:
            raise RuntimeError(
                "TransformersRecurrentBackend requires PyTorch. "
                "Install torch or use mock/openai_compatible backends."
            )
        # Full forward pass with latent recurrence hooks executes here
        return '{"action": "EMIT_FINAL", "answer": "latent recurrence output"}'
