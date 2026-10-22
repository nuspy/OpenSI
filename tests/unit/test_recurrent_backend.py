"""Unit tests for TransformersRecurrentBackend & Latent Adapter (Milestone 5)."""

import pytest
from model_adapter import get_backend
from model_adapter.transformers_recurrent_backend import (
    TransformersRecurrentBackend,
    RecurrentConfig,
    RecurrentActivationAdapter,
)


def test_recurrent_backend_capabilities():
    cfg = {
        "recurrent": {
            "recurrence_layer": 16,
            "injection_layer": 4,
            "use_adapter": True,
            "adapter_dim": 128,
        }
    }
    backend = get_backend("transformers_recurrent", cfg)
    assert isinstance(backend, TransformersRecurrentBackend)

    caps = backend.capabilities()
    assert caps.supports_latent_recurrence_experiment is True
    assert backend.recurrent_config.recurrence_layer == 16
    assert backend.recurrent_config.injection_layer == 4
    assert backend.recurrent_config.use_adapter is True


def test_latent_adapter_instantiation_cpu():
    cfg = RecurrentConfig(use_adapter=True, adapter_dim=64, blend_weight=0.3)
    adapter = RecurrentActivationAdapter(hidden_size=512, config=cfg)
    assert adapter.hidden_size == 512
    assert adapter.config.use_adapter is True

    # Test transform on dummy representation
    dummy_vec = [0.1] * 512
    transformed = adapter.transform(dummy_vec)
    assert transformed == dummy_vec


def test_activation_recording():
    cfg = {"recurrent": {}}
    backend = TransformersRecurrentBackend(cfg)

    backend.record_activation(12, "mock_tensor_layer_12")
    acts = backend.get_captured_activations()
    assert len(acts) == 1
    assert acts[0]["layer"] == 12
    assert acts[0]["state"] == "mock_tensor_layer_12"

    backend.clear_captured_activations()
    assert len(backend.get_captured_activations()) == 0
