"""Runtime compatibility probes (Section 15.4, 15.8).

These tests assert what the runtime layer can and cannot report
*without* requiring a GPU or a downloaded model. When a real serving
endpoint or Transformers install is present, the probes can be re-run
against it; otherwise they verify the contract itself.
"""

from pathlib import Path

import yaml

from tests_helpers import ROOT  # noqa: F401


def load_config() -> dict:
    with open(ROOT / "configs" / "prototype.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_mock_backend_reports_capabilities():
    from model_adapter import get_backend

    backend = get_backend("mock", load_config())
    caps = backend.capabilities()
    assert caps.supports_chat is True
    assert caps.supports_structured_json_output is True
    # The mock must not pretend to expose activations.
    assert caps.supports_hidden_states is False


def test_capability_flags_are_not_silently_faked():
    """A backend whose engine lacks an API must report False for it."""
    from model_adapter import get_backend

    cfg = dict(load_config())
    backend = get_backend("mock", cfg)
    assert backend.capabilities().supports_openai_compatible_serving is False


def test_transformers_backend_present_when_installed():
    import importlib.util

    installed = importlib.util.find_spec("transformers") is not None
    from model_adapter import AVAILABLE

    assert ("transformers" in AVAILABLE) == installed


def test_openai_compatible_probe_when_server_running():
    """Best-effort probe: if a local server answers, the adapter works."""
    import requests

    cfg = load_config()
    base = cfg.get("openai_base_url", "")
    try:
        resp = requests.get(f"{base}/models", timeout=3)
        server_up = resp.status_code == 200
    except Exception:
        server_up = False
    if not server_up:
        import pytest

        pytest.skip("no local OpenAI-compatible server on this host")
    from model_adapter import get_backend

    backend = get_backend("openai_compatible", cfg)
    out = backend.generate([{"role": "user", "content": "1"}])
    assert isinstance(out, str)
