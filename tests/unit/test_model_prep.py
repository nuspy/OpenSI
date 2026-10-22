"""Unit tests for model_prep inspection and manifests."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from model_prep.inspect import inspect_model, _from_hub, ModelInspection
from model_prep.manifest import write_manifest, read_manifest


def test_inspect_local_model(tmp_path):
    model_dir = tmp_path / "MyOrg/TestModel"
    model_dir.mkdir(parents=True)
    config_data = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 32768,
    }
    (model_dir / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    info = inspect_model(str(model_dir))
    assert info.source == "local"
    assert info.architecture == "Qwen3MoeForCausalLM"
    assert info.is_moe() is True
    assert info.num_experts == 64
    assert info.chat_template_present is True
    assert len(info.files) == 3


def test_inspect_hub_model_mocked():
    fake_hf_response = {
        "config": {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_hidden_layers": 28,
            "hidden_size": 2048,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 32768,
        },
        "siblings": [{"rfilename": "model-00001-of-00004.safetensors"}],
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_hf_response).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        info = inspect_model("Qwen/Qwen3-30B-A3B")
        assert info.source == "hub"
        assert info.model_ref == "Qwen/Qwen3-30B-A3B"
        assert info.num_experts == 128
        assert info.is_moe() is True
        assert len(info.files) == 1


def test_inspect_invalid_model_ref():
    with pytest.raises(ValueError, match="must be a local directory or hub id"):
        inspect_model("invalid_model_without_slash")


def test_validate_model_valid(tmp_path):
    from model_prep.validate import validate_model

    model_dir = tmp_path / "MyOrg/ValidMoE"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 32768,
    }), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text('{"chat_template": "yes"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 128)

    val = validate_model(str(model_dir), backend="transformers")
    assert val.valid is True
    assert val.is_moe is True
    assert len(val.errors) == 0


def test_validate_model_negative_cases(tmp_path):
    from model_prep.validate import validate_model

    # Negative 1: Missing weights
    model_dir = tmp_path / "MyOrg/BrokenModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    }), encoding="utf-8")

    val_no_weights = validate_model(str(model_dir), backend="transformers")
    assert val_no_weights.valid is False
    assert any("No weights files" in e for e in val_no_weights.errors)

    # Negative 2: Invalid MoE configuration (top_k > experts)
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 64)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_experts": 4,
        "num_experts_per_tok": 8,  # invalid: 8 > 4
    }), encoding="utf-8")

    val_bad_moe = validate_model(str(model_dir), backend="transformers")
    assert val_bad_moe.valid is False
    assert any("exceeds total num_experts" in e for e in val_bad_moe.errors)


def test_package_config(tmp_path):
    from model_prep.package import package_config

    model_dir = tmp_path / "MyOrg/PackableModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "max_position_embeddings": 8192,
    }), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 64)

    out_dir = tmp_path / "packaged_output"
    artifacts = package_config(str(model_dir), out_dir, profile="test-profile")

    assert artifacts["manifest"].exists()
    assert artifacts["runtime_config"].exists()
    assert artifacts["sidecar"].exists()


def test_cli_subcommands(tmp_path, capsys):
    from model_prep.cli import main

    model_dir = tmp_path / "MyOrg/CLIModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 16,
        "hidden_size": 1024,
        "max_position_embeddings": 4096,
    }), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 32)

    # CLI inspect
    assert main(["inspect", str(model_dir)]) == 0
    captured = capsys.readouterr().out
    assert "Architecture: LlamaForCausalLM" in captured

    # CLI validate
    assert main(["validate", str(model_dir), "--backend", "transformers"]) == 0

    # CLI package-config
    out_dir = tmp_path / "cli_pkg"
    assert main(["package-config", str(model_dir), "--out", str(out_dir)]) == 0
    assert (out_dir / "runtime_config.yaml").exists()

