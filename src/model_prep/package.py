"""Model packaging utility (Milestone 3, Section 6.B).

Packages runtime configurations, sidecar metadata, and reproducible manifests
for a model into a distinct output directory. It never modifies original
weights.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .inspect import inspect_model
from .manifest import write_manifest
from .validate import validate_model


def package_config(
    model_ref: str,
    output_dir: str | Path,
    profile: str = "recurrent-prototype",
    backend: str = "transformers",
) -> dict[str, Path]:
    """Package runtime configuration and sidecar metadata for a model into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    inspection = inspect_model(model_ref)
    validation = validate_model(model_ref, backend=backend, inspection=inspection)

    # 1. Write manifest
    manifest_path = write_manifest(model_ref, out, profile=profile)

    # 2. Generate runtime configuration yaml
    max_ctx = inspection.max_position_embeddings or 32768
    suggested_steps = 12 if inspection.is_moe() else 8

    runtime_cfg = {
        "model": str(model_ref),
        "backend": backend,
        "profile": profile,
        "model_architecture": inspection.architecture,
        "is_moe": inspection.is_moe(),
        "budgets": {
            "max_steps": suggested_steps,
            "max_tool_calls": 6,
            "max_seconds": 90,
            "max_internal_streak": 4,
            "max_repairs": 2,
        },
        "limits": {
            "max_context_tokens": max_ctx,
            "max_generation_tokens": 512,
            "temperature": 0.1,
        },
        "policy": {
            "sandbox_roots": ["data/eval_corpus"],
            "tool_tiers": {
                "calculator": "auto",
                "clock": "auto",
                "file_read": "auto",
            },
            "tool_timeout_s": 15,
        },
    }

    config_path = out / "runtime_config.yaml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime_cfg, fh, sort_keys=False)

    # 3. Generate OpenSI sidecar metadata
    sidecar_data = {
        "schema": "opensi.sidecar/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_ref": str(model_ref),
        "profile": profile,
        "validation_passed": validation.valid,
        "validation_warnings": validation.warnings,
        "validation_errors": validation.errors,
        "capabilities": {
            "is_moe": inspection.is_moe(),
            "num_experts": inspection.num_experts,
            "num_experts_per_tok": inspection.num_experts_per_tok,
            "supports_recurrent_inference": validation.valid,
        },
        "notes": (
            "Generated sidecar metadata for OpenSI cognitive recurrence. "
            "Original model weights were not altered."
        ),
    }

    sidecar_path = out / "opensi_sidecar.json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar_data, fh, indent=2)

    return {
        "manifest": manifest_path,
        "runtime_config": config_path,
        "sidecar": sidecar_path,
    }
