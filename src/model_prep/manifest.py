"""Manifest generation (Milestone 3).

Produces a reproducible manifest describing a model artifact: base
files, sizes, content hashes (first megabyte), and architecture facts
from ``inspect_model``. The manifest is written to a separate output
directory; original model files are never modified.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inspect import inspect_model


def write_manifest(model_ref: str, output_dir: str | Path,
                   profile: str = "recurrent-prototype") -> Path:
    inspection = inspect_model(model_ref)
    manifest = {
        "schema": "opensi.model.manifest/v1",
        "generated_unix": datetime.now(timezone.utc).timestamp(),
        "profile": profile,
        **inspection.to_dict(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = (str(model_ref).replace("/", "__").replace("\\", "__")
            if not str(model_ref).startswith("/") else Path(model_ref).name)
    path = output_dir / f"{name}.manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
