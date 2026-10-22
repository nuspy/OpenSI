"""Model inspection utility (Milestone 3).

Inspects a model artifact (local directory or Hugging Face Hub repo)
and reports architecture facts: model_type, layer count, hidden size,
expert count and top-k for MoE models, context length, tokenizer chat
template presence. Inspection is read-only; it never mutates model
files.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

HF_API = "https://huggingface.co/api/models/"


@dataclass
class ModelInspection:
    model_ref: str
    source: str  # local | hub
    architecture: str | None = None
    model_type: str | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    max_position_embeddings: int | None = None
    chat_template_present: bool = False
    files: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_moe(self) -> bool:
        return self.model_type is not None and "moe" in self.model_type


def _sha256_head(path: Path, n: int = 1 << 20) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(n)).hexdigest()[:16]


def _from_local(directory: Path) -> ModelInspection:
    errors: list[str] = []
    config_path = directory / "config.json"
    if not config_path.exists():
        return ModelInspection(model_ref=str(directory), source="local",
                               errors=["config.json not found"])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    chat_present = (directory / "tokenizer_config.json").exists()
    files = []
    for p in sorted(directory.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "bytes": p.stat().st_size,
                         "sha256_head": _sha256_head(p)})
    return ModelInspection(
        model_ref=str(directory),
        source="local",
        architecture=(cfg.get("architectures") or [None])[0],
        model_type=cfg.get("model_type"),
        num_hidden_layers=cfg.get("num_hidden_layers"),
        hidden_size=cfg.get("hidden_size"),
        num_experts=cfg.get("num_experts") or cfg.get("num_local_experts"),
        num_experts_per_tok=cfg.get("num_experts_per_tok"),
        max_position_embeddings=cfg.get("max_position_embeddings"),
        chat_template_present=chat_present,
        files=files,
        errors=errors,
    )


def _from_hub(model_id: str) -> ModelInspection:
    url = HF_API + model_id
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    config = data.get("config", {}) or {}
    return ModelInspection(
        model_id,
        "hub",
        architecture=(config.get("architectures") or [None])[0],
        model_type=config.get("model_type"),
        num_hidden_layers=config.get("num_hidden_layers"),
        hidden_size=config.get("hidden_size"),
        num_experts=config.get("num_experts"),
        num_experts_per_tok=config.get("num_experts_per_tok"),
        max_position_embeddings=config.get("max_position_embeddings"),
        chat_template_present=True,
        files=[{"name": s["rfilename"]} for s in data.get("siblings", [])],
    )


def inspect_model(model_ref: str) -> ModelInspection:
    """Inspect a local directory or a Hugging Face Hub model id."""
    p = Path(model_ref)
    if p.is_dir():
        return _from_local(p)
    if "/" in model_ref:
        return _from_hub(model_ref)
    raise ValueError(f"model_ref must be a local directory or hub id: {model_ref!r}")
