"""Trace events for evaluation and diagnosis.

Acceptance criterion 10: every loop iteration must log enough to
explain *why* the cycle continued or stopped. Traces are append-only
JSONL under ``data/traces/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, directory: str | Path = "data/traces"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def event(self, session_id: str, kind: str, data: dict[str, Any]) -> None:
        path = self.directory / f"{session_id}.jsonl"
        record = {"unix": time.time(), "kind": kind, **data}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read(self, session_id: str) -> list[dict[str, Any]]:
        path = self.directory / f"{session_id}.jsonl"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
