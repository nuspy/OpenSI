"""Cognitive state model (the J-Space, Section 7).

Explicit, versioned, JSON-serializable, and bounded. Persisted to disk
after every loop step so a process restart can resume the same
cognitive state (acceptance criterion 5). Kept separate from: persistent
long-term memory, the inference KV cache, and ephemeral latent
activations.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

MAX_RECENT_OBSERVATIONS = 8
MAX_TOOL_HISTORY = 32


@dataclass
class CognitiveState:
    """The bounded cognitive state for one goal."""

    session_id: str
    goal: str
    state_version: int = 0
    success_criteria: str = ""
    open_questions: list[str] = field(default_factory=list)
    plan_summary: str = ""
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    budgets: dict[str, float] = field(default_factory=dict)
    stop_conditions: list[str] = field(default_factory=list)
    last_checkpoint: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = time.time()
        self.recent_observations = self.recent_observations[-MAX_RECENT_OBSERVATIONS:]
        self.tool_history = self.tool_history[-MAX_TOOL_HISTORY:]
        path = directory / f"{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json_dict(), fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, directory: str | Path, session_id: str) -> "CognitiveState":
        with open(Path(directory) / f"{session_id}.json", encoding="utf-8") as fh:
            return cls(**json.load(fh))


class StateStore:
    """Persist and resume cognitive states on disk."""

    def __init__(self, directory: str | Path = "data/sessions"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: CognitiveState) -> Path:
        return state.save(self.directory)

    def load(self, session_id: str) -> CognitiveState:
        return CognitiveState.load(self.directory, session_id)

    def exists(self, session_id: str) -> bool:
        return (self.directory / f"{self.session_id}.json").exists()


def new_state(goal: str, success_criteria: str = "",
              budgets: dict[str, float] | None = None) -> CognitiveState:
    return CognitiveState(
        session_id=uuid.uuid4().hex[:12],
        goal=goal,
        success_criteria=success_criteria,
        budgets=budgets or {},
    )
