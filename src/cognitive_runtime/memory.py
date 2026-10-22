"""Cross-session persistent memory store (Section 7 of the handoff).

Maintains long-term durable knowledge (facts, skill references, heuristics)
separately from the active working state (CognitiveState), the serving KV-cache,
and ephemeral latent activations. Cognitive states reference memory items via
compact persistent_memory_refs (IDs), avoiding database bloat in active prompts.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    ref_id: str
    key: str
    content: str
    category: str = "fact"  # fact | skill | heuristic | environment
    confidence: float = 1.0
    created_unix: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersistentMemoryStore:
    """Manages cross-session knowledge persisted as plain JSON."""

    def __init__(self, directory: str | Path = "data/memory"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file_path = self.directory / "knowledge.json"
        self._items: dict[str, MemoryItem] = {}
        self.load()

    def store(
        self,
        key: str,
        content: str,
        category: str = "fact",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Store or update a memory item. Returns the persistent MemoryItem."""
        # Check if an existing item matches the key
        for item in self._items.values():
            if item.key.lower() == key.lower():
                item.content = content
                item.category = category
                item.confidence = confidence
                item.last_accessed = time.time()
                item.access_count += 1
                if metadata:
                    item.metadata.update(metadata)
                self.save()
                return item

        ref_id = f"mem_{uuid.uuid4().hex[:8]}"
        item = MemoryItem(
            ref_id=ref_id,
            key=key,
            content=content,
            category=category,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._items[ref_id] = item
        self.save()
        return item

    def get(self, ref_id: str) -> MemoryItem | None:
        item = self._items.get(ref_id)
        if item:
            item.last_accessed = time.time()
            item.access_count += 1
        return item

    def query(self, query_text: str, limit: int = 5) -> list[MemoryItem]:
        """Search memory items matching query tokens in key or content."""
        tokens = [t.lower() for t in query_text.split() if len(t) > 2]
        scored = []
        for item in self._items.values():
            score = 0
            text_pool = f"{item.key} {item.content} {item.category}".lower()
            for token in tokens:
                if token in text_pool:
                    score += 1
            if score > 0 or not tokens:
                scored.append((score, item))

        scored.sort(key=lambda x: (x[0], x[1].access_count), reverse=True)
        results = [item for _, item in scored[:limit]]
        for item in results:
            item.last_accessed = time.time()
            item.access_count += 1
        return results

    def register_skill_ref(self, skill_name: str, description: str, path_or_uri: str) -> MemoryItem:
        return self.store(
            key=f"skill:{skill_name}",
            content=description,
            category="skill",
            metadata={"path_or_uri": path_or_uri},
        )

    def list_all(self) -> list[MemoryItem]:
        return list(self._items.values())

    def save(self) -> Path:
        data = {ref_id: item.to_dict() for ref_id, item in self._items.items()}
        with open(self.file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return self.file_path

    def load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._items = {
                ref_id: MemoryItem(**item_dict)
                for ref_id, item_dict in data.items()
            }
        except Exception:
            self._items = {}
