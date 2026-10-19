"""Event bus for external events and user messages.

The harness delivers events (user messages, tool completions, external
signals) to the cognitive controller through this bus. The controller
never polls the harness directly. Section 8: new user input has priority
but must not silently interrupt state checkpoints.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_QUEUED_EVENTS = 64


@dataclass
class Event:
    kind: str  # user_message | tool_result | external_signal | interruption
    payload: dict[str, Any]
    unix_time: float = field(default_factory=time.time)


class EventBus:
    """FIFO queue with priority for user interruptions."""

    def __init__(self) -> None:
        self._queue: deque[Event] = deque(maxlen=MAX_QUEUED_EVENTS)
        self._priority: deque[Event] = deque(maxlen=MAX_QUEUED_EVENTS)

    def publish(self, event: Event) -> None:
        if event.kind == "interruption":
            self._priority.append(event)
        else:
            self._queue.append(event)

    def pop(self) -> Event | None:
        if self._priority:
            return self._priority.popleft()
        if self._queue:
            return self._queue.popleft()
        return None

    def pending(self) -> int:
        return len(self._priority) + len(self._queue)

    def wait_for(self, predicate: Callable[[Event], bool], timeout_s: float) -> Event | None:
        """Non-blocking check used by WAIT_FOR_EVENT steps."""
        for queue in (self._priority, self._queue):
            for i, ev in enumerate(queue):
                if predicate(ev):
                    del queue[i]
                    return ev
        return None
