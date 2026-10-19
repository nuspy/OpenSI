"""OpenSI cognitive runtime package.

Owns the cognitive state (the J-Space), the recurrent control loop,
event intake, budget policy, and tracing. It does not embed
model-specific tensor logic (that lives in ``model_adapter``) and does
not execute tool side effects itself (that lives in
``harness_adapter``).
"""

from .state import CognitiveState, StateStore, new_state
from .events import EventBus, Event
from .policies import PolicyEngine, PolicyConfig, PermissionDecision
from .tracing import Tracer
from .controller import CognitiveController, RunOutcome, parse_decision, ParseError

__all__ = [
    "CognitiveState", "StateStore", "new_state",
    "EventBus", "Event",
    "PolicyEngine", "PolicyConfig", "PermissionDecision",
    "Tracer",
    "CognitiveController", "RunOutcome", "parse_decision", "ParseError",
]
