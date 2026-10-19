"""Security and budget policy enforcement (Section 9 of the handoff).

The LLM may decide *what* to do next; this module decides whether that
action is *allowed*. Policy code, not the model, enforces:

- per-step and per-session budgets (steps, tool calls, seconds)
- maximum consecutive internal iterations
- tool allowlist and permission tiers
- approval gates for destructive / irreversible actions
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyConfig:
    sandbox_roots: list[str] = field(default_factory=lambda: ["data"])
    tool_tiers: dict[str, str] = field(default_factory=dict)  # tool -> auto | approval_required
    approval_required_tiers: list[str] = field(
        default_factory=lambda: ["filesystem_write", "network", "shell"]
    )
    max_internal_streak: int = 3


@dataclass
class PermissionDecision:
    allowed: bool
    requires_approval: bool
    reason: str


class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self.config = config

    def check_tool(self, tool_name: str, state_version: int) -> PermissionDecision:
        tier = self.config.tool_tiers.get(tool_name, "approval_required")
        if tier == "auto":
            return PermissionDecision(True, False, "tier=auto")
        if tier in self.config.approval_required_tiers or tier == "approval_required":
            return PermissionDecision(True, True, f"tier={tier} requires approval")
        return PermissionDecision(False, False, f"unknown tier {tier}")

    def check_path(self, path: str) -> PermissionDecision:
        norm = path.replace("\\", "/").lstrip("./")
        for root in self.config.sandbox_roots:
            root_norm = root.replace("\\", "/").strip("/")
            if norm.startswith(root_norm + "/") or norm == root_norm:
                return PermissionDecision(True, False, "inside sandbox")
        return PermissionDecision(False, False, "path outside sandbox roots")
