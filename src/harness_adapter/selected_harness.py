"""Selected harness adapter: Hermes Agent integration (Milestone 1/M0).

Adapts the upstream Hermes Agent environment (cloned in external/hermes-agent)
to act as a passive tool, event, and execution substrate for the OpenSI
cognitive controller. The controller retains ownership of planning and reasoning;
the harness performs execution, approval checks, and event delivery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .tool_gateway import ToolGateway, ToolResult, ToolProposal
from .skill_loader import SkillLoader, HermesSkill
from .mcp_connector import (
    MCPManifestLoader,
    MCPManifest,
    BaseMCPClient,
    bridge_mcp_tools,
)

ROOT = Path(__file__).resolve().parents[2]
HERMES_ROOT = ROOT / "external" / "hermes-agent"


class HarnessAdapter:
    """Passive harness adapter connecting OpenSI to Hermes Agent tool capabilities."""

    def __init__(self, config: dict[str, Any], event_bus=None):
        self.config = config
        self.gateway = ToolGateway(config)
        self.event_bus = event_bus
        self.hermes_path = HERMES_ROOT

        skills_dir = self.hermes_path / "skills" if self.is_hermes_available() else None
        self.skills = SkillLoader(skills_dir)

        mcps_dir = self.hermes_path / "optional-mcps" if self.is_hermes_available() else None
        self.mcps = MCPManifestLoader(mcps_dir)

        self._initialize_hermes_tools()

    def is_hermes_available(self) -> bool:
        """Check if the external Hermes Agent repo is present on disk."""
        return self.hermes_path.is_dir() and (self.hermes_path / "tools").is_dir()

    def _initialize_hermes_tools(self) -> None:
        """Register custom or bridged Hermes tools into the ToolGateway."""
        if not self.is_hermes_available():
            return
        # Hermes tools directory is recognized and ready for bridging.

    def load_skills(self) -> dict[str, HermesSkill]:
        """Discover and return all skills from the harness."""
        return self.skills.discover_skills()

    def bridge_skill(
        self,
        skill_name: str,
        tier: str = "approval_required",
        propose_only: bool = True,
    ) -> ToolProposal | None:
        """Bridge a specific skill into the gateway."""
        return self.skills.bridge_to_gateway(skill_name, self.gateway, tier=tier, propose_only=propose_only)

    def load_mcps(self) -> dict[str, MCPManifest]:
        """Discover and return all MCP catalog entries."""
        return self.mcps.discover()

    def connect_mcp(
        self,
        client: BaseMCPClient,
        server_name: str = "mcp",
        tier: str = "approval_required",
        propose_only: bool = True,
    ) -> list[ToolProposal | str]:
        """Bridge tools from an active MCP client into ToolGateway."""
        return bridge_mcp_tools(client, self.gateway, server_name=server_name, tier=tier, propose_only=propose_only)

    def propose_tool(self, name: str, fn: Callable[..., str], description: str = "",
                     tier: str = "approval_required", proposed_by: str = "model"):
        return self.gateway.propose_tool(name, fn, description=description, tier=tier, proposed_by=proposed_by)

    def list_proposals(self, status: str | None = None):
        return self.gateway.list_proposals(status=status)

    def approve_proposal(self, proposal_id: str) -> bool:
        return self.gateway.approve_proposal(proposal_id)

    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        return self.gateway.reject_proposal(proposal_id, reason=reason)

    def register_tool(self, name: str, fn: Callable[..., str], tier: str = "auto", description: str = "") -> None:
        self.gateway.register(name, fn, tier=tier, description=description)

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.gateway.execute(tool_name, arguments)

    def inject_event(self, kind: str, payload: Any) -> None:
        """Inject an external event (e.g., user interruption, async task completion)."""
        if self.event_bus is not None:
            data = payload if isinstance(payload, dict) else {"content": str(payload)}
            from cognitive_runtime.events import Event
            self.event_bus.publish(Event(kind=kind, payload=data))

    def list_tools(self) -> list[str]:
        return self.gateway.list_tools()
