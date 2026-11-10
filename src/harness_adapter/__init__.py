"""OpenSI harness adapter package.

The harness is the execution environment: it owns tool registries,
sandboxed execution, and the user-facing surface. It does not decide
the next cognitive step (Section 15.6). This package exposes
``ToolGateway`` plus the sandboxed built-in tools used by the Milestone
1-2 prototype.
"""

from .tool_gateway import ToolGateway, ToolResult, ToolProposal
from .selected_harness import HarnessAdapter
from .skill_loader import SkillLoader, Skill
from .tadamoo_catalog import TaDaMooCatalog
from .mcp_connector import (
    MCPManifest,
    MCPManifestLoader,
    BaseMCPClient,
    MockMCPClient,
    StdioMCPClient,
    HttpMCPClient,
    bridge_mcp_tools,
)

__all__ = [
    "ToolGateway",
    "ToolResult",
    "ToolProposal",
    "HarnessAdapter",
    "SkillLoader",
    "Skill",
    "TaDaMooCatalog",
    "MCPManifest",
    "MCPManifestLoader",
    "BaseMCPClient",
    "MockMCPClient",
    "StdioMCPClient",
    "HttpMCPClient",
    "bridge_mcp_tools",
]
