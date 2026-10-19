"""OpenSI harness adapter package.

The harness is the execution environment: it owns tool registries,
sandboxed execution, and the user-facing surface. It does not decide
the next cognitive step (Section 15.6). This package exposes
``ToolGateway`` plus the sandboxed built-in tools used by the Milestone
1-2 prototype.
"""

from .tool_gateway import ToolGateway, ToolResult

__all__ = ["ToolGateway", "ToolResult"]
