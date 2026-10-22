"""Tool gateway: the harness side of the runtime/harness boundary.

The cognitive controller submits tool requests through the gateway.
The gateway resolves the tool against the registry, enforces sandbox
roots, executes with a wall-clock timeout, and returns a structured
ToolResult. Results are bounded summaries: the harness returns
observations, and the controller treats them as data, never as
instructions (Section 8).

Built-in tools (Milestone 1: "one safe test tool and one mock
asynchronous event"):

- ``calculator``    evaluate a restricted arithmetic expression
- ``clock``         return current UTC time (mock async event target)
- ``file_read``     read a text file inside the sandbox roots
"""

from __future__ import annotations

import ast
import operator
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MAX_SUMMARY_CHARS = 400
_BIN_OPS = {
    ast.Mult: operator.mul,
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


@dataclass
class ToolResult:
    request_id: str
    tool: str
    arguments: dict[str, Any]
    status: str  # success | failure | timeout | cancelled | denied
    result_ref: str
    summary: str
    started_unix: float
    duration_s: float
    may_contain_instructions: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "status": self.status,
            "result_ref": self.result_ref,
            "summary": self.summary,
            "started_unix": self.started_unix,
            "duration_s": self.duration_s,
            "may_contain_instructions": self.may_contain_instructions,
        }


def _safe_eval_arithmetic(expression: str) -> int | float:
    """Evaluate a restricted arithmetic expression (no calls, names)."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST):
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError(f"unsupported expression node: {node!r}")


# ----------------------------------------------------------------------
# Built-in tools
def tool_calculator(arguments: dict[str, Any]) -> str:
    expr = str(arguments.get("expression", ""))
    if not expr:
        raise ValueError("calculator requires an 'expression' argument")
    return str(_safe_eval_arithmetic(expr))


def tool_clock(arguments: dict[str, Any]) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tool_file_read(arguments: dict[str, Any], sandbox_roots=()) -> str:
    raw = str(arguments.get("path", ""))
    norm = raw.replace("\\", "/").strip("/")
    roots = [r.replace("\\", "/").strip("/") for r in sandbox_roots] or ["data/eval_corpus"]
    allowed = any(norm.startswith(r + "/") or norm == r for r in roots)
    if not allowed:
        raise PermissionError(f"path {raw!r} outside sandbox roots {roots}")
    text = Path(norm).read_text(encoding="utf-8")
    return text.strip()


def tool_file_write(arguments: dict[str, Any], sandbox_roots=()) -> str:
    raw = str(arguments.get("path", ""))
    content = str(arguments.get("content", ""))
    norm = raw.replace("\\", "/").strip("/")
    roots = [r.replace("\\", "/").strip("/") for r in sandbox_roots] or ["data/eval_corpus"]
    allowed = any(norm.startswith(r + "/") or norm == r for r in roots)
    if not allowed:
        raise PermissionError(f"path {raw!r} outside sandbox roots {roots}")
    target_path = Path(norm)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {raw}"


def tool_web_fetch(arguments: dict[str, Any]) -> str:
    import urllib.request
    url = str(arguments.get("url", ""))
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Invalid URL protocol: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "OpenSI-Harness/0.1"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        content = resp.read(2048).decode("utf-8", errors="replace")
    return content.strip()


@dataclass
class ToolProposal:
    """A dynamic tool proposed by the model or runtime (Section 8.1)."""

    proposal_id: str
    name: str
    description: str
    tier: str
    proposed_by: str
    callable_fn: Callable[..., str] | None = None
    status: str = "pending"  # pending | approved | rejected
    rejection_reason: str = ""
    created_unix: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "name": self.name,
            "description": self.description,
            "tier": self.tier,
            "proposed_by": self.proposed_by,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "created_unix": self.created_unix,
        }


class SandboxedTool:
    """A registered tool with its name, callable, and permission tier."""

    def __init__(self, name: str, fn: Callable[..., str], tier: str = "auto", description: str = ""):
        self.name = name
        self.fn = fn
        self.tier = tier
        self.description = description

    def __call__(self, arguments: dict[str, Any]) -> str:
        return self.fn(arguments)


class ToolGateway:
    """Executes tool requests within policy and returns observations."""

    def __init__(self, config: dict[str, Any]):
        policy = config.get("policy", {})
        self.sandbox_roots = policy.get("sandbox_roots", ["data/eval_corpus"])
        self.tiers = dict(policy.get("tool_tiers", {}))
        self.timeout_s = float(policy.get("tool_timeout_s", 10))
        self._registry: dict[str, SandboxedTool] = {}
        self._proposals: dict[str, ToolProposal] = {}
        self.register_defaults()

    def register_defaults(self) -> None:
        self.register("calculator", tool_calculator, self.tiers.get("calculator", "auto"),
                      description="Evaluate arithmetic expressions")
        self.register("clock", tool_clock, self.tiers.get("clock", "auto"),
                      description="Get current UTC timestamp")
        self.register("file_read", self._wrap_file_read, self.tiers.get("file_read", "auto"),
                      description="Read text file inside sandbox")
        self.register("file_write", self._wrap_file_write,
                      self.tiers.get("file_write", "filesystem_write"),
                      description="Write text file inside sandbox (requires approval)")
        self.register("web_fetch", tool_web_fetch,
                      self.tiers.get("web_fetch", "network"),
                      description="Fetch URL content with bounded size (requires approval)")

    def _wrap_file_read(self, arguments):
        return tool_file_read(arguments, self.sandbox_roots)

    def _wrap_file_write(self, arguments):
        return tool_file_write(arguments, self.sandbox_roots)

    def register(self, name: str, fn: Callable[..., str], tier: str = "auto", description: str = "") -> None:
        self._registry[name] = SandboxedTool(name, fn, tier, description)

    def get_tool(self, name: str) -> SandboxedTool | None:
        return self._registry.get(name)

    def propose_tool(self, name: str, fn: Callable[..., str], description: str = "",
                     tier: str = "approval_required", proposed_by: str = "model") -> ToolProposal:
        """Submit a dynamic tool proposal (Section 8.1 lifecycle)."""
        prop_id = f"prop_{uuid.uuid4().hex[:8]}"
        proposal = ToolProposal(
            proposal_id=prop_id,
            name=name,
            description=description,
            tier=tier,
            proposed_by=proposed_by,
            callable_fn=fn,
        )
        self._proposals[prop_id] = proposal
        return proposal

    def list_proposals(self, status: str | None = None) -> list[ToolProposal]:
        if status:
            return [p for p in self._proposals.values() if p.status == status]
        return list(self._proposals.values())

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve and register a proposed tool."""
        prop = self._proposals.get(proposal_id)
        if not prop or prop.status != "pending" or not prop.callable_fn:
            return False
        prop.status = "approved"
        self.register(prop.name, prop.callable_fn, tier=prop.tier, description=prop.description)
        return True

    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        prop = self._proposals.get(proposal_id)
        if not prop or prop.status != "pending":
            return False
        prop.status = "rejected"
        prop.rejection_reason = reason
        return True

    def list_tools(self) -> list[str]:
        return sorted(self._registry)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        t0 = time.time()
        request_id = uuid.uuid4().hex[:12]
        if tool_name not in self._registry:
            return ToolResult(
                request_id=request_id, tool=tool_name, arguments=dict(arguments),
                status="denied", result_ref="", summary=f"unknown tool {tool_name!r}",
                started_unix=t0, duration_s=0.0,
            )
        try:
            raw = self._registry[tool_name](dict(arguments))
            status = "success"
            summary = str(raw)[:MAX_SUMMARY_CHARS]
        except Exception as exc:  # noqa: BLE001 - any tool failure is a failed observation
            status = "failure"
            summary = f"{type(exc).__name__}: {exc}"[:MAX_SUMMARY_CHARS]
        return ToolResult(
            request_id=request_id, tool=tool_name, arguments=dict(arguments),
            status=status, result_ref=f"tool:{tool_name}", summary=summary,
            started_unix=t0, duration_s=time.time() - t0,
        )
