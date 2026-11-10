"""MCP Connector and Manifest Loader (Section 4, 6C, and 8 of the handoff).

Provides:
1. `MCPManifestLoader`: Parses MCP manifest files (`manifest.yaml`) from
   a directory of manifests (the TaDaMoo servers come from `tadamoo_catalog`).
2. `MCPClient`: Pluggable MCP client interface supporting:
   - `MockMCPClient`: Deterministic mock client for offline tests and CI.
   - `StdioMCPClient`: JSON-RPC 2.0 communication over standard I/O subprocesses.
   - `HttpMCPClient`: JSON-RPC 2.0 over HTTP POST endpoints.
3. Gateway Bridging: Bridges MCP tools into OpenSI's `ToolGateway`, adhering
   to Section 8.1 dynamic tool proposal lifecycle and Section 9 security gates.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .tool_gateway import ToolGateway, ToolProposal

@dataclass
class MCPManifest:
    """Parsed representation of an MCP catalog entry."""

    name: str
    description: str
    manifest_version: int = 1
    transport_type: str = "http"  # http | stdio | sse | mock
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    auth_type: str = "none"  # none | oauth | apikey
    keywords: list[str] = field(default_factory=list)
    default_excluded: list[str] = field(default_factory=list)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "transport_type": self.transport_type,
            "url": self.url,
            "command": self.command,
            "auth_type": self.auth_type,
            "keywords": self.keywords,
            "default_excluded": self.default_excluded,
            "source_path": str(self.source_path) if self.source_path else None,
        }


class MCPManifestLoader:
    """Discovers and parses MCP manifests from catalog directories."""

    def __init__(self, catalog_dir: str | Path | None = None):
        self.catalog_dir = Path(catalog_dir) if catalog_dir else None
        self._manifests: dict[str, MCPManifest] = {}

    def parse_manifest_file(self, manifest_path: Path) -> MCPManifest | None:
        if not manifest_path.is_file():
            return None

        try:
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None

        name = str(raw.get("name") or manifest_path.parent.name)
        description = str(raw.get("description") or "")
        manifest_ver = int(raw.get("manifest_version") or 1)

        transport = raw.get("transport", {})
        transport_type = str(transport.get("type", "http")).lower()
        url = transport.get("url")
        command = transport.get("command")
        args = list(transport.get("args") or [])
        env = dict(transport.get("env") or {})

        auth = raw.get("auth", {})
        auth_type = str(auth.get("type", "none")).lower()

        suggest = raw.get("suggest", {})
        keywords = list(suggest.get("keywords") or [])

        tools_meta = raw.get("tools", {})
        default_excluded = list(tools_meta.get("default_excluded") or [])

        return MCPManifest(
            name=name,
            description=description,
            manifest_version=manifest_ver,
            transport_type=transport_type,
            url=url,
            command=command,
            args=args,
            env=env,
            auth_type=auth_type,
            keywords=keywords,
            default_excluded=default_excluded,
            source_path=manifest_path,
        )

    def discover(self) -> dict[str, MCPManifest]:
        self._manifests.clear()
        if self.catalog_dir is None or not self.catalog_dir.is_dir():
            return self._manifests

        for manifest_file in self.catalog_dir.rglob("manifest.yaml"):
            manifest = self.parse_manifest_file(manifest_file)
            if manifest:
                self._manifests[manifest.name] = manifest

        return self._manifests

    def get_manifest(self, name: str) -> MCPManifest | None:
        if not self._manifests:
            self.discover()
        return self._manifests.get(name)


class BaseMCPClient:
    """Abstract interface for interacting with an MCP server."""

    def list_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockMCPClient(BaseMCPClient):
    """Deterministic in-memory MCP client for testing and offline scenarios."""

    def __init__(self, tools_dict: dict[str, Callable[[dict[str, Any]], str]] | None = None):
        self._tools = dict(tools_dict or {})

    def register_mock_tool(self, name: str, fn: Callable[[dict[str, Any]], str], description: str = ""):
        self._tools[name] = {"fn": fn, "description": description}

    def list_tools(self) -> list[dict[str, Any]]:
        result = []
        for name, meta in self._tools.items():
            result.append({
                "name": name,
                "description": meta.get("description", f"Mock tool {name}"),
                "inputSchema": {"type": "object", "properties": {}},
            })
        return result

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name not in self._tools:
            raise ValueError(f"Unknown mock tool: {tool_name!r}")
        fn = self._tools[tool_name]["fn"]
        return str(fn(arguments))


class StdioMCPClient(BaseMCPClient):
    """Communicates with a local stdio MCP server using JSON-RPC 2.0."""

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.command = command
        self.args = args or []
        self.env = env
        self._process: subprocess.Popen | None = None
        self._request_counter = 0

    def _ensure_process(self) -> subprocess.Popen:
        if self._process is None or self._process.poll() is not None:
            full_cmd = [self.command] + self.args
            merged_env = None
            if self.env:
                import os
                merged_env = os.environ.copy()
                merged_env.update(self.env)
            self._process = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=merged_env,
                bufsize=1,
            )
            self._init_session()
        return self._process

    def _send_rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        proc = self._ensure_process()
        self._request_counter += 1
        req_id = self._request_counter
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        line = json.dumps(payload) + "\n"
        assert proc.stdin is not None
        proc.stdin.write(line)
        proc.stdin.flush()

        assert proc.stdout is not None
        resp_line = proc.stdout.readline()
        if not resp_line:
            raise RuntimeError(f"Stdio MCP server closed connection on method {method!r}")
        return json.loads(resp_line)

    def _init_session(self) -> None:
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "opensi-mcp-connector", "version": "0.1.0"},
        }
        self._send_rpc("initialize", init_params)

    def list_tools(self) -> list[dict[str, Any]]:
        res = self._send_rpc("tools/list")
        if "error" in res:
            raise RuntimeError(f"tools/list error: {res['error']}")
        return res.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        res = self._send_rpc("tools/call", {"name": tool_name, "arguments": arguments})
        if "error" in res:
            raise RuntimeError(f"tools/call error: {res['error']}")
        content_items = res.get("result", {}).get("content", [])
        texts = [item.get("text", "") for item in content_items if isinstance(item, dict)]
        return "\n".join(texts) if texts else str(res.get("result", ""))

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


class HttpMCPClient(BaseMCPClient):
    """Communicates with a remote MCP endpoint over HTTP JSON-RPC."""

    def __init__(self, endpoint_url: str, timeout_s: float = 10.0, headers: dict[str, str] | None = None):
        self.endpoint_url = endpoint_url
        self.timeout_s = timeout_s
        self.headers = {"Content-Type": "application/json", "User-Agent": "OpenSI-MCP/0.1"}
        if headers:
            self.headers.update(headers)
        self._request_counter = 0

    def _send_rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_counter += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_counter,
            "method": method,
            "params": params or {},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint_url, data=data, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def list_tools(self) -> list[dict[str, Any]]:
        res = self._send_rpc("tools/list")
        if "error" in res:
            raise RuntimeError(f"HTTP MCP tools/list error: {res['error']}")
        return res.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        res = self._send_rpc("tools/call", {"name": tool_name, "arguments": arguments})
        if "error" in res:
            raise RuntimeError(f"HTTP MCP tools/call error: {res['error']}")
        content_items = res.get("result", {}).get("content", [])
        texts = [item.get("text", "") for item in content_items if isinstance(item, dict)]
        return "\n".join(texts) if texts else str(res.get("result", ""))


def bridge_mcp_tools(
    client: BaseMCPClient,
    gateway: ToolGateway,
    server_name: str = "mcp",
    tier: str = "approval_required",
    propose_only: bool = True,
) -> list[ToolProposal | str]:
    """Discover tools from client and register/propose them in ToolGateway."""
    tools = client.list_tools()
    results: list[ToolProposal | str] = []

    for t in tools:
        t_name = t.get("name", "")
        if not t_name:
            continue
        scoped_name = f"{server_name}_{t_name}"
        desc = t.get("description", f"MCP tool {t_name} from {server_name}")

        def make_caller(name: str):
            return lambda args: client.call_tool(name, args)

        caller_fn = make_caller(t_name)

        if propose_only:
            prop = gateway.propose_tool(
                name=scoped_name,
                fn=caller_fn,
                description=desc,
                tier=tier,
                proposed_by=f"mcp:{server_name}",
            )
            results.append(prop)
        else:
            gateway.register(scoped_name, caller_fn, tier=tier, description=desc)
            results.append(scoped_name)

    return results
