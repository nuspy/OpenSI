"""TaDaMoo's catalog of skills and MCP servers, read from its user data folder.

TaDaMoo keeps what it has installed under its data folder: `capabilities/skills`
holds `SKILL.md` skills, and `capabilities/installed.json` lists the MCP servers
it installed, each with the command that starts it. This module reads those
files; it imports none of TaDaMoo's code and starts nothing. TaDaMoo's personal
skills (`personal-skills`, a `skill.json` and a `skill.py`) are a different
format and are not read.

The data folder is, in order: the path given, `$TADAMOO_DATA`, then the
platform default (`%APPDATA%\\TaDaMoo` on Windows, `~/Library/Application
Support/TaDaMoo` on macOS, `$XDG_DATA_HOME/TaDaMoo` or `~/.local/share/TaDaMoo`
on Linux).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .mcp_connector import MCPManifest

APP_DIR = "TaDaMoo"
INSTALLED = Path("capabilities") / "installed.json"
SKILL_FOLDERS = (Path("capabilities") / "skills",)


def default_data_dir(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("TADAMOO_DATA"):
        return Path(env["TADAMOO_DATA"])
    home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
    if sys.platform == "win32":
        return Path(env.get("APPDATA") or home / "AppData" / "Roaming") / APP_DIR
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_DIR
    return Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / APP_DIR


class TaDaMooCatalog:
    """Read-only view of the skills and MCP servers TaDaMoo has installed."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()

    def is_available(self) -> bool:
        """The data folder has a skill folder or an installed-servers list."""
        return any((self.data_dir / f).is_dir() for f in SKILL_FOLDERS) or (self.data_dir / INSTALLED).is_file()

    def skill_dirs(self) -> list[Path]:
        """The skill folders that exist."""
        return [self.data_dir / f for f in SKILL_FOLDERS if (self.data_dir / f).is_dir()]

    def mcp_manifests(self) -> dict[str, MCPManifest]:
        """The installed MCP servers as manifests; a server TaDaMoo lists with a command runs over stdio."""
        path = self.data_dir / INSTALLED
        try:
            items = (json.loads(path.read_text(encoding="utf-8")) or {}).get("items") or {}
        except (OSError, ValueError):
            return {}
        out: dict[str, MCPManifest] = {}
        for item_id, item in items.items():
            if not isinstance(item, dict) or not str(item_id).startswith("mcp:"):
                continue
            manifest = self._manifest(str(item_id)[len("mcp:"):], item, path)
            if manifest:
                out[manifest.name] = manifest
        return out

    @staticmethod
    def _manifest(name: str, item: dict[str, Any], source: Path) -> MCPManifest | None:
        command = item.get("command")
        url = item.get("url")
        if isinstance(command, str):
            command = [command]
        command = [str(c) for c in command or []]
        if not command and not url:
            return None
        env: dict[str, str] = {}
        needs_key = False
        for entry in item.get("env") or []:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            if entry.get("value"):
                env[str(entry["name"])] = str(entry["value"])
            elif entry.get("required"):
                needs_key = True
        return MCPManifest(
            name=name,
            description=str(item.get("description") or ""),
            transport_type="stdio" if command else "http",
            url=str(url) if url and not command else None,
            command=command[0] if command else None,
            args=command[1:],
            env=env,
            auth_type="apikey" if needs_key else "none",
            source_path=source,
        )
