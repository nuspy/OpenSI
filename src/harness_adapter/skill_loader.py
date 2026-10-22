"""Skill loader for Hermes Agent skills (Section 4, 6C, and 8 of the handoff).

Discovers, parses, and adapts skills from `external/hermes-agent/skills/`
(or custom skill directories). Each skill provides a `SKILL.md` with YAML
frontmatter, human-readable instructions, and optional helper scripts in
`scripts/`.

Skills can be:
1. Registered into `PersistentMemoryStore` so the cognitive controller can
   reference them via compact `persistent_memory_refs`.
2. Bridged into `ToolGateway` as sandboxed executable commands or proposals.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .tool_gateway import ToolGateway, ToolProposal

DEFAULT_HERMES_SKILLS_DIR = Path(__file__).resolve().parents[2] / "external" / "hermes-agent" / "skills"


@dataclass
class HermesSkill:
    """Parsed representation of a Hermes skill."""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    license: str = ""
    platforms: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    instructions: str = ""
    skill_dir: Path | None = None
    scripts: dict[str, Path] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "platforms": self.platforms,
            "tags": self.tags,
            "instructions": self.instructions[:500] + ("..." if len(self.instructions) > 500 else ""),
            "skill_dir": str(self.skill_dir) if self.skill_dir else None,
            "available_scripts": list(self.scripts.keys()),
        }


class SkillLoader:
    """Discovers, parses, and loads skills from directory hierarchies."""

    def __init__(self, skills_dir: str | Path | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir else DEFAULT_HERMES_SKILLS_DIR
        self._skills: dict[str, HermesSkill] = {}

    def parse_skill_markdown(self, skill_path: Path) -> HermesSkill | None:
        """Parse a SKILL.md file with YAML frontmatter delimited by `---`."""
        if not skill_path.is_file():
            return None

        try:
            content = skill_path.read_text(encoding="utf-8")
        except Exception:
            return None

        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            # No YAML frontmatter
            return None

        end_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx == -1:
            return None

        frontmatter_text = "\n".join(lines[1:end_idx])
        body = "\n".join(lines[end_idx + 1:]).strip()

        try:
            meta = yaml.safe_load(frontmatter_text) or {}
        except Exception:
            return None

        name = str(meta.get("name") or skill_path.parent.name)
        description = str(meta.get("description") or "")
        version = str(meta.get("version") or "1.0.0")
        author = str(meta.get("author") or "")
        license_str = str(meta.get("license") or "")
        platforms = list(meta.get("platforms") or [])

        hermes_meta = meta.get("metadata", {}).get("hermes", {})
        tags = list(hermes_meta.get("tags") or meta.get("tags") or [])

        skill_dir = skill_path.parent
        scripts: dict[str, Path] = {}
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            for script_file in scripts_dir.iterdir():
                if script_file.is_file():
                    scripts[script_file.name] = script_file

        return HermesSkill(
            name=name,
            description=description,
            version=version,
            author=author,
            license=license_str,
            platforms=platforms,
            tags=tags,
            instructions=body,
            skill_dir=skill_dir,
            scripts=scripts,
            raw_metadata=meta,
        )

    def discover_skills(self) -> dict[str, HermesSkill]:
        """Scan skills_dir recursively for SKILL.md and populate registry."""
        self._skills.clear()
        if not self.skills_dir.is_dir():
            return self._skills

        for skill_file in self.skills_dir.rglob("SKILL.md"):
            skill = self.parse_skill_markdown(skill_file)
            if skill:
                self._skills[skill.name] = skill

        return self._skills

    def get_skill(self, name: str) -> HermesSkill | None:
        """Lookup a skill by name, discovering on first access if empty."""
        if not self._skills:
            self.discover_skills()
        return self._skills.get(name)

    def register_with_memory(self, skill: HermesSkill, memory_store: Any) -> Any:
        """Register a discovered skill into PersistentMemoryStore."""
        return memory_store.register_skill_ref(
            skill_name=skill.name,
            description=skill.description,
            path_or_uri=f"skills/{skill.name}/SKILL.md",
        )

    def bridge_to_gateway(
        self,
        skill_name: str,
        gateway: ToolGateway,
        tier: str = "approval_required",
        propose_only: bool = True,
    ) -> ToolProposal | None:
        """Bridge a skill's execution into ToolGateway as a tool proposal or direct tool."""
        skill = self.get_skill(skill_name)
        if not skill:
            return None

        tool_name = f"skill_{skill.name.replace('-', '_')}"

        def skill_runner(arguments: dict[str, Any]) -> str:
            script_name = arguments.get("script")
            args = arguments.get("args", [])
            if not script_name:
                # Return summary instructions
                return f"Skill: {skill.name}\nDescription: {skill.description}\nAvailable scripts: {list(skill.scripts.keys())}"
            if script_name not in skill.scripts:
                return f"Error: script {script_name!r} not found in skill {skill.name!r}. Available: {list(skill.scripts.keys())}"

            script_path = skill.scripts[script_name]
            cmd = [sys.executable, str(script_path)] + [str(a) for a in args]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                output = proc.stdout.strip()
                if proc.returncode != 0:
                    return f"Script error (exit {proc.returncode}): {proc.stderr.strip() or output}"
                return output[:400]
            except Exception as exc:
                return f"Execution failed: {exc}"

        if propose_only:
            return gateway.propose_tool(
                name=tool_name,
                fn=skill_runner,
                description=f"Skill {skill.name}: {skill.description}",
                tier=tier,
                proposed_by="skill_loader",
            )
        gateway.register(tool_name, skill_runner, tier=tier, description=skill.description)
        return None
