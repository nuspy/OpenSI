"""Pytest configuration: expose src/ as an importable package root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import json  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def tadamoo_data(tmp_path):
    """A small TaDaMoo user data folder: two installed skills, two installed MCP servers."""
    data = tmp_path / "TaDaMoo"
    skill = data / "capabilities" / "skills" / "arxiv"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: arxiv\ndescription: Search arXiv for papers.\nversion: 1.0.0\n---\n\n# arXiv\nUse the script to search.\n",
        encoding="utf-8")
    (skill / "scripts" / "search.py").write_text("print('found')\n", encoding="utf-8")
    report = data / "capabilities" / "skills" / "daily-report"
    report.mkdir(parents=True)
    (report / "SKILL.md").write_text(
        "---\nname: daily-report\ndescription: Write the daily report.\n---\n\nSteps.\n", encoding="utf-8")
    (data / "capabilities" / "installed.json").write_text(json.dumps({"items": {
        "mcp:stocktwits": {"command": ["C:/tools/mcp-stocktwits.cmd", "--read-only"], "description": "StockTwits MCP.",
                           "env": [{"name": "PIPEWORX_API_KEY", "required": False, "secret": True, "value": ""}]},
        "mcp:wolfram": {"url": "https://agenttools.wolfram.com/mcp", "description": "Wolfram.",
                        "env": [{"name": "WOLFRAM_KEY", "required": True, "value": ""}]},
        "skill:arxiv": {"description": "not an MCP server"},
    }}), encoding="utf-8")
    return data
