"""Integration tests for InteractiveConsole and user event loop."""

import pytest
from pathlib import Path
from scripts.interactive_cli import InteractiveConsole

ROOT = Path(__file__).resolve().parents[2]


def test_interactive_console_run(tmp_path):
    cfg = {
        "backend": "mock",
        "policy": {"sandbox_roots": [str(tmp_path)]},
        "eval_tasks": [
            {
                "goal": "Compute 41 + 1 and report the result.",
                "expected": "42",
                "tool": "calculator",
                "tool_args": {"expression": "41+1"},
            }
        ],
    }

    console = InteractiveConsole(cfg, auto_approve=True)
    ret = console.run(goal="Compute 41 + 1 and report the result.")
    assert ret == 0


def test_interactive_approval_handler():
    cfg = {"backend": "mock"}
    console = InteractiveConsole(cfg, auto_approve=True)

    approved = console.handle_approval({"tool": "file_write", "arguments": {}})
    assert approved is True
