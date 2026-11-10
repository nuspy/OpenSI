"""Unit tests for HarnessAdapter (TaDaMoo passive integration)."""

import pytest
from harness_adapter import HarnessAdapter, TaDaMooCatalog
from cognitive_runtime.events import EventBus


def test_harness_adapter_initialization(tadamoo_data):
    cfg = {
        "policy": {
            "sandbox_roots": ["data/eval_corpus"],
            "tool_tiers": {"calculator": "auto"},
        }
    }
    events = EventBus()
    adapter = HarnessAdapter(cfg, event_bus=events, catalog=TaDaMooCatalog(tadamoo_data))

    assert adapter.is_tadamoo_available() is True
    assert "calculator" in adapter.list_tools()
    assert "clock" in adapter.list_tools()
    assert "file_read" in adapter.list_tools()

    # Test event injection
    adapter.inject_event("user_interrupt", "pause execution")
    ev = events.pop()
    assert ev is not None
    assert ev.kind == "user_interrupt"
    assert ev.payload == {"content": "pause execution"}


def test_harness_adapter_tool_execution():
    cfg = {"policy": {"sandbox_roots": ["data/eval_corpus"]}}
    adapter = HarnessAdapter(cfg)

    res = adapter.execute_tool("calculator", {"expression": "21 + 21"})
    assert res.status == "success"
    assert res.summary == "42"


def test_tool_proposal_lifecycle():
    cfg = {"policy": {"sandbox_roots": ["data/eval_corpus"]}}
    adapter = HarnessAdapter(cfg)

    def custom_greeter(args):
        return f"Hello {args.get('name', 'world')}"

    # 1. Propose tool
    prop = adapter.propose_tool("greeter", custom_greeter, description="Greet someone")
    assert prop.status == "pending"
    assert "greeter" not in adapter.list_tools()

    # 2. Reject another proposal
    prop_bad = adapter.propose_tool("bad_tool", lambda a: "bad")
    assert adapter.reject_proposal(prop_bad.proposal_id, reason="unsafe") is True
    assert prop_bad.status == "rejected"
    assert prop_bad.rejection_reason == "unsafe"

    # 3. Approve proposal
    assert adapter.approve_proposal(prop.proposal_id) is True
    assert prop.status == "approved"
    assert "greeter" in adapter.list_tools()

    # Execute approved tool
    out = adapter.execute_tool("greeter", {"name": "Alice"})
    assert out.status == "success"
    assert out.summary == "Hello Alice"


def test_file_write_sandboxing(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    cfg = {"policy": {"sandbox_roots": [str(sandbox)]}}
    adapter = HarnessAdapter(cfg)

    # Allowed path
    allowed_file = sandbox / "allowed.txt"
    res = adapter.execute_tool("file_write", {"path": str(allowed_file), "content": "hello openSI"})
    assert res.status == "success"
    assert allowed_file.read_text(encoding="utf-8") == "hello openSI"

    # Denied path outside sandbox
    forbidden = tmp_path / "outside.txt"
    res_bad = adapter.execute_tool("file_write", {"path": str(forbidden), "content": "hack"})
    assert res_bad.status == "failure"
    assert "PermissionError" in res_bad.summary


def test_without_a_tadamoo_data_folder_the_adapter_still_works(tmp_path):
    adapter = HarnessAdapter({}, catalog=TaDaMooCatalog(tmp_path / "missing"))
    assert adapter.is_tadamoo_available() is False
    assert adapter.load_skills() == {}
    assert adapter.load_mcps() == {}
    assert adapter.execute_tool("calculator", {"expression": "1 + 1"}).summary == "2"


def test_the_data_folder_can_come_from_the_config(tadamoo_data):
    adapter = HarnessAdapter({"harness": {"tadamoo_data": str(tadamoo_data)}})
    assert adapter.is_tadamoo_available() is True
    assert set(adapter.load_skills()) == {"arxiv", "daily-report"}
