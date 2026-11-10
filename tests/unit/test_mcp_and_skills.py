import pytest
from pathlib import Path

from harness_adapter import (
    HarnessAdapter,
    SkillLoader,
    Skill,
    TaDaMooCatalog,
    MCPManifestLoader,
    MCPManifest,
    MockMCPClient,
    bridge_mcp_tools,
    ToolGateway,
)
from cognitive_runtime.memory import PersistentMemoryStore


def test_parse_skill_markdown(tmp_path):
    skill_dir = tmp_path / "custom_skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: sample_skill
description: "A test skill for verifying the loader."
version: 1.2.0
author: OpenSI Team
license: Apache-2.0
platforms: [windows, linux]
metadata:
  tags: [Test, Prototype]
---

# Sample Skill Documentation
Detailed instructions go here.
""",
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.parse_skill_markdown(skill_file)

    assert skill is not None
    assert skill.name == "sample_skill"
    assert skill.version == "1.2.0"
    assert skill.author == "OpenSI Team"
    assert "Test" in skill.tags
    assert "Prototype" in skill.tags
    assert "# Sample Skill Documentation" in skill.instructions


def test_discover_tadamoo_skills_and_memory_registration(tmp_path, tadamoo_data):
    loader = SkillLoader(TaDaMooCatalog(tadamoo_data).skill_dirs())
    skills = loader.discover_skills()

    assert set(skills) == {"arxiv", "daily-report"}
    arxiv = skills["arxiv"]
    assert "arXiv" in arxiv.description or "arxiv" in arxiv.description.lower()

    # Test registering skill into PersistentMemoryStore
    mem_store = PersistentMemoryStore(tmp_path / "memory")
    mem_item = loader.register_with_memory(arxiv, mem_store)

    assert mem_item.ref_id.startswith("mem_")
    assert mem_item.category == "skill"
    assert mem_item.key == "skill:arxiv"


def test_bridge_skill_to_gateway(tmp_path, tadamoo_data):
    loader = SkillLoader(TaDaMooCatalog(tadamoo_data).skill_dirs())
    gateway = ToolGateway({})

    # 1. Propose only (default for security)
    proposal = loader.bridge_to_gateway("arxiv", gateway, propose_only=True)
    assert proposal is not None
    assert proposal.name == "skill_arxiv"
    assert proposal.status == "pending"

    # Gateway should not execute it until approved
    res = gateway.execute("skill_arxiv", {})
    assert res.status == "denied"

    # Approve proposal
    approved = gateway.approve_proposal(proposal.proposal_id)
    assert approved is True

    # Now it can be executed
    res_after = gateway.execute("skill_arxiv", {})
    assert res_after.status == "success"
    assert "Skill: arxiv" in res_after.summary


def test_parse_mcp_manifest(tmp_path):
    folder = tmp_path / "catalog" / "wolfram"
    folder.mkdir(parents=True)
    (folder / "manifest.yaml").write_text(
        "name: wolfram\ndescription: Wolfram\ntransport:\n  type: http\n  url: https://agenttools.wolfram.com/mcp\n"
        "suggest:\n  keywords: [wolfram, math]\n", encoding="utf-8")
    loader = MCPManifestLoader(tmp_path / "catalog")
    assert len(loader.discover()) == 1
    wolfram = loader.get_manifest("wolfram")
    assert wolfram is not None
    assert wolfram.transport_type == "http"
    assert wolfram.url == "https://agenttools.wolfram.com/mcp"
    assert "wolfram" in wolfram.keywords
    assert MCPManifestLoader().discover() == {}


def test_mock_mcp_client_and_bridging():
    mock_client = MockMCPClient({
        "echo": {
            "fn": lambda args: f"Echo: {args.get('text', '')}",
            "description": "Echo input text",
        },
        "add_ten": {
            "fn": lambda args: str(int(args.get('val', 0)) + 10),
            "description": "Add 10 to a value",
        },
    })

    tools = mock_client.list_tools()
    assert len(tools) == 2
    assert {t["name"] for t in tools} == {"echo", "add_ten"}

    gateway = ToolGateway({})

    # Bridge with propose_only=False for immediate registration
    registered = bridge_mcp_tools(mock_client, gateway, server_name="calc_server", propose_only=False)
    assert len(registered) == 2
    assert "calc_server_echo" in registered
    assert "calc_server_add_ten" in registered

    # Execute via ToolGateway
    res1 = gateway.execute("calc_server_echo", {"text": "hello mcp"})
    assert res1.status == "success"
    assert res1.summary == "Echo: hello mcp"

    res2 = gateway.execute("calc_server_add_ten", {"val": 42})
    assert res2.status == "success"
    assert res2.summary == "52"


def test_harness_adapter_mcp_and_skills_integration(tadamoo_data):
    adapter = HarnessAdapter({}, catalog=TaDaMooCatalog(tadamoo_data))
    assert adapter.is_tadamoo_available() is True

    skills = adapter.load_skills()
    assert "arxiv" in skills

    mcps = adapter.load_mcps()
    assert set(mcps) == {"stocktwits", "wolfram"}

    mock_client = MockMCPClient({
        "ping": {
            "fn": lambda args: "pong",
            "description": "Ping test",
        }
    })

    proposals = adapter.connect_mcp(mock_client, server_name="network_server", propose_only=True)
    assert len(proposals) == 1
    assert proposals[0].name == "network_server_ping"

    adapter.approve_proposal(proposals[0].proposal_id)
    res = adapter.execute_tool("network_server_ping", {})
    assert res.status == "success"
    assert res.summary == "pong"
