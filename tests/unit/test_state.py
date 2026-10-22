"""Unit tests: cognitive state persistence and boundedness."""

import json

from cognitive_runtime.state import new_state, MAX_RECENT_OBSERVATIONS
from cognitive_runtime.state import StateStore


def test_state_serializes_to_json(tmp_path):
    state = new_state("test goal", "criteria")
    path = state.save(tmp_path)
    assert path.exists()
    store = StateStore(tmp_path)
    assert store.exists(state.session_id) is True
    assert store.exists("nonexistent_session") is False
    loaded = store.load(state.session_id)
    assert loaded.goal == state.goal
    assert loaded.success_criteria == state.success_criteria


def test_recent_observations_are_bounded(tmp_path):
    state = new_state("goal")
    for i in range(MAX_RECENT_OBSERVATIONS + 10):
        state.recent_observations.append({"kind": "tool", "content": str(i)})
    StateStore(tmp_path).save(state)
    loaded = StateStore(tmp_path).load(state.session_id)
    assert len(loaded.recent_observations) == MAX_RECENT_OBSERVATIONS


def test_state_version_increases(monkeypatch, tmp_path):
    state = new_state("goal")
    v0 = state.state_version
    state.state_version += 1
    assert state.state_version > v0


def test_state_compression_condenses_older_observations():
    state = new_state("compression goal")
    for i in range(15):
        state.recent_observations.append({"kind": "tool_read", "content": f"data_item_{i}"})

    assert len(state.recent_observations) == 15
    state.compress_state(max_recent=8)

    assert len(state.recent_observations) == 8
    assert len(state.working_hypotheses) > 0
    assert "[tool_read] data_item_0" in state.working_hypotheses


def test_persistent_memory_store(tmp_path):
    from cognitive_runtime.memory import PersistentMemoryStore

    mem = PersistentMemoryStore(tmp_path / "memory")
    item1 = mem.store("project_codename", "Project Phoenix", category="fact", confidence=0.95)
    assert item1.ref_id.startswith("mem_")
    assert item1.key == "project_codename"

    item2 = mem.register_skill_ref("python_exec", "Execute safe python scripts", "skills/python_exec")
    assert item2.category == "skill"

    # Query
    results = mem.query("phoenix")
    assert len(results) == 1
    assert results[0].key == "project_codename"

    # Reload from disk
    mem2 = PersistentMemoryStore(tmp_path / "memory")
    reloaded = mem2.get(item1.ref_id)
    assert reloaded is not None
    assert reloaded.content == "Project Phoenix"


def test_manifest_roundtrip(tmp_path):
    from model_prep import write_manifest, read_manifest

    # A fake local model dir with a config.json and one weight stub.
    model_dir = tmp_path / "FakeOrg/FakeModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["FakeMoeForCausalLM"],
        "model_type": "fake_moe",
        "num_hidden_layers": 4,
        "hidden_size": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 1024,
    }))
    (model_dir / "weights.bin").write_bytes(b"\x00" * 64)

    out = tmp_path / "manifests"
    path = write_manifest(str(model_dir), out)
    manifest = read_manifest(path)
    assert manifest["model_type"] == "fake_moe"
    assert manifest["num_experts"] == 8
    assert manifest["num_experts_per_tok"] == 2
    assert len(manifest["files"]) == 2
