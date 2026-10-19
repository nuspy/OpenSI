"""Unit tests: cognitive state persistence and boundedness."""

import json

from cognitive_runtime.state import new_state, MAX_RECENT_OBSERVATIONS
from cognitive_runtime.state import StateStore


def test_state_serializes_to_json(tmp_path):
    state = new_state("test goal", "criteria")
    path = state.save(tmp_path)
    assert path.exists()
    loaded = StateStore(tmp_path).load(state.session_id)
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
