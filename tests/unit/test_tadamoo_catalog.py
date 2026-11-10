"""The catalog of what TaDaMoo has installed, read from its data folder."""

from harness_adapter import TaDaMooCatalog
from harness_adapter.tadamoo_catalog import default_data_dir


def test_the_skill_folder_is_listed_only_when_it_exists(tadamoo_data, tmp_path):
    assert [p.name for p in TaDaMooCatalog(tadamoo_data).skill_dirs()] == ["skills"]
    assert TaDaMooCatalog(tmp_path / "none").skill_dirs() == []


def test_installed_servers_become_manifests(tadamoo_data):
    manifests = TaDaMooCatalog(tadamoo_data).mcp_manifests()
    assert set(manifests) == {"stocktwits", "wolfram"}          # the skill entry is not a server
    local = manifests["stocktwits"]
    assert (local.transport_type, local.command, local.args) == ("stdio", "C:/tools/mcp-stocktwits.cmd", ["--read-only"])
    assert local.auth_type == "none" and local.env == {}         # an empty optional key is not passed
    remote = manifests["wolfram"]
    assert (remote.transport_type, remote.url, remote.command) == ("http", "https://agenttools.wolfram.com/mcp", None)
    assert remote.auth_type == "apikey"                           # a required key with no value


def test_a_missing_or_broken_list_gives_no_servers(tmp_path):
    assert TaDaMooCatalog(tmp_path / "none").mcp_manifests() == {}
    folder = tmp_path / "TaDaMoo" / "capabilities"
    folder.mkdir(parents=True)
    (folder / "installed.json").write_text("{not json", encoding="utf-8")
    assert TaDaMooCatalog(tmp_path / "TaDaMoo").mcp_manifests() == {}


def test_the_data_folder_follows_the_environment(tmp_path):
    assert default_data_dir({"TADAMOO_DATA": str(tmp_path / "x")}) == tmp_path / "x"
    assert default_data_dir({"APPDATA": str(tmp_path), "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
                             "XDG_DATA_HOME": str(tmp_path)}).name == "TaDaMoo"
