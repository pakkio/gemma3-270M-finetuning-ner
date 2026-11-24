"""Test di scripts/weights_hub.py senza rete: huggingface_hub e' sostituito da un finto modulo."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weights_hub as wh  # noqa: E402


class FakeHub(types.ModuleType):
    def __init__(self, materialise=None):
        super().__init__("huggingface_hub")
        self.calls = []
        self.materialise = materialise or []
        outer = self

        class HfApi:
            def create_repo(self, **kw):
                outer.calls.append(("create_repo", kw))

            def upload_folder(self, **kw):
                outer.calls.append(("upload_folder", kw))

            def list_repo_files(self, **kw):
                return ["outputs/run-a/adapter_model.safetensors", "outputs/run-a/adapter_config.json",
                        "models/production/best/README.md"]

        self.HfApi = HfApi

    def snapshot_download(self, **kw):
        self.calls.append(("snapshot_download", kw))
        for rel in self.materialise:
            target = Path(kw["local_dir"]) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "ROOT", tmp_path)
    monkeypatch.delenv(wh.ENV_REPO, raising=False)
    return tmp_path


def install(monkeypatch, hub):
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


@pytest.mark.parametrize("name", ["outputs/run", "models/production/x/", "outputs\\run"])
def test_check_name_accepts(name):
    assert wh.check_name(name) in ("outputs/run", "models/production/x")


@pytest.mark.parametrize("name", ["../x", "/abs/path", "data/x", "outputs", "outputs/../data", "outputs/../../etc"])
def test_check_name_rejects(name):
    with pytest.raises(ValueError):
        wh.check_name(name)


def test_ensure_does_not_touch_the_hub_when_present(root, monkeypatch):
    (root / "outputs" / "run").mkdir(parents=True)
    (root / "outputs" / "run" / "adapter_config.json").write_text("{}")
    hub = install(monkeypatch, FakeHub())
    assert wh.ensure("outputs/run") == str(root / "outputs" / "run")
    assert hub.calls == []


def test_ensure_pulls_when_missing(root, monkeypatch):
    hub = install(monkeypatch, FakeHub(materialise=["outputs/run/adapter_config.json"]))
    assert wh.ensure("outputs/run", repo="u/r") == str(root / "outputs" / "run")
    kind, kw = hub.calls[0]
    assert kind == "snapshot_download"
    assert kw["repo_id"] == "u/r" and kw["local_dir"] == str(root) and kw["allow_patterns"] == ["outputs/run/*"]


def test_pull_raises_if_hub_has_nothing(root, monkeypatch):
    install(monkeypatch, FakeHub(materialise=[]))
    with pytest.raises(FileNotFoundError):
        wh.pull(["outputs/run"], repo="u/r")


def test_pull_all_uses_both_top_levels(root, monkeypatch):
    hub = install(monkeypatch, FakeHub())
    wh.pull(repo="u/r", everything=True)
    assert hub.calls[0][1]["allow_patterns"] == ["outputs/*", "models/*"]


def test_pull_without_target_exits(root, monkeypatch):
    install(monkeypatch, FakeHub())
    with pytest.raises(SystemExit):
        wh.pull(repo="u/r")


def test_push_private_by_default_and_skips_training_state(root, monkeypatch):
    (root / "outputs" / "run").mkdir(parents=True)
    (root / "outputs" / "run" / "adapter_model.safetensors").write_text("w")
    hub = install(monkeypatch, FakeHub())
    wh.push(["outputs/run"], repo="u/r")
    (_, create), (_, upload) = hub.calls
    assert create["private"] is True and create["exist_ok"] is True
    assert upload["path_in_repo"] == "outputs/run" and upload["folder_path"] == str(root / "outputs" / "run")
    assert "*optimizer.pt" in upload["ignore_patterns"]


def test_push_with_training_state_and_public(root, monkeypatch):
    (root / "models" / "m").mkdir(parents=True)
    (root / "models" / "m" / "f").write_text("w")
    hub = install(monkeypatch, FakeHub())
    wh.push(["models/m"], repo="u/r", private=False, with_training_state=True)
    assert hub.calls[0][1]["private"] is False
    assert "*optimizer.pt" not in hub.calls[1][1]["ignore_patterns"]


def test_push_refuses_missing_local_folder(root, monkeypatch):
    hub = install(monkeypatch, FakeHub())
    with pytest.raises(SystemExit):
        wh.push(["outputs/ghost"], repo="u/r")
    assert hub.calls == []


def test_repo_from_env_and_missing(root, monkeypatch):
    monkeypatch.setenv(wh.ENV_REPO, "env/repo")
    assert wh.resolve_repo() == "env/repo" and wh.resolve_repo("cli/repo") == "cli/repo"
    monkeypatch.delenv(wh.ENV_REPO)
    with pytest.raises(SystemExit):
        wh.resolve_repo()


def test_list_remote_groups(root, monkeypatch, capsys):
    install(monkeypatch, FakeHub())
    wh.list_remote(repo="u/r")
    out = capsys.readouterr().out
    assert "outputs/run-a" in out and "models/production/best" in out
