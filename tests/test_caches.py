"""Coherence of the hot-path memos: config.load, scan_cache_dir, weights_size.

Each memo must return fresh data after the underlying state changes and may
reuse work only while it is unchanged.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from ollajev import config, manager, store


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    return tmp_path


class _FakeApi:
    """HfApi double counting model_info calls; siblings/failure configured per repo."""

    def __init__(self):
        self.calls: list[str] = []
        self.files: dict[str, list[str]] = {}
        self.missing: set[str] = set()

    def model_info(self, repo_id, **kwargs):
        self.calls.append(repo_id)
        if repo_id in self.missing:
            raise httpx.ConnectError("gone")  # caught like RepositoryNotFoundError
        return SimpleNamespace(
            sha="b" * 40,
            created_at=None,
            downloads=7,
            siblings=[SimpleNamespace(rfilename=f) for f in self.files.get(repo_id, [])],
            base_models=None,
        )


@pytest.fixture
def hub(monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(store, "HfApi", lambda: api)
    monkeypatch.setattr(store, "_repo_files_memo", {})
    monkeypatch.setattr(store, "_listing_memo", {})
    return api


def test_config_reloads_after_an_external_write(home):
    config.save({"keep_alive": "5m"})
    assert config.load()["keep_alive"] == "5m"
    (home / "config.json").write_text(json.dumps({"keep_alive": "1h"}))
    assert config.load()["keep_alive"] == "1h"


def test_config_mutating_the_result_does_not_poison_the_memo(home):
    config.save({"pins": {"u/r": "abc"}})
    config.load()["pins"]["u/r"] = "MUTATED"
    assert config.load()["pins"] == {"u/r": "abc"}


def test_scan_memo_reuses_one_walk_and_refreshes_on_demand(monkeypatch):
    monkeypatch.setattr(store, "_scan_memo", None)
    calls = []
    monkeypatch.setattr(store, "_scan_cache_dir", lambda d=None: calls.append(d) or "INFO")
    assert store.scan_cache_dir() == "INFO"
    assert store.scan_cache_dir() == "INFO"
    assert len(calls) == 1
    assert store.scan_cache_dir(refresh=True) == "INFO"
    assert len(calls) == 2
    store.refresh_scan_cache()
    assert store.scan_cache_dir() == "INFO"
    assert len(calls) == 3


def test_repo_files_reuse_one_call_until_the_ttl_lapses(hub, monkeypatch):
    hub.files["u/r"] = ["m.gguf"]
    assert store._repo_files("u/r") == ("m.gguf",)
    assert store._repo_files("u/r") == ("m.gguf",)
    assert hub.calls == ["u/r"]
    monkeypatch.setattr(store, "LOOKUP_TTL", 0)
    assert store._repo_files("u/r") == ("m.gguf",)
    assert hub.calls == ["u/r", "u/r"]


def test_repo_files_failures_expire_too(hub, monkeypatch):
    hub.missing.add("u/gone")
    assert store._repo_files("u/gone") == ()
    assert store._repo_files("u/gone") == ()
    assert hub.calls == ["u/gone"]
    monkeypatch.setattr(store, "LOOKUP_TTL", 0)
    assert store._repo_files("u/gone") == ()
    assert hub.calls == ["u/gone", "u/gone"]


def test_listing_reuses_downloads_and_family(hub):
    hub.files["u/r"] = ["m.gguf"]
    first = store.listing("u/r")
    assert first[0] == 7
    assert store.listing("u/r") == first
    assert hub.calls == ["u/r"]


def test_weights_size_rewalks_when_the_snapshot_grows(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"0" * 100)
    assert manager.weights_size(str(tmp_path)) == 100
    assert manager.weights_size(str(tmp_path)) == 100  # memo hit, same answer
    (tmp_path / "other.gguf").write_bytes(b"1" * 50)  # a new file bumps the dir mtime
    assert manager.weights_size(str(tmp_path)) == 150
    assert manager.weights_size("/no/such/dir") == 0
