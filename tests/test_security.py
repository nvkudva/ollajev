"""Trust gate, API key, Host check, name parsing, config safety and manager races. No weights load."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ollajev import config, manager, registry, store
from ollajev.names import parse
from ollajev.server import admin, api
from ollajev.ui import cli


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAJEV_API_KEY", raising=False)
    monkeypatch.setattr(api, "allowed_hosts", None)
    return tmp_path


@pytest.fixture
def client(home, monkeypatch):
    class Idle:
        def loaded(self):
            return []

        def unload_all(self):
            pass

    monkeypatch.setattr(api, "Manager", Idle)
    monkeypatch.setattr(api, "preload", None)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


def fake_resolved(runs_code: bool = True):
    family = SimpleNamespace(name="julia", runs_repo_code=runs_code)
    return SimpleNamespace(
        repo_id="user/repo", revision="a" * 40, family=family, weights=None, ref=SimpleNamespace(name="user/repo")
    )


def test_pull_never_trusts_over_http(client, monkeypatch):
    r = fake_resolved()
    monkeypatch.setattr(store, "resolve", lambda *a, **k: r)
    monkeypatch.setattr(store, "download", lambda r: pytest.fail("downloaded an untrusted repo"))
    body = client.post("/api/pull", json={"model": "user/repo", "stream": False, "trust": True})
    assert body.status_code == 400
    assert "--trust" in body.json()["error"]
    assert config.load().get("trusted", []) == []


def test_pull_of_the_same_repo_twice_is_refused(client, monkeypatch):
    r = fake_resolved(runs_code=False)
    monkeypatch.setattr(store, "resolve", lambda *a, **k: r)
    monkeypatch.setattr(store, "download", lambda r: None)
    monkeypatch.setattr(admin, "canonical", lambda r: r.repo_id)
    admin._pulling.add("user/repo")
    try:
        out = client.post("/api/pull", json={"model": "user/repo", "stream": False})
    finally:
        admin._pulling.discard("user/repo")
    assert out.status_code == 400 and "already being pulled" in out.json()["error"]


def test_pull_hides_unexpected_errors(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("/secret/path exploded")

    monkeypatch.setattr(store, "resolve", boom)
    out = client.post("/api/pull", json={"model": "user/repo", "stream": False})
    assert "secret" not in out.text


def test_api_key_is_required_when_set(client, monkeypatch):
    monkeypatch.setenv("OLLAJEV_API_KEY", "s3cret")
    assert client.get("/").status_code == 200  # health probe stays open
    assert client.get("/api/tags").status_code == 401
    assert client.get("/api/tags", headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/tags", headers={"authorization": "Bearer s3cret"}).status_code == 200


def test_no_key_means_open(client):
    assert client.get("/api/ps").status_code == 200


@pytest.mark.parametrize("host,status", [("localhost:8000", 200), ("[::1]:8000", 200), ("evil.example", 403)])
def test_host_header_is_checked_on_loopback(client, monkeypatch, host, status):
    monkeypatch.setattr(api, "allowed_hosts", frozenset({"localhost", "127.0.0.1", "[::1]"}))
    assert client.get("/api/ps", headers={"host": host}).status_code == status


def test_serve_refuses_a_public_bind_without_a_key(home, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("OLLAJEV_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["serve", "--host", "0.0.0.0", "--no-browser"])
    with pytest.raises(SystemExit, match="OLLAJEV_API_KEY"):
        cli.cmd_serve(args)


@pytest.mark.parametrize("name", ["a/..", "../a", "./b", "https://evil.example/a/b", "a/b/c", "x"])
def test_parse_rejects(name):
    with pytest.raises(ValueError):
        parse(name)


@pytest.mark.parametrize("name", ["https://huggingface.co/u/r", "hf.co/u/r", "u/r"])
def test_parse_accepts_hf_forms(name):
    assert parse(name).repo_id == "u/r"


def test_corrupt_config_is_backed_up(home):
    (home / "config.json").write_text("{not json")
    assert config.load() == {}
    assert (home / "config.json.bad").read_text() == "{not json"


def test_concurrent_edits_do_not_lose_updates(home):
    def add(i):
        with config.edit() as data:
            data.setdefault("n", []).append(i)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(config.load()["n"]) == list(range(40))


def test_legacy_config_is_read_when_home_is_default(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "config.json"
    legacy.parent.mkdir()
    legacy.write_text('{"default_model": "u/r"}')
    monkeypatch.delenv("OLLAJEV_HOME", raising=False)
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    monkeypatch.setattr(config, "_legacy_path", lambda: legacy)
    assert config.load() == {"default_model": "u/r"}


class FakeAdapter:
    limits: dict = {}
    name = ""

    def system_one(self, state, questions):
        return {"answers": {}}


def test_run_retries_when_the_slot_was_unloaded(home, monkeypatch):
    mgr = manager.Manager.__new__(manager.Manager)
    dead = SimpleNamespace(
        adapter=FakeAdapter(), lock=threading.Lock(), closed=True, name="m", pinned=False, expires=0.0
    )
    live = SimpleNamespace(
        adapter=FakeAdapter(), lock=threading.Lock(), closed=False, name="m", pinned=False, expires=0.0
    )
    slots = iter([dead, live])
    mgr.get = lambda name, keep_alive=None: next(slots)  # type: ignore[method-assign]
    mgr._touch = lambda slot, keep_alive: None  # type: ignore[method-assign]
    slot, result = mgr.run("m", "state", {})
    assert slot is live and result == {"answers": {}}


def test_oversized_body_is_413(client, monkeypatch):
    monkeypatch.setenv("OLLAJEV_MAX_BODY_BYTES", "100")
    out = client.post("/v1/systemone", content=b"x" * 500, headers={"content-type": "application/json"})
    assert out.status_code == 413
    assert out.json()["detail"][0]["type"] == "payload_too_large"


def test_playground_page_sends_a_csp(client):
    assert "default-src 'self'" in client.get("/playground").headers["content-security-policy"]


def test_the_old_demo_address_redirects_to_the_playground(client):
    reply = client.get("/demo", follow_redirects=False)
    assert reply.status_code == 308 and reply.headers["location"] == "/playground"


@pytest.mark.parametrize(
    "var,value",
    [
        ("OLLAJEV_HOST", "127.0.0.1:abc"),
        ("OLLAJEV_HOST", "[::1]:99999"),
        ("OLLAJEV_MAX_LOADED_MODELS", "two"),
        ("OLLAJEV_KEEP_ALIVE", "soon"),
        ("OLLAJEV_MAX_BODY_BYTES", "big"),
    ],
)
def test_bad_env_names_the_variable(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        config.host(), config.keep_alive(), config.max_loaded_models(), config.max_body_bytes()


def make_cache(tmp_path, rev_files):
    """A fake Hugging Face snapshot dir: blobs/<sha> plus symlinks under snapshots/<rev>/."""
    root = tmp_path / "models--u--r"
    (root / "blobs").mkdir(parents=True)
    for rev, files in rev_files.items():
        (root / "snapshots" / rev).mkdir(parents=True)
        for name, blob in files.items():
            (root / "blobs" / blob).write_bytes(b"w" * 10)
            (root / "snapshots" / rev / name).symlink_to(root / "blobs" / blob)
    return root


def delete_with(monkeypatch, root, rev):
    snap = root / "snapshots" / rev
    fake = SimpleNamespace(snapshot_path=snap)
    monkeypatch.setattr(store, "snapshot", lambda repo_id, revision: fake)
    return store.delete_file(SimpleNamespace(repo_id="u/r", revision=rev, weights="m-Q4_K_M.gguf"))


def test_delete_file_keeps_a_blob_another_revision_uses(tmp_path, monkeypatch):
    root = make_cache(tmp_path, {"r1": {"m-Q4_K_M.gguf": "b1"}, "r2": {"m-Q4_K_M.gguf": "b1"}})
    assert delete_with(monkeypatch, root, "r1") == 0
    assert (root / "blobs" / "b1").exists()


def test_delete_file_frees_an_unshared_blob(tmp_path, monkeypatch):
    root = make_cache(tmp_path, {"r1": {"m-Q4_K_M.gguf": "b1", "m-Q8_0.gguf": "b2"}})
    assert delete_with(monkeypatch, root, "r1") == 10
    assert not (root / "blobs" / "b1").exists() and (root / "blobs" / "b2").exists()


def test_failed_download_leaves_no_pin(home, monkeypatch):
    def fail(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(store, "snapshot_download", fail)
    r = SimpleNamespace(repo_id="u/r", revision="a" * 40, allow=None, created="2026-01-01", base=None)
    with pytest.raises(OSError):
        store.download(r)
    assert registry.pins() == {}
    monkeypatch.setattr(store, "snapshot_download", lambda *a, **k: "/x")
    store.download(r)
    assert registry.pins() == {"u/r": "a" * 40} and registry.released("u/r") == "2026-01-01"


def test_pull_stream_reports_bytes_as_the_cache_grows(client, monkeypatch):
    import json
    import time

    r = fake_resolved(runs_code=False)
    grown = {"n": 0}
    monkeypatch.setattr(store, "resolve", lambda *a, **k: r)
    monkeypatch.setattr(store, "download_size", lambda r: 100)
    monkeypatch.setattr(store, "bytes_on_disk", lambda repo_id: grown["n"])
    monkeypatch.setattr(admin, "canonical", lambda r: r.repo_id)

    def download(r):
        grown["n"] = 60
        time.sleep(1.2)

    monkeypatch.setattr(store, "download", download)
    out = client.post("/api/pull", json={"model": "user/repo", "stream": True})
    events = [json.loads(line) for line in out.text.splitlines()]
    assert any(e.get("total") == 100 and e.get("completed") == 60 for e in events)
    assert events[-1]["status"] == "success"
