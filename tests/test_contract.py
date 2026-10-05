"""The wire contract, model naming and answer normalisation, without loading any weights.

`api.manager` is replaced by a stub, so routes run for real while no model loads. Anything that
needs inference is checked by running the server against real models.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ollajev import names, normalize, store
from ollajev.adapters import laya
from ollajev.manager import NotDownloaded, NotEnoughMemory, NotTrusted, check_limits
from ollajev.names import parse, pick_weights
from ollajev.server import api

NOUL = {"type": "noul", "noul": 0.9, "x_extra": 1}


class StubManager:
    def __init__(self):
        self.calls = []
        self.error = None

    def get(self, name, keep_alive=None):
        return None

    def run(self, name, state, questions, keep_alive=None):
        self.calls.append((name, state, questions))
        if self.error:
            raise self.error
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = NOUL
            elif q["type"] == "choice":
                answers[qid] = {
                    "choice": next(iter(q["criteria"])),
                    "probabilities": {k: 1 / len(q["criteria"]) for k in q["criteria"]},
                }
            else:
                answers[qid] = {"probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))}}
        slot = SimpleNamespace(name="user/model", loaded_at=self.loaded_at)
        return slot, {"answers": answers, "usage": {"input_tokens": 3}}

    loaded_at = 0.0  # long ago: the model was already in memory

    def loaded(self):
        return []

    def unload_all(self):
        pass


@pytest.fixture
def stub(monkeypatch):
    m = StubManager()
    monkeypatch.setattr(api, "Manager", lambda: m)
    monkeypatch.setattr(api, "preload", None)
    return m


@pytest.fixture
def client(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


def ask(client, questions, **body):
    return client.post("/v1/systemone", json={"state": "hello", "questions": questions, **body})


def test_response_matches_the_typesafe_schema(client):
    models = pytest.importorskip("typesafe_sdk._schemas.models")
    r = ask(
        client,
        {
            "n": {"type": "noul", "instructions": "y"},
            "c": {"type": "choice", "criteria": {"a": None, "b": "B"}},
            "s": {"type": "score", "criteria": ["low", "high"]},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    models.SystemOneResponse.model_validate(body)
    assert body["model"] == "user/model"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 0}
    assert "x_extra" not in body["answers"]["n"]
    assert body["answers"]["s"]["legend"] == {"0": "low", "1": "high"}
    models.ModelMetadataList.model_validate(client.get("/v1/models").json())


def test_choice_options_without_description_reach_the_model(client, stub):
    ask(client, {"c": {"type": "choice", "criteria": {"a": None, "b": "B"}}})
    assert stub.calls[-1][2]["c"]["criteria"] == {"a": None, "b": "B"}


@pytest.mark.parametrize(
    "error,status,kind",
    [
        (NotDownloaded("x is not downloaded"), 404, "model_not_found"),
        (NotTrusted("x is not trusted"), 403, "model_not_trusted"),
        (NotEnoughMemory("x needs 9 GiB"), 503, "model_out_of_memory"),
        (ValueError("too many options"), 422, "value_error"),
    ],
)
def test_model_errors_keep_the_error_shape(client, stub, error, status, kind):
    stub.error = error
    r = ask(client, {"n": {"type": "noul"}})
    assert r.status_code == status
    assert r.json()["detail"][0]["type"] == kind


def test_under_two_criteria_is_422(client):
    assert ask(client, {"s": {"type": "score", "criteria": ["one"]}}).status_code == 422


def test_500_does_not_leak_the_exception(client, stub):
    stub.error = RuntimeError("secret detail")
    r = ask(client, {"n": {"type": "noul"}})
    assert r.status_code == 500 and "secret" not in r.text


def test_404_and_405_use_the_error_shape(client):
    assert client.get("/nope").json()["detail"][0]["type"] == "http_error"
    r = client.get("/v1/systemone")
    assert r.status_code == 405 and "allow" in r.headers


def test_every_preset_is_a_valid_request(client):
    for name, example in client.get("/ui/presets").json().items():
        assert ask(client, example["questions"]).status_code == 200, name


@pytest.mark.parametrize(
    "name,repo,tag",
    [
        ("SupersonicLabs/Julia-1", "SupersonicLabs/Julia-1", None),
        ("Mapika/decider-4b-GGUF:Q4_K_M", "Mapika/decider-4b-GGUF", "Q4_K_M"),
        ("hf.co/Mapika/decider-4b-GGUF:q8_0", "Mapika/decider-4b-GGUF", "q8_0"),
        ("https://huggingface.co/a/b:x.gguf", "a/b", "x.gguf"),
        ("https://huggingface.co/a/b/", "a/b", None),
        ("https://huggingface.co/a/b/tree/main", "a/b", None),
        ("https://huggingface.co/a/b/blob/main/x_q4_k_m.gguf", "a/b", "x_q4_k_m.gguf"),
        ("https://huggingface.co/a/b/resolve/main/sub/x.gguf?download=true", "a/b", "x.gguf"),
        ("https://huggingface.co/a/b/blob/main/README.md", "a/b", None),
    ],
)
def test_parse_names(name, repo, tag):
    ref = parse(name)
    assert (ref.repo_id, ref.tag) == (repo, tag)


def test_parse_rejects_non_repo_names():
    with pytest.raises(ValueError):
        parse("decider-4b")


FILES = ["m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-BF16.gguf", "config.json"]


def test_gguf_quant_selection_follows_ollama():
    assert pick_weights(FILES, None) == "m-Q4_K_M.gguf"
    assert pick_weights(FILES, "q8_0") == "m-Q8_0.gguf"
    assert pick_weights(FILES, "m-BF16.gguf") == "m-BF16.gguf"
    assert pick_weights(["m-Q8_0.gguf", "m-Q5_K_M.gguf"], None) == "m-Q5_K_M.gguf"
    assert pick_weights(["m_f16.gguf", "m_ud_q4_k_m.gguf"], "Q4_K_M") == "m_ud_q4_k_m.gguf"
    assert pick_weights(["m.mmproj-Q8_0.gguf", "m.Q8_0.gguf"], "Q8_0") == "m.Q8_0.gguf"
    with pytest.raises(ValueError):
        pick_weights(FILES, "Q2_K")


def test_confidence_uses_typesafe_formulas():
    assert normalize.choice_confidence([0.5, 0.5]) == 0
    assert normalize.choice_confidence([1.0, 0.0, 0.0]) == 1
    assert normalize.score_confidence([0.0, 1.0, 0.0]) == 1
    a = normalize.answer({"type": "noul"}, {"noul": 0.75, "decision": "yes"})
    assert a == {"type": "noul", "noul": 0.75, "confidence": 0.5}


def test_per_question_model_errors_become_value_errors():
    with pytest.raises(ValueError):
        normalize.answers({"q": {"type": "noul"}}, {"q": {"type": "noul", "error": "max_length_exceeded"}})


def test_limits_are_checked_before_the_model_runs():
    questions = {"c": {"type": "choice", "criteria": {str(i): None for i in range(21)}}}
    with pytest.raises(ValueError, match="at most 20"):
        check_limits({"max_options": 20}, questions)
    check_limits({"max_options": 255}, questions)
    with pytest.raises(ValueError, match="at most 1 questions"):
        check_limits({"max_questions": 1}, {**questions, "d": {"type": "noul"}})


def test_configure_sets_startup_state_and_resets_it():
    try:
        api.configure(preload_model="u/r", pin_preload_model=True, allowed=frozenset({"localhost"}))
        assert (api.preload, api.pin_preload, api.allowed_hosts) == ("u/r", True, frozenset({"localhost"}))
    finally:
        api.configure()
    assert (api.preload, api.pin_preload, api.allowed_hosts) == (None, False, None)


def test_fresh_machine_without_a_model_cache_lists_no_models(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "never-created"))
    r = client.get("/v1/models")
    assert r.status_code == 200 and r.json() == {"models": []}
    assert client.get("/api/tags").json() == {"models": []}


def test_variants_name_each_quant_and_size_its_download():
    sizes = {
        "m-Q4_K_M.gguf": 4,
        "m-Q8_0.gguf": 8,
        "m-Q4_K_M-imat.gguf": 5,
        "decider_config.json": 1,
        "tokenizer.json": 1,
    }
    found = store._variants("Mapika/decider-x-GGUF", "sha", sizes, store._family("Mapika/decider-x-GGUF", list(sizes)))
    assert [(v.name, v.size) for v in found] == [
        ("Mapika/decider-x-GGUF:Q4_K_M", 6),
        ("Mapika/decider-x-GGUF:m-Q4_K_M-imat.gguf", 7),
        ("Mapika/decider-x-GGUF:Q8_0", 10),
    ]
    assert [v.name for v in store._variants("u/r", "sha", {"weights.bin": 3}, None)] == ["u/r"]


def test_search_lists_supported_models_first(monkeypatch):
    def model(repo, files):
        return SimpleNamespace(id=repo, downloads=0, siblings=[SimpleNamespace(rfilename=f) for f in files])

    found = [
        model("u/a-GGUF", ["a.gguf"]),
        model("u/a-decider", ["decider_config.json"]),
        model("u/b-GGUF", ["b.gguf"]),
    ]

    class Api:
        def model_info(self, repo_id, **kwargs):
            return next(m for m in found if m.id == repo_id)

        def list_models(self, **kwargs):
            return found

    monkeypatch.setattr(store, "HfApi", Api)
    assert [h.repo_id for h in store.search("a")] == ["u/a-decider", "u/a-GGUF"]

    class Gone(Api):  # a repo deleted after the listing does not fail the whole search
        def model_info(self, repo_id, **kwargs):
            if repo_id == "u/a-GGUF":
                raise store.RepositoryNotFoundError.__new__(store.RepositoryNotFoundError)
            return super().model_info(repo_id, **kwargs)

    monkeypatch.setattr(store, "HfApi", Gone)
    assert [(h.repo_id, h.family) for h in store.search("a")] == [("u/a-decider", "decider"), ("u/a-GGUF", None)]


def test_a_quantized_copy_runs_on_its_base_family(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    repos = {
        "q/decider-GGUF": ("c1", ["decider.Q4_K_M.gguf", "README.md"], ["Mapika/decider-x"]),
        "Mapika/decider-x": ("b1", ["decider_config.json", "model.safetensors", "tokenizer.json"], []),
        "q/laya-GGUF": ("c2", ["laya.Q4_K_M.gguf"], ["convaiinnovations/laya"]),
        "convaiinnovations/laya": ("b2", ["rl_agent_config.json", "model.safetensors"], []),
    }
    monkeypatch.setattr(store, "_remote_files", lambda repo, rev: (repos[repo][0], None, *repos[repo][1:]))
    r = store.resolve("q/decider-GGUF")
    assert (r.family.name, r.weights, r.allow) == ("decider", "decider.Q4_K_M.gguf", ["decider.Q4_K_M.gguf"])
    assert r.base is not None and (r.base.repo_id, r.base.revision) == ("Mapika/decider-x", "b1")
    assert "*.json" in (r.base.allow or [])
    with pytest.raises(LookupError):  # laya cannot run a GGUF with its base's files
        store.resolve("q/laya-GGUF")


def test_only_quantizations_inherit_a_family():
    models = [{"id": "Mapika/decider-x"}]
    assert store._quantized_from({"relation": "quantized", "models": models}) == ["Mapika/decider-x"]
    assert store._quantized_from({"relation": "finetune", "models": models}) == []
    assert store._quantized_from(None) == []


ONNX = ["model.onnx", "model_fp16.onnx", "model_fp16.onnx_data", "model_int8.onnx", "tokenizer.json"]


def test_onnx_exports_are_variants_named_by_precision():
    assert pick_weights(ONNX, None) == "model.onnx"
    assert pick_weights(ONNX, "FP16") == "model_fp16.onnx"
    assert pick_weights(["onnx/model-int8.onnx", "onnx/model-fp32.onnx"], None) == "onnx/model-fp32.onnx"
    assert pick_weights(["a/model.onnx", "b/model.onnx"], "b/model.onnx") == "b/model.onnx"
    assert pick_weights(["m.Q4_K_M.gguf", "model.onnx"], None) == "m.Q4_K_M.gguf"  # GGUF repos stay GGUF repos
    with pytest.raises(ValueError, match=r"fp16, int8, model\.onnx"):
        pick_weights(ONNX, "q4")
    assert names.labels(["a/model.onnx", "b/model.onnx", "x.int8.onnx"]) == {
        "a/model.onnx": "a/model.onnx",
        "b/model.onnx": "b/model.onnx",
        "x.int8.onnx": "int8",
    }
    assert names.sidecars("onnx/model.onnx") == ["onnx/model.onnx_data", "onnx/model.onnx.data"]
    assert names.sidecars("m.Q4_K_M.gguf") == []
    assert parse("https://huggingface.co/a/b/blob/main/onnx/model_fp16.onnx").tag == "model_fp16.onnx"


def test_an_onnx_variant_downloads_its_external_data():
    sizes = {"onnx/model.onnx": 3, "onnx/model.onnx_data": 100, "onnx/model_fp16.onnx": 3, "laya.onnx": 2}
    sizes |= {"onnx/model_fp16.onnx_data": 50, "laya.onnx.data": 90, "config.json": 1}
    found = store._variants("u/laya-ONNX", "sha", sizes, None)
    assert [(v.name, v.size) for v in found] == [
        ("u/laya-ONNX:fp16", 53),
        ("u/laya-ONNX:laya.onnx", 92),
        ("u/laya-ONNX:model.onnx", 103),
    ]


def test_a_quantized_onnx_copy_runs_on_the_laya_family(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    torch_files = ["rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json"]
    repos = {
        "q/laya-onnx": ("c1", ["onnx/model.onnx", "onnx/model.onnx_data", "onnx/model_int8.onnx"], ["c/laya"]),
        "q/laya-split": ("c2", ["encoder.onnx", "head.onnx", "rl_agent_config.json"], ["c/laya"]),
        "q/laya-int4": ("c3", ["laya-int4-blk32.onnx", "laya-int4-blk32.onnx.data"], ["c/laya"]),
        "c/laya": ("b1", torch_files, []),
    }
    monkeypatch.setattr(store, "_remote_files", lambda repo, rev: (repos[repo][0], None, *repos[repo][1:]))
    r = store.resolve("q/laya-onnx")
    assert (r.family.name, r.weights) == ("laya", "onnx/model.onnx")
    assert r.allow == ["onnx/model.onnx", "onnx/model.onnx_data", "onnx/model.onnx.data"]
    assert r.base is not None and r.base.repo_id == "c/laya"
    assert r.base.allow == ["rl_agent_config.json", "tokenizer/*"]
    assert store.resolve("q/laya-onnx:int8").weights == "onnx/model_int8.onnx"
    for unsupported in ("q/laya-split", "q/laya-int4"):
        with pytest.raises(LookupError, match="does not run"):
            store.resolve(unsupported)
    torch = store.resolve("c/laya")  # the PyTorch repo matches by itself, as before
    assert (torch.family.name, torch.weights, torch.base) == ("laya", None, None)
    assert not laya.FAMILY.matches("q/laya-onnx", repos["q/laya-onnx"][1])


def test_load_refuses_a_model_that_does_not_fit_in_free_memory(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ollajev import manager, registry, store

    (tmp_path / "m.safetensors").write_bytes(b"x" * 1000)
    monkeypatch.setattr(registry, "is_trusted", lambda r: True)
    monkeypatch.setattr(store, "local_path", lambda r: str(tmp_path))
    monkeypatch.setattr(manager, "pick_device", lambda requested: "cpu")
    monkeypatch.setattr(manager, "free_memory", lambda device: 999)
    r = SimpleNamespace(revision="r" * 12, family=None)
    with pytest.raises(NotEnoughMemory, match=r"needs about .* but only .* free on cpu"):
        manager.Manager()._load("m", r)


def test_a_cancelled_download_stops_at_its_next_progress_update(monkeypatch):
    import threading

    def snapshot_download(*args, tqdm_class, **kwargs):
        bar = tqdm_class(total=10, disable=True)
        bar.update(1)
        cancel.set()
        bar.update(1)

    cancel = threading.Event()
    monkeypatch.setattr(store, "snapshot_download", snapshot_download)
    r = SimpleNamespace(repo_id="u/r", revision="a" * 40, allow=None, base=None)
    with pytest.raises(store.Cancelled):
        store.download(r, cancel)


def test_model_loads_show_on_the_server_console(tmp_path, capsys):
    import logging

    from ollajev.ui import cli

    root, manager_log = logging.getLogger(), logging.getLogger("ollajev.manager")
    saved = root.handlers[:], root.level, manager_log.handlers[:]
    try:
        cli.configure_logging(str(tmp_path / "server.log"))
        manager_log.info("Loading a/b on cpu …")
        manager_log.warning("low memory")
    finally:
        root.handlers, manager_log.handlers = saved[0], saved[2]
        root.setLevel(saved[1])
    err = capsys.readouterr().err
    assert "==> Loading a/b on cpu …" in err
    assert err.count("low memory") == 1


def test_server_timing_separates_a_model_load_from_the_answer(client, stub):
    import time

    already_loaded = ask(client, {"n": {"type": "noul"}}).headers["server-timing"]
    assert already_loaded.startswith("run;dur=") and "load" not in already_loaded
    stub.loaded_at = time.time() + 60  # loaded during this request
    just_loaded = ask(client, {"n": {"type": "noul"}}).headers["server-timing"]
    assert just_loaded.startswith("load;dur=") and ", run;dur=" in just_loaded


def test_ipv6_is_skipped_when_it_does_not_connect(monkeypatch):
    import socket

    def no_route(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET6, 1, 6, "", ("::1", 443, 0, 0))])
    monkeypatch.setattr(socket, "create_connection", no_route)
    assert store._ipv6_reaches("huggingface.co") is False
    monkeypatch.setattr(socket, "getaddrinfo", no_route)  # no IPv6 address at all: nothing tries IPv6
    assert store._ipv6_reaches("huggingface.co") is True


def test_ipv6_probe_connects_to_the_first_address_host_and_port(monkeypatch):
    import contextlib
    import socket

    tried = []

    def connect(address, timeout):
        tried.append(address)
        return contextlib.nullcontext()

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET6, 1, 6, "", ("2001:db8::1", 443, 0, 0))]
    )
    monkeypatch.setattr(socket, "create_connection", connect)
    assert store._ipv6_reaches("huggingface.co") is True
    assert tried == [("2001:db8::1", 443)]


def test_saved_settings_apply_unless_the_environment_overrides(tmp_path, monkeypatch):
    from ollajev import config

    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAJEV_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OLLAJEV_MAX_LOADED_MODELS", raising=False)
    config.update(keep_alive="1h", max_loaded_models=3)
    assert (config.keep_alive(), config.max_loaded_models()) == (3600, 3)
    monkeypatch.setenv("OLLAJEV_MAX_LOADED_MODELS", "2")
    assert config.max_loaded_models() == 2


def test_aliases_are_saved_lower_case_and_matched_ignoring_case(tmp_path, monkeypatch):
    from ollajev import config
    from ollajev.manager import lookup

    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    assert config.set_alias("Julia", "SupersonicLabs/Julia-1") == "julia"
    assert config.load()["aliases"] == {"julia": "SupersonicLabs/Julia-1"}
    assert lookup("JULIA") == "SupersonicLabs/Julia-1"
    assert lookup("JuliA") == "SupersonicLabs/Julia-1"
    assert config.remove_alias("JuLiA") == "julia"
    assert config.aliases() == {}


def test_an_older_capitalised_alias_is_matched_and_replaced(tmp_path, monkeypatch):
    from ollajev import config

    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    config.update(aliases={"Jev": "u/a"})
    assert config.alias("jev") == "u/a"
    config.set_alias("jev", "u/b")
    assert config.load()["aliases"] == {"jev": "u/b"}


def test_a_short_name_may_not_clash_with_a_model_even_in_case(tmp_path, monkeypatch):
    import asyncio

    from ollajev.ui import tui

    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "m"))
    app = tui.Models()
    app.names = ["SupersonicLabs/Julia-1"]

    async def go():
        async with app.run_test(size=(120, 30)):
            return {short: app.short_name_error(short) for short in ("julia", "supersoniclabs/Julia-1", "two words")}

    errors = asyncio.run(go())
    assert errors["julia"] is None  # not a model's own name, so it is free
    assert errors["supersoniclabs/Julia-1"] == "'supersoniclabs/Julia-1' is already a model's own name"
    assert errors["two words"] == "'two words' cannot be a short name: no spaces in it"
