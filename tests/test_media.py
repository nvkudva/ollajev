"""Media in requests: data URL decoding, per-model input checks, decoding, and the API's handling. No weights load."""

from __future__ import annotations

import base64
import io
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ollajev import manager, media
from ollajev.adapters import d1, detect
from ollajev.server import api


def data_url(kind: str, data: bytes) -> str:
    return f"data:{kind};base64," + base64.b64encode(data).decode()


def png() -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(out, "PNG")
    return out.getvalue()


def wav(seconds: float = 0.5, rate: int = 22050) -> bytes:
    import wave

    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x10" * 2 * int(seconds * rate))
    return out.getvalue()


def test_decode_takes_only_base64_data_urls():
    assert media.decode("images", [data_url("image/png", b"abc")]) == [b"abc"]
    assert media.decode("audio", ["data:audio/wav;codecs=pcm;base64,YWJj"]) == [b"abc"]
    for bad in ("https://example.com/a.png", "/etc/passwd", "file:///etc/passwd", "data:image/png,abc"):
        with pytest.raises(ValueError, match="data URL"):
            media.decode("images", [bad])
    with pytest.raises(ValueError, match="base64"):
        media.decode("images", ["data:image/png;base64,@@@"])


def test_check_follows_the_model_inputs():
    media.check({}, {"images": [], "audio": [], "videos": []})
    with pytest.raises(ValueError, match="reads text; it takes no images"):
        media.check({}, {"images": [b"x"]})
    media.check(d1.LIMITS, {"images": [b"x", b"y"], "audio": []})
    with pytest.raises(ValueError, match="no videos"):
        media.check(d1.LIMITS, {"videos": [b"x"]})
    with pytest.raises(ValueError, match="at most 1 audio"):
        media.check(d1.LIMITS, {"audio": [b"x", b"y"]})


def test_image_and_audio_decode():
    assert media.image(png()).size == (8, 6)
    samples = media.audio(wav(0.5))
    assert samples.dtype.name == "float32"
    assert abs(len(samples) - media.AUDIO_RATE // 2) < 400  # resampled from 22.05 kHz stereo to 16 kHz mono
    with pytest.raises(media.Invalid, match=r"not an image this server can read\. Send a PNG"):
        media.image(b"not an image")
    with pytest.raises(ValueError, match="could not be read"):
        media.audio(b"not audio")


def test_d1_repos_are_detected():
    files = ["config.json", "modeling_d1.py", "audio.py", "vision.py", "model.safetensors", "tokenizer.json"]
    assert detect("LiquidAI/d1-omni-600M", files) is d1.FAMILY
    assert not d1.FAMILY.matches("someone/other", ["config.json", "model.safetensors"])


class MediaAdapter:
    limits = {"inputs": ["text", "image"]}
    name = "m"

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, media=None):
        self.calls.append(media)
        return {"answers": {"q": {"noul": 0.75}}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAJEV_API_KEY", raising=False)
    monkeypatch.setattr(api, "allowed_hosts", None)
    monkeypatch.setattr(api, "preload", None)
    adapter = MediaAdapter()
    slot = SimpleNamespace(adapter=adapter, lock=threading.Lock(), closed=False, name="m", loaded_at=0.0)

    class Fake(manager.Manager):
        def __init__(self):
            pass

        def get(self, name, keep_alive=None):
            return slot

        def resolve(self, name):
            return SimpleNamespace(family=SimpleNamespace(limits=lambda r: adapter.limits))

        def _touch(self, slot, keep_alive):
            pass

        def loaded(self):
            return []

        def unload_all(self):
            pass

    monkeypatch.setattr(api, "Manager", Fake)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c, adapter


QUESTION = {"q": {"type": "noul", "instructions": "Is it red?"}}


def test_api_passes_decoded_media(client):
    c, adapter = client
    r = c.post("/v1/systemone", json={"state": "", "images": [data_url("image/png", b"abc")], "questions": QUESTION})
    assert r.status_code == 200, r.text
    assert adapter.calls == [{"images": [b"abc"], "audio": [], "videos": []}]
    r = c.post("/v1/systemone", json={"state": "x", "questions": QUESTION})
    assert r.status_code == 200 and adapter.calls[-1] is None  # text only: the two-argument call


def test_api_rejects_media_the_model_cannot_read(client):
    c, adapter = client
    r = c.post("/v1/systemone", json={"state": "x", "audio": [data_url("audio/wav", b"abc")], "questions": QUESTION})
    assert r.status_code == 422 and "takes no audio" in r.json()["detail"][0]["msg"]
    r = c.post("/v1/systemone", json={"state": "x", "images": ["https://example.com/a.png"], "questions": QUESTION})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "images"]
    assert adapter.calls == []


def test_playground_catalog_lists_every_curated_model(client):
    from ollajev.catalog import CATALOG

    c, _ = client
    body = c.get("/ui/catalog").json()["models"]
    assert [m["name"] for m in body] == [e.name for e in CATALOG]
    assert {"name", "size_gb", "languages", "description"} <= set(body[0])


def mp4(seconds: int = 2, fps: int = 10) -> bytes:
    import av
    import numpy as np

    out = io.BytesIO()
    with av.open(out, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * fps):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10 % 255, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return out.getvalue()


def test_video_frames_are_sampled_at_two_per_second():
    frames = media.video(mp4(seconds=2, fps=10))
    assert frames.shape == (4, 48, 64, 3)  # 2 s at media.VIDEO_FPS
    with pytest.raises(ValueError, match="no video stream"):
        media.video(wav())
    with pytest.raises(ValueError, match="could not be read"):
        media.video(b"not a video")


def test_loaded_passes_media_only_when_there_is_some():
    from ollajev.adapters.base import Loaded

    calls = []
    loaded = Loaded("m", "", None, {}, lambda state, questions, **media: calls.append(media) or {"answers": {}})
    loaded.system_one("s", {})
    loaded.system_one("s", {}, {"images": [], "audio": []})
    loaded.system_one("s", {}, {"images": [b"x"], "videos": []})
    assert calls == [{}, {}, {"images": [b"x"]}]


def test_unload_clears_the_mlx_cache(monkeypatch):
    import sys

    cleared = []
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(clear_cache=lambda: cleared.append(True)))
    manager._empty_device_cache()
    assert cleared == [True]


def test_media_the_family_cannot_read_is_refused_before_a_load(monkeypatch):
    mgr = manager.Manager.__new__(manager.Manager)
    family = SimpleNamespace(limits=lambda r: {"inputs": ["text"]})
    mgr.resolve = lambda name: SimpleNamespace(family=family)  # type: ignore[method-assign]
    mgr.get = lambda *a, **k: pytest.fail("loaded a model for media it cannot read")  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="takes no images"):
        mgr.run("m", "s", {}, inputs={"images": [b"x"]})


def test_long_videos_are_refused_from_the_header(monkeypatch):
    monkeypatch.setattr(media, "MAX_VIDEO_SECONDS", 1)
    with pytest.raises(ValueError, match="videos\\[0\\] is 2 s long"):
        media.check({"inputs": ["text", "video"]}, {"videos": [mp4(seconds=2)]})
    with pytest.raises(ValueError, match="longer than 1 s"):  # decoding stops at the cap too
        media.video(mp4(seconds=2))
    media.check({"inputs": ["text", "video"]}, {"videos": [b"not a video"]})  # unreadable: the decoder reports it
