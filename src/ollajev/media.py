"""Images, audio and video sent with a request: base64 data URLs in, decoded media out.

Only data URLs are accepted. A URL or a file path in a request would let any caller make the server
fetch a remote address or read a local file, so neither is ever opened.
"""

from __future__ import annotations

import base64
import binascii
import io
import itertools
import math
import re
from typing import Any

KINDS = {"images": "image", "audio": "audio", "videos": "video"}  # request field -> limits["inputs"] entry
AUDIO_RATE = 16_000
VIDEO_FPS = 2.0
VIDEO_FRAME_PIXELS = 256 * 32 * 32
# Longest video taken: at 2 fps its frames alone are ~0.5 GB decoded, and longer clips overflow every model's
# context anyway (clef-omni spends ~150-260 tokens per second of video).
MAX_VIDEO_SECONDS = 300

NOUNS = {"images": "image", "audio": "audio clip", "videos": "video"}
PLURALS = {"text": "text", "image": "images", "audio": "audio", "video": "videos"}  # limits["inputs"] entries
FORMATS = {
    "images": "PNG, JPEG, WebP, GIF or BMP",
    "audio": "WAV, MP3, FLAC, OGG or M4A",
    "videos": "MP4, MOV, WebM or MKV",
}


class Invalid(ValueError):
    """Media a request cannot use. `field` is the request field it came from; `kind` is set when the model reads no
    media of that kind, so the caller can name models that do."""

    def __init__(self, field: str, message: str, kind: str | None = None):
        super().__init__(message)
        self.field, self.kind = field, kind


_DATA_URL = re.compile(r"^data:[\w.+-]+/[\w.+-]+(?:;[\w.+-]+=[\w.+-]+)*;base64,(?P<data>.*)$", re.DOTALL)


def decode(field: str, urls: list[str]) -> list[bytes]:
    """A request field's data URLs as raw bytes."""
    out = []
    for i, url in enumerate(urls):
        match = _DATA_URL.match(url)
        if match is None:
            raise Invalid(
                field,
                f"{field}[{i}] is not a base64 data URL. Send the file itself as data:<type>;base64,<data>; "
                "web addresses and file paths are never fetched.",
            )
        try:
            out.append(base64.b64decode(match["data"], validate=True))
        except binascii.Error:
            raise Invalid(
                field,
                f"{field}[{i}] is not valid base64. Encode the whole file as standard base64, without line breaks.",
            ) from None
    return out


def check(limits: dict[str, Any], media: dict[str, list[bytes]]) -> None:
    """Reject media the model cannot read, before it loads."""
    inputs = limits.get("inputs", ["text"])
    for field, items in media.items():
        if items and KINDS[field] not in inputs:
            raise Invalid(
                field,
                f"this model reads {_reads(inputs)}; it takes no {field}. Remove the {NOUNS[field]}s "
                "or pick a model that reads them.",
                kind=KINDS[field],
            )
        if items and (n := limits.get(f"max_{field}")) and len(items) > n:
            what = NOUNS[field] if n == 1 else f"{NOUNS[field]}s"
            raise Invalid(
                field,
                f"this model takes at most {n} {what} per request, got {len(items)}. "
                "Send the others in separate requests.",
            )
    for i, data in enumerate(media.get("videos") or []):
        if (seconds := _seconds(data)) is not None and seconds > MAX_VIDEO_SECONDS:
            raise Invalid(
                "videos",
                f"videos[{i}] is {seconds:.0f} s long; the limit is {MAX_VIDEO_SECONDS} s. Trim it or send a shorter clip.",
            )


def _reads(inputs: list[str]) -> str:
    """["text", "image"] -> "text and images"."""
    words = [PLURALS.get(k, k) for k in inputs]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _seconds(data: bytes) -> float | None:
    """A media file's duration from its container header, without decoding; None when unknown or unreadable."""
    import av
    from av.error import FFmpegError

    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            return None if container.duration is None else container.duration / av.time_base
    except FFmpegError:
        return None  # the decoder reports the real error


def image(data: bytes) -> Any:
    """RGB pixels as a PIL image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("RGB")
    except UnidentifiedImageError:
        raise Invalid(
            "images", f"the file is not an image this server can read. Send a {FORMATS['images']} file."
        ) from None
    except (Image.DecompressionBombError, OSError) as exc:
        raise Invalid("images", f"image could not be read ({exc}). Send a {FORMATS['images']} file.") from None


def audio(data: bytes) -> Any:
    """16 kHz mono float32 samples from an audio file, or from a video's soundtrack."""
    import av
    import numpy as np
    from av.error import FFmpegError

    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                raise Invalid("audio", f"audio has no audio stream. Send a {FORMATS['audio']} file.")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
            chunks = [
                out.to_ndarray().reshape(-1)
                for frame in itertools.chain(container.decode(stream), (None,))
                for out in resampler.resample(frame)
            ]
    except FFmpegError as exc:
        raise Invalid("audio", f"audio could not be read ({exc}). Send a {FORMATS['audio']} file.") from None
    if not chunks:
        raise Invalid("audio", "audio has no decodable samples; the clip is empty or silent-only. Send another file.")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def video(data: bytes) -> Any:
    """RGB frames sampled at VIDEO_FPS, each scaled to at most VIDEO_FRAME_PIXELS, as one (T, H, W, 3) array."""
    import av
    import numpy as np
    from av.error import FFmpegError

    frames: list[Any] = []
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.video:
                raise Invalid("videos", f"video has no video stream. Send a {FORMATS['videos']} file.")
            stream = container.streams.video[0]
            start = None
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                start = frame.time if start is None else start  # streams need not start at t=0
                if frame.time - start + 1e-6 < len(frames) / VIDEO_FPS:
                    continue
                if len(frames) >= MAX_VIDEO_SECONDS * VIDEO_FPS:  # no duration in the header: stop here
                    raise Invalid(
                        "videos", f"video is longer than {MAX_VIDEO_SECONDS} s. Trim it or send a shorter clip."
                    )
                scale = min(1.0, math.sqrt(VIDEO_FRAME_PIXELS / (frame.width * frame.height)))
                width = max(2, round(frame.width * scale / 2) * 2)
                height = max(2, round(frame.height * scale / 2) * 2)
                frames.append(frame.to_ndarray(format="rgb24", width=width, height=height))
    except FFmpegError as exc:
        raise Invalid("videos", f"video could not be read ({exc}). Send a {FORMATS['videos']} file.") from None
    if not frames:
        raise Invalid("videos", f"video has no decodable frames. Send a {FORMATS['videos']} file.")
    return np.stack(frames)
