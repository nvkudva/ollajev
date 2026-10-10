"""MLX port of Cloudflare Clef-Omni (Qwen3-Omni-30B-A3B thinker + joint schema head).

Torch-free. The backbone (MoE text model with its vision and audio encoders) is loaded with
mlx-vlm, bf16 or quantized. The joint head is loaded from the original
``joint_head.safetensors`` / ``joint_head_config.json`` with no conversion. Prompt layout,
media preprocessing and multimodal rotary positions follow the reference
``joint_schema_model.py`` + transformers' Qwen3-Omni processor/thinker.

    import clef_mlx
    model = clef_mlx.load("mlx-community/clef-omni-4bit")
    model.predict(record)        # {question_id: {option_id: probability}}
    model.systemone(request)     # Jev/SystemOne /v1/systemone response body

Records may carry ``images``, ``audio`` and ``videos`` (paths, URLs, base64 data URLs or raw
bytes; images may also be PIL images, audio 16 kHz mono sample arrays, videos RGB frame arrays
sampled at 2 fps). Video soundtracks are heard alongside the frames when every video in the
record has one, exactly like the reference.

Command line (from a downloaded repo, --model defaults to that repo):

    python clef_mlx.py predict --state "..." --questions '{"q": {"type": "noul"}}' [--image x.jpg] [--audio x.wav] [--video x.mp4]
    python clef_mlx.py serve --port 8000     # POST /v1/systemone on 127.0.0.1
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import math
import os
import re
import urllib.request

# transformers is only used for the tokenizer and the (numpy) Whisper feature extractor; silence
# its "PyTorch was not found" advisory and config chatter unless the caller configured logging.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>\n"
AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>\n"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>\n"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
DEFAULT_MAX_LENGTH = 64000
AUDIO_SAMPLE_RATE = 16_000
VIDEO_FPS = 2.0
MAX_VIDEO_FRAME_PIXELS = 256 * 32 * 32
# Reference image size; video size is the transformers Qwen3-Omni processor default.
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_048_576}
VIDEO_SIZE = {"shortest_edge": 128 * 32 * 32, "longest_edge": 768 * 32 * 32}
POSITION_ID_PER_SECONDS = 13.0
_DATA_URL = re.compile(r"^data:[\w.+-]+/[\w.+-]+;base64,(?P<data>.+)$", re.DOTALL)

# This loader uses mlx-vlm internals (thinker.get_input_embeddings, language_model.model with
# inputs_embeds/position_ids/deepstack), which can change between minor releases.
# Tested: mlx 0.32.3, mlx-vlm 0.7.6.
TESTED_VERSIONS = {"mlx-vlm": (0, 7)}


class ContextTooLong(ValueError):
    pass


def _check_versions() -> None:
    import importlib.metadata
    import warnings

    for name, (major, minor) in TESTED_VERSIONS.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        major_minor = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
        if major_minor != (major, minor):
            warnings.warn(
                f"clef_mlx was tested with {name} {major}.{minor}.x but {version} is installed; "
                f'if loading fails or results look wrong, run: pip install "{name}>={major}.{minor},<{major}.{minor + 1}"',
                stacklevel=3,
            )


# --------------------------------------------------------------------------- encoding
# Mirrors joint_schema_model.encode_record exactly.


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    media: dict[str, Any] | None = field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


# --------------------------------------------------------------------------- media decoding
# Same decoding as the reference (PIL for images, PyAV for audio/video).


def _media_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    text = str(value)
    match = _DATA_URL.match(text)
    if match is not None:
        return base64.b64decode(match["data"])
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text, timeout=60) as response:
            return response.read()
    return Path(text).expanduser().read_bytes()


def _is_encoded(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, str, Path))


def load_image(value: Any):
    """RGB PIL image from a PIL image, path, URL, base64 data URL, or encoded bytes."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    with Image.open(io.BytesIO(_media_bytes(value))) as image:
        return image.convert("RGB")


def _decode_audio(data: bytes) -> np.ndarray | None:
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_SAMPLE_RATE)
        chunks = []
        for frame in itertools.chain(container.decode(stream), (None,)):
            chunks.extend(r.to_ndarray().reshape(-1) for r in resampler.resample(frame))
    if not chunks:
        return None
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    return samples if samples.size else None


def load_audio(value: Any) -> np.ndarray:
    """16 kHz mono samples from an encoded clip (path, URL, data URL, bytes) or a sample array."""
    if not _is_encoded(value):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    samples = _decode_audio(_media_bytes(value))
    if samples is None:
        raise ValueError("audio has no decodable audio stream")
    return samples


def load_video(value: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """RGB frames sampled at ``VIDEO_FPS`` and the clip's 16 kHz audio track, if any.

    Frame arrays are used as given (already sampled at ``VIDEO_FPS``) and have no audio track.
    """
    if not _is_encoded(value):
        return np.asarray(value), None
    import av

    data = _media_bytes(value)
    frames: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("video has no video stream")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < len(frames) / VIDEO_FPS:
                continue
            scale = min(1.0, math.sqrt(MAX_VIDEO_FRAME_PIXELS / (frame.width * frame.height)))
            width = max(2, round(frame.width * scale / 2) * 2)
            height = max(2, round(frame.height * scale / 2) * 2)
            frames.append(frame.to_ndarray(format="rgb24", width=width, height=height))
    if not frames:
        raise ValueError("video has no decodable frames")
    if len(frames) % 2:
        frames.append(frames[-1])
    return np.stack(frames), _decode_audio(data)


# --------------------------------------------------------------------------- media preprocessing
# Numpy ports of transformers' Qwen2VLImageProcessor / Qwen2VLVideoProcessor (what the
# checkpoint's processor_config.json selects) and WhisperFeatureExtractor (from transformers,
# numpy-only).


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class VisualPreprocessor:
    def __init__(self, config: dict[str, Any]):
        self.patch_size = int(config.get("patch_size", 16))
        self.temporal_patch_size = int(config.get("temporal_patch_size", 2))
        self.merge_size = int(config.get("merge_size", 2))
        self.mean = np.asarray(config.get("image_mean", [0.5] * 3), dtype=np.float32)[:, None, None]
        self.std = np.asarray(config.get("image_std", [0.5] * 3), dtype=np.float32)[:, None, None]
        self.rescale = float(config.get("rescale_factor", 1 / 255))

    def _resize(self, frames: np.ndarray, size: dict[str, int]) -> np.ndarray:
        """(T, H, W, 3) uint8 -> (T, 3, h, w) float32, bicubic + rescale + normalize."""
        from PIL import Image

        _, height, width, _ = frames.shape
        factor = self.patch_size * self.merge_size
        h, w = smart_resize(height, width, factor, size["shortest_edge"], size["longest_edge"])
        out = np.empty((len(frames), 3, h, w), dtype=np.float32)
        for i, frame in enumerate(frames):
            if (h, w) != (height, width):
                frame = np.asarray(Image.fromarray(frame).resize((w, h), resample=Image.BICUBIC))
            out[i] = frame.transpose(2, 0, 1).astype(np.float32)
        return (out * self.rescale - self.mean) / self.std

    def _patchify(self, video: np.ndarray) -> tuple[np.ndarray, list[int]]:
        tps, ps, ms = self.temporal_patch_size, self.patch_size, self.merge_size
        if pad := -len(video) % tps:
            video = np.concatenate([video, np.repeat(video[-1:], pad, axis=0)], axis=0)
        t, c, h, w = video.shape
        gt, gh, gw = t // tps, h // ps, w // ps
        patches = video.reshape(gt, tps, c, gh // ms, ms, ps, gw // ms, ms, ps)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(gt * gh * gw, c * tps * ps * ps), [gt, gh, gw]

    def images(self, images: list, size: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        patches, grids = [], []
        for image in images:
            p, g = self._patchify(self._resize(np.asarray(image)[None], size))
            patches.append(p)
            grids.append(g)
        return np.concatenate(patches), np.asarray(grids, dtype=np.int64)

    def videos(self, videos: list[np.ndarray], size: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
        patches, grids = [], []
        for video in videos:
            video = np.asarray(video)
            if video.ndim != 4 or video.shape[-1] not in (3, 4):
                raise ValueError(f"videos must be (frames, height, width, 3) RGB arrays, got {video.shape}")
            if video.dtype != np.uint8:
                video = np.clip(video, 0, 255).astype(np.uint8)
            p, g = self._patchify(self._resize(video[..., :3], size))
            patches.append(p)
            grids.append(g)
        return np.concatenate(patches), np.asarray(grids, dtype=np.int64)


def audio_output_lengths(mel_lengths):
    leave = mel_lengths % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lengths // 100) * 13


def _expand_placeholders(
    text: str,
    audio_lengths: list[int],
    image_grids: np.ndarray | None,
    video_grids: np.ndarray | None,
    use_audio_in_video: bool,
    merge_size: int,
    temporal_patch_size: int,
) -> str:
    """transformers Qwen3OmniMoeProcessor.replace_multimodal_special_tokens."""
    audio_it = iter(audio_lengths)
    image_it = iter(image_grids if image_grids is not None else [])
    video_it = iter(video_grids if video_grids is not None else [])
    merge = merge_size**2
    second_per_grid = temporal_patch_size / VIDEO_FPS
    pattern = "|".join(re.escape(t) for t in ("<|audio_pad|>", "<|image_pad|>", "<|video_pad|>"))
    for match in sorted(re.finditer(pattern, text), key=lambda m: m.start()):
        token = match.group()
        if token == "<|audio_pad|>":
            text = text.replace(token, "<|audio_placeholder|>" * int(next(audio_it)), 1)
        elif token == "<|image_pad|>":
            text = text.replace(token, "<|image_placeholder|>" * int(np.prod(next(image_it)) // merge), 1)
        elif not use_audio_in_video:
            text = text.replace(token, "<|video_placeholder|>" * int(np.prod(next(video_it)) // merge), 1)
        else:
            audio_idx = np.arange(next(audio_it))
            grid = next(video_it)
            h, w = grid[1] // merge_size, grid[2] // merge_size
            video_idx = np.broadcast_to(np.arange(grid[0]).reshape(-1, 1, 1), (grid[0], h, w)).reshape(-1)
            video_idx = video_idx * second_per_grid * POSITION_ID_PER_SECONDS
            vi = ai = 0
            parts = ["<|vision_start|><|audio_start|>"]
            while vi < len(video_idx) and ai < len(audio_idx):
                if video_idx[vi] <= audio_idx[ai]:
                    parts.append("<|video_placeholder|>")
                    vi += 1
                else:
                    parts.append("<|audio_placeholder|>")
                    ai += 1
            parts.append("<|video_placeholder|>" * (len(video_idx) - vi))
            parts.append("<|audio_placeholder|>" * (len(audio_idx) - ai))
            parts.append("<|audio_end|><|vision_end|>")
            text = text.replace("<|vision_start|><|video_pad|><|vision_end|>", "".join(parts), 1)
    return (
        text.replace("<|audio_placeholder|>", "<|audio_pad|>")
        .replace("<|image_placeholder|>", "<|image_pad|>")
        .replace("<|video_placeholder|>", "<|video_pad|>")
    )


class MediaProcessor:
    """Torch-free stand-in for the transformers Qwen3OmniMoeProcessor call in the reference."""

    def __init__(self, path: Path, tokenizer: Any):
        from transformers import WhisperFeatureExtractor

        config = json.loads((path / "processor_config.json").read_text())
        self.tokenizer = tokenizer
        self.visual = VisualPreprocessor(config.get("image_processor", {}))
        fe = dict(config.get("feature_extractor", {}))
        self.feature_extractor = WhisperFeatureExtractor(
            **{k: fe[k] for k in ("feature_size", "sampling_rate", "hop_length", "chunk_length", "n_fft",
                                  "padding_value", "dither", "return_attention_mask") if k in fe}
        )

    def __call__(self, record: dict[str, Any]) -> tuple[list[int], dict[str, Any] | None]:
        images = [load_image(image) for image in record.get("images") or []]
        audio = [load_audio(clip) for clip in record.get("audio") or []]
        decoded = [load_video(video) for video in record.get("videos") or []]
        if not images and not audio and not decoded:
            return [], None
        videos = [frames for frames, _ in decoded]
        tracks = [track for _, track in decoded]
        text = IMAGE_PLACEHOLDER * len(images) + AUDIO_PLACEHOLDER * len(audio) + VIDEO_PLACEHOLDER * len(videos)
        use_audio_in_video = bool(videos) and all(track is not None for track in tracks)
        if use_audio_in_video:
            audio.extend(tracks)

        kwargs = dict(record.get("media_kwargs") or {})
        image_size, video_size = dict(IMAGE_SIZE), dict(VIDEO_SIZE)
        if not videos:  # the reference passes size= to every modality when there is no video
            video_size = dict(IMAGE_SIZE)
        if "size" in kwargs:
            image_size = video_size = dict(kwargs.pop("size"))
        image_size.update((kwargs.pop("images_kwargs", {}) or {}).get("size", {}))
        video_size.update((kwargs.pop("videos_kwargs", {}) or {}).get("size", {}))
        if kwargs:
            raise ValueError(f"unsupported media_kwargs: {sorted(kwargs)}")

        media: dict[str, Any] = {"use_audio_in_video": use_audio_in_video}
        image_grids = video_grids = None
        audio_lengths: list[int] = []
        if images:
            pixel_values, image_grids = self.visual.images(images, image_size)
            media["pixel_values"] = mx.array(pixel_values)
            media["image_grid_thw"] = mx.array(image_grids)
        if videos:
            pixel_values_videos, video_grids = self.visual.videos(videos, video_size)
            media["pixel_values_videos"] = mx.array(pixel_values_videos)
            media["video_grid_thw"] = mx.array(video_grids)
            media["video_second_per_grid"] = [self.visual.temporal_patch_size / VIDEO_FPS] * len(videos)
        if audio:
            features = self.feature_extractor(
                audio, sampling_rate=AUDIO_SAMPLE_RATE, padding=True, truncation=False,
                return_attention_mask=True, return_tensors="np",
            )
            mask = np.asarray(features["attention_mask"])
            media["input_features"] = mx.array(np.asarray(features["input_features"], dtype=np.float32))
            media["feature_attention_mask"] = mx.array(mask.astype(np.int32))
            media["audio_seqlens"] = mask.sum(-1).astype(np.int64)
            audio_lengths = [int(n) for n in audio_output_lengths(media["audio_seqlens"])]
        text = _expand_placeholders(
            text, audio_lengths, image_grids, video_grids, use_audio_in_video,
            self.visual.merge_size, self.visual.temporal_patch_size,
        )
        return _tokens(self.tokenizer, text), media


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = DEFAULT_MAX_LENGTH,
    max_state_tokens: int | None = None,
    processor: MediaProcessor | None = None,
    truncate: bool = True,
) -> EncodedRecord:
    """Encode a record. Like the reference, the state is truncated to fit ``max_length``
    unless ``truncate=False``, in which case ``ContextTooLong`` is raised instead."""
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for qi, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(tokenizer, f"\nFIELD {qi + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ")
        )
        q_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        q_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans, option_ids = [], []
        for oi, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {oi + 1}: "))
            o_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((o_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(str(question_id), QUESTION_TYPES[str(question["type"])], (q_start, q_end),
                            tuple(option_spans), tuple(option_ids))
        )

    prefix_ids = _tokens(tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix_ids = _tokens(tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    if processor is None and any(record.get(k) for k in ("images", "audio", "videos")):
        raise ValueError("records with images, audio, or videos require a processor")
    media_ids, media = processor(record) if processor is not None else ([], None)
    prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed > max_length:
        raise ContextTooLong(f"schema requires {fixed} tokens before state; maximum is {max_length}")
    if not truncate and fixed + len(state_ids) > max_length:
        raise ContextTooLong(f"request needs {fixed + len(state_ids)} tokens; maximum is {max_length}")
    state_ids = state_ids[: max_length - fixed]
    off = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            q.question_id, q.question_type,
            (q.question_span[0] + off, q.question_span[1] + off),
            tuple((s + off, e + off) for s, e in q.option_spans),
            q.option_ids,
        )
        for q in questions
    )
    return EncodedRecord(tuple(prefix_ids + state_ids + schema_ids + suffix_ids), shifted, media)


# --------------------------------------------------------------------------- rotary positions


def rope_index(
    input_ids: list[int],
    config: Any,
    image_grid_thw: np.ndarray | None = None,
    video_grid_thw: np.ndarray | None = None,
    use_audio_in_video: bool = False,
    audio_seqlens: np.ndarray | None = None,
    second_per_grids: list[float] | None = None,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Port of transformers Qwen3-Omni ``get_rope_index`` for one unpadded sequence -> (3, L)."""
    n = len(input_ids)
    if image_grid_thw is None and video_grid_thw is None:
        return np.broadcast_to(np.arange(n), (3, n)).copy()
    image_id, video_id, audio_id = config.image_token_id, config.video_token_id, config.audio_token_id
    vision_start, audio_start = config.vision_start_token_id, config.audio_start_token_id
    pps = POSITION_ID_PER_SECONDS
    ids = np.asarray(input_ids)
    tokens = list(input_ids)
    vs_idx = np.flatnonzero(ids == vision_start)
    vision_tokens = ids[vs_idx + 1]
    audio_nums = int((ids == audio_start).sum())
    image_nums = int((vision_tokens == image_id).sum())
    video_nums = int((vision_tokens == (audio_start if use_audio_in_video else video_id)).sum())
    pos: list[np.ndarray] = []
    st = 0
    remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
    multimodal = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
    image_idx = video_idx = audio_idx = 0

    def nxt():
        return pos[-1].max() + 1 if pos else 0

    def vision_pos(start, grid, t_index):
        gh, gw = int(grid[1]) // spatial_merge_size, int(grid[2]) // spatial_merge_size
        t = np.repeat(np.asarray(t_index, dtype=np.float64), gh * gw)
        h = np.tile(np.repeat(np.arange(gh), gw), len(t_index)).astype(np.float64)
        w = np.tile(np.arange(gw), gh * len(t_index)).astype(np.float64)
        return np.stack([t, h, w]) + start

    def seq(length, start):
        return np.broadcast_to(np.arange(length, dtype=np.float64), (3, length)) + start

    for _ in range(multimodal):
        st_idx = nxt()
        if (image_id in tokens or video_id in tokens) and (remain_videos > 0 or remain_images > 0):
            ed_vision = tokens.index(vision_start, st)
        else:
            ed_vision = n + 1
        if audio_id in tokens and remain_audios > 0:
            ed_audio = tokens.index(audio_start, st)
        else:
            ed_audio = n + 1
        min_ed = min(ed_vision, ed_audio)
        text_len = min_ed - st
        if text_len:
            pos.append(seq(text_len, st_idx))
            st_idx += text_len
        in_video = min_ed == ed_vision and ed_vision + 1 == ed_audio
        bos_len = eos_len = 2 if in_video else 1
        pos.append(seq(bos_len, st_idx))
        st_idx += bos_len
        if min_ed == ed_audio:
            audio_len = int(audio_output_lengths(int(audio_seqlens[audio_idx])))
            pos.append(seq(audio_len, st_idx))
            st += text_len + bos_len + audio_len + eos_len
            audio_idx += 1
            remain_audios -= 1
        elif min_ed == ed_vision and tokens[ed_vision + 1] == image_id:
            grid = image_grid_thw[image_idx]
            pos.append(vision_pos(st_idx, grid, np.arange(int(grid[0])) * 1 * pps))
            st += text_len + bos_len + int(np.prod(grid)) // spatial_merge_size**2 + eos_len
            image_idx += 1
            remain_images -= 1
        elif min_ed == ed_vision and tokens[ed_vision + 1] == video_id:
            grid = video_grid_thw[video_idx]
            pos.append(vision_pos(st_idx, grid, np.arange(int(grid[0])) * float(second_per_grids[video_idx]) * pps))
            st += text_len + bos_len + int(np.prod(grid)) // spatial_merge_size**2 + eos_len
            video_idx += 1
            remain_videos -= 1
        elif in_video:
            audio_len = int(audio_output_lengths(int(audio_seqlens[audio_idx])))
            audio_pos = seq(audio_len, st_idx)
            grid = video_grid_thw[video_idx]
            video_pos = vision_pos(st_idx, grid, np.arange(int(grid[0])) * float(second_per_grids[video_idx]) * pps)
            vi = ai = 0
            while vi < video_pos.shape[1] and ai < audio_pos.shape[1]:
                if video_pos[0, vi] <= audio_pos[0, ai]:
                    pos.append(video_pos[:, vi : vi + 1])
                    vi += 1
                else:
                    pos.append(audio_pos[:, ai : ai + 1])
                    ai += 1
            if vi < video_pos.shape[1]:
                pos.append(video_pos[:, vi:])
            if ai < audio_pos.shape[1]:
                pos.append(audio_pos[:, ai:])
            st += text_len + bos_len + audio_len + int(np.prod(grid)) // spatial_merge_size**2 + eos_len
            audio_idx += 1
            video_idx += 1
            remain_videos -= 1
            remain_audios -= 1
        pos.append(seq(eos_len, nxt()))
    if st < n:
        pos.append(seq(n - st, nxt()))
    out = np.concatenate(pos, axis=1)
    if out.shape[1] != n:
        raise ValueError(f"rope index covers {out.shape[1]} tokens but the prompt has {n}")
    return out


# --------------------------------------------------------------------------- head


class MultiheadAttention(nn.Module):
    """torch.nn.MultiheadAttention (batch_first, packed in_proj) in MLX."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        w, b = self.in_proj_weight, self.in_proj_bias
        d = w.shape[1]
        q = q @ w[:d].T + b[:d]
        k = k @ w[d : 2 * d].T + b[d : 2 * d]
        v = v @ w[2 * d :].T + b[2 * d :]
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        hd = d // self.heads
        q = q.reshape(B, Lq, self.heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, Lk, self.heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, Lk, self.heads, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, Lq, d))


class FeedForward(nn.Module):
    def __init__(self, width: int, feedforward: int):
        super().__init__()
        self.fc1 = nn.Linear(width, feedforward)
        self.fc2 = nn.Linear(feedforward, width)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = MultiheadAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(width, feedforward)

    def __call__(self, queries, memory):
        m = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), m, m)
        return queries + self.feedforward(self.feedforward_norm(queries))


class TransformerDecoderLayer(nn.Module):
    """torch.nn.TransformerDecoderLayer(norm_first=True, activation='gelu')."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = MultiheadAttention(width, heads)
        self.multihead_attn = MultiheadAttention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x, memory):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


def _l2norm(x, eps=1e-12):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [TransformerDecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.scorer1 = nn.Linear(width * 4, width)
        self.scorer2 = nn.Linear(width, 1)
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            k = k.replace(".feedforward.0.", ".feedforward.fc1.").replace(".feedforward.3.", ".feedforward.fc2.")
            k = k.replace("residual_scorer.0.", "scorer1.").replace("residual_scorer.3.", "scorer2.")
            out[k] = v
        return out

    def __call__(self, hidden, input_ids, record: EncodedRecord, lexical_lookup):
        """hidden: (L, H) final backbone states for one record. Returns list of (n_opts,)."""
        h = self.hidden_norm(hidden)
        memory = self.memory_projection(h)[None]
        global_vector = h[-1]
        qs = record.questions
        question_vectors = mx.stack([h[s:e].mean(0) for s, e in (q.question_span for q in qs)])
        type_ids = mx.array([q.question_type for q in qs])

        option_contexts, lexical_options, counts = [], [], []
        for q in qs:
            option_contexts.append(mx.stack([h[s:e].mean(0) for s, e in q.option_spans]))
            lexical_options.append(mx.stack([lexical_lookup(input_ids[s:e]).mean(0) for s, e in q.option_spans]))
            counts.append(len(q.option_spans))

        option_queries = [
            self.option_context_projection(c)
            + self.option_lexical_projection(lx)
            + self.option_question_projection(question_vectors[i])[None]
            for i, (c, lx) in enumerate(zip(option_contexts, lexical_options))
        ]
        routed = mx.concatenate(option_queries, axis=0)[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]
        splits = list(itertools.accumulate(counts))[:-1]
        split_options = mx.split(routed, splits, axis=0) if splits else [routed]

        base_fields = self.question_projection(question_vectors)
        summaries = []
        for fld, options in zip(base_fields, split_options):
            w = mx.softmax((options @ fld) / math.sqrt(options.shape[-1]), axis=0)
            summaries.append((w[:, None] * options).sum(0))
        fields = (
            base_fields
            + self.option_summary_norm(mx.stack(summaries))
            + self.global_projection(global_vector)[None]
            + self.type_embedding(type_ids)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        gate = mx.sigmoid(self.residual_gate)
        logits = []
        for i, (fld, lexical, routed_opts) in enumerate(zip(fields, lexical_options, split_options)):
            anchor = _l2norm(question_vectors[i] + global_vector)
            prior = prior_scale * (_l2norm(lexical) @ anchor)
            options = self.option_norm(routed_opts)
            rf = mx.broadcast_to(fld[None], options.shape)
            cosine = (rf * options).sum(-1) / mx.maximum(
                mx.linalg.norm(rf, axis=-1) * mx.linalg.norm(options, axis=-1), 1e-8
            )
            feats = mx.concatenate([rf, options, rf * options, mx.abs(rf - options)], axis=-1)
            residual = self.scorer2(nn.gelu(self.scorer1(feats)))[:, 0]
            logits.append(prior + gate * (joint_scale * cosine + residual))
        return logits


# --------------------------------------------------------------------------- model


class ClefMLX:
    def __init__(self, backbone, tokenizer, head: JointSchemaHead, processor: MediaProcessor):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.processor = processor
        self.head = head
        self.thinker = backbone.thinker
        lm = self.thinker.language_model
        self._text_model = lm.model
        self._out = lm.lm_head if hasattr(lm, "lm_head") else lm.model.embed_tokens
        self._merge = self.thinker.config.vision_config.spatial_merge_size

    def _lexical(self, ids: mx.array) -> mx.array:
        """Rows of the output-embedding matrix (thinker lm_head), dequantized if needed."""
        m = self._out
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return mx.dequantize(
                m.weight[ids], m.scales[ids], m.get("biases")[ids] if "biases" in m else None,
                group_size=m.group_size, bits=m.bits, mode=getattr(m, "mode", "affine"),
            )
        return m.weight[ids]

    def _hidden(self, enc: EncodedRecord) -> mx.array:
        ids = mx.array(enc.input_ids)[None]
        media = enc.media
        if not media:
            positions = mx.broadcast_to(mx.arange(ids.shape[1])[None, None], (3, 1, ids.shape[1]))
            return self._text_model(ids, position_ids=positions)[0]
        feats = self.thinker.get_input_embeddings(
            ids,
            pixel_values=media.get("pixel_values"),
            pixel_values_videos=media.get("pixel_values_videos"),
            image_grid_thw=media.get("image_grid_thw"),
            video_grid_thw=media.get("video_grid_thw"),
            input_features=media.get("input_features"),
            feature_attention_mask=media.get("feature_attention_mask"),
        )
        positions = rope_index(
            list(enc.input_ids),
            self.thinker.config,
            image_grid_thw=np.asarray(media["image_grid_thw"]) if "image_grid_thw" in media else None,
            video_grid_thw=np.asarray(media["video_grid_thw"]) if "video_grid_thw" in media else None,
            use_audio_in_video=media.get("use_audio_in_video", False),
            audio_seqlens=media.get("audio_seqlens"),
            second_per_grids=media.get("video_second_per_grid"),
            spatial_merge_size=self._merge,
        )
        positions = mx.array(positions.astype(np.int32))[:, None, :]
        return self._text_model(
            ids,
            inputs_embeds=feats.inputs_embeds,
            position_ids=positions,
            visual_pos_masks=feats.visual_pos_masks,
            deepstack_visual_embeds=feats.deepstack_visual_embeds,
        )[0]

    def logits(self, record: dict[str, Any], **encode_kwargs) -> tuple[EncodedRecord, list[mx.array]]:
        enc = encode_record(self.tokenizer, record, processor=self.processor, **encode_kwargs)
        hidden = self._hidden(enc)
        out = self.head(hidden, mx.array(enc.input_ids), enc, self._lexical)
        mx.eval(out)
        return enc, out

    def predict(self, record: dict[str, Any], **kw) -> dict[str, dict[str, float]]:
        enc, logits = self.logits(record, **kw)
        return {
            q.question_id: dict(zip(q.option_ids, mx.softmax(lg.astype(mx.float32)).tolist()))
            for q, lg in zip(enc.questions, logits)
        }

    def systemone(self, request: dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH, truncate: bool = True) -> dict[str, Any]:
        """Answer a Jev/SystemOne ``POST /v1/systemone`` request body (same response body)."""
        questions = request.get("questions")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise ValueError("model and state are required")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for qid, q in questions.items():
            if q.get("type") not in QUESTION_TYPES:
                raise ValueError(f"{qid}: type must be noul, choice, or score")
            if q["type"] != "noul" and not q.get("criteria"):
                raise ValueError(f"{qid}: criteria must not be empty")
        enc, logits = self.logits(request, max_length=max_length, truncate=truncate)
        answers = {
            q.question_id: systemone_answer(
                questions[q.question_id], dict(zip(q.option_ids, mx.softmax(lg.astype(mx.float32)).tolist()))
            )
            for q, lg in zip(enc.questions, logits)
        }
        return {"model": request["model"], "answers": answers, "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(o) for o in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {o: round(probabilities[o], 4) for o in options},
        }
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(i * probabilities[lv] for i, lv in enumerate(levels)), 4),
        "confidence": round(max(probabilities[lv] for lv in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {lv: round(probabilities[lv], 4) for lv in levels},
    }


def load(path: str | Path, head_dtype=mx.bfloat16) -> ClefMLX:
    """Load an MLX Clef-Omni checkpoint (local dir or HF repo id)."""
    path = Path(path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(path)))
    _check_versions()
    from mlx_vlm.utils import load_model
    from transformers import AutoTokenizer

    backbone = load_model(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    cfg = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**cfg)
    weights = JointSchemaHead.sanitize(mx.load(str(path / "joint_head.safetensors")))
    head.load_weights(list(weights.items()), strict=True)
    head.set_dtype(head_dtype)
    mx.eval(head.parameters())
    return ClefMLX(backbone, tokenizer, head, MediaProcessor(path, tokenizer))


# --------------------------------------------------------------------------- CLI and server


def _default_model() -> str | None:
    """When run from inside a downloaded repo, default to that repo."""
    # absolute(), not resolve(): Hub cache snapshots are symlinks into a blob store.
    here = Path(__file__).absolute().parent
    return str(here) if (here / "joint_head.safetensors").exists() else None


def _model_name(model: str) -> str:
    """Short name for responses: 'clef-omni-4bit' for a repo id, a local dir, or a Hub cache snapshot."""
    path = Path(model)
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name.split("--")[-1]
    return path.name


def _json_arg(value: str) -> Any:
    """A JSON string, a path to a JSON file, or '-' for stdin."""
    if value == "-":
        import sys

        return json.load(sys.stdin)
    try:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return json.loads(candidate.read_text())
    except OSError:  # e.g. inline JSON too long to be a file name
        pass
    return json.loads(value)


def _state_arg(value: str) -> Any:
    """State text; JSON objects/arrays are parsed so they are rendered like the reference does."""
    stripped = value.strip()
    if stripped[:1] in "{[":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


def _remote_media(items: Any, kind: str) -> list[str]:
    """Media over HTTP: data URLs, raw base64 or http(s) URLs (never local paths)."""
    if not isinstance(items, list):
        raise ValueError(f"{kind} must be a list")
    out = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{kind} must be strings: data URLs, base64, or http(s) URLs")
        if item.startswith(("http://", "https://")) or _DATA_URL.match(item):
            out.append(item)
            continue
        try:
            out.append(base64.b64decode(item, validate=True))
        except ValueError:
            raise ValueError(f"could not decode {kind}: expected a data URL, base64, or http(s) URL") from None
    return out


def _cmd_predict(args) -> None:
    import sys

    if args.request:
        request = _json_arg(args.request)
    else:
        if args.state is None or args.questions is None:
            sys.exit("predict needs --state and --questions, or --request")
        request = {"state": _state_arg(args.state), "questions": _json_arg(args.questions)}
    request.setdefault("model", args.name or _model_name(args.model))
    for flag, key in (("image", "images"), ("audio", "audio"), ("video", "videos")):
        if getattr(args, flag):
            request[key] = list(request.get(key) or []) + [str(Path(p).expanduser()) for p in getattr(args, flag)]
    model = load(args.model)
    try:
        response = model.systemone(request, max_length=args.max_length, truncate=not args.no_truncate)
    except ValueError as error:
        sys.exit(f"error: {error}")
    print(json.dumps(response, indent=2, ensure_ascii=False))


def _cmd_serve(args) -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model = load(args.model)
    served_name = args.name or _model_name(args.model)
    model.systemone({"model": served_name, "state": "warmup",
                     "questions": {"w": {"type": "noul", "instructions": "Is this a warmup?"}}})
    lock = threading.Lock()  # one GPU: run requests one at a time
    max_body = args.max_body_mb * 1024 * 1024

    class Handler(BaseHTTPRequestHandler):
        server_version = "clef-mlx"

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": {"message": message}})

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": served_name})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            else:
                self._error(404, f"not found: {self.path}")

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._error(404, f"not found: {self.path}")
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_body:
                return self._error(413, f"request body larger than {args.max_body_mb} MB")
            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                request.setdefault("model", served_name)
                for key in ("images", "audio", "videos"):
                    if request.get(key):
                        request[key] = _remote_media(request[key], key)
                truncate = request.pop("truncate", not args.no_truncate)
            except (json.JSONDecodeError, ValueError, OSError) as error:
                return self._error(400, str(error))
            started = time.perf_counter()
            try:
                with lock:
                    response = model.systemone(request, max_length=args.max_length, truncate=bool(truncate))
            except ContextTooLong as error:
                return self._error(413, f"maximum context length exceeded: {error}")
            except (ValueError, KeyError, TypeError) as error:
                return self._error(400, str(error))
            except Exception as error:  # noqa: BLE001 - report, keep serving
                return self._error(500, f"{type(error).__name__}: {error}")
            response["usage"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self._send(200, response)

        def log_message(self, fmt, *fmt_args):
            if not args.quiet:
                super().log_message(fmt, *fmt_args)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {served_name} on http://{args.host}:{args.port}/v1/systemone "
          f"(truncate={'off' if args.no_truncate else 'on'}, max_length={args.max_length})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    import argparse

    default_model = _default_model()
    parser = argparse.ArgumentParser(prog="clef_mlx.py", description="Run Clef-Omni locally with MLX.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--model", default=default_model, required=default_model is None,
                       help="Hub repo id or local dir (default: the repo this file is in)")
        p.add_argument("--name", help="model name echoed in responses (default: the model dir/repo name)")
        p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help=f"prompt token limit (default {DEFAULT_MAX_LENGTH}, like the reference)")
        p.add_argument("--no-truncate", action="store_true",
                       help="refuse over-length inputs instead of truncating the state")

    p = sub.add_parser("predict", help="answer one request and print the SystemOne response")
    common(p)
    p.add_argument("--state", help="state text (JSON objects/arrays are parsed)")
    p.add_argument("--questions", help="questions as JSON, a .json file, or '-' for stdin")
    p.add_argument("--request", help="a full /v1/systemone request body as JSON, a .json file, or '-'")
    p.add_argument("--image", action="append", help="image file to attach (repeatable)")
    p.add_argument("--audio", action="append", help="audio file (wav, mp3, ...) to attach (repeatable)")
    p.add_argument("--video", action="append", help="video file (mp4, webm, ...) to attach (repeatable)")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("serve", help="serve POST /v1/systemone over HTTP")
    common(p)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1, local only)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-body-mb", type=int, default=256)
    p.add_argument("--quiet", action="store_true", help="no per-request access log")
    p.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
