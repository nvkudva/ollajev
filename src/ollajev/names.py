"""Model names, Ollama style: `<user>/<repo>`, `<user>/<repo>:<quant>` or `<user>/<repo>:<file>`.

An `hf.co/` or `huggingface.co/` prefix is accepted, so names copied from an Ollama command work,
and so are browser URLs of a repo or one of its files (`.../tree/main`, `.../blob/main/x.gguf`).
A repo's weight files are its variants: GGUF quants, or ONNX exports named by precision (`fp16`, `int8`).
The quant is matched case-insensitively against the repo's file names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PREFIXES = (
    "https://huggingface.co/",
    "https://hf.co/",
    "http://huggingface.co/",
    "http://hf.co/",
    "hf.co/",
    "huggingface.co/",
)
# Ollama's order when a repo has no Q4_K_M: the first quant found, best compromise first.
QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"]
# An ONNX repo's main export (no precision in its name) first, then full precision.
PRECISION_PREFERENCE = [None, "fp32"]
FORMATS = (".gguf", ".onnx")
_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
_URL_PATH = re.compile(r"^([^/]+/[^/]+)/(?:blob|resolve|tree)/[^/]+(?:/(.*))?$")


@dataclass(frozen=True)
class Ref:
    repo_id: str
    tag: str | None = None  # quant or file name as written, None for the repo's default

    @property
    def name(self) -> str:
        return f"{self.repo_id}:{self.tag}" if self.tag else self.repo_id


def parse(name: str) -> Ref:
    text = name.strip()
    for prefix in PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].split("?", 1)[0].split("#", 1)[0].rstrip("/")
            if m := _URL_PATH.match(text):
                file = (m.group(2) or "").rsplit("/", 1)[-1]
                text = f"{m.group(1)}:{file}" if file.lower().endswith(FORMATS) else m.group(1)
            break
    repo_id, _, tag = text.partition(":")
    if not _REPO.match(repo_id) or ".." in repo_id or any(part.strip(".") == "" for part in repo_id.split("/")):
        raise ValueError(f"not a Hugging Face model name: {name!r} (expected <user>/<repo>[:<quant>])")
    return Ref(repo_id, tag or None)


# One vocabulary each, shared by the file-name matchers below and the bare-tag matcher in runtime_of.
_QUANT = r"I?Q\d[\w]*|BF16|F16|F32"
_PRECISION = r"fp16|fp32|bf16|int8|uint8|int4|q4|q4f16|q8|bnb4|quantized"


def quant_of(filename: str) -> str | None:
    """`decider-4b-v2.1-Q4_K_M.gguf` or `laya_english_ud_q4_k_m.gguf` -> `Q4_K_M`."""
    match = re.search(rf"[._-]({_QUANT})\.gguf$", filename, re.IGNORECASE)
    return match.group(1).upper() if match else None


def precision_of(filename: str) -> str | None:
    """`model_fp16.onnx`, `model-int8.onnx` or `laya.int8.onnx` -> `fp16` / `int8`; None for `model.onnx`."""
    match = re.search(rf"[._-]({_PRECISION})\.onnx$", filename, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _is_tag(pattern: str, tag: str | None) -> bool:
    """Whether a bare `<repo>:<tag>` tag is a quant/precision name (no file extension to read)."""
    return re.fullmatch(pattern, tag or "", re.IGNORECASE) is not None


def tag_of(filename: str) -> str | None:
    """The quant or precision a weight file is named by, or None when its name says neither."""
    return quant_of(filename) or precision_of(filename)


def label_of(filename: str) -> str:
    """What names a weight file in `<repo>:<tag>`: its quant or precision, else its file name."""
    return tag_of(filename) or filename.rsplit("/", 1)[-1]


def format_of(filename: str) -> str | None:
    """`gguf` or `onnx` for a weight file, else None."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower()
    return suffix[1:] if suffix in FORMATS else None


# The runtime a weight file needs; `safetensors` is the repo's full PyTorch weights.
RUNTIMES = {"gguf": "llama.cpp", "onnx": "ONNX", "safetensors": "PyTorch"}


def runtime_of(tag: str | None) -> str:
    """The runtime a `<repo>:<tag>` name runs on: llama.cpp for a GGUF file or quant tag, ONNX Runtime for an
    export, PyTorch for the repo's full weights."""
    if tag:
        if format_of(tag) == "gguf" or _is_tag(_QUANT, tag):
            return RUNTIMES["gguf"]
        if format_of(tag) == "onnx" or _is_tag(_PRECISION, tag):
            return RUNTIMES["onnx"]
    return RUNTIMES["safetensors"]


def weight_files(files: list[str]) -> list[str]:
    """A repo's variants, one weight file each: its .gguf files, or its .onnx files when it has no GGUF.
    Vision projectors (`mmproj`) are not variants: no family loads them as a model."""
    ggufs = [f for f in files if format_of(f) == "gguf" and "mmproj" not in f.rsplit("/", 1)[-1].lower()]
    return ggufs or [f for f in files if format_of(f) == "onnx"]


def sidecars(weights: str) -> list[str]:
    """Files that must sit next to a weight file: an ONNX graph's external data (`x.onnx_data`, `x.onnx.data`)."""
    return [f"{weights}_data", f"{weights}.data"] if format_of(weights) == "onnx" else []


def labels(files: list[str]) -> dict[str, str]:
    """Each weight file's tag in `<repo>:<tag>`: its quant or precision when no other file shares it, else its
    file name, else its path in the repo."""
    found = weight_files(files)
    tags = [tag_of(path) for path in found]
    file_names = [path.rsplit("/", 1)[-1] for path in found]
    result = {}
    for path, tag, file_name in zip(found, tags, file_names, strict=True):
        if tag and tags.count(tag) == 1:
            result[path] = tag
        elif file_names.count(file_name) == 1:
            result[path] = file_name
        else:
            result[path] = path
    return result


def pick_weights(files: list[str], tag: str | None) -> str:
    """The one weight file `tag` names in a repo's file list, or the default variant when tag is None."""
    found = sorted(weight_files(files), key=lambda f: f.lower())
    if not found:
        raise ValueError("repo has no .gguf or .onnx files")
    if tag is None:
        by_tag: dict[str | None, str] = {}
        for f in found:
            by_tag.setdefault(tag_of(f), f)
        preference = QUANT_PREFERENCE if format_of(found[0]) == "gguf" else PRECISION_PREFERENCE
        for t in preference:
            if t in by_tag:
                return by_tag[t]
        return found[0]
    if tag.lower().endswith(FORMATS):
        matches = [f for f in found if f.lower() == tag.lower()] or [
            f for f in found if f.rsplit("/", 1)[-1].lower() == tag.lower()
        ]
    else:
        matches = [f for f in found if (tag_of(f) or "").lower() == tag.lower()]
    if len(matches) != 1:
        available = ", ".join(sorted(labels(files).values()))
        raise ValueError(f"no single file matches {tag!r}; available: {available}")
    return matches[0]
