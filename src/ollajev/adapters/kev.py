"""jaredpalmer/kev-*: LoRA adapter + pointer head on a Qwen base, scored by kev's own loader.

The repos ship no code; kev's loader is vendored from GitHub at a pinned commit (ollajev/_vendor/kev).
The base model is fetched at the revision head.pt names.
"""

from __future__ import annotations

import collections
import functools
import io
import os
import pickle
import zipfile
from typing import Any

from . import cached_repo
from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 8192}
BASE_FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"]


def _skip(*args: Any, **kwargs: Any) -> None:
    return None


class _MetadataOnly(pickle.Unpickler):
    """Reads a torch.save file's pickle without torch: tensors and storages come back as None, and only plain
    containers can be built, so nothing in the file runs."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" or module.startswith("torch."):
            return _skip
        raise pickle.UnpicklingError(f"unexpected {module}.{name}")

    def persistent_load(self, pid: Any) -> None:
        return None


def _head_metadata(head: str) -> dict[str, Any]:
    """head.pt's metadata. Read without importing torch, which costs most of a second on every listing of the
    downloads; torch.load only for a file this reader does not understand."""
    try:
        with zipfile.ZipFile(head) as archive:
            pickled = next(name for name in archive.namelist() if name.endswith("data.pkl"))
            meta = _MetadataOnly(io.BytesIO(archive.read(pickled))).load()
        if isinstance(meta, dict) and isinstance(meta.get("base"), str):
            return meta
    except (zipfile.BadZipFile, StopIteration, pickle.UnpicklingError, EOFError, ValueError, TypeError):
        pass
    import torch

    return torch.load(head, map_location="cpu", weights_only=True)


@functools.lru_cache(maxsize=32)
def _base(head: str) -> list[tuple[str, str | None, list[str]]]:
    """The base repo head.pt names, with the revision and the files it needs. Cached: the path carries its commit."""
    meta = _head_metadata(head)
    return [(meta["base"], meta.get("base_revision"), BASE_FILES)]


class _Kev:
    name = "kev"
    runs_repo_code = False

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "head.pt" in files and "adapter_config.json" in files

    def allow_patterns(self, resolved) -> list[str]:
        return ["*.json", "*.safetensors", "head.pt", "*.txt", "*.jinja"]

    def extras(self, path: str) -> list[tuple[str, str | None, list[str]]]:
        """The base model head.pt names, which has to sit in the cache beside this repo. Empty until downloaded."""
        head = os.path.join(path, "head.pt")
        return _base(head) if os.path.isfile(head) else []

    def prefetch(self, path: str, tqdm_class=None) -> None:
        """Download the base model head.pt names, so `pull` leaves nothing to fetch at load time. It goes to the
        default Hugging Face cache, where the vendored loader looks for it."""
        from huggingface_hub import snapshot_download

        for repo, revision, allow in self.extras(path):
            snapshot_download(repo, revision=revision, allow_patterns=allow, tqdm_class=tqdm_class)

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        import torch

        from .._vendor.kev.api import SystemOneRequest, to_answers, to_record
        from .._vendor.kev.checkpoint import Checkpoint, LoadOptions
        from .._vendor.kev.model import admit

        device = device or "cpu"
        # kev.serve's own defaults: bf16 off the CPU, sdpa attention on Apple GPUs.
        dtype = None if device == "cpu" else torch.bfloat16
        attention = "sdpa" if device == "mps" else None
        opts = LoadOptions(dtype=dtype, attn=attention)
        ck = Checkpoint(path)
        _use_cached_base(ck.meta)
        tok, model = ck.load(device, opts)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            rec, meta = to_record(SystemOneRequest.model_validate({"state": state, "questions": questions}))
            enc = admit(model, tok, rec)
            with torch.inference_mode():
                probs = model.probs(enc)
            return {
                "answers": to_answers([p.tolist() for p in probs], meta),
                "usage": {"input_tokens": len(enc["ids"]), "output_tokens": 0},
            }

        return Loaded(resolved.name, f"Kev pointer head on {ck.meta.base}", None, self.limits(resolved), predict)


def _use_cached_base(meta: Any) -> None:
    """Point the checkpoint at the base model's downloaded folder instead of its repo name. A base that is not
    downloaded keeps its name and downloads."""
    folder = cached_repo(meta.base, meta.base_revision)
    if folder:
        meta.base, meta.base_revision = folder, None


FAMILY = _Kev()
