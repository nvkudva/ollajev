"""One adapter per model family. Each turns a Jev request into that family's inference call."""

from __future__ import annotations

import importlib
import math
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..store import Resolved
    from .base import Adapter


class Family(Protocol):
    name: str
    runs_repo_code: bool  # the model repo ships Python we import; needs the user's trust first

    def matches(self, repo_id: str, files: list[str]) -> bool: ...

    def allow_patterns(self, resolved: Resolved) -> list[str] | None: ...

    def limits(self, resolved: Resolved) -> dict[str, Any]: ...

    def load(self, path: str, resolved: Resolved, device: str | None) -> Adapter: ...


def families() -> list[Family]:
    from . import (
        clef,
        clef_mlx,
        d1,
        decider,
        decision1,
        intern,
        jevk5,
        julia,
        kev,
        laya,
        onejev,
        openjev,
        rlcd,
        vega,
        von,
    )

    found = [
        vega.FAMILY,
        laya.FAMILY,
        decider.FAMILY,
        clef.FAMILY,
        clef_mlx.FAMILY,
        julia.FAMILY,
        openjev.FAMILY,
        kev.FAMILY,
        intern.FAMILY,
        decision1.FAMILY,
        d1.FAMILY,
        jevk5.FAMILY,
        onejev.FAMILY,
        rlcd.FAMILY,
        von.FAMILY,
    ]
    return [f for f in found if f is not None]


def detect(repo_id: str, files: list[str]) -> Family:
    for family in families():
        if family.matches(repo_id, files):
            return family
    names = ", ".join(f.name for f in families())
    raise LookupError(f"{repo_id} is not a supported System One model (supported families: {names})")


def pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_import_lock = threading.Lock()


def import_from(root: str, module: str) -> Any:
    """Import `module` from a model snapshot. Drops a same-named package loaded from another snapshot,
    since bundled packages import themselves by absolute name."""
    top = module.split(".")[0]
    with _import_lock:  # sys.path and sys.modules are process-wide
        loaded = sys.modules.get(top)
        if loaded is not None and not str(getattr(loaded, "__file__", "") or "").startswith(root):
            for name in [n for n in sys.modules if n == top or n.startswith(top + ".")]:
                del sys.modules[name]
        sys.path.insert(0, root)
        try:
            importlib.invalidate_caches()
            return importlib.import_module(module)
        finally:
            sys.path.remove(root)


def cached_repo(repo: str, revision: str | None) -> str | None:
    """The downloaded folder of a base model `prefetch` fetched, or None when it is not on disk. Loading from the
    folder instead of the repo name skips a Hugging Face round trip per from_pretrained call (kev-0.5b: 20-30 s
    against 2.6 s)."""
    # Not snapshot_download(local_files_only=True): it refuses a snapshot missing README.md and the like, which
    # prefetch skips on purpose.
    from huggingface_hub import try_to_load_from_cache

    config_file = try_to_load_from_cache(repo, "config.json", revision=revision)
    return os.path.dirname(config_file) if isinstance(config_file, str) else None


# transformers' modeling modules whose linear-attention layers run causal_conv1d_fn, a depthwise conv1d. Without the
# CUDA-only causal-conv1d package they use the PyTorch reference, and PyTorch's CPU depthwise conv1d costs ~230 ms
# per call whatever the length (torch 2.14, macOS arm64): Vega 0.8B took 4 s per request on the CPU, 3.6 s of it here.
CONV_MODULES = ("qwen3_5", "qwen3_5_moe", "qwen3_next", "lfm2", "olmo_hybrid")


def fast_cpu_convs() -> None:
    """On the CPU, compute causal_conv1d_fn as K shifted multiply-adds (K is 4): the same sums in fp32, 40-4000x
    faster. Other devices keep transformers' own path. Patches only modules already imported, after a model loads;
    the call sites look the function up in their module on each call."""
    for name in CONV_MODULES:
        module = sys.modules.get(f"transformers.models.{name}.modeling_{name}")
        original = getattr(module, "causal_conv1d_fn", None)
        if original is None or getattr(original, "_ollajev_cpu", False):
            continue
        module.causal_conv1d_fn = _cpu_conv(original)  # type: ignore[union-attr]


def _cpu_conv(original: Any) -> Any:
    import torch.nn.functional as F
    from transformers.activations import ACT2FN

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if hidden_states.device.type != "cpu":
            return original(hidden_states, weight, bias=bias, activation=activation, **kwargs)
        size, length = weight.shape[-1], hidden_states.shape[-1]
        x = F.pad(hidden_states.float(), (size - 1, 0))
        w = weight.float()
        out = x[..., :length] * w[:, :1]
        for k in range(1, size):
            out = out + x[..., k : k + length] * w[:, k : k + 1]
        if bias is not None:
            out = out + bias.float()[:, None]
        out = out.to(weight.dtype)
        if activation is not None:
            out = ACT2FN[activation](out)
        return out.to(hidden_states.dtype)

    causal_conv1d_fn._ollajev_cpu = True  # type: ignore[attr-defined]
    return causal_conv1d_fn


def text_state(state: Any) -> str:
    """Models that take only text get JSON states as compact JSON text."""
    import json

    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def instructions_or_name(qid: str, question: dict[str, Any]) -> Any:
    """Jev leaves instructions optional; most models need text. Fall back to the humanized question id."""
    return question.get("instructions") if question.get("instructions") not in (None, "") else qid.replace("_", " ")


def wire_questions(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Jev questions -> each model's `{type, instructions, criteria}`: instructions fall back to the
    question id, and None criteria are dropped (Jev leaves them optional)."""
    qs: dict[str, Any] = {}
    for qid, q in questions.items():
        out: dict[str, Any] = {"type": q["type"], "instructions": instructions_or_name(qid, q)}
        if q.get("criteria") is not None:
            out["criteria"] = q["criteria"]
        qs[qid] = out
    return qs


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over letter logits; temperature 1 is the plain one.

    Shared by the letter-logit readouts (jevk5 calibrated, onejev plain). The engines themselves
    stay separate: cached prefix-continue vs fresh full-prompt forwards are different numerics.
    """
    top = max(logits)
    weights = [math.exp((z - top) / temperature) for z in logits]
    total = sum(weights)
    return [w / total for w in weights]


def has(files: list[str], *names: str) -> bool:
    present = {Path(f).name for f in files} | set(files)
    return all(n in present for n in names)
