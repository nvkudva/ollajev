"""One adapter per model family. Each turns a Jev request into that family's inference call."""

from __future__ import annotations

import importlib
import math
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
    from . import clef, decider, decision1, intern, jevk5, julia, kev, laya, onejev, openjev, rlcd, von

    found = [
        laya.FAMILY,
        decider.FAMILY,
        clef.FAMILY,
        julia.FAMILY,
        openjev.FAMILY,
        kev.FAMILY,
        intern.FAMILY,
        decision1.FAMILY,
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
