"""The downloaded-model library: what is on disk, for the CLI, the TUI and the server alike.

This lives outside `server` so the front ends can list downloads without importing the HTTP
layer (which pulls in FastAPI) and without reaching across the UI->server boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import names, store
from .catalog import CATALOG
from .manager import canonical

DESCRIPTIONS = {e.name: e.description for e in CATALOG}


def _describe(name: str, resolved: store.Resolved, size: int, modified: float) -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "size": size,
        "modified_at": datetime.fromtimestamp(modified, UTC).isoformat(),
        "digest": resolved.revision,
        "description": DESCRIPTIONS.get(name, f"{resolved.family.name} typed-decision model"),
        # TypeSafe clients require a string; fall back to the download date for repos pinned offline.
        "release_date": store.released(resolved.repo_id) or datetime.fromtimestamp(modified, UTC).date().isoformat(),
        "details": {
            "family": resolved.family.name,
            "format": names.format_of(resolved.weights) if resolved.weights else "safetensors",
            "quantization_level": names.tag_of(resolved.weights) if resolved.weights else None,
        },
        "limits": resolved.family.limits(resolved),
    }


def tags() -> list[dict[str, Any]]:
    """One entry per downloaded weight file: a GGUF repo with two quants on disk is two models, and so is an ONNX
    repo with two exports."""
    out = []
    pins = store.pins()  # one config read, not one per downloaded repo
    for repo_id, (_, modified) in sorted(store.downloaded().items()):
        revision = pins[repo_id]
        rev = store.snapshot(repo_id, revision)
        if rev is None:
            continue
        snap = Path(rev.snapshot_path)
        local = names.labels(sorted(str(p.relative_to(snap)) for p in snap.glob("**/*")))
        for name in [f"{repo_id}:{tag}" for tag in local.values()] or [repo_id]:
            try:
                resolved = store.resolve(name, online=False)
            except (LookupError, ValueError):
                continue
            if resolved.weights:
                weight = sum(
                    (snap / f).stat().st_size
                    for f in [resolved.weights, *names.sidecars(resolved.weights)]
                    if (snap / f).is_file()
                )
            else:
                # No one weight file: the whole repo, plus the base a family needs beside it (kev's Qwen).
                weight = store.on_disk(resolved)
            out.append(_describe(canonical(resolved), resolved, weight, modified))
    return out
