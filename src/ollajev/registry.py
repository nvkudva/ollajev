"""Which repos are pinned and trusted: the download registry, without the download machinery.

Pins record the commit a repo resolved to; trust records the exact commits whose repo code
may run. Both live in the config file, so this module needs only config — unlike store,
it imports neither the Hub client nor the adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .store import Resolved


def pins() -> dict[str, str]:
    return config.load().get("pins", {})


def pin(repo_id: str, sha: str, created: str | None) -> None:
    with config.edit() as data:
        data.setdefault("pins", {})[repo_id] = sha
        if created:
            data.setdefault("released", {})[repo_id] = created


def released(repo_id: str) -> str | None:
    """The repo's creation date, reported as release_date like the hosted API does."""
    return config.load().get("released", {}).get(repo_id)


def bases() -> dict[str, dict[str, str]]:
    """copy repo -> {"repo", "revision"} of the base it was resolved against, so copies resolve offline."""
    return config.load().get("bases", {})


def trust_label(resolved: Resolved) -> str:
    """How `show` and the model manager describe whether a model runs code from its repo."""
    if not resolved.family.runs_repo_code:
        return "no repo code"
    if is_trusted(resolved):
        return "trusted"
    return "NOT trusted"


def is_trusted(resolved: Resolved) -> bool:
    return not resolved.family.runs_repo_code or f"{resolved.repo_id}@{resolved.revision}" in config.load().get(
        "trusted", []
    )


def trust(resolved: Resolved) -> None:
    key = f"{resolved.repo_id}@{resolved.revision}"
    with config.edit() as data:
        if key not in data.setdefault("trusted", []):
            data["trusted"].append(key)


def forget(repo_id: str) -> None:
    """Drop every record of `repo_id`: its pin, release date, base and trusted commits."""
    with config.edit() as data:
        data.get("pins", {}).pop(repo_id, None)
        data.get("released", {}).pop(repo_id, None)
        data.get("bases", {}).pop(repo_id, None)
        data["trusted"] = [t for t in data.get("trusted", []) if not t.startswith(f"{repo_id}@")]
