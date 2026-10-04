"""Resolve model names to a pinned Hugging Face snapshot, download it, list and delete downloads."""

from __future__ import annotations

import functools
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from huggingface_hub import CachedRevisionInfo, HfApi, HFCacheInfo, set_client_factory, snapshot_download
from huggingface_hub import scan_cache_dir as _scan_cache_dir
from huggingface_hub.constants import ENDPOINT as HF_ENDPOINT
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.utils import tqdm as hf_tqdm
from huggingface_hub.utils._http import default_client_factory  # the Hub's stock client, to adjust

from . import config, names
from .adapters import Family, detect, families
from .names import Ref, parse

IPV6_PROBE_SECONDS = 1.5
HF_TIMEOUT = 10  # seconds: a slow Hub lookup must not hold up a search or listing
SEARCH_POOL = 16  # base repos fetched in parallel when a search needs files they have
SEARCH_CANDIDATES = 100  # most-downloaded matches checked for support before ranking


def _ipv6_reaches(host: str) -> bool:
    """Whether a TCP connection to `host` over IPv6 opens. True when the host has no IPv6 address, since then
    nothing tries IPv6."""
    try:
        addresses = socket.getaddrinfo(host, 443, socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return True
    try:
        with socket.create_connection(addresses[0][4][:2], timeout=IPV6_PROBE_SECONDS):
            return True
    except OSError:
        return False


def _hub_client() -> httpx.Client:
    """Hugging Face's own HTTP client, made to use IPv4 when IPv6 to the Hub does not connect. httpx tries IPv6
    first and waits 7-15 s for it to fail on every new connection, where browsers and curl race both; on such a
    network every search, lookup and download started that much later. Checked once, on the first request."""
    client = default_client_factory()
    if _ipv6_reaches(urlparse(HF_ENDPOINT).hostname or "huggingface.co"):
        return client
    hooks = client.event_hooks
    client.close()
    transport = httpx.HTTPTransport(local_address="0.0.0.0")  # noqa: S104 a client-side bind, which only picks IPv4
    return httpx.Client(event_hooks=hooks, follow_redirects=True, timeout=None, transport=transport)  # noqa: S113 the stock client has no timeout either: a download can take hours


set_client_factory(_hub_client)


@dataclass
class Resolved:
    ref: Ref
    family: Family
    revision: str
    files: list[str]
    weights: str | None = None  # the one .gguf or .onnx file this name selects; None for full weights
    allow: list[str] | None = field(default=None)
    created: str | None = None  # the repo's creation date, recorded as release_date once it is downloaded
    base: Resolved | None = None  # a quantized copy's base repo, which supplies config and tokenizer files

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def repo_id(self) -> str:
        return self.ref.repo_id


SCAN_TTL = 2.0  # seconds a full cache walk is reused; downloads and deletes invalidate it
_scan_memo: tuple[float, str | None, HFCacheInfo | None] | None = None
_scan_lock = threading.Lock()


def scan_cache_dir(cache_dir: str | None = None, *, refresh: bool = False) -> HFCacheInfo | None:
    """The Hugging Face cache, or None before anything was ever downloaded to it.

    A full cache walk is reused for SCAN_TTL seconds, so listing calls that scan once per
    repo (snapshot/downloaded/tags) do one walk instead of N. Mutations refresh it; see download/delete.
    """
    global _scan_memo
    now = time.monotonic()
    with _scan_lock:
        if not refresh and _scan_memo is not None:
            at, memo_dir, memo = _scan_memo
            if memo_dir == cache_dir and now - at < SCAN_TTL:
                return memo
    try:
        info: HFCacheInfo | None = _scan_cache_dir(cache_dir)
    except CacheNotFound:
        info = None
    with _scan_lock:
        _scan_memo = (now, cache_dir, info)
    return info


def refresh_scan_cache(cache_dir: str | None = None) -> None:
    """Forget the memoized cache walk, after a download or delete changed what is on disk."""
    global _scan_memo
    with _scan_lock:
        if _scan_memo is None or _scan_memo[1] == cache_dir:
            _scan_memo = None


def pins() -> dict[str, str]:
    return config.load().get("pins", {})


def _pin(repo_id: str, sha: str, created: str | None) -> None:
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


def _quantized_from(base_models: dict | None) -> list[str]:
    """The repos Hugging Face lists as this repo's base when it is a quantization of them, else none."""
    if not base_models or base_models.get("relation") != "quantized":
        return []
    return [m["id"] for m in base_models.get("models", []) if m.get("id")]


def _remote_files(repo_id: str, revision: str | None) -> tuple[str, str | None, list[str], list[str]]:
    try:
        info = HfApi().model_info(repo_id, revision=revision, expand=["sha", "createdAt", "siblings", "baseModels"])
    except RepositoryNotFoundError:
        raise LookupError(
            f"model {repo_id!r} not found on Hugging Face (private or gated repos need `hf auth login`)"
        ) from None
    if info.sha is None:
        raise LookupError(f"Hugging Face returned no commit for {repo_id!r}")
    created = info.created_at.date().isoformat() if info.created_at else None
    files = [s.rfilename for s in info.siblings or []]
    return info.sha, created, files, _quantized_from(getattr(info, "base_models", None))


def snapshot(repo_id: str, revision: str) -> CachedRevisionInfo | None:
    """The downloaded snapshot of a repo at a commit, or None."""
    info = scan_cache_dir(config.models_dir())
    for repo in info.repos if info else ():
        if repo.repo_id != repo_id:
            continue
        for cached in repo.revisions:
            if cached.commit_hash == revision:
                return cached
    return None


def _local_files(repo_id: str, revision: str) -> list[str] | None:
    """The file list of a downloaded snapshot, so resolving works offline."""
    rev = snapshot(repo_id, revision)
    return [str(f.file_path.relative_to(rev.snapshot_path)) for f in rev.files] if rev else None


def resolve(name: str, *, online: bool = True) -> Resolved:
    """Name -> family, pinned revision, files to fetch. The first resolve of a repo pins its commit.

    Offline (`online=False`) it uses only what is downloaded and raises LookupError otherwise.
    """
    ref = parse(name)
    return _resolve(ref, pins().get(ref.repo_id), online=online, allow_base=True)


def _resolve(ref: Ref, revision: str | None, *, online: bool, allow_base: bool) -> Resolved:
    created = None
    base_ids: list[str] = []
    files = _local_files(ref.repo_id, revision) if revision else None
    weights = None
    if files is not None:
        try:
            weights = _pick(ref, files)
        except ValueError:
            if not online:
                raise LookupError(f"{ref.name} is not downloaded; run: ollajev pull {ref.name}") from None
            files = None  # a quant of a downloaded repo that is not on disk yet
    if files is None:
        if not online:
            raise LookupError(f"{ref.name} is not downloaded; run: ollajev pull {ref.name}")
        sha, created, files, base_ids = _remote_files(ref.repo_id, revision)
        if revision is None:
            revision = sha  # pinned by download(), once the weights are on disk
        weights = _pick(ref, files)
    if revision is None:  # unreachable: set from the pin or from the remote lookup above
        raise LookupError(f"no revision for {ref.name}")
    try:
        family = detect(ref.repo_id, files)
    except LookupError:
        base = _base(ref.repo_id, weights, files, base_ids, online=online) if allow_base else None
        if base is None or weights is None:
            raise
        allow = [weights, *names.sidecars(weights)]
        return Resolved(ref, base.family, revision, files, weights, allow=allow, created=created, base=base)
    if weights and not _runs(family, weights, files):
        if ref.tag:
            raise ValueError(f"the {family.name} family cannot run {weights}")
        weights = None  # e.g. an ONNX export next to the weights the family loads
    resolved = Resolved(ref, family, revision, files, weights, created=created)
    resolved.allow = family.allow_patterns(resolved)
    return resolved


def _runs(family: Family, weights: str, files: list[str]) -> bool:
    """Whether `family` loads this one weight file (a GGUF quant, an ONNX graph) of a repo with `files`."""
    runs = getattr(family, "runs_weights", None)
    return bool(runs and runs(weights, files))


def _inherits(family: Family) -> bool:
    """Whether quantized copies of this family's repos can run with the base repo's config files."""
    return bool(getattr(family, "base_files", None)) and not family.runs_repo_code


def _base(repo_id: str, weights: str | None, files: list[str], base_ids: list[str], *, online: bool) -> Resolved | None:
    """The base repo of a quantization that no family recognises by itself, when the base's family can run the
    copy's weight file with the base's config files. Offline it uses the base recorded at download time. A base
    whose family declines the file's layout makes this a LookupError that says so."""
    if weights is None:
        return None
    recorded = bases().get(repo_id)
    candidates = [(recorded["repo"], recorded["revision"])] if recorded else [(b, None) for b in base_ids]
    declined = None
    for base_id, revision in candidates:
        try:
            base = _resolve(Ref(base_id), revision, online=online, allow_base=False)
        except (LookupError, ValueError):
            continue
        if not _inherits(base.family):
            continue
        if not _runs(base.family, weights, files):
            declined = base.family.name
            continue
        base.allow = base.family.base_files  # type: ignore[attr-defined]
        return base
    if declined:
        raise LookupError(f"{repo_id}: the {declined} family does not run {weights} (unsupported file or layout)")
    return None


@dataclass(frozen=True)
class Hit:
    """A Hugging Face repo found by `search`. family is None when no adapter runs it."""

    repo_id: str
    downloads: int
    family: str | None
    files: tuple[str, ...] = ()  # the repo's file names, from the search itself
    copy: bool = False  # a quantized copy that runs on its base repo's family


@dataclass(frozen=True)
class Variant:
    """One downloadable form of a repo: a GGUF quant, an ONNX export, or the full weights. size is the download in
    bytes."""

    name: str  # what `pull` takes
    label: str
    size: int


def _family(repo_id: str, files: list[str], base_models: dict | None = None, files_of: Any = None) -> Family | None:
    """The family that runs a repo, through its quantized base when no family recognises the repo itself.
    files_of(repo_id) gives a base repo's file names; by default one Hugging Face call, cached."""
    try:
        return detect(repo_id, files)
    except LookupError:
        pass
    weights = names.weight_files(files)
    for base_id in _quantized_from(base_models) if weights else []:
        family = _family(base_id, list((files_of or _repo_files)(base_id)))
        if family and _inherits(family) and any(_runs(family, w, files) for w in weights):
            return family
    return None


@functools.lru_cache(maxsize=256)
def _repo_files(repo_id: str) -> tuple[str, ...]:
    """A repo's file names, or none when Hugging Face does not answer within 10 s: a slow lookup of a base repo
    must not hold up a search, and its copies then just show as unsupported."""
    try:
        info = HfApi().model_info(repo_id, expand=["siblings"], timeout=HF_TIMEOUT)
    except (RepositoryNotFoundError, httpx.HTTPError):
        return ()
    return tuple(s.rfilename for s in info.siblings or [])


@functools.lru_cache(maxsize=256)
def listing(repo_id: str) -> tuple[int | None, str | None]:
    """A repo's download count and the family that runs it, in one call; (None, None) when Hugging Face does not
    answer within 10 s. A model list needs both per repo."""
    try:
        info = HfApi().model_info(repo_id, expand=["downloads", "siblings", "baseModels"], timeout=HF_TIMEOUT)
    except (RepositoryNotFoundError, httpx.HTTPError):
        return None, None
    family = _family(repo_id, [s.rfilename for s in info.siblings or []], getattr(info, "base_models", None))
    return info.downloads or 0, family.name if family else None


def search(query: str, limit: int = 40) -> list[Hit]:
    """Repos matching every word of `query`, supported ones first, then most downloaded first. Only the 100 most
    downloaded matches are checked for support. A repo name or URL finds that repo."""
    api = HfApi()
    expand: list = ["siblings", "downloads", "baseModels"]
    try:
        found = [api.model_info(parse(query).repo_id, expand=expand)]
    except (ValueError, RepositoryNotFoundError):
        words = query.lower().split()
        if not words:
            return []
        # One listing with the file names and base models in it: under a second for 100 repos, where listing
        # ids alone took 13 s and then reading each repo's files took another call per repo.
        found = [
            model
            for model in api.list_models(
                search=max(words, key=len), sort="downloads", limit=SEARCH_CANDIDATES, expand=expand
            )
            if all(word in model.id.lower() for word in words)
        ]

    # A quantized copy is checked through its base repo. The base is usually in the results already; any other
    # base is fetched once, all of them in parallel, before the copies that need it are checked.
    files_by_repo = {model.id: tuple(sibling.rfilename for sibling in model.siblings or []) for model in found}
    missing = {
        base_id
        for model in found
        for base_id in _quantized_from(getattr(model, "base_models", None))
        if base_id not in files_by_repo
    }
    with ThreadPoolExecutor(SEARCH_POOL) as pool:
        files_by_repo.update(zip(missing, pool.map(_repo_files, missing), strict=True))

    def hit(model) -> Hit:
        files = list(files_by_repo[model.id])
        family = _family(model.id, files)
        copy = False
        if family is None:
            family = _family(
                model.id, files, getattr(model, "base_models", None), lambda base_id: files_by_repo.get(base_id, ())
            )
            copy = family is not None
        return Hit(model.id, model.downloads or 0, family.name if family else None, tuple(files), copy)

    hits = [hit(model) for model in found]
    return sorted(hits, key=lambda h: h.family is None)[:limit]  # stable: keeps the download order


def listed_variants(hit: Hit) -> list[Variant]:
    """A search hit's quants from the file names the search returned, with sizes still unknown (0)."""
    family = next((f for f in families() if f.name == hit.family), None)
    return _variants(hit.repo_id, "", dict.fromkeys(hit.files, 0), family, copy=hit.copy)


def variants(repo_id: str) -> list[Variant]:
    """A repo's GGUF quants or ONNX exports, or its full weights, with download sizes, smallest first."""
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True, expand=["baseModels"])
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    family = _family(repo_id, list(sizes))
    copy = False
    if family is None:
        family = _family(repo_id, list(sizes), getattr(info, "base_models", None))
        copy = family is not None
    return _variants(repo_id, info.sha or "", sizes, family, copy=copy)


def _variants(
    repo_id: str, sha: str, sizes: dict[str, int], family: Family | None, *, copy: bool = False
) -> list[Variant]:
    """copy: the weights come from this repo and the config files from its base, so only the weights count."""
    files = list(sizes)
    tags = {w: t for w, t in names.labels(files).items() if family is None or _runs(family, w, files)}
    found = []
    for weights, tag in tags.items() if tags else [(None, None)]:
        ref = Ref(repo_id, tag)
        if family and not copy:
            allow = family.allow_patterns(Resolved(ref, family, sha, files, weights))
        else:
            allow = [weights, *names.sidecars(weights)] if weights else None
        size = sum(sizes[f] for f in filter_repo_objects(files, allow_patterns=allow))
        found.append(Variant(ref.name, tag or "full weights", size))
    return sorted(found, key=lambda v: v.size)


def _pick(ref: Ref, files: list[str]) -> str | None:
    if names.weight_files(files):
        return names.pick_weights(files, ref.tag)
    if ref.tag:
        raise ValueError(f"{ref.repo_id} has no quantized files; drop ':{ref.tag}'")
    return None


def base_snapshot(resolved: Resolved) -> str | None:
    """The base snapshot a quantized copy reads config and tokenizer files from, or None for a plain repo
    (or until the base files are on disk)."""
    return local_path(resolved.base) if resolved.base else None


def local_path(resolved: Resolved) -> str | None:
    """The snapshot folder holding `r`'s weights, or None until it and, for a copy, its base files are on disk."""
    if resolved.base and local_path(resolved.base) is None:
        return None
    try:
        return snapshot_download(
            resolved.repo_id,
            revision=resolved.revision,
            allow_patterns=resolved.allow,
            cache_dir=config.models_dir(),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None


class Cancelled(Exception):
    pass


def _cancellable(cancel: threading.Event | None) -> type[hf_tqdm] | None:
    """A progress bar class that aborts the download at its next update once `cancel` is set. Partial files stay
    in the cache, so the next pull resumes."""
    if cancel is None:
        return None

    class Bar(hf_tqdm):
        def update(self, n: float | None = 1) -> bool | None:
            if cancel.is_set():
                raise Cancelled
            return super().update(n)

    return Bar


def download(resolved: Resolved, cancel: threading.Event | None = None) -> str:
    """Fetch the snapshot, then pin the repo to this commit if it has no pin yet. A copy also fetches its base's
    config files and records which base commit they came from. Setting `cancel` aborts it with Cancelled."""
    fetch = functools.partial(snapshot_download, cache_dir=config.models_dir(), tqdm_class=_cancellable(cancel))
    if resolved.base:
        fetch(resolved.base.repo_id, revision=resolved.base.revision, allow_patterns=resolved.base.allow)
    path = fetch(resolved.repo_id, revision=resolved.revision, allow_patterns=resolved.allow)
    if resolved.base:  # recorded once both are on disk, so a cancelled pull leaves no half record
        with config.edit() as data:
            data.setdefault("bases", {})[resolved.repo_id] = {
                "repo": resolved.base.repo_id,
                "revision": resolved.base.revision,
            }
    if resolved.repo_id not in pins():
        _pin(resolved.repo_id, resolved.revision, resolved.created)
    refresh_scan_cache(config.models_dir())
    return path


def prefetch(resolved: Resolved, cancel: threading.Event | None = None) -> None:
    """Fetch the extra files some families need beyond the repo (kev's base model); see needs_prefetch.
    Setting `cancel` aborts it with Cancelled."""
    resolved.family.prefetch(local_path(resolved), tqdm_class=_cancellable(cancel))


def needs_prefetch(resolved: Resolved) -> bool:
    return hasattr(resolved.family, "prefetch")


def download_size(resolved: Resolved) -> int:
    """Bytes `download` fetches for `r`'s own repo."""
    info = HfApi().model_info(resolved.repo_id, revision=resolved.revision, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    return sum(sizes[f] for f in filter_repo_objects(list(sizes), allow_patterns=resolved.allow))


def bytes_on_disk(repo_id: str) -> int:
    """Bytes of `repo_id` in the cache, partial downloads included."""
    blobs = Path(config.models_dir() or HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model") / "blobs"
    return sum(p.stat().st_size for p in blobs.glob("*") if p.is_file()) if blobs.is_dir() else 0


def extras(resolved: Resolved) -> list[tuple[str, str | None, list[str]]]:
    """The repos `r` needs on disk beside its own, as (repo, revision, files): kev's base model. Empty until `r`
    is downloaded, because only its own files say which base it is."""
    read = getattr(resolved.family, "extras", None)
    path = local_path(resolved)
    return read(path) if read and path else []


def on_disk(resolved: Resolved) -> int:
    """Bytes the model takes in the cache: its own repo plus the repos its family needs beside it."""
    return bytes_on_disk(resolved.repo_id) + sum(bytes_on_disk(repo) for repo, _, _ in extras(resolved))


def downloaded() -> dict[str, tuple[int, float]]:
    """repo_id -> (bytes on disk, last modified) for pinned repos in the cache."""
    wanted = pins()
    info = scan_cache_dir(config.models_dir())
    return {
        repo.repo_id: (repo.size_on_disk, repo.last_modified)
        for repo in (info.repos if info else ())
        if repo.repo_id in wanted and repo.repo_type == "model"
    }


def delete_file(resolved: Resolved) -> int:
    """Remove one weight file of a downloaded repo, with its ONNX external data. A blob goes only when no other
    snapshot links to it."""
    rev = snapshot(resolved.repo_id, resolved.revision)
    if rev is None or resolved.weights is None:
        return 0
    freed = 0
    for name in [resolved.weights, *names.sidecars(resolved.weights)]:
        link = rev.snapshot_path / name
        if not link.is_symlink():
            continue
        blob = link.resolve()
        shared = any(
            other != link and other.resolve() == blob
            for other in rev.snapshot_path.parent.glob("*/**/*")
            if other.is_symlink()
        )
        link.unlink()
        if not shared:
            freed += blob.stat().st_size
            blob.unlink()
    return freed


def remove(resolved: Resolved) -> int:
    """Delete what `r` names: one variant when the repo has others on disk, else the whole repo. Bytes freed."""
    path = local_path(resolved)
    if resolved.weights and path:
        on_disk = names.weight_files([str(p.relative_to(path)) for p in Path(path).glob("**/*")])
        if any(f != resolved.weights for f in on_disk):
            return delete_file(resolved)
    return delete(resolved.repo_id)


def delete(repo_id: str) -> int:
    """Remove every downloaded revision of `repo_id` and forget its pin. Returns bytes freed."""
    info = scan_cache_dir(config.models_dir())
    revisions = []
    for repo in info.repos if info else ():
        if repo.repo_id == repo_id:
            revisions.extend(revision.commit_hash for revision in repo.revisions)
    freed = 0
    if info and revisions:
        strategy = info.delete_revisions(*revisions)
        freed = strategy.expected_freed_size
        strategy.execute()
    with config.edit() as data:
        data.get("pins", {}).pop(repo_id, None)
        data.get("released", {}).pop(repo_id, None)
        data.get("bases", {}).pop(repo_id, None)
        data["trusted"] = [t for t in data.get("trusted", []) if not t.startswith(f"{repo_id}@")]
    refresh_scan_cache(config.models_dir())
    return freed


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
