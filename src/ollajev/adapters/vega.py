"""nandakishorm/vega-*: a small trained physics engine over a frozen Qwen3.5, run by the repo's own `vega_api.py`.

One repo holds two sizes: the 0.8B at the root and the 4B under `4b/`, picked with the tag `:4b`. The frozen
backbone named in vega_config.json is a separate Hugging Face repo, fetched by `prefetch`.

Images go through the repo's `vega_vision.py`: zero-shot (the engine was trained on text), one image per request,
choice and noul questions only. It loads a second, multimodal copy of the backbone, so only on the first image.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from .. import media
from . import cached_repo, has, import_from, instructions_or_name, text_state
from .base import Loaded

LIMITS = {"max_tokens": 73728, "inputs": ["text", "image"], "max_images": 1}
BASE_FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"]
VISION_STATE = "<|vision_start|><|image_pad|><|vision_end|>"  # vega_vision.VISION_STATE
MODEL_FILES = ["vega_config.json", "config.json", "engine.safetensors", "adapters/*"]


def _folder(path: str, resolved) -> str:
    return os.path.join(path, resolved.ref.tag) if resolved.ref.tag else path


def _backbone(folder: str) -> str:
    with open(os.path.join(folder, "vega_config.json")) as f:
        return json.load(f)["backbone"]


def _remote_backbone(repo_id: str, revision: str | None, tag: str | None) -> str:
    """The backbone a size names, read from the Hub without writing to the cache: a cached file would make the
    size look downloaded."""
    from huggingface_hub import HfFileSystem

    path = f"{repo_id}/{tag}/vega_config.json" if tag else f"{repo_id}/vega_config.json"
    return json.loads(HfFileSystem().read_text(path, revision=revision))["backbone"]


def _question(qid: str, q: dict[str, Any]) -> dict[str, Any]:
    """Jev's noul is Vega's boolean; choice and score criteria pass through as they are."""
    out: dict[str, Any] = {
        "type": "boolean" if q["type"] == "noul" else q["type"],
        "instructions": instructions_or_name(qid, q),
    }
    if q.get("criteria") is not None:
        out["criteria"] = q["criteria"]
    return out


def _answer(kind: str, a: dict[str, Any]) -> dict[str, Any]:
    if kind == "noul":
        return {"noul": a["boolean"]}
    return {"probabilities": a["probs"], **({"choice": a["choice"]} if kind == "choice" else {})}


class _Vega:
    name = "vega"
    runs_repo_code = True

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return has(files, "vega_api.py", "vega_common.py")

    def folders(self, files: list[str]) -> list[str]:
        """The sizes besides the root one, each a folder with its own vega_config.json."""
        return sorted(f.split("/")[0] for f in files if f.count("/") == 1 and f.endswith("/vega_config.json"))

    def allow_patterns(self, resolved) -> list[str]:
        tag = resolved.ref.tag
        return ["vega_api.py", "vega_common.py", "vega_vision.py", *(f"{tag}/{f}" if tag else f for f in MODEL_FILES)]

    def extras(self, path: str) -> list[tuple[str, str | None, list[str]]]:
        """The backbone each downloaded size names; a size not on disk adds nothing."""
        folders = [path, *(os.path.join(path, d) for d in sorted(os.listdir(path)))]
        return [
            (_backbone(f), None, BASE_FILES) for f in folders if os.path.isfile(os.path.join(f, "vega_config.json"))
        ]

    def remote_extras(
        self, repo_id: str, revision: str | None, tag: str | None
    ) -> list[tuple[str, str | None, list[str]]]:
        """What `extras` will name once this size is downloaded, for download sizes before it is."""
        return [(_remote_backbone(repo_id, revision, tag), None, BASE_FILES)]

    def prefetch(self, path: str, tqdm_class=None) -> None:
        """Download the frozen backbone into the default Hugging Face cache, so `pull` leaves nothing to fetch."""
        from huggingface_hub import snapshot_download

        for repo, revision, allow in self.extras(path):
            snapshot_download(repo, revision=revision, allow_patterns=allow, tqdm_class=tqdm_class)

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        folder = _folder(path, resolved)
        backbone = _backbone(folder)
        Vega = import_from(path, "vega_api").Vega
        device = device or "cpu"
        backbone_path = cached_repo(backbone, None) or backbone
        model = Vega(folder, backbone=backbone_path, device=device)
        vision: list[Any] = []
        vision_lock = threading.Lock()

        def see() -> Any:
            with vision_lock:
                if not vision:
                    VegaVision = import_from(path, "vega_vision").VegaVision
                    vision.append(VegaVision(folder, backbone=backbone_path, device=device))
                return vision[0]

        def decide(state: Any, qs: dict[str, Any], images=()) -> dict[str, Any]:
            if not images:
                return model.decide(state, qs)
            if scores := [qid for qid, q in qs.items() if q["type"] == "score"]:
                raise media.Invalid(
                    "images",
                    f"Vega reads images only with choice and noul questions; {', '.join(map(repr, scores))} "
                    f"{'is a score question' if len(scores) == 1 else 'are score questions'}. Remove "
                    f"{'it' if len(scores) == 1 else 'them'} or the image, or use a model that scores images.",
                )
            vv = see()
            text = text_state(state) if state not in (None, "") else ""
            # The image takes the state's place in Vega's prompt; a text state goes before it.
            state_text = f"{text}\n{VISION_STATE}" if text else VISION_STATE
            return vv._decide(state_text, qs, image=media.image(images[0]))

        def predict(state: Any, questions: dict[str, dict[str, Any]], images=()) -> dict[str, Any]:
            out = decide(state, {qid: _question(qid, q) for qid, q in questions.items()}, images)
            return {
                "answers": {qid: _answer(q["type"], out["answers"][qid]) for qid, q in questions.items()},
                "usage": {"input_tokens": out["n_tokens"], "output_tokens": 0},
            }

        return Loaded(resolved.name, f"Vega physics engine on a frozen {backbone}", None, LIMITS, predict)


FAMILY = _Vega()
