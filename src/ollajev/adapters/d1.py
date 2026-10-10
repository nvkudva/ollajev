"""LiquidAI/d1-omni-*: the repo's own modeling code through transformers' trust_remote_code.

Reads text with either images or one voice clip (up to 30 s), not both in one request.
"""

from __future__ import annotations

from typing import Any

from .. import media
from . import has, wire_questions
from .base import Loaded

LIMITS = {
    "max_levels": 10,
    "max_tokens": 16384,
    "languages": "Multilingual (audio: English)",
    "inputs": ["text", "image", "audio"],
    "max_audio": 1,
}


class _D1:
    name = "d1"
    runs_repo_code = True

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return has(files, "modeling_d1.py", "config.json")

    def allow_patterns(self, resolved) -> list[str]:
        return ["*.json", "*.py", "*.safetensors"]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        import torch
        from transformers import AutoModel

        device = device or "cpu"
        # Trained in float32; float16 keeps its answers on GPUs, bfloat16 does not (per the model card).
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype, local_files_only=True).to(device)

        def predict(state: Any, questions: dict[str, dict[str, Any]], images=(), audio=()) -> dict[str, Any]:
            if images and audio:
                raise ValueError("this model takes images or audio in one request, not both")
            return model.system_one(
                state,
                wire_questions(questions),
                images=[media.image(x) for x in images] or None,
                audio=media.audio(audio[0]) if audio else None,
            )

        return Loaded(resolved.name, "d1-omni LFM2.5 encoder decision model", None, LIMITS, predict, device=device)


FAMILY = _D1()
