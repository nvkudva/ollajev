"""wfzyx/von: Option-Marker head on ModernBERT-large, run by the `von-sdk` PyPI package. No repo code is imported.

The SDK reads the snapshot's marker_calibration.json for the temperature map, the zero-shot Noul prior
and the order-invariant attention mode the weights were trained under. option_marker.pt holds every
weight (encoder and head) and is loaded strictly, so the repo's 1.6 GB model.safetensors is not fetched.
"""

from __future__ import annotations

from typing import Any

from . import has, text_state, wire_questions
from .base import Loaded

LIMITS = {"max_levels": 10, "max_tokens": 8192, "languages": "English"}
ALLOW = ["config.json", "option_marker.pt", "marker_calibration.json", "tokenizer*"]


class _EncoderFromConfig:
    """Stands in for AutoModel in the SDK: builds the encoder without weights; option_marker.pt fills it."""

    @staticmethod
    def from_pretrained(path: str, config: Any = None, **kwargs: Any) -> Any:
        from transformers import AutoModel

        return AutoModel.from_config(config)


def questions_for(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Wire questions in the SDK's shape: instructions are required there."""
    return wire_questions(questions)


def answers_from(response: Any) -> dict[str, Any]:
    """SDK response -> {"answers", "usage"}; Noul keeps the band-committed probability the SDK serves."""
    answers = {qid: a.model_dump(exclude_none=True) for qid, a in response.answers.items()}
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": 0}
    return {"answers": answers, "usage": usage}


class _Von:
    name = "von"
    runs_repo_code = False

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return has(files, "option_marker.pt", "marker_calibration.json")

    def allow_patterns(self, resolved) -> list[str]:
        return ALLOW

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        from von.backends.option_marker_backend import OptionMarkerBackend
        from von.models import option_marker

        backend = OptionMarkerBackend(checkpoint_dir=path, device=device or "cpu")
        # An over-long state becomes a 422 instead of a silent middle cut.
        backend.on_overflow = "refuse"
        stock = option_marker.AutoModel
        option_marker.AutoModel = _EncoderFromConfig
        try:
            backend._get_model()  # load now, not on the first request
        finally:
            option_marker.AutoModel = stock

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            text = state if isinstance(state, (str, dict)) else text_state(state)
            return answers_from(backend.evaluate(text, questions_for(questions)))

        return Loaded(
            resolved.name, "Von Option-Marker decision model, ModernBERT-large", None, self.limits(resolved), predict
        )


FAMILY = _Von()
