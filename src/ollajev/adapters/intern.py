"""internlm/Intern-Decision-*: imports the `inference.py` bundled in the repo.

The checkpoint is a vision-language model; text-only requests never touch the vision tower, but its
weights are part of the checkpoint and load with it.
"""

from __future__ import annotations

from typing import Any

from . import import_from, wire_questions
from .base import Loaded

LIMITS = {"max_options": 62, "max_levels": 62, "max_questions": 16, "max_tokens": 8192, "languages": "Multilingual"}


class _Intern:
    name = "intern-decision"
    runs_repo_code = True

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "inference.py" in files and "video_preprocessor_config.json" in files

    def allow_patterns(self, resolved) -> None:
        return None

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        DecisionEngine = import_from(path, "inference").DecisionEngine
        device = device or "cpu"
        engine = DecisionEngine(path, device=device, dtype="float32" if device == "cpu" else "bfloat16")

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            qs = wire_questions(questions)
            for qid, q in questions.items():
                if q["type"] == "choice":
                    # None would be rendered as the text "None"
                    qs[qid]["criteria"] = {n: "" if d is None else d for n, d in q["criteria"].items()}
            return engine.predict({"state": state, "questions": qs})

        return Loaded(
            resolved.name, "Intern-Decision Qwen3.5 typed-decision model", None, self.limits(resolved), predict
        )


FAMILY = _Intern()
