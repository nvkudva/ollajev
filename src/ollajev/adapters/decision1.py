"""llm-semantic-router/Decision-1.0-*: the repo's own modeling code through transformers' trust_remote_code."""

from __future__ import annotations

from typing import Any

from . import wire_questions
from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 1024}


class _Decision1:
    name = "decision1"
    runs_repo_code = True

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "decision1_system_one.py" in files

    def allow_patterns(self, resolved) -> list[str]:
        return ["config.json", "*.py", "native/*", "native/**/*"]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(path, trust_remote_code=True, device=device or "cpu", local_files_only=True)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            return model.system_one(state=state, questions=wire_questions(questions))

        return Loaded(
            resolved.name, "Decision-1.0 Vela-encoder typed-decision model", None, self.limits(resolved), predict
        )


FAMILY = _Decision1()
