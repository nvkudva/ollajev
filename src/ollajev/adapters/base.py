"""The contract every model family's adapter implements."""

from __future__ import annotations

from typing import Any, Protocol


class Adapter(Protocol):
    name: str
    description: str
    released: str | None
    limits: dict[str, Any]  # max_options, max_levels, max_tokens, languages: shown by `show` and the playground

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Answer every question about `state`.

        `questions` is the request's map in wire form: {id: {"type", "instructions"?, "criteria"?}}.
        Returns {"answers": {id: answer}, "usage"?: {...}}; answers may carry model-specific extras,
        the server normalises them. Raise ValueError for a request this model cannot take (too many
        options, too long); it becomes a 422.
        """
        ...


class Loaded:
    """An Adapter built from a family's predict function."""

    def __init__(
        self,
        name: str,
        description: str,
        released: str | None,
        limits: dict[str, Any],
        predict,
        close=None,
        device: str | None = None,
    ):
        self.name, self.description, self.released, self.limits = name, description, released, limits
        self.device = device  # where it actually runs, when the family overrides the requested device
        self._predict, self._close = predict, close

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return self._predict(state, questions)

    def close(self) -> None:
        if self._close is not None:
            self._close()
