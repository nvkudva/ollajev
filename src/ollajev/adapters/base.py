"""The contract every model family's adapter implements."""

from __future__ import annotations

from typing import Any, Protocol


class Adapter(Protocol):
    name: str
    description: str
    released: str | None
    limits: dict[str, Any]  # max_options, max_levels, max_tokens, languages, inputs: shown by `show` and the playground

    def system_one(
        self, state: Any, questions: dict[str, dict[str, Any]], media: dict[str, list[bytes]] | None = None
    ) -> dict[str, Any]:
        """Answer every question about `state`, and about its images, audio and videos when the model reads them.

        `questions` is the request's map in wire form: {id: {"type", "instructions"?, "criteria"?}}.
        Returns {"answers": {id: answer}, "usage"?: {...}}; answers may carry model-specific extras,
        the server normalises them. Raise ValueError for a request this model cannot take (too many
        options, too long); it becomes a 422. `media` maps `images`, `audio` and `videos` to raw file bytes;
        it only carries kinds listed in limits["inputs"].
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

    def system_one(
        self, state: Any, questions: dict[str, dict[str, Any]], media: dict[str, list[bytes]] | None = None
    ) -> dict[str, Any]:
        media = {k: v for k, v in (media or {}).items() if v}
        return self._predict(state, questions, **media) if media else self._predict(state, questions)

    def close(self) -> None:
        if self._close is not None:
            self._close()
