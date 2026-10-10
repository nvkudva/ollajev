"""mlx-community/clef, clef-flash and clef-omni in 4 and 8 bits: Clef quantized for MLX on Apple Silicon, no PyTorch.

The repos ship `clef_mlx.py`, one script for the dense Qwen3.5 models and another for the Qwen3-Omni MoE. Unchanged
copies are vendored (ollajev/_vendor/clef_mlx_dense, ollajev/_vendor/clef_mlx), so no repo code is imported.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, cast

from .. import media
from . import clef
from .base import Loaded


class _ClefMLX:
    name = "clef-mlx"
    runs_repo_code = False

    def limits(self, resolved) -> dict:
        """By name until loaded; the loaded model's limits follow its config.json."""
        return clef.OMNI_LIMITS if "omni" in resolved.repo_id.lower() else clef.LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return all(f in files for f in (*clef.HEAD, "clef_mlx.py", "model.safetensors.index.json"))

    def allow_patterns(self, resolved) -> list[str]:
        # clef_mlx.py is fetched only so offline detection sees it; the vendored copy is what runs.
        return ["*.json", "*.safetensors", "*.jinja", "tokenizer*", "clef_mlx.py"]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise ValueError(f"{resolved.name} runs on MLX, which needs a Mac with Apple Silicon")
        from .._vendor.clef_mlx import clef_mlx as omni_runtime
        from .._vendor.clef_mlx_dense import clef_mlx as dense_runtime

        omni = clef.is_omni(path)
        runtime: Any = omni_runtime if omni else dense_runtime
        limits = clef.OMNI_LIMITS if omni else clef.LIMITS
        model = runtime.load(path)

        def predict(state: Any, questions: dict[str, dict[str, Any]], images=(), audio=(), videos=()) -> dict[str, Any]:
            rec = clef.record(state, questions)
            if images:
                rec["images"] = [media.image(x) for x in images]
            if audio:
                rec["audio"] = [media.audio(x) for x in audio]
            if videos and omni:  # bytes: the runtime decodes them, soundtrack included
                rec["videos"] = list(videos)
            elif videos:
                if images:
                    raise ValueError("this model takes images or videos in one request, not both")
                # Frames already sampled at media.VIDEO_FPS: no resampling, and timestamps at that rate.
                rec["videos"] = [media.video(x) for x in videos]
                rec["media_kwargs"] = {"do_sample_frames": False, "source_fps": media.VIDEO_FPS}
            enc, logits = model.logits(rec, max_length=limits["max_tokens"], truncate=False)
            answers = {
                q.question_id: clef.answer(
                    questions[q.question_id]["type"],
                    dict(
                        zip(
                            q.option_ids,
                            cast(list, runtime.mx.softmax(x.astype(runtime.mx.float32)).tolist()),
                            strict=True,
                        )
                    ),
                )
                for q, x in zip(enc.questions, logits, strict=True)
            }
            return {"answers": answers, "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}

        return Loaded(
            resolved.name,
            f"Clef{'-Omni' if omni else ''} joint schema model (MLX, Apple GPU)",
            None,
            limits,
            predict,
            device="mlx",
        )


FAMILY = _ClefMLX()
