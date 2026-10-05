"""heman10x/rlcd-*: OpenJev (Verdict), a GLiClass ModernBERT encoder run with the `gliclass` PyPI package.

Prompts, the abstention slot and calibration follow the reference engine (Verdict-open-jev
core/formatting.py and core/engine_encoder.py). No repo code is imported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import has, instructions_or_name, text_state
from .base import Loaded

ABSTAIN = "insufficient evidence"  # the extra candidate the model was trained to pick when it cannot tell
SLOTS = 25  # max_capacity_logits: 24 options plus the abstention slot
MAX_TOKENS = 512  # the reference engine's tokenizer max_length
LIMITS = {"max_options": SLOTS - 1, "max_levels": SLOTS - 1, "max_tokens": MAX_TOKENS}


def prompt(question: str, text: str, labels: list[str]) -> str:
    """<<LABEL>>label1<<LABEL>>label2<<SEP>>text, labels ending with the abstention slot."""
    return (
        "".join(f"<<LABEL>>{label}" for label in [*labels, ABSTAIN])
        + "<<SEP>>"
        + (f"Question: {question}\n\nContext:\n{text}" if question else text)
    )


def build(context: str, qid: str, question: dict[str, Any]) -> tuple[str, list[str]]:
    """The model input for one wire question and the answer key of each substantive candidate."""
    kind, text = question["type"], instructions_or_name(qid, question)
    if kind == "choice":
        criteria = question.get("criteria") or {}
        keys = list(criteria)
        return prompt(text, context, [f"It is {criteria[k] or k}" for k in keys]), keys
    if kind == "score":
        levels = question.get("criteria") or []
        labels = [f"{level} (Value: {i})" for i, level in enumerate(levels)]
        return prompt(text, context, labels), [str(i) for i in range(len(levels))]
    if kind == "noul":
        body = f"Context:\n{context}\n\nEvaluate proposition: {text}"
        return prompt("", body, [f"true: {text}", f"false: not {text}"]), ["true", "false"]
    raise ValueError(f"question {qid!r}: unsupported type {kind!r}")


def temperature(calibrator: dict[str, Any], kind: int) -> float:
    """Per-candidate-count temperature when fitted for k (abstention slot included), else the global one."""
    per_k = calibrator.get("per_k") or {}
    if str(kind) in per_k:
        return float(per_k[str(kind)])
    if "log_temperature" in calibrator:
        return math.exp(float(calibrator["log_temperature"]))
    return float(calibrator.get("temperature", 1.0))


def shape(kind: str, keys: list[str], logits: list[float], temperature: float) -> dict[str, Any]:
    """Calibrated softmax over the candidates, then the answer over the substantive ones.

    The abstention mass is reported as p_abstain and left out of the answer, as the reference engine
    conditions noul and score on sufficient evidence."""
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exp = [math.exp(x - top) for x in scaled]
    total = sum(exp)
    probabilities = [x / total for x in exp]
    substantive = sum(probabilities[:-1]) or 1.0
    probs = {k: x / substantive for k, x in zip(keys, probabilities[:-1], strict=True)}
    out: dict[str, Any] = {"p_abstain": probabilities[-1]}
    if kind == "noul":
        out["noul"] = probs["true"]
    else:
        out["probabilities"] = probs
        if kind == "choice":
            out["choice"] = max(probs, key=probs.__getitem__)
    return out


class _Rlcd:
    name = "rlcd"
    runs_repo_code = False

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return has(files, "bundle_manifest.json", "calibrator.json", "model.safetensors", "tokenizer.json")

    def allow_patterns(self, resolved) -> list[str]:
        return ["*.json", "model.safetensors"]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        import torch
        from gliclass import GLiClassModel
        from transformers import AutoTokenizer

        root = Path(path)
        calibrator = json.loads((root / "calibrator.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = GLiClassModel.from_pretrained(path).to(device).eval()  # pyright: ignore[reportArgumentType]  # gliclass's stubs type the classmethod as an instance method

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            context = text_state(state)
            built = {qid: build(context, qid, q) for qid, q in questions.items()}
            for qid, (_, keys) in built.items():
                if not 2 <= len(keys) < SLOTS:
                    raise ValueError(f"question {qid!r}: this model takes 2 to {SLOTS - 1} candidates, got {len(keys)}")
            batch = tokenizer(
                [text for text, _ in built.values()],
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                logits = model(**batch).logits.float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("the model returned non-finite logits")
            answers = {}
            for row, (qid, (_, keys)) in enumerate(built.items()):
                slots = len(keys) + 1
                answers[qid] = shape(
                    questions[qid]["type"], keys, logits[row, :slots].tolist(), temperature(calibrator, slots)
                )
            return {"answers": answers}

        return Loaded(resolved.name, "OpenJev (Verdict) GLiClass ModernBERT decision encoder", None, LIMITS, predict)


FAMILY = _Rlcd()
