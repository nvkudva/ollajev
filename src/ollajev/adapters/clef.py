"""Cloudflare/clef, clef-flash and clef-omni: a multimodal Qwen backbone with a joint schema head, scored in one
forward pass.

The repos ship `joint_schema_model.py`; unchanged copies are vendored (ollajev/_vendor/clef for the dense Qwen3.5
models, ollajev/_vendor/clef_omni for the Qwen3-Omni MoE), so no repo code is imported. clef and clef-flash read
images and videos beside the state; clef-omni also reads audio, and hears a video's soundtrack with its frames.
"""

from __future__ import annotations

from typing import Any

from .. import media
from .base import Loaded

HEAD = ("joint_head_config.json", "joint_head.safetensors")
LIMITS = {
    "max_options": 255,
    "max_levels": 255,
    "max_tokens": 16384,
    "languages": "Multilingual",
    "inputs": ["text", "image", "video"],
}
OMNI_LIMITS = {**LIMITS, "max_tokens": 64000, "inputs": ["text", "image", "audio", "video"]}


def is_omni(path: str) -> bool:
    import json
    from pathlib import Path

    return json.loads((Path(path) / "config.json").read_text()).get("model_type") == "qwen3_omni_moe"


def record(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The model's input record. Wire questions keep their shape; a noul keeps the model's default wording for
    whichever of true and false the request does not describe."""
    qs = {}
    for qid, q in questions.items():
        out = {k: q[k] for k in ("type", "instructions", "criteria") if q.get(k) is not None}
        if q["type"] == "noul":
            crit = q.get("criteria")
            crit = crit if isinstance(crit, dict) else {}
            out["criteria"] = {k: v for k, v in crit.items() if k in ("true", "false") and v is not None}
        qs[qid] = out
    return {"state": state, "questions": qs}


def answer(kind: str, probabilities: dict[str, float]) -> dict[str, Any]:
    """One question's option probabilities as an adapter answer; normalize adds confidence and the legend."""
    if kind == "noul":
        return {"noul": probabilities["true"]}
    if kind == "choice":
        return {"choice": max(probabilities, key=probabilities.__getitem__), "probabilities": probabilities}
    return {"probabilities": probabilities}


class _Clef:
    name = "clef"
    runs_repo_code = False

    def limits(self, resolved) -> dict:
        """By name until loaded; the loaded model's limits follow its config.json."""
        return OMNI_LIMITS if "omni" in resolved.repo_id.lower() else LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        """Cloudflare's layout in bf16. MLX, vLLM, EXL3, OpenVINO and int8 copies keep the joint head but ship
        their own runtime script, and llm-compressor (FP8/NVFP4) copies a recipe.yaml; transformers runs none."""
        official = all(f in files for f in (*HEAD, "joint_schema_model.py", "model.safetensors.index.json"))
        scripts = [f for f in files if f.endswith(".py") and f != "joint_schema_model.py"]
        return official and not scripts and "recipe.yaml" not in files

    def allow_patterns(self, resolved) -> list[str]:
        # joint_schema_model.py is fetched only so offline detection sees it; the vendored copy is what runs.
        return ["*.json", "*.safetensors", "*.jinja", "tokenizer*", "joint_schema_model.py"]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        import torch

        from .._vendor.clef import joint_schema_model as dense
        from .._vendor.clef_omni import joint_schema_model as omni_model

        omni = is_omni(path)
        clef: Any = omni_model if omni else dense
        limits = OMNI_LIMITS if omni else LIMITS
        device = device or "cpu"
        model, processor = clef.load_release_model(
            path, device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16
        )
        tok = processor.tokenizer

        def predict(state: Any, questions: dict[str, dict[str, Any]], images=(), audio=(), videos=()) -> dict[str, Any]:
            rec = record(state, questions)
            if images:
                rec["images"] = [media.image(x) for x in images]
            if audio:
                rec["audio"] = [media.audio(x) for x in audio]
            if videos and omni:  # clef-omni decodes the bytes itself, so it can hear the soundtrack too
                rec["videos"] = list(videos)
            elif videos:  # frames already sampled at media.VIDEO_FPS: say so, or Qwen3.5 assumes 24 fps
                rec["videos"] = [media.video(x) for x in videos]
                rec["media_kwargs"] = {
                    "do_sample_frames": False,
                    "video_metadata": [
                        {"fps": media.VIDEO_FPS, "total_num_frames": len(v), "frames_indices": list(range(len(v)))}
                        for v in rec["videos"]
                    ],
                }
            enc = clef.encode_record(tok, rec, max_length=1 << 30, processor=processor)
            if len(enc.input_ids) > limits["max_tokens"]:
                raise ValueError(f"request is {len(enc.input_ids)} tokens; this model takes {limits['max_tokens']}")
            batch = clef.collate_records([enc], tok.pad_token_id, torch.device(device))
            with torch.inference_mode():
                logits = model(batch)[0]
            answers = {
                q.question_id: answer(
                    questions[q.question_id]["type"],
                    dict(zip(q.option_ids, x.float().softmax(-1).tolist(), strict=True)),
                )
                for q, x in zip(enc.questions, logits, strict=True)
            }
            return {"answers": answers, "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}

        return Loaded(
            resolved.name,
            f"Clef{'-Omni' if omni else ''} joint schema model (PyTorch {device})",
            None,
            limits,
            predict,
            device=device,
        )


FAMILY = _Clef()
