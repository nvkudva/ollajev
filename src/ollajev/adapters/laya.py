"""convaiinnovations/laya*: the `laya` PyPI package, on PyTorch or on onnxruntime for ONNX copies. No repo code is
imported.

An ONNX export of laya's DecisionModel takes the same five tensors and returns the same logits and act head, so
laya's own `Agent.system_one` builds the batch and reads the answers; only the forward pass changes.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .. import names
from . import wire_questions
from .base import Loaded

# Context lengths from the model cards; laya does not expose them.
CONTEXT = {"convaiinnovations/laya": 512, "convaiinnovations/laya-multilingual": 1024}
ALLOW = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
BASE_FILES = ["rl_agent_config.json", "tokenizer/*"]
INPUTS = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
# Graphs laya's forward pass does not describe: an encoder and a head exported apart, block-quantized layouts
# for browser runtimes, external data split in parts.
SPLIT = re.compile(r"(^|/)(encoder|head)[^/]*\.onnx$", re.IGNORECASE)
CUSTOM = re.compile(r"blk|block|int4|q4e8|q8e8", re.IGNORECASE)
DTYPES = {"tensor(int64)": "int64", "tensor(int32)": "int32", "tensor(bool)": "bool", "tensor(float)": "float32"}


class _Laya:
    name = "laya"
    runs_repo_code = False
    # A quantized ONNX copy runs with the base repo's config and tokenizer.
    base_files = BASE_FILES

    def limits(self, resolved) -> dict:
        repo = resolved.base.repo_id if resolved.base else resolved.repo_id
        return {"max_tokens": CONTEXT.get(repo, 1024)}

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "rl_agent_config.json" in files and "model.safetensors" in files

    def runs_weights(self, weights: str, files: list[str]) -> bool:
        if names.format_of(weights) != "onnx" or CUSTOM.search(weights):
            return False
        folder = weights.rpartition("/")[0]
        split = any(SPLIT.search(f) for f in files if f.rpartition("/")[0] == folder)
        parts = any(f.startswith(weights) and ".part" in f for f in files)
        return not split and not parts

    def allow_patterns(self, resolved) -> list[str]:
        return ALLOW

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        if resolved.weights:
            from .. import store

            # A copy's config and tokenizer come from its base's snapshot; the graph stays in the copy's.
            config_dir = store.base_snapshot(resolved) or path
            agent = _onnx_agent(config_dir, os.path.join(path, resolved.weights))
            backend = runs_on = "onnxruntime"
        else:
            import laya
            from transformers.initialization import no_init_weights

            # from_config randomly initialises the encoder, then a strict load_state_dict overwrites every
            # tensor of it; skipping the init saves ~24 s and is safe because that load is strict.
            with no_init_weights():
                agent = laya.load(path, device=device)
            backend, runs_on = f"PyTorch {device}", None

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            return agent.system_one(state, wire_questions(questions))

        return Loaded(
            resolved.name,
            f"Laya typed-decision encoder ({backend})",
            None,
            self.limits(resolved),
            predict,
            device=runs_on,
        )


def _onnx_agent(config_dir: str, graph: str) -> Any:
    """A laya Agent whose model is an onnxruntime session of `graph`, configured from `config_dir`."""
    import laya
    import numpy as np
    import onnxruntime
    import torch
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    session = onnxruntime.InferenceSession(graph, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != len(INPUTS) or len(session.get_outputs()) < 2:
        found = ", ".join(i.name for i in inputs)
        raise ValueError(f"{os.path.basename(graph)} is not a laya decision graph (inputs: {found})")
    dtypes = [np.dtype(DTYPES.get(i.type, "int64")) for i in inputs]

    def forward(*tensors):
        feed = {i.name: t.cpu().numpy().astype(d) for i, t, d in zip(inputs, tensors, dtypes, strict=True)}
        logits, act = session.run(None, feed)[:2]  # the act output is `act` or `act_logits`
        return torch.from_numpy(logits), torch.from_numpy(act)

    class OnnxAgent(laya.Agent):
        def __init__(self):  # what Agent.__init__ sets, with the session in place of the torch module
            _fix_tokenizer_config(config_dir)
            with open(os.path.join(config_dir, "rl_agent_config.json")) as f:
                self.cfg = json.load(f)
            self.tok = AutoTokenizer.from_pretrained(os.path.join(config_dir, "tokenizer"))
            self.model = forward  # type: ignore[assignment]
            self.device, self.dtype = torch.device("cpu"), torch.float32
            self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
            self.temperature_by_options = self.cfg.get("temperature_by_options", {})

    return OnnxAgent()


FAMILY = _Laya()
