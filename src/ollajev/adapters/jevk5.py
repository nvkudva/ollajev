"""alibiserikbay/JevK5*: Qwen3.5 with a merged LoRA, read out as the answer letters' next-token logits.

The prompt, the option mapping and the many-option knockout come from the JevK5 runtime's `prompt.py`,
vendored at a pinned commit (ollajev/_vendor/jevk5). One forward pass per question of up to 16 options;
the letter logits are divided by the model's calibration temperature from jevk5_config.json. Runs on
PyTorch for safetensors repos, and on llama.cpp for the GGUF copies in alibiserikbay/JevK5-GGUF.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import threading
from pathlib import Path
from typing import Any

from .. import names
from .._vendor.jevk5 import prompt
from . import instructions_or_name
from .base import Loaded

MAX_TOKENS = 16384  # the runtime refuses longer inputs rather than cutting them
LIMITS = {"max_options": 255, "max_levels": 16, "max_tokens": MAX_TOKENS}
CONFIG = "jevk5_config.json"
BASE_FILES = [CONFIG]
# JevK5-GGUF holds several models and versions, but Hugging Face names only JevK5 (4B) as its base, so the
# base's jevk5_config.json fits only the 4B v0.3 files. Each file's own temperatures, from the JevK5-GGUF card:
GGUF_TEMPERATURES = {
    "jevk5-4b-v0.3": (1.22, 0.93),
    "jevk5-9b-v0.3.3": (1.316, 1.05),
    "jevk5-9b-v0.3": (1.049, 1.2),
    "jevk5-4b-v0.2": (1.532, 0.77),
    "jevk5-2b-v0.2": (1.42, 0.77),
}


def temperatures(config_dir: str, gguf: str | None = None) -> tuple[float, float | None]:
    """(letter temperature, knockout temperature) for a checkpoint: a known GGUF file's own, else jevk5_config.json's."""
    if gguf:
        known = GGUF_TEMPERATURES.get(Path(gguf).name.rsplit("-", 1)[0].lower())
        if known:
            return known
    values = json.loads((Path(config_dir) / CONFIG).read_text())
    knockout = values.get("knockout_temperature")
    return float(values.get("temperature", 1.0)), None if knockout is None else float(knockout)


def question(qid: str, question_spec: dict[str, Any]) -> dict[str, Any]:
    """A wire question in the runtime's form: instructions always present, choice criteria as a dict."""
    crit = question_spec.get("criteria")
    if question_spec["type"] == "choice" and isinstance(crit, list):
        crit = dict.fromkeys(crit)
    return {"type": question_spec["type"], "instructions": instructions_or_name(qid, question_spec), "criteria": crit}


def softmax(logits: list[float], temperature: float) -> list[float]:
    top = max(logits)
    weights = [math.exp((z - top) / temperature) for z in logits]
    total = sum(weights)
    return [w / total for w in weights]


def shaped(kind: str, probs: dict[str, float]) -> dict[str, Any]:
    """The adapter answer for a distribution over the option ids of prompt.decision_options."""
    if kind == "noul":
        return {"noul": probs["true"]}
    if kind == "choice":
        return {"choice": max(probs, key=probs.__getitem__), "probabilities": probs}
    return {"probabilities": probs}


def system_one(
    encode, letter_logits, temperature: float, knockout: float | None, state: Any, questions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Answer every question with `encode(prompt text) -> ids` and `letter_logits(ids, n) -> n logits`."""
    answers, tokens = {}, 0  # a JSON state goes into the prompt's JSON payload as is, as the runtime does
    for qid, q in questions.items():
        wire = question(qid, q)
        options = prompt.decision_options(wire)

        def read(texts: list[str], wire=wire) -> list[float]:
            nonlocal tokens
            ids = encode(prompt.prompt_text(state, wire["instructions"], texts))
            if len(ids) > MAX_TOKENS:
                raise ValueError(f"the prompt is {len(ids)} tokens; this model takes at most {MAX_TOKENS}")
            tokens += len(ids)
            return softmax(letter_logits(ids, len(texts)), temperature)

        probs = prompt.spread(read, [text for _, text in options], "knockout", knockout)
        answers[qid] = shaped(wire["type"], {key: p for (key, _), p in zip(options, probs, strict=True)})
    return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


class _Torch:
    """The safetensors checkpoint on PyTorch, read as the runtime reads it without CUDA graphs."""

    def __init__(self, path: str, device: str):
        import torch
        import transformers

        config = transformers.AutoConfig.from_pretrained(path, local_files_only=True)
        cls: Any = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
        self.tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = cls.from_pretrained(path, config=config, dtype=dtype, local_files_only=True).to(device).eval()
        self.device = device
        slots = [self.tok.encode(letter, add_special_tokens=False) for letter in prompt.LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("every answer letter must be one token")
        self.slot_weight = self.model.lm_head.weight[[ids[0] for ids in slots]].detach().contiguous()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def letter_logits(self, ids: list[int], count: int) -> list[float]:
        import torch

        with torch.inference_mode():
            hidden = self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
            logits = hidden.last_hidden_state[0, -1] @ self.slot_weight[:count].T
        return logits.float().cpu().tolist()


_quiet_callback = None


class _Llama:
    """A GGUF on llama.cpp's C API: one decode per prompt, logits read at the last position only."""

    def __init__(self, gguf_path: str, n_ctx: int = MAX_TOKENS):
        global _quiet_callback
        import llama_cpp as llama

        self.llama = llama
        if _quiet_callback is None:  # llama.cpp logs to stderr process-wide; the callback must stay referenced
            _quiet_callback = llama.llama_log_callback(lambda level, text, data: None)
            llama.llama_log_set(_quiet_callback, ctypes.c_void_p(0))
        llama.llama_backend_init()
        mp = llama.llama_model_default_params()
        mp.n_gpu_layers = -1
        model = llama.llama_model_load_from_file(os.fsencode(gguf_path), mp)
        if not model:
            raise RuntimeError(f"llama.cpp could not load {gguf_path}")
        self.model = model
        cp = llama.llama_context_default_params()
        # llama.cpp aborts when one decode exceeds n_batch, so a whole prompt fits in one batch.
        cp.n_ctx = cp.n_batch = n_ctx
        cp.n_ubatch = 2048
        cp.n_seq_max = 1
        ctx = llama.llama_init_from_model(model, cp)
        if not ctx:
            llama.llama_model_free(model)
            raise RuntimeError("llama.cpp could not create a context")
        self.ctx = ctx
        vocab = llama.llama_model_get_vocab(model)
        if not vocab:
            raise RuntimeError("llama.cpp found no vocabulary in the GGUF")
        self.vocab = vocab
        self.n_vocab = llama.llama_vocab_n_tokens(self.vocab)
        self.batch = llama.llama_batch_init(n_ctx, 0, 1)
        self.n_ctx = n_ctx
        self.lock = threading.Lock()
        slots = [self.encode(letter) for letter in prompt.LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("every answer letter must be one token")
        self.slots = [ids[0] for ids in slots]

    def encode(self, text: str) -> list[int]:
        """Tokens of `text` with special tokens parsed, so the chat markers stay single tokens."""
        data = text.encode()
        buf = (self.llama.llama_token * (len(data) + 8))()
        count = self.llama.llama_tokenize(self.vocab, data, len(data), buf, len(buf), False, True)
        if count < 0:
            raise RuntimeError("llama.cpp could not tokenize the prompt")
        return list(buf[:count])

    def letter_logits(self, ids: list[int], count: int) -> list[float]:
        if len(ids) > self.n_ctx:
            raise ValueError(f"the prompt is {len(ids)} tokens; this model takes at most {self.n_ctx}")
        llama, batch = self.llama, self.batch
        with self.lock:
            llama.llama_memory_clear(llama.llama_get_memory(self.ctx), True)
            for i, t in enumerate(ids):
                batch.token[i], batch.pos[i], batch.n_seq_id[i], batch.logits[i] = t, i, 1, False
                batch.seq_id[i][0] = 0
            batch.logits[len(ids) - 1] = True
            batch.n_tokens = len(ids)
            if (rc := llama.llama_decode(self.ctx, batch)) != 0:
                raise RuntimeError(f"llama_decode returned {rc}")
            row = ctypes.cast(llama.llama_get_logits_ith(self.ctx, len(ids) - 1), ctypes.POINTER(ctypes.c_float))
            return [float(row[s]) for s in self.slots[:count]]

    def close(self) -> None:
        llama = self.llama
        llama.llama_batch_free(self.batch)
        llama.llama_free(self.ctx)
        llama.llama_model_free(self.model)


class _JevK5:
    name = "jevk5"
    runs_repo_code = False
    # A GGUF copy (alibiserikbay/JevK5-GGUF) brings its own tokenizer; the base adds only the temperatures.
    base_files = BASE_FILES

    def runs_weights(self, weights: str, files: list[str]) -> bool:
        return names.format_of(weights) == "gguf"

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        weights = any(f.endswith((".safetensors", ".gguf")) for f in files)
        return CONFIG in files and weights

    def allow_patterns(self, resolved) -> list[str]:
        if resolved.weights:
            return [resolved.weights, CONFIG]
        return ["*.safetensors", "*.json", "*.jinja"]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        if resolved.weights:
            from .. import store

            config_dir = store.base_snapshot(resolved) or path
            engine: Any = _Llama(os.path.join(path, resolved.weights))
            backend, device = "llama.cpp", "llama.cpp"
        else:
            config_dir = path
            device = device or "cpu"
            engine = _Torch(path, device)
            backend = f"PyTorch {device}"
        temperature, knockout = temperatures(config_dir, resolved.weights)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            return system_one(engine.encode, engine.letter_logits, temperature, knockout, state, questions)

        return Loaded(
            resolved.name,
            f"JevK5 typed-decision model ({backend})",
            None,
            self.limits(resolved),
            predict,
            close=getattr(engine, "close", None),
            device=device,
        )


FAMILY = _JevK5()
