"""OmniJev/OneJev-*: Qwen3.5 finetunes that answer with the letter of an option, read off the next-token logits.

Prompt and readout follow the official `qev` engine (github.com/OmniJev/OneJev, prompt qev-labels-v2): one chat turn
per question, the state first so every question shares its prefix, and the probabilities are the softmax of the LM
head's logits for the option letters at the end of the turn. No token is sampled. On PyTorch the state prefix is run
once and each question continues from a copy of its cache; GGUF copies run every question in full on llama.cpp, with
the tokenizer and chat template of their base repo. The repos ship no calibration.json, so the temperature is 1.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any

from .. import names
from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 32768}
META = ["*.json", "*.jinja", "tokenizer*"]
REPO = re.compile(r"OmniJev/OneJev-\d+(\.\d+)?B")  # not OneJev-27B-FP8: compressed-tensors FP8 needs CUDA kernels

SYSTEM = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SPLIT = {
    "BQ",
    "BZ",
    "CJ",
    "CQ",
    "CZ",
    "DQ",
    "DZ",
    "EJ",
    "EY",
    "FJ",
    "FQ",
    "FV",
    "FZ",
    "GJ",
    "GK",
    "GQ",
    "GZ",
    "HJ",
    "IY",
}
_SPLIT |= {"JF", "JG", "JH", "JL", "JN", "JQ"}


# A..Z, then the two-letter labels the Qwen tokenizer holds as one token: 255 slots, TypeSafe's choice limit.
MAX_SLOTS = 255
LLAMA_BATCH = 2048  # prompt tokens per llama.cpp batch; the prefix is fed once, then cached


def _slots() -> list[str]:
    slots = list(LETTERS)
    for first in LETTERS:
        for second in LETTERS:
            label = first + second
            if label not in _SPLIT:
                slots.append(label)
    return slots[:MAX_SLOTS]


SLOTS = _slots()
NOUL_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_TRUE = "the statement is true / the answer is yes"
NOUL_FALSE = "the statement is false / the answer is no"
MARK = "QEV_PREFIX_MARK"


def entry(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def state_text(state: Any) -> str:
    if state is None:
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def question(question_spec: dict[str, Any]) -> tuple[list[str], str]:
    """A question's answer labels and its turn text after the state block."""
    kind, crit = question_spec["type"], question_spec.get("criteria")
    if kind == "noul":
        crit = crit if isinstance(crit, dict) else {}
        text = entry(question_spec.get("instructions")) or NOUL_INSTRUCTIONS
        labels, names = ["yes", "no"], ["yes", "no"]
        descs = [
            entry(crit["true"]) if crit.get("true") is not None else NOUL_TRUE,
            entry(crit["false"]) if crit.get("false") is not None else NOUL_FALSE,
        ]
        header = "Question:"
    elif kind == "choice":
        text = entry(question_spec.get("instructions")) or "Which option applies to the state?"
        crit = crit if isinstance(crit, dict) else dict.fromkeys(crit or [])
        labels, names, descs = list(crit), list(crit), [entry(d) for d in crit.values()]
        header = "Question:"
    elif kind == "score":
        text = entry(question_spec.get("instructions")) or "Which level describes the state?"
        levels = list(crit or [])
        labels, names = [str(i) for i in range(len(levels))], [f"level {i}" for i in range(len(levels))]
        descs = [entry(level) for level in levels]
        header = "Rate the state:"
    else:
        raise ValueError(f"unknown question type {kind!r}")
    if len(labels) > len(SLOTS):
        raise ValueError(f"{len(labels)} options exceed OneJev's {len(SLOTS)} answer slots")
    slots = SLOTS[: len(labels)]
    options = "\n".join(f"{s}. {n}: {d}" if d else f"{s}. {n}" for s, n, d in zip(slots, names, descs, strict=True))
    answer = "Answer with one letter" if len(slots) <= len(LETTERS) else "Answer with one label"
    return labels, f"{header} {text}\n\nOptions:\n{options}\n\n{answer}: {', '.join(slots)}."


def messages(state: Any, suffix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"<state>\n{state_text(state)}\n</state>\n\n{suffix}"},
    ]


def softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    return [x / sum(weights) for x in weights]


def shape(question: dict[str, Any], labels: list[str], logits: list[float]) -> dict[str, Any]:
    """The adapter answer for one question from its slot logits."""
    probabilities = softmax(logits)
    if question["type"] == "noul":
        return {"noul": probabilities[0]}
    probs = dict(zip(labels, probabilities, strict=True))
    if question["type"] == "choice":
        return {"choice": max(probs, key=probs.__getitem__), "probabilities": probs}
    return {"probabilities": probs}


class _Prompts:
    """Renders and tokenizes requests with the repo's tokenizer and chat template."""

    def __init__(self, tok):
        self.tok = tok
        self.slot_ids = self._slots()

    def _slots(self) -> list[int]:
        def one(label: str) -> int | None:
            ids = self.tok.encode(label, add_special_tokens=False)
            return ids[0] if len(ids) == 1 and self.tok.decode(ids) == label else None

        ids = [one(s) for s in SLOTS]
        if any(i is None for i in ids[: len(LETTERS)]):
            raise ValueError("the tokenizer splits an option letter; this is not a OneJev tokenizer")
        if any(i is None for i in ids):  # two-letter labels are all or nothing, as in qev
            ids = ids[: len(LETTERS)]
        return [i for i in ids if i is not None]

    def render(self, state: Any, suffix: str) -> str:
        return self.tok.apply_chat_template(
            messages(state, suffix), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def build(self, state: Any, questions: dict[str, dict[str, Any]]):
        """(prefix ids, [(qid, labels, suffix ids)]): the prefix is the longest token prefix every question shares
        that ends by the end of the state block."""
        rows = []
        for qid, q in questions.items():
            labels, suffix = question(q)
            if len(labels) > len(self.slot_ids):
                raise ValueError(f"question {qid!r} has {len(labels)} options; this tokenizer has {len(self.slot_ids)}")
            rows.append((qid, labels, self.encode(self.render(state, suffix))))
        text = self.render(state, MARK)
        prefix = self.encode(text[: text.index(MARK)])
        shared = len(prefix)
        while shared > 0 and not all(ids[:shared] == prefix[:shared] for _, _, ids in rows):
            shared -= 1
        longest = max(len(ids) for _, _, ids in rows)
        if longest > LIMITS["max_tokens"]:
            raise ValueError(f"state plus question is {longest} tokens, limit is {LIMITS['max_tokens']}")
        return prefix[:shared], [(qid, labels, ids[shared:]) for qid, labels, ids in rows]


def _torch_engine(path: str, device: str):
    """predict on transformers: the text tower of the checkpoint, its LM head in float32 for the readout."""
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(path)
    prompts = _Prompts(tok)
    config = transformers.AutoConfig.from_pretrained(path).get_text_config()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = transformers.Qwen3_5ForCausalLM.from_pretrained(path, config=config, dtype=dtype, device_map=device)
    model.eval()
    # bf16 logits step by 0.125 at the magnitudes the letters reach, which moves two-option probabilities in
    # 3% steps; qev computes the head in float32 for that reason.
    head = model.lm_head
    head.weight = torch.nn.Parameter(head.weight.detach().clone().float(), requires_grad=False)
    head.register_forward_pre_hook(lambda module, args: (args[0].float(), *args[1:]))
    slots = torch.tensor(prompts.slot_ids, device=device)

    def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        prefix, rows = prompts.build(state, questions)
        answers = {}
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([prefix], device=device), use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            for qid, labels, ids in rows:
                output = model(
                    input_ids=torch.tensor([ids], device=device),
                    past_key_values=copy.deepcopy(cache),
                    use_cache=True,
                    logits_to_keep=1,
                )
                logits = output.logits[0, -1][slots[: len(labels)]].float().tolist()
                answers[qid] = shape(questions[qid], labels, logits)
        tokens = len(prefix) + sum(len(ids) for _, _, ids in rows)
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    return predict, None


def _gguf_engine(gguf: str, tokenizer_dir: str):
    """predict on llama.cpp: each question decoded in full, logits read at its last token."""
    import ctypes

    import llama_cpp
    import numpy as np
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    prompts = _Prompts(tok)
    llm = llama_cpp.Llama(gguf, n_ctx=LIMITS["max_tokens"], n_batch=LLAMA_BATCH, n_gpu_layers=-1, verbose=False)
    sample = prompts.render({"task": "Pay the invoice"}, "Question: Is it paid?\n\nOptions:\nA. yes\nB. no")
    if llm.tokenize(sample.encode(), add_bos=False, special=True) != prompts.encode(sample):
        raise ValueError(f"{os.path.basename(gguf)} does not use the OneJev tokenizer of its base repo")
    n_vocab = llm.n_vocab()
    slots = np.asarray(prompts.slot_ids)

    def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        prefix, rows = prompts.build(state, questions)
        answers = {}
        for qid, labels, ids in rows:
            llm.reset()
            llm.eval(prefix + ids)
            ptr = ctypes.cast(llama_cpp.llama_get_logits_ith(llm.ctx, -1), ctypes.POINTER(ctypes.c_float))
            logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,))[slots[: len(labels)]]
            answers[qid] = shape(questions[qid], labels, logits.astype(float).tolist())
        tokens = sum(len(prefix) + len(ids) for _, _, ids in rows)
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}

    return predict, llm.close


class _OneJev:
    name = "onejev"
    runs_repo_code = False
    # A llama.cpp quantization of a OneJev repo runs too: the readout is the LM head's letter logits, and the
    # tokenizer and chat template come from these files of the base repo.
    base_files = META

    def runs_weights(self, weights: str, files: list[str]) -> bool:
        return names.format_of(weights) == "gguf"

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        # The repos are plain Qwen3.5 checkpoints with no file of their own, so the repo name has to say it.
        weights = "model.safetensors" in files or "model.safetensors.index.json" in files
        return bool(REPO.fullmatch(repo_id)) and weights and "tokenizer.json" in files

    def allow_patterns(self, resolved) -> list[str]:
        return [resolved.weights, *META] if resolved.weights else ["*.safetensors", *META]

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        if resolved.base or resolved.weights:
            from .. import store

            tokenizer_dir = store.base_snapshot(resolved) if resolved.base else path
            if tokenizer_dir is None:
                raise LookupError(f"the base files of {resolved.name} are not downloaded")
            predict, close = _gguf_engine(os.path.join(path, resolved.weights), tokenizer_dir)
            backend, device = "llama.cpp", "llama.cpp"
        else:
            device = device or "cpu"
            predict, close = _torch_engine(path, device)
            backend = f"PyTorch {device}"
        return Loaded(
            resolved.name,
            f"OneJev System One decision model ({backend})",
            None,
            self.limits(resolved),
            predict,
            close,
            device,
        )


FAMILY = _OneJev()
