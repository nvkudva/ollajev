"""The curated models: each one checked to answer /v1/systemone."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_MODEL


@dataclass(frozen=True)
class Entry:
    name: str
    size_gb: float
    languages: str
    description: str


CATALOG: list[Entry] = [
    Entry(DEFAULT_MODEL, 2.7, "English", "Qwen3.5-4B decider, 4-bit GGUF on llama.cpp"),
    Entry("Mapika/decider-2b-GGUF:Q4_K_M", 1.2, "English", "Qwen3.5-2B decider, 4-bit GGUF on llama.cpp"),
    Entry("Mapika/decider-2b-GGUF:Q8_0", 2.0, "English", "Qwen3.5-2B decider, 8-bit GGUF on llama.cpp"),
    Entry("Mapika/decider-2b", 3.76, "English", "Qwen3.5-2B decider, bf16 on PyTorch"),
    Entry("Mapika/decider-0.8b", 1.5, "English", "Qwen3.5-0.8B decider, bf16 on PyTorch"),
    Entry("convaiinnovations/laya", 0.85, "English", "Laya, ModernBERT-large, general purpose"),
    Entry("convaiinnovations/laya-multilingual", 0.68, "100+ languages", "Laya, mmBERT-base"),
    Entry("convaiinnovations/laya-typed-decisions", 0.85, "English", "Laya tuned for agent traces, support, invoices"),
    Entry("SupersonicLabs/Julia-1", 0.57, "Multilingual", "Julia 1, mmBERT-small, 2-20 options"),
    Entry("com-kotobalabs/open-jev-deberta-v3-large", 1.74, "English", "open-jev, DeBERTa-v3-large, 512 tokens"),
    Entry("jaredpalmer/kev-0.5b", 1.0, "English", "Kev, LoRA + pointer head on Qwen2.5-0.5B"),
    Entry("jaredpalmer/kev-0.6b", 1.2, "English", "Kev, LoRA + pointer head on Qwen3-0.6B"),
    Entry("jaredpalmer/kev-0.8b", 1.7, "English", "Kev, LoRA + pointer head on Qwen3.5-0.8B"),
    Entry("jaredpalmer/kev-4b", 9.5, "English", "Kev, LoRA + pointer head on Qwen3.5-4B"),
    Entry("jaredpalmer/kev-9b", 19.5, "English", "Kev, LoRA + pointer head on Qwen3.5-9B"),
    Entry("internlm/Intern-Decision-0.8B", 1.7, "Multilingual", "Intern-Decision, Qwen3.5-0.8B"),
    Entry("internlm/Intern-Decision-2B", 4.5, "Multilingual", "Intern-Decision, Qwen3.5-2B"),
    Entry("internlm/Intern-Decision-4B", 9.1, "Multilingual", "Intern-Decision, Qwen3.5-4B"),
    Entry("llm-semantic-router/Decision-1.0-Kai-0.6B", 2.28, "English", "Decision-1.0 Kai, Vela encoder"),
    Entry(
        "llm-semantic-router/Decision-1.0-Lex-0.6B",
        2.28,
        "English",
        "Decision-1.0 Lex, Kai tuned for support, invoices, agent traces",
    ),
    Entry("wfzyx/von", 1.6, "English", "Von, option-marker head on ModernBERT-large"),
    Entry("heman10x/rlcd-modernbert-151m", 0.7, "English", "rlcd, GLiClass ModernBERT encoder, OpenJev Verdict"),
    Entry("alibiserikbay/JevK5", 8.4, "English", "JevK5, Qwen3.5-4B with the runtime's letter readout"),
    Entry("alibiserikbay/JevK5-2B", 3.8, "English", "JevK5, Qwen3.5-2B with the runtime's letter readout"),
    Entry("OmniJev/OneJev-0.8B", 2.2, "Multilingual", "OneJev, Qwen3.5-0.8B, letters read from the LM head"),
    Entry("OmniJev/OneJev-4B", 10.4, "Multilingual", "OneJev, Qwen3.5-4B, letters read from the LM head"),
    Entry("Cloudflare/clef-flash", 19.1, "Multilingual", "clef-flash, the smaller clef, joint schema head"),
    Entry("Cloudflare/clef", 55.0, "Multilingual", "clef, Qwen3.5 backbone with a joint schema head"),
    Entry("Cloudflare/clef-omni", 70.9, "Multilingual", "clef-omni, Qwen3-Omni 30B-A3B; images, audio, video"),
    Entry("mlx-community/clef-flash-4bit", 6.2, "Multilingual", "clef-flash, 4-bit MLX for Apple Silicon"),
    Entry("mlx-community/clef-flash-8bit", 10.7, "Multilingual", "clef-flash, 8-bit MLX for Apple Silicon"),
    Entry("mlx-community/clef-4bit", 16.3, "Multilingual", "clef, 4-bit MLX for Apple Silicon"),
    Entry("mlx-community/clef-8bit", 29.8, "Multilingual", "clef, 8-bit MLX for Apple Silicon"),
    Entry("mlx-community/clef-omni-4bit", 19.8, "Multilingual", "clef-omni, 4-bit MLX for Apple Silicon"),
    Entry("mlx-community/clef-omni-8bit", 35.0, "Multilingual", "clef-omni, 8-bit MLX for Apple Silicon"),
    Entry("LiquidAI/d1-omni-600M", 2.35, "Multilingual", "d1-omni, LFM2.5 encoder; images or 30 s of speech"),
]
