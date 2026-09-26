"""Candidate lightweight models. Compare them with scripts/bench.py before choosing.

GGUF files are resolved on Hugging Face by pattern (see scripts/download_model.py),
so exact filenames don't need to be hardcoded.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    pattern: str  # substring matched (case-insensitive) against .gguf filenames
    template: str
    params: str
    note: str


REGISTRY: dict[str, ModelSpec] = {
    s.key: s
    for s in [
        ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "q4_k_m", "chatml", "0.5B",
                  "Tiny floor; use only if latency is everything."),
        ModelSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "q4_k_m", "chatml", "1.5B",
                  "73% on evals/english.json; fast but weaker."),
        ModelSpec("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct-GGUF", "q4_k_m", "chatml", "3B",
                  "Noticeably smarter, ~2 GB."),
        ModelSpec("qwen3-1.7b", "unsloth/Qwen3-1.7B-GGUF", "q4_k_m", "qwen3", "1.7B",
                  "Default primary: 82%, ~150 ms on Metal."),
        ModelSpec("qwen3-4b", "unsloth/Qwen3-4B-GGUF", "q4_k_m", "qwen3", "4B",
                  "91%; solid alternative escalation model."),
        ModelSpec("llama3.2-3b", "bartowski/Llama-3.2-3B-Instruct-GGUF", "q4_k_m", "llama3", "3B",
                  "Solid English instruction following."),
        ModelSpec("gemma3-1b", "unsloth/gemma-3-1b-it-GGUF", "q4_k_m", "gemma", "1B",
                  "Very small; good multilingual coverage."),
        ModelSpec("gemma3-4b", "unsloth/gemma-3-4b-it-GGUF", "q4_k_m", "gemma", "4B",
                  "Default escalation: 93%, best calibrated (ECE 0.018)."),
        ModelSpec("phi4-mini", "unsloth/Phi-4-mini-instruct-GGUF", "q4_k_m", "phi4", "3.8B",
                  "Strong reasoning for size, English-leaning."),
    ]
}


def local_path(models_dir: str, key: str) -> Path | None:
    """Find a downloaded GGUF for a registry key (models/<key>/*.gguf)."""
    d = Path(models_dir) / key
    files = sorted(d.glob("*.gguf")) if d.is_dir() else []
    return files[0] if files else None
