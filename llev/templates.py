"""Chat templates, split so the STATE can be a shared prefix across questions.

prompt = start(system) + <user text> + end + "Answer:"
The next-token distribution after "Answer:" is what we read.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    name: str
    start: str  # contains {system}
    end: str
    add_bos: bool = False

    def open(self, system: str) -> str:
        return self.start.replace("{system}", system)


TEMPLATES: dict[str, Template] = {
    "chatml": Template(
        "chatml",
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n",
    ),
    # Qwen3 is a hybrid thinking model: prefill an empty think block so it answers directly.
    "qwen3": Template(
        "qwen3",
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    "llama3": Template(
        "llama3",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    # Gemma has no system role; fold it into the user turn.
    "gemma": Template(
        "gemma",
        "<start_of_turn>user\n{system}\n\n",
        "<end_of_turn>\n<start_of_turn>model\n",
        add_bos=True,
    ),
    "phi4": Template(
        "phi4",
        "<|system|>{system}<|end|><|user|>",
        "<|end|><|assistant|>",
    ),
    "raw": Template("raw", "{system}\n\n", "\n\n", add_bos=True),
}


def get_template(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise ValueError(f"unknown template {name!r}; choose from {sorted(TEMPLATES)}") from None
