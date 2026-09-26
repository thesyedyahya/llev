"""Model backends. An engine reads next-token log-probabilities for candidate labels.

It never generates text: one forward pass per job, over a KV cache that already holds
the shared STATE prefix, so extra questions cost only their own suffix tokens.
"""

import math
import re
import threading
from typing import Callable, Protocol

from .templates import Template


class ContextOverflow(ValueError):
    pass


class Engine(Protocol):
    name: str
    template: Template

    def label_logprobs(self, prefix: str, jobs: list[tuple[str, list[str]]]) -> list[dict[str, float]]:
        """For each (suffix, labels) return {label: log P(label | prefix + suffix)}."""
        ...


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class LlamaEngine:
    def __init__(
        self,
        name: str,
        model_path: str,
        template: Template,
        n_ctx: int = 8192,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ):
        import numpy as np
        from llama_cpp import Llama

        self._np = np
        self.name = name
        self.template = template
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=512,
            logits_all=False,
            verbose=False,
        )
        self._lock = threading.Lock()  # llama.cpp contexts are not thread-safe
        self._prefix: list[int] = []  # tokens of the prefix currently in the KV cache
        self._label_ids: dict[str, list[int]] = {}

    def _tok(self, text: str, bos: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=True)

    def _ids(self, label: str) -> list[int]:
        """First token of the label with and without a leading space ("A" / " A")."""
        if label not in self._label_ids:
            ids = set()
            for variant in (label, " " + label):
                toks = self.llm.tokenize(variant.encode("utf-8"), add_bos=False, special=False)
                if toks:
                    ids.add(toks[0])
            self._label_ids[label] = sorted(ids)
        return self._label_ids[label]

    def count_tokens(self, text: str) -> int:
        return len(self._tok(text))

    def label_logprobs(self, prefix, jobs):
        np = self._np
        with self._lock:
            ptoks = self._tok(prefix, self.template.add_bos)
            stoks = [self._tok(s) for s, _ in jobs]
            need = len(ptoks) + max(len(t) for t in stoks)
            if need > self.llm.n_ctx():
                raise ContextOverflow(f"prompt needs {need} tokens, context is {self.llm.n_ctx()}")

            # Reuse whatever part of the prefix is already cached (system prompt, same state).
            keep = _lcp(self._prefix, ptoks)
            self._prefix = []  # invalid until the eval below succeeds
            self.llm.n_tokens = keep
            self.llm.eval(ptoks[keep:])
            self._prefix = ptoks

            n_vocab = self.llm.n_vocab()
            out = []
            for toks, (_, labels) in zip(stoks, jobs):
                self.llm.n_tokens = len(ptoks)  # rewind to end of prefix; eval drops stale KV
                self.llm.eval(toks)
                # logits_all=False: llama.cpp holds only the last token's logits.
                logits = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(n_vocab,))
                logits = logits.astype(np.float64)
                m = logits.max()
                lse = m + math.log(np.exp(logits - m).sum())
                res = {}
                for label in labels:
                    sel = logits[self._ids(label)]
                    sm = sel.max()
                    res[label] = float(sm + math.log(np.exp(sel - sm).sum()) - lse)
                out.append(res)
            return out


class FakeEngine:
    """Model-free engine for tests and local dev: word overlap between state and option."""

    def __init__(self, name: str = "fake", template: Template | None = None,
                 scorer: Callable[[str, str, list[str]], dict[str, float]] | None = None):
        from .templates import get_template

        self.name = name
        self.template = template or get_template("chatml")
        self.scorer = scorer or self._overlap
        self.calls = 0

    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}

    def _overlap(self, prefix: str, suffix: str, labels: list[str]) -> dict[str, float]:
        state = self._words(prefix.split("TEXT:", 1)[-1])
        if labels == ["Yes", "No"]:
            m = re.search(r"(?:about the TEXT\? |Option: )(.*)", suffix)
            hit = len(self._words(m.group(1) if m else "") & state)
            return {"Yes": math.log(0.2 + 0.6 * (hit > 0)), "No": math.log(0.8 - 0.6 * (hit > 0))}
        lines = dict(re.findall(r"^([A-Z])\) (.*)$", suffix, re.M))
        return {lbl: float(len(self._words(lines.get(lbl, "")) & state)) - 3.0 for lbl in labels}

    def label_logprobs(self, prefix, jobs):
        self.calls += 1
        return [self.scorer(prefix, s, labels) for s, labels in jobs]
