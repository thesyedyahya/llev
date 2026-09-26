"""Few-shot memory: human-corrected answers become examples for similar future states.

Retrieval is TF-IDF cosine over unicode word tokens (language-agnostic).
It reliably finds near-duplicates (templated complaints, resubmissions); generic
embeddings were measured and did not separate labels better, see README.
Persisted as append-only JSONL; loaded into memory at startup.
"""

import json
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STORED_STATE_CHARS = 2000


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class Example:
    task: str
    state: str
    label: Any
    tf: Counter


class ExampleMemory:
    def __init__(self, path: Path, max_per_task: int = 500):
        self.path = path
        self.max_per_task = max_per_task
        self._by_task: dict[str, list[Example]] = defaultdict(list)
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._append(r["task"], r["state"], r["label"])

    def _append(self, task: str, state: str, label: Any) -> None:
        bucket = self._by_task[task]
        bucket.append(Example(task, state, label, Counter(_tokens(state))))
        if len(bucket) > self.max_per_task:
            del bucket[0]

    def add(self, task: str, state: str, label: Any) -> None:
        state = state[:STORED_STATE_CHARS]
        with self._lock:
            self._append(task, state, label)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"task": task, "state": state, "label": label}, ensure_ascii=False) + "\n")

    def search(self, task: str, state: str, k: int) -> list[tuple[str, Any, float]]:
        """Top-k (state, label, cosine similarity), most similar last."""
        with self._lock:
            bucket = list(self._by_task.get(task, ()))
        if not bucket or k <= 0:
            return []
        n = len(bucket)
        df: Counter = Counter()
        for ex in bucket:
            df.update(ex.tf.keys())
        idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

        def vec(tf: Counter) -> dict[str, float]:
            return {w: c * idf.get(w, math.log(n + 1) + 1) for w, c in tf.items()}

        q = vec(Counter(_tokens(state)))
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored = []
        for ex in bucket:
            d = vec(ex.tf)
            dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
            sim = sum(q[w] * d[w] for w in q.keys() & d.keys()) / (qn * dn)
            scored.append((sim, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Most similar last: closest to the real question, where small models weigh it most.
        return [(ex.state, ex.label, sim) for sim, ex in reversed(scored[:k]) if sim > 0]

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {t: len(v) for t, v in self._by_task.items()}
