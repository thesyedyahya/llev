"""Append-only decision + feedback logs (JSONL). Training data for future fine-tuning."""

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

RECENT = 20_000


class DecisionStore:
    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.decisions = self.dir / "decisions.jsonl"
        self.feedback = self.dir / "feedback.jsonl"
        self._recent: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def _write(self, path: Path, rec: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self._write(self.decisions, rec)
            self._recent[rec["id"]] = rec
            if len(self._recent) > RECENT:
                self._recent.popitem(last=False)

    def get(self, decision_id: str) -> dict | None:
        with self._lock:
            if decision_id in self._recent:
                return self._recent[decision_id]
        if not self.decisions.exists():
            return None
        with self.decisions.open(encoding="utf-8") as f:
            for line in f:
                if decision_id in line:
                    rec = json.loads(line)
                    if rec.get("id") == decision_id:
                        return rec
        return None

    def add_feedback(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self._write(self.feedback, rec)
