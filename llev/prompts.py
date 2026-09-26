"""Prompt construction.

Each question becomes one or more *jobs*: a suffix after the shared STATE prefix plus the
labels whose next-token probabilities we read. Choice/score questions get several jobs
with different option orderings (position debiasing); multi questions get one yes/no job
per option.
"""

import json
import string
from dataclasses import dataclass
from typing import Any

from .schemas import ChoiceQ, MultiQ, NoulQ, ScoreQ
from .templates import Template

LETTERS = string.ascii_uppercase
YES_NO = ["Yes", "No"]
EXAMPLE_STATE_CHARS = 700

SYSTEM = (
    "You are a careful, accurate classifier. The user gives you a text and one question about it. "
    "Judge what the text actually says or clearly implies. Reply with only the answer label."
)


@dataclass
class Job:
    suffix: str
    labels: list[str]
    order: list[int]  # order[i] = original option index shown under labels[i]
    group: int = 0  # multi: option index; otherwise 0


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2, default=str)


def build_prefix(t: Template, state_text: str) -> str:
    return t.open(SYSTEM) + "TEXT:\n\"\"\"\n" + state_text + "\n\"\"\"\n\n"


def _finish(t: Template, body: str) -> str:
    return body + t.end + "Answer:"


def _orders(n: int, k: int) -> list[list[int]]:
    """k distinct cyclic shifts of range(n): each option visits different positions."""
    k = max(1, min(k, n))
    step = max(1, n // k)
    return [[(i + s * step) % n for i in range(n)] for s in range(k)]


def _examples_block(examples: list[tuple[str, str]]) -> str:
    if not examples:
        return ""
    lines = ["Solved examples of this same question (reference only):"]
    for i, (st, ans) in enumerate(examples, 1):
        st = st if len(st) <= EXAMPLE_STATE_CHARS else st[:EXAMPLE_STATE_CHARS] + "…"
        lines.append(f"Example {i} state: <<<{st}>>>\nExample {i} answer: {ans}")
    return "\n".join(lines) + "\n\nNow answer for the TEXT above.\n"


def choice_jobs(t: Template, q: ChoiceQ, k: int, examples: list[tuple[str, Any]]) -> list[Job]:
    keys = list(q.criteria)
    jobs = []
    for order in _orders(len(keys), k):
        opts = "\n".join(f"{LETTERS[i]}) {keys[o]}: {q.criteria[keys[o]]}" for i, o in enumerate(order))
        ex = [(s, LETTERS[order.index(keys.index(lbl))]) for s, lbl in examples if lbl in keys]
        body = (
            f"QUESTION: {q.instructions}\nOptions:\n{opts}\n\n{_examples_block(ex)}"
            "Answer with the letter of the single best option."
        )
        jobs.append(Job(_finish(t, body), [LETTERS[i] for i in range(len(keys))], order))
    return jobs


def score_jobs(t: Template, q: ScoreQ, k: int, examples: list[tuple[str, Any]]) -> list[Job]:
    n = len(q.criteria)
    # Ordered scales can't be shuffled without losing meaning; debias by also asking reversed.
    orders = [list(range(n))] + ([list(reversed(range(n)))] if k >= 2 else [])
    jobs = []
    for order in orders:
        direction = "lowest to highest" if order[0] == 0 else "highest to lowest"
        levels = "\n".join(f"{LETTERS[i]}) {q.criteria[o]}" for i, o in enumerate(order))
        ex = [
            (s, LETTERS[order.index(int(lbl))])
            for s, lbl in examples
            if isinstance(lbl, int) and 0 <= lbl < n
        ]
        body = (
            f"QUESTION: {q.instructions}\nLevels ({direction}):\n{levels}\n\n{_examples_block(ex)}"
            "Answer with the letter of the level that fits best."
        )
        jobs.append(Job(_finish(t, body), [LETTERS[i] for i in range(n)], order))
    return jobs


def noul_jobs(t: Template, q: NoulQ, examples: list[tuple[str, Any]]) -> list[Job]:
    ex = [(s, "Yes" if lbl else "No") for s, lbl in examples if isinstance(lbl, bool)]
    body = (
        f"{_examples_block(ex)}QUESTION: Is the following true about the TEXT? {q.instructions}.\n"
        "Answer Yes or No."
    )
    return [Job(_finish(t, body), YES_NO, [0, 1])]


def multi_jobs(t: Template, q: MultiQ, examples: list[tuple[str, Any]]) -> list[Job]:
    jobs = []
    for gi, (key, desc) in enumerate(q.criteria.items()):
        ex = [(s, "Yes" if key in lbl else "No") for s, lbl in examples if isinstance(lbl, list)]
        body = (
            f"QUESTION: {q.instructions}\nDoes this option apply to the TEXT?\n"
            f"Option: {key}: {desc}\n\n{_examples_block(ex)}Answer Yes or No."
        )
        jobs.append(Job(_finish(t, body), YES_NO, [0, 1], group=gi))
    return jobs
