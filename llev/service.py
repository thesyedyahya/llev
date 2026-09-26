"""Orchestration: plan jobs -> run primary engine -> escalate low-confidence answers ->
log the decision. Feedback turns corrected answers into few-shot memory."""

import time
import uuid
from typing import Any

from .config import Settings
from .engine import Engine
from .memory import ExampleMemory
from .prompts import Job, build_prefix, choice_jobs, multi_jobs, noul_jobs, render_state, score_jobs
from .schemas import Answer, ChoiceQ, DecideRequest, DecideResponse, MultiQ, NoulQ, ScoreQ
from .scoring import binary_confidence, combine, knn_blend, normalize, score_confidence
from .store import DecisionStore

MIN_MASS = 0.5  # below this the model didn't follow the answer format; treat as unsure


def _onehot(n: int, i: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


class LLEV:
    def __init__(self, settings: Settings, primary: Engine, escalation: Engine | None,
                 memory: ExampleMemory, store: DecisionStore):
        self.s = settings
        self.primary = primary
        self.escalation = escalation
        self.memory = memory
        self.store = store

    # ---- planning -------------------------------------------------------------------

    def _jobs(self, engine: Engine, q, examples: list[tuple[str, Any]]) -> list[Job]:
        t = engine.template
        if isinstance(q, ChoiceQ):
            return choice_jobs(t, q, self.s.debias_permutations, examples)
        if isinstance(q, ScoreQ):
            return score_jobs(t, q, self.s.debias_permutations, examples)
        if isinstance(q, NoulQ):
            return noul_jobs(t, q, examples)
        if isinstance(q, MultiQ):
            return multi_jobs(t, q, examples)
        raise TypeError(type(q))

    def _assemble(self, q, jobs: list[Job], logps: list[dict[str, float]], engine: str,
                  examples: list[tuple[str, Any, float]]) -> Answer:
        normed = [normalize(j, lp) for j, lp in zip(jobs, logps)]
        knn = ([e for e in examples if e[2] >= self.s.knn_min_sim]
               if self.s.fewshot_mode in ("knn", "both") else [])
        base = {"type": q.type, "engine": engine, "fewshot": len(examples)}

        if isinstance(q, ChoiceQ):
            keys = list(q.criteria)
            probs, mass = combine(normed)
            probs = knn_blend(probs, [(_onehot(len(keys), keys.index(lbl)), sim)
                                      for _, lbl, sim in knn if lbl in keys], self.s.knn_strength)
            best = max(range(len(keys)), key=probs.__getitem__)
            return Answer(**base, choice=keys[best], confidence=probs[best], mass=mass,
                          probabilities={k: round(p, 6) for k, p in zip(keys, probs)})

        if isinstance(q, ScoreQ):
            n = len(q.criteria)
            probs, mass = combine(normed)
            probs = knn_blend(probs, [(_onehot(n, lbl), sim) for _, lbl, sim in knn
                                      if isinstance(lbl, int) and not isinstance(lbl, bool) and 0 <= lbl < n],
                              self.s.knn_strength)
            expected = sum(i * p for i, p in enumerate(probs))
            level = max(range(n), key=probs.__getitem__)
            return Answer(**base, score=round(expected, 4), level=level,
                          confidence=score_confidence(probs), mass=mass,
                          probabilities=[round(p, 6) for p in probs])

        if isinstance(q, NoulQ):
            (probs, mass), = normed
            probs = knn_blend(probs, [(_onehot(2, 0 if lbl else 1), sim) for _, lbl, sim in knn
                                      if isinstance(lbl, bool)], self.s.knn_strength)
            return Answer(**base, noul=round(probs[0], 6), confidence=binary_confidence(probs[0]),
                          mass=mass)

        # MultiQ: one yes/no job per option
        keys = list(q.criteria)
        p_yes = {}
        for j, (probs, _) in zip(jobs, normed):
            key = keys[j.group]
            probs = knn_blend(probs, [(_onehot(2, 0 if key in lbl else 1), sim) for _, lbl, sim in knn
                                      if isinstance(lbl, list)], self.s.knn_strength)
            p_yes[key] = probs[0]
        return Answer(**base, probabilities={k: round(p, 6) for k, p in p_yes.items()},
                      selected=[k for k, p in p_yes.items() if p >= q.threshold],
                      confidence=min(binary_confidence(p) for p in p_yes.values()),
                      mass=min(m for _, m in normed))

    def _run(self, engine: Engine, state_text: str, questions: dict, fewshot: bool) -> dict[str, Answer]:
        prefix = build_prefix(engine.template, state_text)
        use_prompt = self.s.fewshot_mode in ("prompt", "both")
        plan: list[tuple[str, Any, list[Job], list]] = []
        for key, q in questions.items():
            examples = (
                self.memory.search(q.task, state_text, self.s.fewshot_k)
                if fewshot and q.task and self.s.fewshot_k > 0 else []
            )
            shown = [(st, lbl) for st, lbl, _ in examples] if use_prompt else []
            plan.append((key, q, self._jobs(engine, q, shown), examples))

        flat = [(j.suffix, j.labels) for _, _, jobs, _ in plan for j in jobs]
        logps = engine.label_logprobs(prefix, flat)

        answers, i = {}, 0
        for key, q, jobs, examples in plan:
            answers[key] = self._assemble(q, jobs, logps[i:i + len(jobs)], engine.name, examples)
            i += len(jobs)
        return answers

    # ---- public API -----------------------------------------------------------------

    def decide(self, req: DecideRequest) -> DecideResponse:
        t0 = time.perf_counter()
        state_text = render_state(req.state)
        big = self.escalation if req.escalate and self.escalation is not None else None
        accurate = {k: q for k, q in req.questions.items() if q.tier == "accurate" and big}
        fast = {k: q for k, q in req.questions.items() if k not in accurate}

        answers = self._run(self.primary, state_text, fast, req.fewshot) if fast else {}
        unsure = {
            k: req.questions[k] for k, a in answers.items()
            if big and (a.confidence < self.s.escalate_threshold or a.mass < MIN_MASS)
        }
        if unsure:
            for k, a in self._run(big, state_text, unsure, req.fewshot).items():
                a.escalated = True
                answers[k] = a
        if accurate:
            answers.update(self._run(big, state_text, accurate, req.fewshot))
        answers = {k: answers[k] for k in req.questions}  # keep request order

        resp = DecideResponse(
            id="dec_" + uuid.uuid4().hex,
            model=self.primary.name,
            answers=answers,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        self.store.record({
            "id": resp.id,
            "ts": time.time(),
            "model": resp.model,
            "state": state_text if self.s.log_state else None,
            "questions": {k: q.model_dump() for k, q in req.questions.items()},
            "answers": {k: a.model_dump(exclude_none=True) for k, a in answers.items()},
            "latency_ms": resp.latency_ms,
        })
        return resp

    def feedback(self, decision_id: str, key: str, label: Any, source: str | None) -> tuple[bool, str | None]:
        """Record the human's final answer. Returns (learned, detail)."""
        rec = self.store.get(decision_id)
        if rec is None:
            raise KeyError("unknown decision id")
        q = rec["questions"].get(key)
        if q is None:
            raise KeyError("unknown question key")
        label = _validate_label(q, label)
        self.store.add_feedback({"id": decision_id, "key": key, "label": label, "source": source,
                                 "predicted": rec["answers"].get(key), "ts": time.time()})
        if not q.get("task"):
            return False, "question has no task id; logged for training but not added to memory"
        if not rec.get("state"):
            return False, "state logging is off; logged but not added to memory"
        self.memory.add(q["task"], rec["state"], label)
        return True, None


def _validate_label(q: dict, label: Any) -> Any:
    t = q["type"]
    if t == "choice":
        if label not in q["criteria"]:
            raise ValueError(f"label must be one of {list(q['criteria'])}")
    elif t == "score":
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(q["criteria"]):
            raise ValueError(f"label must be a level index 0..{len(q['criteria']) - 1}")
    elif t == "noul":
        if not isinstance(label, bool):
            raise ValueError("label must be true or false")
    elif t == "multi":
        if not isinstance(label, list) or any(x not in q["criteria"] for x in label):
            raise ValueError(f"label must be a list drawn from {list(q['criteria'])}")
    return label
