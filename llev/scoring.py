"""Turns raw label log-probabilities into typed answers. Pure math, no model."""

import math

from .prompts import Job


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def normalize(job: Job, logps: dict[str, float]) -> tuple[list[float], float]:
    """Probabilities over the job's labels, mapped back to original option order,
    plus the total probability mass the model put on valid labels."""
    raw = [logps[label] for label in job.labels]
    lse = logsumexp(raw)
    shown = [math.exp(x - lse) for x in raw]
    probs = [0.0] * len(job.order)
    for i, orig in enumerate(job.order):
        probs[orig] = shown[i]
    return probs, math.exp(lse)


def combine(results: list[tuple[list[float], float]]) -> tuple[list[float], float]:
    """Average probabilities across option orderings (cancels position bias)."""
    n = len(results[0][0])
    probs = [sum(r[0][i] for r in results) / len(results) for i in range(n)]
    mass = sum(r[1] for r in results) / len(results)
    return probs, mass


def score_confidence(probs: list[float]) -> float:
    """1 - normalized std-dev of the level distribution: peaked -> 1, spread -> 0."""
    n = len(probs)
    mean = sum(i * p for i, p in enumerate(probs))
    var = sum(p * (i - mean) ** 2 for i, p in enumerate(probs))
    max_std = (n - 1) / 2
    return max(0.0, 1.0 - math.sqrt(var) / max_std) if max_std else 1.0


def binary_confidence(p: float) -> float:
    return abs(p - 0.5) * 2


def knn_blend(probs: list[float], votes: list[tuple[list[float], float]], strength: float) -> list[float]:
    """Blend model probabilities with similarity-weighted votes from labelled neighbours.

    Weight = strength * best similarity, so near-duplicates of past corrected cases pull
    hard while loosely related memories barely move the answer.
    """
    votes = [(v, sim) for v, sim in votes if sim > 0]
    if not votes or strength <= 0:
        return probs
    total = sum(sim for _, sim in votes)
    p_knn = [sum(v[i] * sim for v, sim in votes) / total for i in range(len(probs))]
    w = min(1.0, strength * max(sim for _, sim in votes))
    return [(1 - w) * p + w * k for p, k in zip(probs, p_knn)]
