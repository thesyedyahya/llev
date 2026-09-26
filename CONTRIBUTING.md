# Contributing to LLEV

Thanks for helping! Issues and pull requests are welcome.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # builds llama-cpp-python; see README for Metal/CUDA flags
.venv/bin/python -m pytest -q         # no model needed, tests use FakeEngine
```

## Guidelines

- **Measure, don't guess.** Changes to prompts, scoring or model defaults need before/after numbers from
  `scripts/bench.py`. Include the table in the PR.
- **Keep answers typed.** LLEV reads next-token probabilities and never generates text. Features that need
  generation belong in a separate step, not in the decision path.
- **Tests stay model-free.** Add a `FakeEngine` scorer for new behaviour so CI runs without downloading models.
- **Eval data:** new cases for `evals/` must be hand-labelled, contain no personal data, and be English unless
  you're adding a clearly separated language set.
- Match the surrounding style: small modules, type hints, config only through `LLEV_*` env vars.

## Good first issues

- More chat templates and models in `llev/models.py` (with bench results)
- Batching concurrent requests into one llama.cpp batch
- Per-task temperature calibration from feedback
- Eval sets for new domains
