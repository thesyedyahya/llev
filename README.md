# LLEV: open-source Jev alternative for self-hosted LLM classification

[![CI](https://github.com/thesyedyahya/llev/actions/workflows/ci.yml/badge.svg)](https://github.com/thesyedyahya/llev/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![llama.cpp](https://img.shields.io/badge/runs%20on-llama.cpp-orange.svg)](https://github.com/ggml-org/llama.cpp)

**LLEV is a free, self-hosted "System One" decision engine: a local LLM classification API that returns typed
answers with calibrated confidence in about 150 ms.** Use it for zero-shot text classification, content moderation,
support ticket triage, AI agent guardrails, sentiment analysis, spam and phishing detection, lead scoring and game NPC
decisions. It runs on your own Mac, Linux box or GPU with open models (Qwen, Gemma, Llama, Phi) via llama.cpp.

LLEV never generates text. It runs one forward pass and reads the model's next-token distribution over the
answer labels. That makes it fast, cheap and private, and its confidence scores are meaningful enough to automate on.

**[Website](https://thesyedyahya.github.io/llev/)** · [Quick start](#quick-start) · [API](#api) ·
[Benchmarks](#benchmarks) · [Use cases](#use-cases) · [FAQ](#faq)

> Inspired by [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): same
> Choice / Score / Noul question model, plus a `/v1/systemone` compatibility path. LLEV is an independent
> open-source project and is not affiliated with TypeSafe AI.

```bash
curl localhost:8088/v1/decide -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{
  "state": "Nice try deleting my comment. I know which gym you go to, see you there tomorrow.",
  "questions": {
    "threat":        {"type": "noul", "tier": "accurate", "instructions": "The text threatens or intimidates someone, even implicitly"},
    "personal_info": {"type": "noul", "tier": "accurate", "instructions": "The text hints at knowing where someone lives, works or goes"}
  }
}'
# → threat 0.999, personal_info 1.000   (your code: escalate to trust & safety)
```

## Why LLEV

| | Jev (hosted, per its public docs) | LLEV |
|---|---|---|
| Runs on your servers, data never leaves | ✗ | ✓ |
| Choice / Score / Noul questions | ✓ | ✓ |
| Multi-label questions | not in public docs | ✓ |
| Swap models (Qwen, Gemma, Llama, Phi GGUF), benchmark them on your data | ✗ (proprietary model) | ✓ |
| Confidence-gated **cascade**: small model first, big model only when unsure | build it yourself | built in |
| Per-question **tier** to force the accurate model for safety or money decisions | not in public docs | ✓ |
| **Position-bias correction**: options asked in several orders and averaged | not in public docs | ✓ |
| Feedback endpoint, decision logs and a dataset export for fine-tuning | not in public docs | ✓ |
| Cost | per token | your hardware |

## Use cases

| Use case | Example questions | Question types |
|---|---|---|
| **Support ticket triage** | Which team? How urgent? Is the customer about to churn? | choice, score, noul |
| **Content moderation** | Is it a threat? Harassment? Hate speech? Spam? | noul, multi (`tier: accurate`) |
| **AI agent guardrails** | Is this tool call destructive? Does it target production? Does it match the goal? | noul, decided in code |
| **Sentiment analysis** | How positive is this review, from 1 to 5? | score |
| **Spam and phishing detection** | Is this email a phishing attempt? Which folder does it go in? | noul, choice |
| **Lead scoring and routing** | Large deal? Wants a meeting? Decision-maker? | noul, multi |
| **Intent classification** | Question, task, complaint, cancellation? | choice |
| **Game NPCs and simulations** | How does the NPC react? What's its mood? | choice, score |

Six of these have ready-made presets in the [playground](#quick-start): support, moderation, guardrails, leads, NPCs and phishing.

## Quick start

```bash
git clone https://github.com/thesyedyahya/llev && cd llev
python3 -m venv .venv
# Apple Silicon (Metal). Linux + NVIDIA: CMAKE_ARGS="-DGGML_CUDA=on". Plain CPU: no CMAKE_ARGS.
CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install -e ".[dev]"

.venv/bin/python scripts/download_model.py qwen3-1.7b gemma3-4b   # ~3.6 GB
cp .env.example .env                                              # set LLEV_API_KEYS
.venv/bin/python -m llev                                          # http://localhost:8088
```

Open **http://localhost:8088/playground** to try six ready-made examples: support triage, content moderation,
an AI agent guardrail, sales lead scoring, a game NPC and phishing triage. Each one shows the model's answers and the
few lines of code that turn them into a decision.

No model downloaded yet? `LLEV_ENGINE=fake LLEV_ALLOW_NO_AUTH=true .venv/bin/python -m llev` runs the whole API on a stub.
Docker: `docker compose up -d --build` (downloads the configured models on first boot).

## API

### `POST /v1/decide`

Also available at `/v1/systemone` for compatibility. Auth: `Authorization: Bearer <key>` or `x-api-key`.

```json
{
  "state": {"customer": {"plan": "Pro"}, "message": "Second double charge this month. Fix it today or I'm leaving."},
  "questions": {
    "team":    {"type": "choice", "task": "support.route", "instructions": "Which team should handle this message",
                "criteria": {"billing": "Charges, refunds", "technical": "Bugs, errors", "other": "Anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent", "criteria": ["Days", "Today", "Within the hour", "Now"]},
    "churn":   {"type": "noul", "tier": "accurate", "instructions": "The customer is threatening to leave"},
    "topics":  {"type": "multi", "instructions": "Problems mentioned",
                "criteria": {"double_charge": "charged twice", "slow_support": "slow or unhelpful support"}}
  }
}
```

| Question type | Returns | Use for |
|---|---|---|
| `choice` | `choice`, `probabilities` per option | routing and categorisation (up to 26 options) |
| `score` | `score` (fractional), `level`, `probabilities` per level | urgency, sentiment, quality (2–10 ordered levels) |
| `noul` | `noul` = P(true) | any yes/no fact |
| `multi` | `selected`, `probabilities` per option | tags, policies, extracted requirements |

Every answer also includes `confidence` (0–1, comparable across types), `mass` (how much probability the model put on valid
labels), `engine` and `escalated`. Optional per-question fields:

- `tier`: `"fast"` (default; small model, escalates when confidence < `LLEV_ESCALATE_THRESHOLD`) or `"accurate"` (always the big model).
- `task`: a stable id such as `support.route`. It enables feedback memory for that question.

### `POST /v1/feedback`

```json
{"id": "dec_…", "key": "team", "label": "billing"}
```

`label` is an option key (choice), a level index (score), `true`/`false` (noul) or a list of keys (multi). Feedback is logged
for training. If the question had a `task`, near-duplicate future inputs are also nudged toward the corrected answer.

### Other endpoints

`GET /health` · `GET /v1/memory` (examples per task) · `GET /playground`

Runnable examples: [`examples/try.sh`](examples/try.sh) (curl, every endpoint) and
[`examples/llev-client.mjs`](examples/llev-client.mjs) (Node client, with a fallback for when LLEV is down).

## Use it well: atomic questions, decisions in code

The single most important lesson from building LLEV: **small models are excellent at atomic facts and poor at composite judgements.**

Asked "what should happen to this tool call?", the model **approved** `DELETE FROM customers` on *production* for a
goal that said *staging*. Asked three atomic questions instead ("is it destructive?", "does it target production?",
"is the goal about production?"), it answered all three correctly, and three lines of code blocked the call:

```js
if (a.destructive.noul > 0.5 && a.production.noul > 0.5 && a.goal_is_production.noul < 0.5) block();
```

More rules that held up in testing:

- **Put safety and money questions on `"tier": "accurate"`.** Confidence gating can't catch a small model that is confidently
  wrong. In the benchmark, qwen3-1.7b rated "If you post that again I will find where you live" as 0% toxic, with full confidence.
- **Gate automation on `confidence`**, with thresholds set by the stakes.
- **Always give choice questions an `other` / `none` option**, but don't put `none` inside a `multi` question.
- **Keep `state` to the relevant context.** Irrelevant text lowers accuracy.
- **Don't use it for counting, date arithmetic, extraction or writing text.** Use code or a generative LLM for those.

## How it works

```
state ─► [system + STATE prefix] ─(KV-cached once)─┬─► question 1, options in order A ─► P(A…Z) ┐
                                                   ├─► question 1, options in order B ─► P(A…Z) ┴─► average ─► answer
                                                   └─► question 2 …
fast question, confidence < threshold ─► re-asked on the escalation model
accurate question ─────────────────────► escalation model directly
```

- **Choice:** options are labelled A–Z. Probabilities are renormalised over the valid letters, mapped back through each ordering and averaged, which cancels the small models' bias toward option A.
- **Score:** levels are asked both ascending and descending. `score` is the expected level, and `confidence` = 1 − the normalised standard deviation.
- **Noul:** P(Yes) / (P(Yes) + P(No)).
- **Multi:** one yes/no pass per option.
- The state is evaluated once and held in the KV cache, so each extra question costs only its own tokens.

## Benchmarks

[`evals/english.json`](evals/english.json) has 56 hand-labelled cases over 6 tasks: support routing, urgency, sentiment,
toxicity, spam and intent. Measured on an Apple M4 Pro (Metal) with Q4_K_M models and debiasing on.
`auto@0.8` is the share of answers with confidence ≥ 0.8, and `auto acc` is the accuracy of those answers.

| setup | accuracy | auto@0.8 | auto acc | ECE ↓ | p50 ms |
|---|---|---|---|---|---|
| gemma3-4b | **93%** | 86% | **98%** | **0.018** | 327 |
| qwen3-4b | 91% | 86% | 98% | 0.037 | 372 |
| **qwen3-1.7b → gemma3-4b, toxic + spam on `accurate`** (recommended) | 89% | 91% | 94% | 0.077 | **149** |
| qwen3-1.7b → gemma3-4b | 86% | 91% | 90% | 0.110 | 149 |
| qwen3-1.7b | 82% | 82% | 91% | 0.101 | 151 |
| qwen2.5-1.5b | 73% | 61% | 91% | 0.086 | 148 |
| gemma3-1b | 66% | 43% | 79% | 0.153 | 96 |

Per-task numbers are in [`bench.md`](bench.md). How to pick:

- **Accuracy first:** `LLEV_MODEL=gemma3-4b` with no escalation model.
- **Latency first:** the cascade, with safety questions on `accurate`.
- **Your data:** always benchmark on your own domain first. `scripts/export_dataset.py` turns feedback into an eval set.

```bash
python scripts/bench.py qwen3-1.7b gemma3-4b --gpu-layers -1 -v
python scripts/bench.py qwen3-1.7b --escalate gemma3-4b --accurate toxic,spam --gpu-layers -1
```

Server CPUs are several times slower than Metal. Benchmark with `--gpu-layers 0` on your target machine.

### What didn't work (so you don't have to try it)

- **Few-shot examples in the prompt** lowered qwen3-1.7b from 82% to 73%, because small models copy the example labels.
  They're still available (`LLEV_FEWSHOT_MODE=prompt`) for larger models.
- **Unfiltered nearest-neighbour voting** from feedback also hurt. LLEV only lets near-duplicates vote (TF-IDF similarity ≥ `LLEV_KNN_MIN_SIM`).
- **Generic embeddings (bge-m3)** were no better than TF-IDF at finding same-label examples. They measure topic, not the decision.
- **Low-resource languages and slang** are weak spots for 1–4B models. Test yours before trusting it.

## FAQ

### What is LLEV?
LLEV is an open-source, self-hosted decision engine. You send text or JSON plus typed questions (multiple choice,
a score on a scale, yes/no, or multi-label), and a small local LLM returns typed answers with probabilities and a calibrated
confidence score. It's built for high-volume, real-time classification where calling a large hosted LLM would be too slow,
too expensive or not private enough.

### Is there an open-source alternative to TypeSafe AI's Jev?
Yes, LLEV. It follows the same "System One model" idea (typed questions in, typed answers with confidence out) and accepts
requests on a `/v1/systemone` path, but it runs entirely on your own hardware with open models. It is an independent project
and is not affiliated with TypeSafe AI.

### How do I run LLM text classification locally?
Install LLEV, download a model (`python scripts/download_model.py qwen3-1.7b`), start the server with `python -m llev`
and POST your text and labels to `/v1/decide`. No GPU is required, though Apple Silicon (Metal) and NVIDIA (CUDA) are much faster.

### How is this different from JSON mode or structured outputs?
Structured outputs make a generative LLM *write* JSON token by token. LLEV doesn't generate anything. It reads the
probability of each label in a single forward pass, so you get a real probability distribution rather than one
sampled answer. It's also faster, and you can gate automation on the confidence.

### How is it different from zero-shot classifiers like BART-MNLI?
NLI-based zero-shot classifiers score each label separately with a small encoder model. LLEV uses an instruction-tuned LLM
that reads your instructions and option descriptions, supports ordered scales and multi-label questions, corrects option-order
bias and can escalate unsure answers to a bigger model.

### Which model should I use?
On the bundled English benchmark, Gemma 3 4B scored 93% with the best calibration. A Qwen3 1.7B → Gemma 3 4B cascade scored
89% at half the latency. Benchmark on your own data with `scripts/bench.py`.

### Can I use it for AI agent guardrails?
Yes. Ask atomic questions about a proposed tool call ("is it destructive?", "does it target production?") and make the
decision in code. See [Use it well](#use-it-well-atomic-questions-decisions-in-code) and the guardrail preset in the playground.

### Is it free?
Yes. It's Apache-2.0 licensed, and you only pay for your own hardware.

## Configuration

All settings are `LLEV_*` environment variables (see [`.env.example`](.env.example)):

| Variable | Default | |
|---|---|---|
| `LLEV_MODEL` | `qwen3-1.7b` | primary model key (`scripts/download_model.py --list`), or `LLEV_MODEL_PATH` + `LLEV_TEMPLATE` for any GGUF |
| `LLEV_ESCALATE_MODEL` | unset | escalation / accurate-tier model |
| `LLEV_ESCALATE_THRESHOLD` | `0.6` | fast-tier answers below this confidence escalate |
| `LLEV_N_GPU_LAYERS` | `0` | `-1` offloads everything to Metal or CUDA |
| `LLEV_DEBIAS_PERMUTATIONS` | `2` | option orderings averaged per choice question |
| `LLEV_FEWSHOT_K` / `LLEV_FEWSHOT_MODE` | `3` / `knn` | feedback memory: `knn`, `prompt` or `both` |
| `LLEV_API_KEYS` | required | comma-separated keys |
| `LLEV_LOG_STATE` | `true` | store request text so feedback can build memory and training data |

Supported chat templates: `chatml`, `qwen3`, `llama3`, `gemma`, `phi4`, `raw`.

## Project layout

```
llev/
  api.py          FastAPI app, auth, endpoints, playground
  service.py      planning → fast tier → escalation / accurate tier → logging; feedback → memory
  engine.py       LlamaEngine (llama.cpp, prefix KV reuse, label log-prob readout), FakeEngine for tests
  prompts.py      templates for each question type, option orderings
  scoring.py      normalisation, debias averaging, confidence, kNN blending
  memory.py       per-task feedback memory
  store.py        decision / feedback logs
  templates.py    chat templates        models.py   model registry
  playground.html browser playground
scripts/          download_model.py · bench.py · export_dataset.py
evals/            labelled eval sets
examples/         curl walkthrough, Node client
tests/            API and logic tests (no model needed)
```

## Roadmap

- [ ] Fine-tune on feedback: LoRA or a small classifier head per high-volume task, from `export_dataset.py`
- [ ] Per-task temperature calibration, so a confidence of 0.8 means 80% correct on *your* data
- [ ] Batch concurrent requests into one llama.cpp batch
- [ ] More eval domains and languages
- [ ] OpenAI-compatible and CUDA/vLLM backends

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE)
