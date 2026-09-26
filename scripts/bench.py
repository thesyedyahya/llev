"""Compare local models on a labelled eval set.

  python scripts/bench.py qwen2.5-1.5b qwen3-1.7b gemma3-1b [--eval evals/english.json]
                          [--debias 2] [--gpu-layers -1] [--out bench.md]

Reports accuracy, calibration (ECE), latency and how many answers are safe to
automate at confidence >= 0.8 (and how accurate those are).
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import TypeAdapter  # noqa: E402

from llev.config import Settings  # noqa: E402
from llev.engine import LlamaEngine  # noqa: E402
from llev.memory import ExampleMemory  # noqa: E402
from llev.models import REGISTRY, local_path  # noqa: E402
from llev.schemas import DecideRequest, Question  # noqa: E402
from llev.service import LLEV  # noqa: E402
from llev.store import DecisionStore  # noqa: E402
from llev.templates import get_template  # noqa: E402

GATE = 0.8


def correct(qtype: str, ans, expected) -> bool:
    if qtype == "choice":
        return ans.choice == expected
    if qtype == "score":
        return abs(ans.score - expected) <= 0.5
    if qtype == "noul":
        return (ans.noul >= 0.5) == expected
    raise ValueError(qtype)


def ece(rows: list[tuple[float, bool]], bins: int = 5) -> float:
    total, err = len(rows), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in rows if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(ok for _, ok in sel) / len(sel)
            err += len(sel) / total * abs(conf - acc)
    return err


def load(key: str, args) -> tuple[LlamaEngine, Path]:
    path = local_path(args.models_dir, key)
    if path is None:
        raise FileNotFoundError(key)
    engine = LlamaEngine(key, str(path), get_template(REGISTRY[key].template), n_ctx=4096,
                         n_gpu_layers=args.gpu_layers)
    return engine, path


def run_model(key: str, data: dict, args) -> dict:
    spec = REGISTRY[key]
    t0 = time.perf_counter()
    try:
        engine, path = load(key, args)
        escalation = load(args.escalate, args)[0] if args.escalate else None
    except FileNotFoundError as e:
        return {"model": key, "error": f"{e.args[0]} not downloaded"}
    load_s = time.perf_counter() - t0
    label = f"{key} → {args.escalate}" if args.escalate else key

    s = Settings(engine="llama", debias_permutations=args.debias, fewshot_k=args.fewshot, log_state=False,
                 escalate_threshold=args.threshold)
    if args.fewshot:
        label += f" +fewshot{args.fewshot}"
    if args.accurate:
        label += f" (accurate: {','.join(sorted(args.accurate))})"
    with tempfile.TemporaryDirectory() as tmp:
        svc = LLEV(s, engine, escalation, ExampleMemory(Path(tmp) / "m.jsonl"), DecisionStore(Path(tmp)))
        qa = TypeAdapter(Question)
        questions = {k: qa.validate_python({**v, **({"tier": "accurate"} if k in args.accurate else {})})
                     for k, v in data["questions"].items()}

        # warm-up (first Metal/CPU kernels are slow)
        svc.decide(DecideRequest(state="warm up", questions={"w": next(iter(questions.values()))}))

        rows, lat, by_q, escalated = [], [], {}, 0
        for ci, case in enumerate(data["cases"]):
            q = questions[case["q"]]
            if args.fewshot:
                # leave-one-out: every other labelled case of this task is "past feedback"
                svc.memory = ExampleMemory(Path(tmp) / f"loo{ci}.jsonl")
                for cj, other in enumerate(data["cases"]):
                    if cj != ci and other["q"] == case["q"]:
                        svc.memory.add(case["q"], other["state"], other["expected"])
                q = q.model_copy(update={"task": case["q"]})
            t = time.perf_counter()
            ans = svc.decide(DecideRequest(state=case["state"], questions={"x": q})).answers["x"]
            escalated += ans.escalated
            lat.append((time.perf_counter() - t) * 1000)
            ok = correct(q.type, ans, case["expected"])
            rows.append((ans.confidence, ok))
            by_q.setdefault(case["q"], []).append(ok)
            if args.verbose and not ok:
                got = ans.choice if ans.choice is not None else ans.score if ans.score is not None else ans.noul
                print(f"  ✗ {label} {case['q']}: expected {case['expected']!r} got {got!r} "
                      f"(conf {ans.confidence:.2f}) — {case['state'][:70]}")

    gated = [ok for c, ok in rows if c >= GATE]
    return {
        "model": label,
        "escalated": escalated / len(rows),
        "params": spec.params,
        "size_gb": path.stat().st_size / 1e9,
        "load_s": load_s,
        "accuracy": sum(ok for _, ok in rows) / len(rows),
        "per_task": {k: sum(v) / len(v) for k, v in by_q.items()},
        "ece": ece(rows),
        "auto_rate": len(gated) / len(rows),
        "auto_acc": (sum(gated) / len(gated)) if gated else float("nan"),
        "p50_ms": statistics.median(lat),
        "p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))],
    }


def table(results: list[dict], tasks: list[str]) -> str:
    head = ["model", "params", "GB", "accuracy", *tasks, "ECE↓", f"auto@{GATE}", "auto acc", "escalated",
            "p50 ms", "p95 ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(results, key=lambda r: -r.get("accuracy", -1)):
        if "error" in r:
            lines.append(f"| {r['model']} | {r['error']} |")
            continue
        cells = [r["model"], r["params"], f"{r['size_gb']:.1f}", f"{r['accuracy']:.0%}",
                 *(f"{r['per_task'].get(t, float('nan')):.0%}" for t in tasks),
                 f"{r['ece']:.3f}", f"{r['auto_rate']:.0%}", f"{r['auto_acc']:.0%}",
                 f"{r['escalated']:.0%}", f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--eval", default="evals/english.json")
    ap.add_argument("--debias", type=int, default=2)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--escalate", help="escalation model key (benchmarks the cascade)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--accurate", type=lambda v: set(v.split(",")), default=set(),
                    help="comma-separated question keys sent straight to the escalation model")
    ap.add_argument("--fewshot", type=int, default=0, help="k learned examples (leave-one-out)")
    ap.add_argument("--out")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    results = []
    for key in args.models:
        print(f"→ {key}", flush=True)
        results.append(run_model(key, data, args))
    md = table(results, list(data["questions"]))
    print("\n" + md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
