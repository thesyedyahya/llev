"""Join decisions + human feedback into a labelled JSONL for evals or fine-tuning.

  python scripts/export_dataset.py [--data data] [--out dataset.jsonl]

Each line: {task, type, question, state, label, predicted, correct}
Also prints per-task agreement between LLEV and humans.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dataset.jsonl")
    a = ap.parse_args()
    d = Path(a.data)
    if not (d / "feedback.jsonl").exists():
        sys.exit("no feedback yet")

    fb = [json.loads(l) for l in (d / "feedback.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    want = {f["id"] for f in fb}
    decisions = {}
    with (d / "decisions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in want:
                decisions[rec["id"]] = rec

    agree = defaultdict(lambda: [0, 0])
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for f in fb:
            rec = decisions.get(f["id"])
            if not rec or not rec.get("state"):
                continue
            q = rec["questions"][f["key"]]
            pred = rec["answers"].get(f["key"], {})
            got = {"choice": pred.get("choice"), "score": pred.get("level"),
                   "noul": (pred.get("noul") or 0) >= 0.5, "multi": pred.get("selected")}[q["type"]]
            ok = got == f["label"] if q["type"] != "multi" else sorted(got or []) == sorted(f["label"])
            task = q.get("task") or f["key"]
            agree[task][0] += ok
            agree[task][1] += 1
            out.write(json.dumps({"task": task, "type": q["type"], "question": q, "state": rec["state"],
                                  "label": f["label"], "predicted": got, "correct": ok},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} examples to {a.out}")
    for t, (ok, tot) in sorted(agree.items()):
        print(f"  {t:30} {ok}/{tot} agree ({ok / tot:.0%})")


if __name__ == "__main__":
    main()
