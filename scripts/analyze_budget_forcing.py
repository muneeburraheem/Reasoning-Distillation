"""Summarise scripts/eval_budget_forcing.py: RD vs LKD at each budget, paired per question.

For every budget and each score (hard_strict, hard_lenient, forced) the per-question
accuracy is the mean over that question's samples; RD - LKD is bootstrapped over
questions. Also reports how samples ended (state mix) and the forced score by level.

Run: python scripts/analyze_budget_forcing.py eval_outputs/budget_forcing
"""

import json
import random
import sys
from pathlib import Path

SCORES = ("hard_strict", "hard_lenient", "forced")
STATES = ("natural", "closed_cut", "eos_in_think", "budget")


def per_question(records, key):
    return {r["id"]: sum(c[key] for c in r["candidates"]) / len(r["candidates"]) for r in records}


def boot(d, b=20000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / n, m[int(0.025 * b)], m[int(0.975 * b)], min(1.0, max(p, 1 / b))


def main(out_dir, models=("lkd", "rd")):
    out_dir = Path(out_dir)
    budgets = sorted({int(p.stem.split("__B")[1]) for p in out_dir.glob("*__B*.json")})
    for budget in budgets:
        paths = {m: out_dir / f"{m}__B{budget}.json" for m in models}
        if not all(p.exists() for p in paths.values()):
            print(f"\nB={budget}: incomplete, skipping")
            continue
        data = {m: json.load(open(p))["records"] for m, p in paths.items()}
        A, B = models
        data = {"lkd": data[A], "rd": data[B]}
        ids = [r["id"] for r in data["lkd"]]
        level = {r["id"]: r["metadata"].get("level") for r in data["lkd"]}
        print(f"\n######## B = {budget}  ({len(ids)} questions, "
              f"{len(data['lkd'][0]['candidates'])} samples each)")
        for m in ("lkd", "rd"):
            cs = [c for r in data[m] for c in r["candidates"]]
            mix = " ".join(f"{s}={sum(c['state'] == s for c in cs) / len(cs):.3f}" for s in STATES)
            acc = " ".join(f"{k}={sum(c[k] for c in cs) / len(cs):.3f}" for k in SCORES)
            forced = [c for c in cs if c["state"] != "natural"]
            f_acc = sum(c["forced"] for c in forced) / max(len(forced), 1)
            print(f"  {m:3s} {acc} | {mix} | acc of forced-only samples={f_acc:.3f}")
        for k in SCORES:
            a, b = per_question(data["lkd"], k), per_question(data["rd"], k)
            mean, lo, hi, p = boot([b[i] - a[i] for i in ids])
            print(f"  {models[1]}-{models[0]} {k:12s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}")
        a, b = per_question(data["lkd"], "forced"), per_question(data["rd"], "forced")
        row = []
        for lv in range(1, 6):
            q = [i for i in ids if level[i] == lv]
            if q:
                row.append(f"L{lv}:{sum(b[i] - a[i] for i in q) / len(q):+.3f}(n={len(q)})")
        print("  forced RD-LKD by level: " + "  ".join(row))


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2:4]) if len(sys.argv) > 3 else ("lkd", "rd"))
