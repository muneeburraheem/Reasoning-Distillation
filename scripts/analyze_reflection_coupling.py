"""Is RD's extra reflection linked to success? Follow-up to bonds Family E (RUN_LOG 8n), RD moderate suite.

Data: eval_outputs/rd_newsae/eval_mod/bonds/steps.json (one finished pass@8 MATH-500 trace per question per model,
steps labelled with the Chen et al. 2026 prompt). Models: LKD, RD_mod, RD_new (= October RD).

1. Coupling: per model, level-stratified (correct - wrong) difference of a trace feature, exactly as Family E
   (mean over levels 1-5 with >= 5 correct and >= 5 wrong traces). Bootstrap over questions (10,000; the same
   resampled question set for every model), so the RD - LKD difference in coupling gets a paired CI.
2. Both-correct control: questions where both models' traces are correct; paired RD - LKD of each feature.
Features: reflection share (R), exploration share (E), normal share (N), oscillation (switches between
{reflection, exploration} and {normal, deep} per 100 steps), number of steps.

Run: python scripts/analyze_reflection_coupling.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BONDS = HERE.parent / "eval_outputs/rd_newsae/eval_mod/bonds"
SHORT = {"normal operation": "N", "deep reasoning": "D", "self-reflection": "R", "exploration": "E"}
FEATS = ("R", "E", "N", "osc", "steps")
B = 10000


def features():
    tr = defaultdict(list)
    for r in json.load(open(BONDS / "steps.json")):
        tr[(r["model"], r["qid"])].append(r)
    F = {}
    for (m, q), items in tr.items():
        items.sort(key=lambda r: r["step"])
        labs = [SHORT[r["label"]] for r in items]
        n = len(labs)
        meta = [c in "RE" for c in labs]
        F[(m, q)] = {"R": labs.count("R") / n, "E": labs.count("E") / n, "N": labs.count("N") / n,
                     "osc": 100 * sum(a != b for a, b in zip(meta, meta[1:])) / max(n - 1, 1), "steps": n,
                     "correct": bool(items[0]["correct"]), "level": items[0]["level"]}
    return F


def coupling(F, m, qs, k):
    diffs = []
    for lv in range(1, 6):
        c = [F[(m, q)][k] for q in qs if F[(m, q)]["level"] == lv and F[(m, q)]["correct"]]
        w = [F[(m, q)][k] for q in qs if F[(m, q)]["level"] == lv and not F[(m, q)]["correct"]]
        if len(c) >= 5 and len(w) >= 5:
            diffs.append(np.mean(c) - np.mean(w))
    return np.mean(diffs) if diffs else np.nan


def ci(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return np.percentile(x, 2.5), np.percentile(x, 97.5), max(2 * min((x <= 0).mean(), (x >= 0).mean()), 1 / len(x))


def main():
    F = features()
    models = ("LKD", "RD_mod", "RD_new")
    qs = sorted(set.intersection(*[{q for (mm, q) in F if mm == m} for m in models]))
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(qs), len(qs)) for _ in range(B)]
    L = [f"# Reflection-success coupling, {len(qs)} questions with one finished trace from every model "
         f"(RD_new = October RD)",
         "  wrong traces: " + "  ".join(f"{m} {sum(not F[(m, q)]['correct'] for q in qs)}" for m in models)]
    L.append("\n## 1. Coupling = correct minus wrong, level-stratified, per model; bootstrap over questions")
    for k in FEATS:
        pt = {m: coupling(F, m, qs, k) for m in models}
        bs = {m: [] for m in models}
        for ix in idx:
            sq = [qs[i] for i in ix]
            for m in models:
                bs[m].append(coupling(F, m, sq, k))
        line = f"  {k:5s} " + "  ".join(f"{m} {pt[m]:+.4f} [{ci(bs[m])[0]:+.4f},{ci(bs[m])[1]:+.4f}]" for m in models)
        L.append(line)
        for a in ("RD_mod", "RD_new"):
            d = np.array(bs[a]) - np.array(bs["LKD"])
            lo, hi, p = ci(d)
            L.append(f"        {a} - LKD coupling {pt[a] - pt['LKD']:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
        d = np.array(bs["RD_mod"]) - np.array(bs["RD_new"])
        lo, hi, p = ci(d)
        L.append(f"        RD_mod - RD_new coupling {pt['RD_mod'] - pt['RD_new']:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")

    L.append("\n## 2. Both-correct control: questions where both traces are correct, paired RD - LKD")
    for a in ("RD_mod", "RD_new"):
        bq = [q for q in qs if F[(a, q)]["correct"] and F[("LKD", q)]["correct"]]
        L.append(f"  {a} vs LKD, n={len(bq)}")
        for k in FEATS:
            d = np.array([F[(a, q)][k] - F[("LKD", q)][k] for q in bq])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
            lo, hi, p = ci(bs)
            L.append(f"    {k:5s} {np.mean([F[(a, q)][k] for q in bq]):8.4f} vs {np.mean([F[('LKD', q)][k] for q in bq]):8.4f}"
                     f"  diff {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
    out = "\n".join(L)
    print(out)
    (BONDS / "reflection_coupling.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
