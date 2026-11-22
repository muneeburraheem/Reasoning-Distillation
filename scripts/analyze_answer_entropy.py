"""Family D (RUN_LOG 8n): answer-token entropy, RD vs LKD, self- and cross-scored.

From eval_outputs/rd_newsae/eval_mod/answer_entropy/rows.json (scripts/eval_answer_entropy.py):
  self-scored   each model reads its own traces (what the model "felt" when it
                wrote the answer). Per question: mean over that model's samples;
                RD - LKD paired over questions.
  cross-scored  both models read the SAME text; RD-scorer minus LKD-scorer
                paired per text, separately for texts written by each model.
                This isolates the model from the text.
Metrics: first_ent (first answer token), ans_ent (mean over answer tokens),
trace_ent (mean over the whole generated trace). Also split by correct/wrong.
Holm adjustment across the primary tests (D1, D2 self and cross).

Run: python scripts/analyze_answer_entropy.py [eval_outputs/rd_newsae/eval_mod/answer_entropy]
"""

import json
import random
import statistics as st
import sys
from collections import defaultdict

PATH = (sys.argv[1] if len(sys.argv) > 1 else "eval_outputs/rd_newsae/eval_mod/answer_entropy") + "/rows.json"
rows = json.load(open(PATH))


def boot(d, b=10000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    m = sorted(sum(rng.choices(d, k=n)) / n for _ in range(b))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / b
    return sum(d) / n, m[int(0.025 * b)], m[int(0.975 * b)], min(1.0, max(p, 1 / b))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


by = defaultdict(list)
for r in rows:
    if "first_ent" in r:
        by[(r["scorer"], r["author"])].append(r)

print("Means (texts with a final box), scorer reading author's texts:")
for (sc, au), rs in sorted(by.items()):
    print(f"  {sc:6s} reads {au:6s}: n={len(rs):5d} first_ent {st.mean(r['first_ent'] for r in rs):.4f}"
          f"  ans_ent {st.mean(r['ans_ent'] for r in rs):.4f}  trace_ent {st.mean(r['trace_ent'] for r in rs):.4f}")

tests = []
for rd in sorted({r["scorer"] for r in rows} - {"LKD"}):  # rows: RD_new (October RD), RD_mod
    print(f"\n== {rd} vs LKD")
    for metric in ("first_ent", "ans_ent", "trace_ent"):
        for subset in ("all", "correct", "wrong"):
            keep = (lambda r: True) if subset == "all" else (lambda r, s=subset: r["correct"] == (s == "correct"))
            qa, qb = defaultdict(list), defaultdict(list)
            for r in by[(rd, rd)]:
                if keep(r):
                    qa[r["qid"]].append(r[metric])
            for r in by[("LKD", "LKD")]:
                if keep(r):
                    qb[r["qid"]].append(r[metric])
            ids = sorted(set(qa) & set(qb))
            if len(ids) < 10:
                continue
            m, lo, hi, p = boot([st.mean(qa[i]) - st.mean(qb[i]) for i in ids])
            print(f"  self   {metric:9s} {subset:7s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} (n_q={len(ids)})")
            if subset == "all" and metric != "trace_ent":
                tests.append((f"{rd} self {metric}", p))
        for author in (rd, "LKD"):
            a = {(r["qid"], r["idx"]): r[metric] for r in by[(rd, author)]}
            b = {(r["qid"], r["idx"]): r[metric] for r in by[("LKD", author)]}
            keys = sorted(set(a) & set(b))
            m, lo, hi, p = boot([a[k] - b[k] for k in keys])
            print(f"  cross  {metric:9s} on {author:6s} texts: {rd}-scorer minus LKD-scorer {m:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} (n={len(keys)})")
            if metric != "trace_ent":
                tests.append((f"{rd} cross {metric} on {author}", p))

print("\nHolm-adjusted p (primary answer-entropy tests):")
for (name, p), pa in zip(tests, holm([p for _, p in tests])):
    print(f"  {name:40s} p={p:.4f}  Holm {pa:.4f}")
