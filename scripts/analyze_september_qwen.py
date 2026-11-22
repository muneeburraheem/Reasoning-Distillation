"""September benchmark set (GSM8K, MMLU, ARC-C, HellaSwag): new RD vs LKD and base.

These runs had no answer-stopper, so models keep writing after their answer
until the cap and truncation is not meaningful; the harness's own per-question
`correct` is the measure. Paired difference with a bootstrap CI over questions
and an exact sign test on discordant questions.

Run: python scripts/analyze_september_qwen.py
"""

import json
import os
import random
from math import comb
from pathlib import Path

OLD = Path("eval_outputs/september/20260922T180514Z-1112953")
NEW = Path("eval_outputs/september_rd_october")
NAMES = {"RD_new": "qwen-reasondistill-october-continue-epoch_1-{m}",
         "LKD": "qwen-logit-kd-continue-epoch_1-{m}", "base": "qwen-3.5-2b-{m}"}
DIRS = {"RD_new": NEW, "LKD": OLD, "base": OLD}
# RD_MOD_SET=1: new-SAE RD moderate (docs/rd_newsae.md) as "RD_new"; October RD as a third reference.
if os.environ.get("RD_MOD_SET"):
    NAMES = {"RD_new": "qwen-rd-newsae-moderate-final-{m}", "LKD": NAMES["LKD"], "base": NAMES["base"],
             "RD_oct": NAMES["RD_new"]}
    DIRS = {"RD_new": Path("eval_outputs/rd_newsae/eval_mod/september"),
            "LKD": OLD, "base": OLD, "RD_oct": NEW}


def load(path):
    return {r["id"]: bool(r["candidates"][0]["correct"]) for r in json.load(open(path))["records"]}


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(d, b=20000):
    rng = random.Random(0)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(b))
    return sum(d) / len(d), m[int(0.025 * b)], m[int(0.975 * b)]


for mode in ("think", "no_think"):
    for bench in ("gsm8k", "mmlu", "arc-c", "hellaswag"):
        paths = {k: DIRS[k] / mode / f"{NAMES[k].format(m=mode)}__{bench}.json" for k in NAMES}
        if not all(p.exists() for p in paths.values()):
            continue
        res = {k: load(p) for k, p in paths.items()}
        ids = sorted(set.intersection(*(set(v) for v in res.values())))
        acc = {k: sum(v[i] for i in ids) / len(ids) for k, v in res.items()}
        line = f"{mode:8s} {bench:9s} n={len(ids)} " + " ".join(f"{k}={v:.3f}" for k, v in acc.items())
        for ref in [k for k in NAMES if k != "RD_new"]:
            a = sum(res["RD_new"][i] and not res[ref][i] for i in ids)
            b = sum(res[ref][i] and not res["RD_new"][i] for i in ids)
            m, lo, hi = boot([int(res["RD_new"][i]) - int(res[ref][i]) for i in ids])
            line += f" | RD-{ref} {100 * m:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] ({a} vs {b}, p={sign_p(a, b):.3f})"
        print(line)
