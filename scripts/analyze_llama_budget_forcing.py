"""Analysis of scripts/eval_llama_budget_forcing.py (Llama arm version of RUN_LOG 8i).

Per bench tag and budget: hard_strict / hard_lenient / forced accuracy per model (4 samples),
RD - LKD paired per question (bootstrap over questions); forced-only accuracy (samples that had to
be forced) on questions where both models have one; end-state mix.

Run: python scripts/analyze_llama_budget_forcing.py [out_dir]
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_outputs/llama_budget_forcing")
M = ("logit_kd_epoch2", "reasondistill_epoch2")


def boot(d, B=5000):
    rng = random.Random(0)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / B
    return sum(d) / len(d), m[int(.025 * B)], m[int(.975 * B)], min(1.0, max(p, 1 / B))


def main():
    runs = {}
    for f in D.glob("*__B*.json"):
        m, tag, b = re.fullmatch(r"(.+?)__(.+)__B(\d+)\.json", f.name).groups()
        runs[(m, tag, int(b))] = {r["id"]: r for r in json.load(open(f))["records"]}
    for tag in sorted({t for _, t, _ in runs}):
        print(f"\n######## {tag}")
        for B in sorted({b for _, t, b in runs if t == tag}):
            if not all((m, tag, B) in runs for m in M):
                continue
            R = [runs[(m, tag, B)] for m in M]
            qs = sorted(set(R[0]) & set(R[1]))
            line = [f"B={B:5d} n={len(qs)}"]
            for k in ("hard_strict", "hard_lenient", "forced"):
                per = [{q: sum(c[k] for c in r[q]["candidates"]) / len(r[q]["candidates"]) for q in qs} for r in R]
                mm, lo, hi, p = boot([per[1][q] - per[0][q] for q in qs])
                line.append(f"{k} {sum(per[0].values())/len(qs):.3f}/{sum(per[1].values())/len(qs):.3f} "
                            f"{100*mm:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] p={p:.4f}")
            print("  " + " | ".join(line))
            # forced-only, questions where both models had at least one forced sample
            fo = [{q: [c["forced"] for c in r[q]["candidates"] if c["state"] != "answered"] for q in qs} for r in R]
            both = [q for q in qs if fo[0][q] and fo[1][q]]
            if len(both) >= 20:
                mm, lo, hi, p = boot([sum(fo[1][q]) / len(fo[1][q]) - sum(fo[0][q]) / len(fo[0][q]) for q in both])
                a, b = (sum(sum(f[q]) / len(f[q]) for q in both) / len(both) for f in fo)
                print(f"           forced-only (both forced, n={len(both)}) {a:.3f}/{b:.3f} {100*mm:+.1f} pp "
                      f"[{100*lo:+.1f}, {100*hi:+.1f}] p={p:.4f}")
            for m, r in zip(("LKD", "RD"), R):
                c = Counter(x["state"] for q in qs for x in r[q]["candidates"])
                n = sum(c.values())
                empty = sum(not x["forced_answer"] for q in qs for x in r[q]["candidates"] if x["state"] != "answered")
                print(f"           {m} states: " + ", ".join(f"{s} {v/n:.3f}" for s, v in c.most_common())
                      + f" | empty forced answer {empty}")


if __name__ == "__main__":
    main()
